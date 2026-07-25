/**
 * @file rate_matching.cu
 * @brief 5G NR Rate Matching CUDA Implementation
 *
 * Implements circular buffer rate matching and de-rate matching
 * per 3GPP TS 38.212 Section 5.4.2.
 */

#include "rate_matching.h"
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <new>

// ============================================================================
// Rate Matcher Context
// ============================================================================

struct rate_matcher_ctx {
    nr_ldpc_config_t ldpc_cfg;
    nr_rate_match_config_t rm_cfg;

    int N;              // Circular buffer size (after puncturing: 50*Z for BG2, 66*Z for BG1)
    int N_full;         // Full codeword size (52*Z for BG2, 68*Z for BG1)
    int N_cols;         // Number of columns (52 for BG2, 68 for BG1)
    int E;              // Rate matched output length
    int k0;             // Starting position in circular buffer
    int Z;              // Lifting size
    int Kb;             // Information columns
    int Kd;             // Actual info bits (excluding filler)
    int F;              // Filler bits
    int puncture_offset; // Offset for punctured bits (2*Z)
    bool is_tx;         // TX (encoding) or RX (decoding) direction
};

// ============================================================================
// Device Constants for Bit Interleaving
// ============================================================================

// Bit interleaving pattern for different modulation orders
__device__ __constant__ int INTERLEAVE_QPSK[2] = {0, 1};
__device__ __constant__ int INTERLEAVE_16QAM[4] = {0, 2, 1, 3};
__device__ __constant__ int INTERLEAVE_64QAM[6] = {0, 3, 1, 4, 2, 5};
__device__ __constant__ int INTERLEAVE_256QAM[8] = {0, 4, 1, 5, 2, 6, 3, 7};

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * @brief Saturating atomic add for FP32 using CAS loop
 *
 * Clamps accumulated result to [-127.0f, 127.0f] to prevent decoder instability
 * when E > N_cb (rate matching repetition) causes multiple LLRs to accumulate.
 *
 * This is critical for high repetition cases where accumulated values could
 * otherwise be 3-4x the input LLR magnitude.
 *
 * @param address Pointer to float to update
 * @param val Value to add
 */
__device__ __forceinline__ void atomicAddSaturatingFloat(float* address, float val)
{
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old_val, new_val;

    do {
        old_val = *address_as_uint;
        float old_float = __uint_as_float(old_val);
        float sum = old_float + val;
        // Clamp to [-120, 120] to match srsRAN LLR_MAX (120, not 127)
        // 127 is reserved for "infinity" (fixed bits)
        sum = fmaxf(-120.0f, fminf(120.0f, sum));
        new_val = __float_as_uint(sum);
    } while (atomicCAS(address_as_uint, old_val, new_val) != old_val);
}

/**
 * @brief Saturating atomic add for FP16 using CAS loop
 *
 * Clamps accumulated result to [-127.0f, 127.0f] for decoder stability.
 *
 * @param address Pointer to __half to update
 * @param val Value to add
 */
__device__ __forceinline__ void atomicAddSaturatingHalf(__half* address, __half val)
{
#if __CUDA_ARCH__ >= 700
    // Use native fp16 atomicCAS on Volta+ (requires SM 7.0+)
    unsigned short* address_as_ushort = (unsigned short*)address;
    unsigned short old_val, new_val;

    do {
        old_val = *address_as_ushort;
        float old_float = __half2float(*(__half*)&old_val);
        float sum = old_float + __half2float(val);
        // Clamp to safe FP16 range — large enough to preserve LLR magnitudes
        // at high SNR, while preventing FP16 overflow during accumulation.
        sum = fmaxf(-2000.0f, fminf(2000.0f, sum));
        __half sum_half = __float2half(sum);
        new_val = *(unsigned short*)&sum_half;
    } while (atomicCAS(address_as_ushort, old_val, new_val) != old_val);
#else
    // Fallback for older architectures: just store (no accumulation)
    *address = val;
#endif
}

/**
 * @brief Calculate de-rate match output index with filler bit handling
 *
 * Per 3GPP TS 38.212 Section 5.4.2.1, the circular buffer has a "hole"
 * for filler bits that must be skipped during de-rate matching.
 *
 * Layout: [Systematic (Kd)] [HOLE/Filler (F)] [Parity (Ncb - K)]
 * where K = Kd + F
 *
 * The key insight is that the EFFECTIVE circular buffer size is Ncb - F,
 * since F filler bits are skipped on each wrap. The algorithm:
 * 1. Compute position in effective buffer: eff_pos = (k0_eff + inIdx) % (Ncb - F)
 * 2. Map eff_pos to actual CB position by adding F if eff_pos >= Kd
 *
 * @param inIdx Input index (0 to E-1)
 * @param Kd Actual info bits (excluding filler) in CB coordinates
 * @param F Filler bits
 * @param k0 Starting position in circular buffer
 * @param Ncb Circular buffer size
 * @return Output index in the LDPC codeword buffer (CB coordinates)
 */
__device__ __forceinline__ int derate_match_calc_index(int inIdx, int Kd, int F, int k0, int Ncb)
{
    // Effective buffer size (excluding filler bits)
    int Ncb_eff = Ncb - F;

    // Convert k0 to effective coordinates
    // If k0 >= Kd + F, we're in parity region - subtract F to get effective position
    // If k0 < Kd, k0_eff = k0 (before filler hole)
    // If Kd <= k0 < Kd + F, starting in filler (shouldn't happen per spec)
    int k0_eff;
    if (k0 < Kd) {
        k0_eff = k0;
    } else if (k0 >= Kd + F) {
        k0_eff = k0 - F;  // Convert parity position to effective coordinates
    } else {
        // Starting in filler hole - shouldn't happen, but handle it
        k0_eff = Kd;  // Jump to end of filler
    }

    // Compute position in effective (filler-excluded) circular buffer
    int eff_pos = (k0_eff + inIdx) % Ncb_eff;

    // Map effective position back to actual CB position
    // If eff_pos >= Kd, we're past where filler would be, so add F
    int outIdx;
    if (eff_pos >= Kd) {
        outIdx = eff_pos + F;
    } else {
        outIdx = eff_pos;
    }

    return outIdx;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Calculate rate match input index with filler bit handling (TX direction)
 *
 * Per 3GPP TS 38.212, filler bits are "known" values that should be skipped
 * during rate matching. This function maps output index to input position,
 * skipping the filler bit "hole".
 *
 * @param outIdx Output index (0 to E-1)
 * @param Kd Actual info bits (excluding filler)
 * @param F Filler bits
 * @param k0 Starting position in circular buffer
 * @param Ncb Circular buffer size
 * @return Input position in the LDPC codeword buffer
 */
__device__ __forceinline__ int rate_match_calc_index(int outIdx, int Kd, int F, int k0, int Ncb)
{
    // Same logic as derate_match_calc_index - symmetric operation
    return derate_match_calc_index(outIdx, Kd, F, k0, Ncb);
}

/**
 * @brief Rate matching kernel (TX direction)
 *
 * Per 3GPP TS 38.212 Section 5.4.2.1:
 * e[k] = d[(k0 + k) mod N_cb] for k = 0, 1, ..., E-1
 *
 * Selects E bits from the circular buffer (LDPC codeword) starting at k0.
 *
 * IMPORTANT: The LDPC encoder uses COLUMN-MAJOR layout:
 * - Each column (VN) has Z bits stored in Z_words = (Z+31)/32 words
 * - Total output = N * Z_words words per codeword
 * - Bit position i is at: column = i/Z, bit_in_col = i%Z,
 *   word = col * Z_words + bit_in_col/32, bit = bit_in_col%32
 */
__global__ void rate_match_tx_kernel(
    const uint32_t* __restrict__ d_encoded,
    uint32_t* __restrict__ d_output,
    int N_cb,           // Circular buffer size (after puncturing: 50*Z for BG2)
    int N_full,         // Full codeword size (52*Z for BG2)
    int k0,             // Starting position in circular buffer
    int E,              // Number of output bits
    int Z,              // Lifting size (needed for column-major addressing)
    int N_cols,         // Number of columns (52 for BG2, 68 for BG1)
    int puncture_offset, // Offset for punctured bits (2*Z)
    int Kd,             // Actual info bits in CB coordinates (for filler skipping)
    int F,              // Filler bits (for filler skipping)
    int num_cbs,
    int output_word_stride  // Output stride per CB in words (0 = use E_words)
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= E) return;

    // Use filler-skipping algorithm per 3GPP TS 38.212 Section 5.4.2.1
    // Filler bits at positions [Kd, Kd+F) in the circular buffer are NOT transmitted
    // This must match the RX de-rate matching algorithm for consistency
    int cb_pos = rate_match_calc_index(out_idx, Kd, F, k0, N_cb);

    // Add puncture offset to get position in full codeword
    int enc_pos = cb_pos + puncture_offset;

    // FLAT addressing for LDPC encoder output (uses MSB-first bit ordering within each byte)
    // Bit index enc_pos is at word enc_pos/32
    // Within the word, use MSB-first per byte: byte_in_word*8 + (7 - bit_in_byte)
    int in_word = enc_pos / 32;
    int bit_in_word = enc_pos % 32;
    int byte_in_word = bit_in_word / 8;
    int bit_in_byte = 7 - (bit_in_word % 8);  // MSB-first within byte
    int in_bit = byte_in_word * 8 + bit_in_byte;

    int input_stride = (N_full + 31) / 32;  // Flat word count for input
    uint32_t encoded_word = d_encoded[cb_idx * input_stride + in_word];
    uint32_t bit = (encoded_word >> in_bit) & 1u;

    // Write to output (flat sequential format, MSB-first within each byte)
    int out_word = out_idx / 32;
    int out_bit_in_word = out_idx % 32;
    int out_byte = out_bit_in_word / 8;
    int out_bit_in_byte = 7 - (out_bit_in_word % 8);  // MSB-first within byte
    int out_bit = out_byte * 8 + out_bit_in_byte;

    // Use configurable output stride (for uniform per-CB layout with variable E)
    int E_words = (output_word_stride > 0) ? output_word_stride : (E + 31) / 32;

    atomicOr(&d_output[cb_idx * E_words + out_word], bit << out_bit);
}

/**
 * @brief De-rate matching kernel (RX direction)
 *
 * Per 3GPP TS 38.212 Section 5.4.2.1:
 * e[k] = d[(k0 + k) mod N_cb] for k = 0, 1, ..., E-1
 *
 * This kernel does the reverse: accumulates received LLRs into the circular buffer.
 * Uses the filler-skipping algorithm from cuPHY to correctly handle the filler
 * "hole" in the systematic region - this is critical for repetition (E > N_cb).
 *
 * Filler bit LLRs are set separately after this kernel completes.
 *
 * IMPORTANT: The LDPC decoder expects COLUMN-MAJOR layout:
 * - Each column (VN) has Z LLRs stored contiguously
 * - Total buffer = N_cols * Z floats per codeword
 * - LLR position i is at: column = i/Z, offset_in_col = i%Z,
 *   buffer_index = col * Z + offset_in_col
 */
__global__ void rate_match_rx_kernel(
    const float* __restrict__ d_received_llrs,
    float* __restrict__ d_output_llrs,
    int N_cb,           // Circular buffer size (after puncturing: 50*Z for BG2)
    int N_full,         // Full LLR buffer size (52*Z for BG2)
    int k0,             // Starting position in circular buffer
    int E,              // Number of received LLRs
    int Z,              // Lifting size (needed for column-major addressing)
    int N_cols,         // Number of columns (52 for BG2, 68 for BG1)
    int puncture_offset, // Offset for punctured bits (2*Z)
    int Kd,             // Actual info bits (excluding filler) - for filler skipping
    int F,              // Filler bits - for filler skipping
    int num_cbs
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int recv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (recv_idx >= E) return;

    // Use filler-skipping algorithm to calculate output position
    // This correctly handles the filler "hole" at [Kd, Kd+F) in the circular buffer
    // Critical for repetition (E > N_cb) where we wrap around multiple times
    int cb_pos = derate_match_calc_index(recv_idx, Kd, F, k0, N_cb);

    // Add puncture offset to get position in full LLR buffer
    int dec_pos = cb_pos + puncture_offset;

    // Column-major addressing for LDPC decoder input
    // The decoder expects LLRs in column-major format: all Z LLRs for column 0,
    // then all Z LLRs for column 1, etc.
    int column = dec_pos / Z;
    int offset_in_col = dec_pos % Z;
    int llr_idx = column * Z + offset_in_col;

    // Read received LLR
    float llr = d_received_llrs[cb_idx * E + recv_idx];

    // Accumulate into output (for combining repetitions when E > N_cb).
    // No saturation for FP32 path - the decoder's own llr_clamp handles value range.
    int output_stride = N_cols * Z;
    atomicAdd(&d_output_llrs[cb_idx * output_stride + llr_idx], llr);
}

/**
 * @brief Set filler bit LLRs to +infinity (known to be 0)
 *
 * Filler bits are set to 0 before encoding and are not transmitted.
 * For the decoder, these LLRs should indicate high confidence that bit=0.
 *
 * Filler bits are at the END of the systematic region in the codeword:
 * - Encoder input bits 0 to Kd-1: actual info (TB + CRC)
 * - Encoder input bits Kd to Kd+F-1: filler bits (always 0)
 * - These map directly to codeword positions 0-Kd-1 and Kd to Kd+F-1
 *
 * NOTE: puncture_offset is NOT added here - that's for the circular buffer.
 * The decoder expects LLRs in flat codeword order where systematic bits
 * (including filler) are at positions 0 to Kb*Z-1.
 */
__global__ void set_filler_llrs_kernel(
    float* __restrict__ d_llrs,
    int Kd,             // Number of actual info bits (filler starts at position Kd in CB coords)
    int F,              // Number of filler bits
    int Z,              // Lifting size (unused but kept for consistency)
    int N_cols,         // Number of columns (52 for BG2, 68 for BG1)
    int puncture_offset, // Offset for punctured bits (2*Z), needed to convert to encoder coords
    float filler_llr,   // LLR value for filler bits (positive = bit=0)
    int num_cbs
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int filler_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (filler_idx >= F) return;

    // Filler bit position in encoder coordinates.
    // Kd is in CB coords (position where filler starts in circular buffer).
    // Add puncture_offset to convert to encoder coordinates.
    // Filler bits are at encoder positions [puncture_offset + Kd, puncture_offset + K).
    int pos = puncture_offset + Kd + filler_idx;

    // Flat LLR addressing
    int output_stride = N_cols * Z;
    d_llrs[cb_idx * output_stride + pos] = filler_llr;
}

/**
 * @brief Initialize output LLRs to zero for rate dematching
 *
 * We initialize to 0.0f because atomicAdd will add received LLRs.
 * Punctured positions that don't receive any LLRs will remain at 0.0f.
 * Those are handled separately by set_punctured_llrs_kernel after dematching.
 */
__global__ void init_llrs_zero_kernel(
    float* __restrict__ d_llrs,
    int N,
    int num_cbs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * num_cbs) return;
    d_llrs[idx] = 0.0f;
}

/**
 * @brief Set punctured column LLRs for min-sum stability
 *
 * Per srsRAN and cuPHY reference implementations, punctured bits (first 2*Z)
 * should be initialized to LLR=0.0 representing erasure. However, in the
 * layered min-sum decoder, when V2C = 0 for punctured columns, it becomes
 * the minimum and zeros out C2V messages for all other columns in that row.
 *
 * To enable proper information propagation, we use a small non-zero value
 * that prevents punctured columns from dominating the min-sum while still
 * being effectively "erasure-like". The value 0.1 is chosen to be:
 * - Much smaller than typical channel LLRs (~127)
 * - Large enough to not always be the minimum
 * - Positive (representing slight preference for bit=0)
 */
__global__ void set_punctured_llrs_kernel(
    float* __restrict__ d_llrs,
    int puncture_count,    // Number of positions to check
    int stride,            // Stride per codeblock
    int num_cbs
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= puncture_count) return;

    // Punctured positions (first 2*Z bits) have no channel information
    // Per 3GPP TS 38.212, punctured bits are always the first 2*Z bits (columns 0,1)
    // These are systematic bits that are not transmitted but must be decoded.
    // Set to 0 (erasure) - the decoder excludes punctured columns from min computation
    int pos = cb_idx * stride + idx;
    d_llrs[pos] = 0.0f;  // Erasure - decoder learns from parity checks
}

/**
 * @brief Bit interleaving for rate matching (TX)
 *
 * Per 3GPP TS 38.212 Section 5.4.2.2:
 *   e(j*Qm + i) = f(i*K + j)  for i=0..Qm-1, j=0..K-1
 * where K = E/Qm (number of symbols).
 *
 * This is a row-column transpose where input bits are viewed as a matrix
 * with Qm rows and K columns, and output reads column-by-column.
 */
__global__ void bit_interleave_kernel(
    const uint32_t* __restrict__ d_input,
    uint32_t* __restrict__ d_output,
    int E,
    int Q_m,            // Modulation order (bits per symbol)
    int num_cbs
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= E) return;

    // Per 3GPP TS 38.212 Section 5.4.2.2:
    // Output position out_idx = j*Qm + i maps to input position i*K + j
    // where j = symbol index, i = bit within symbol, K = E/Qm
    int K = E / Q_m;  // Number of symbols
    int j = out_idx / Q_m;   // Symbol index
    int i = out_idx % Q_m;   // Bit index within symbol

    // Calculate input position: i*K + j
    int in_idx = i * K + j;

    // Read input bit
    int in_word = in_idx / 32;
    int in_bit = in_idx % 32;
    int E_words = (E + 31) / 32;

    uint32_t bit = (d_input[cb_idx * E_words + in_word] >> in_bit) & 1u;

    // Write to output
    int out_word = out_idx / 32;
    int out_bit = out_idx % 32;
    atomicOr(&d_output[cb_idx * E_words + out_word], bit << out_bit);
}

/**
 * @brief Bit de-interleaving for de-rate matching (RX)
 *
 * Per 3GPP TS 38.212 Section 5.4.2.2:
 *   TX: e(j*Qm + i) = f(i*K + j)
 * The inverse is:
 *   RX: output at position i*K + j reads from input position j*Qm + i
 *
 * For output position out_idx = i*K + j:
 *   i = out_idx / K, j = out_idx % K
 *   in_idx = j*Qm + i = (out_idx % K)*Qm + (out_idx / K)
 */
__global__ void bit_deinterleave_llr_kernel(
    const float* __restrict__ d_input_llrs,
    float* __restrict__ d_output_llrs,
    int E,
    int Q_m,
    int num_cbs
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= E) return;

    // Per 3GPP TS 38.212 Section 5.4.2.2 (inverse of interleaving):
    // Output position out_idx = i*K + j maps to input position j*Qm + i
    // where i = row index (0..Qm-1), j = column index (0..K-1), K = E/Qm
    int K = E / Q_m;  // Number of symbols
    int i = out_idx / K;   // Row index in de-interleaved matrix
    int j = out_idx % K;   // Column index in de-interleaved matrix

    // Calculate input position: j*Qm + i
    int in_idx = j * Q_m + i;

    // Copy LLR
    d_output_llrs[cb_idx * E + out_idx] = d_input_llrs[cb_idx * E + in_idx];
}

// ============================================================================
// API Implementation
// ============================================================================

extern "C" {

nr_ldpc_status_t rate_matcher_create(rate_matcher_handle_t* handle) {
    if (!handle) return NR_LDPC_ERROR_INVALID_CONFIG;

    rate_matcher_ctx* ctx = new (std::nothrow) rate_matcher_ctx;
    if (!ctx) return NR_LDPC_ERROR_ALLOC_FAILED;

    memset(ctx, 0, sizeof(*ctx));
    *handle = ctx;
    return NR_LDPC_SUCCESS;
}

void rate_matcher_destroy(rate_matcher_handle_t handle) {
    delete handle;
}

int rate_matcher_compute_k0(int bg, int Z, int rv, int N_cb) {
    // Per TS 38.212 Table 5.4.2.1-2
    // k0 depends on rv, base graph, and lifting size

    // Compute k0 based on rv
    int k0;
    if (rv == 0) {
        k0 = 0;
    } else if (rv == 1) {
        k0 = (bg == 1) ?
             (17 * N_cb) / (66 * Z) * Z :
             (13 * N_cb) / (50 * Z) * Z;
    } else if (rv == 2) {
        k0 = (bg == 1) ?
             (33 * N_cb) / (66 * Z) * Z :
             (25 * N_cb) / (50 * Z) * Z;
    } else {  // rv == 3
        k0 = (bg == 1) ?
             (56 * N_cb) / (66 * Z) * Z :
             (43 * N_cb) / (50 * Z) * Z;
    }

    return k0;
}

nr_ldpc_status_t rate_matcher_configure_tx(rate_matcher_handle_t handle,
                                           const nr_ldpc_config_t* ldpc_cfg,
                                           const nr_rate_match_config_t* rm_cfg) {
    if (!handle || !ldpc_cfg || !rm_cfg) return NR_LDPC_ERROR_INVALID_CONFIG;

    handle->ldpc_cfg = *ldpc_cfg;
    handle->rm_cfg = *rm_cfg;
    handle->is_tx = true;

    handle->Z = ldpc_cfg->lifting_size;
    handle->Kb = (ldpc_cfg->base_graph == 1) ? 22 : 10;

    // Circular buffer size (after puncturing first 2*Z bits)
    // Per 3GPP TS 38.212: N_cb = N - 2*Z where N is the full codeword
    int full_var_nodes = (ldpc_cfg->base_graph == 1) ? 68 : 52;
    int punctured_nodes = 2;  // First 2*Z bits are always punctured
    int cb_var_nodes = full_var_nodes - punctured_nodes;

    handle->N_cols = full_var_nodes;              // Number of columns (for column-major addressing)
    handle->N_full = full_var_nodes * handle->Z;  // Full codeword (encoder output)
    handle->N = cb_var_nodes * handle->Z;         // Circular buffer (after puncturing)
    handle->puncture_offset = punctured_nodes * handle->Z;  // 2*Z

    // Filler bit handling per 3GPP TS 38.212
    // Kd = actual info bits (excluding filler)
    // F = filler bits
    // K = Kd + F = Kb * Z (total systematic region)
    handle->Kd = ldpc_cfg->num_info_bits;
    handle->F = ldpc_cfg->num_filler_bits;

    // Use limited buffer if specified
    int N_cb = rm_cfg->limited_buffer ? rm_cfg->N_cb : handle->N;
    N_cb = (N_cb < handle->N) ? N_cb : handle->N;

    handle->E = rm_cfg->E;
    handle->k0 = rate_matcher_compute_k0(ldpc_cfg->base_graph, handle->Z, rm_cfg->rv, N_cb);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t rate_matcher_configure_rx(rate_matcher_handle_t handle,
                                           const nr_ldpc_config_t* ldpc_cfg,
                                           const nr_rate_match_config_t* rm_cfg) {
    nr_ldpc_status_t status = rate_matcher_configure_tx(handle, ldpc_cfg, rm_cfg);
    if (status == NR_LDPC_SUCCESS) {
        handle->is_tx = false;
    }
    return status;
}

nr_ldpc_status_t rate_matcher_match(rate_matcher_handle_t handle,
                                    const uint32_t* d_encoded_bits,
                                    uint32_t* d_rate_matched,
                                    cudaStream_t stream) {
    if (!handle || !d_encoded_bits || !d_rate_matched) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    int E = handle->E;
    int N_cb = handle->N;
    int N_full = handle->N_full;
    int N_cols = handle->N_cols;
    int k0 = handle->k0;
    int Z = handle->Z;
    int Kd = handle->Kd;
    int F = handle->F;
    int puncture_offset = handle->puncture_offset;

    // Kd is in ENCODER coordinates (includes punctured region).
    // Convert to CIRCULAR BUFFER coordinates for the rate_match algorithm.
    // The filler hole is at CB positions [Kd_cb, Kd_cb + F), not encoder positions.
    int Kd_cb = Kd - puncture_offset;  // Convert Kd from encoder to CB coordinates

    // Clear output
    int E_words = (E + 31) / 32;
    cudaMemsetAsync(d_rate_matched, 0, E_words * sizeof(uint32_t), stream);

    // Launch rate matching kernel with filler-skipping algorithm
    int block_size = 256;
    dim3 grid((E + block_size - 1) / block_size, 1);

    rate_match_tx_kernel<<<grid, block_size, 0, stream>>>(
        d_encoded_bits, d_rate_matched,
        N_cb, N_full, k0, E, Z, N_cols, puncture_offset, Kd_cb, F, 1, 0);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t rate_matcher_match_batch(rate_matcher_handle_t handle,
                                           const uint32_t* d_encoded_bits,
                                           uint32_t* d_rate_matched,
                                           int num_cbs,
                                           cudaStream_t stream) {
    return rate_matcher_match_batch_strided(handle, d_encoded_bits, d_rate_matched, num_cbs, 0, stream);
}

nr_ldpc_status_t rate_matcher_match_batch_strided(rate_matcher_handle_t handle,
                                                   const uint32_t* d_encoded_bits,
                                                   uint32_t* d_rate_matched,
                                                   int num_cbs,
                                                   int output_word_stride,
                                                   cudaStream_t stream) {
    if (!handle || !d_encoded_bits || !d_rate_matched || num_cbs <= 0) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    int E = handle->E;
    int N_cb = handle->N;
    int N_full = handle->N_full;
    int N_cols = handle->N_cols;
    int k0 = handle->k0;
    int Z = handle->Z;
    int Kd = handle->Kd;
    int F = handle->F;
    int puncture_offset = handle->puncture_offset;

    // Convert Kd from encoder to circular buffer coordinates
    int Kd_cb = Kd - puncture_offset;

    // Use configured stride or default to E_words
    int E_words = (E + 31) / 32;
    int stride = (output_word_stride > 0) ? output_word_stride : E_words;

    // Clear output for all CBs (using the actual stride)
    cudaMemsetAsync(d_rate_matched, 0, num_cbs * stride * sizeof(uint32_t), stream);

    // Launch batched rate matching kernel - process ALL codeblocks in one launch!
    int block_size = 256;
    dim3 grid((E + block_size - 1) / block_size, num_cbs);

    rate_match_tx_kernel<<<grid, block_size, 0, stream>>>(
        d_encoded_bits, d_rate_matched,
        N_cb, N_full, k0, E, Z, N_cols, puncture_offset, Kd_cb, F, num_cbs, stride);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t rate_matcher_dematch(rate_matcher_handle_t handle,
                                      const float* d_received_llrs,
                                      float* d_output_llrs,
                                      cudaStream_t stream) {
    if (!handle || !d_received_llrs || !d_output_llrs) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    int E = handle->E;
    int N_cb = handle->N;
    int N_full = handle->N_full;
    int N_cols = handle->N_cols;
    int k0 = handle->k0;
    int Z = handle->Z;
    int Kd = handle->Kd;
    int F = handle->F;
    int puncture_offset = handle->puncture_offset;

    // For the derate match algorithm, Kd needs to be in CIRCULAR BUFFER coordinates
    // (i.e., after removing punctured bits from the systematic region).
    // The circular buffer layout is:
    //   [0, Kd_cb) = payload bits (in CB coordinates)
    //   [Kd_cb, Kd_cb + F) = filler hole
    //   [Kd_cb + F, N_cb) = parity bits
    // Convert Kd from encoder to circular buffer coordinates
    int Kd_cb = Kd - puncture_offset;

    // Initialize output LLRs to zero using cudaMemsetAsync (faster than kernel)
    // 0.0f in IEEE 754 is all zeros, so cudaMemset works correctly
    cudaMemsetAsync(d_output_llrs, 0, N_full * sizeof(float), stream);

    int block_size = 256;

    // Launch de-rate matching kernel with filler-skipping algorithm
    dim3 grid((E + block_size - 1) / block_size, 1);

    rate_match_rx_kernel<<<grid, block_size, 0, stream>>>(
        d_received_llrs, d_output_llrs,
        N_cb, N_full, k0, E, Z, N_cols, puncture_offset, Kd_cb, F, 1);

    // Set punctured positions (first 2*Z) to 0 - uninformative prior
    // The decoder excludes these from min computation and learns values via parity
    {
        int puncture_count = 2 * Z;
        dim3 punct_grid((puncture_count + block_size - 1) / block_size, 1);
        set_punctured_llrs_kernel<<<punct_grid, block_size, 0, stream>>>(
            d_output_llrs, puncture_count, N_full, 1);
    }

    // Set filler bit LLRs to +infinity (known to be 0)
    // Filler bits are at encoder positions Kd to K-1 (= Kd to Kd + F - 1)
    // Use very high value (10000) to match cuPHY/Aerial reference implementation
    // IMPORTANT: Pass Kd_cb (circular buffer coords) to the kernel, not Kd (encoder coords).
    // The kernel adds puncture_offset internally to convert to encoder coords.
    if (F > 0) {
        dim3 filler_grid((F + block_size - 1) / block_size, 1);
        float filler_llr = 10000.0f;  // High positive value = bit is definitely 0
        set_filler_llrs_kernel<<<filler_grid, block_size, 0, stream>>>(
            d_output_llrs, Kd_cb, F, Z, N_cols, puncture_offset, filler_llr, 1);
    }

    return NR_LDPC_SUCCESS;
}

int rate_matcher_get_output_bits(rate_matcher_handle_t handle) {
    return handle ? handle->E : 0;
}

nr_ldpc_status_t rate_matcher_interleave(rate_matcher_handle_t handle,
                                          const uint32_t* d_input,
                                          uint32_t* d_output,
                                          int E,
                                          int Q_m,
                                          cudaStream_t stream) {
    if (!handle || !d_input || !d_output) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // Per 3GPP TS 38.212 Section 5.4.2.2, interleaving applies for Q_m >= 2
    // (QPSK, 16QAM, 64QAM, 256QAM)
    if (Q_m < 2 || Q_m > 8) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // Clear output
    int E_words = (E + 31) / 32;
    cudaMemsetAsync(d_output, 0, E_words * sizeof(uint32_t), stream);

    // Launch interleaving kernel
    int block_size = 256;
    dim3 grid((E + block_size - 1) / block_size, 1);

    bit_interleave_kernel<<<grid, block_size, 0, stream>>>(
        d_input, d_output, E, Q_m, 1);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t rate_matcher_deinterleave_llr(rate_matcher_handle_t handle,
                                                const float* d_input_llrs,
                                                float* d_output_llrs,
                                                int E,
                                                int Q_m,
                                                cudaStream_t stream) {
    if (!handle || !d_input_llrs || !d_output_llrs) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // Per 3GPP TS 38.212 Section 5.4.2.2, interleaving applies for Q_m >= 2
    // (QPSK, 16QAM, 64QAM, 256QAM)
    if (Q_m < 2 || Q_m > 8) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // Launch de-interleaving kernel
    int block_size = 256;
    dim3 grid((E + block_size - 1) / block_size, 1);

    bit_deinterleave_llr_kernel<<<grid, block_size, 0, stream>>>(
        d_input_llrs, d_output_llrs, E, Q_m, 1);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t rate_matcher_interleave_batch(rate_matcher_handle_t handle,
                                                const uint32_t* d_input,
                                                uint32_t* d_output,
                                                int E,
                                                int Q_m,
                                                int num_cbs,
                                                cudaStream_t stream) {
    if (!handle || !d_input || !d_output) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // Per 3GPP TS 38.212 Section 5.4.2.2, interleaving applies for Q_m >= 2
    // (QPSK, 16QAM, 64QAM, 256QAM)
    if (Q_m < 2 || Q_m > 8) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    int E_words = (E + 31) / 32;
    cudaMemsetAsync(d_output, 0, num_cbs * E_words * sizeof(uint32_t), stream);

    int block_size = 256;
    dim3 grid((E + block_size - 1) / block_size, num_cbs);

    bit_interleave_kernel<<<grid, block_size, 0, stream>>>(
        d_input, d_output, E, Q_m, num_cbs);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t rate_matcher_deinterleave_llr_batch(rate_matcher_handle_t handle,
                                                      const float* d_input_llrs,
                                                      float* d_output_llrs,
                                                      int E,
                                                      int Q_m,
                                                      int num_cbs,
                                                      cudaStream_t stream) {
    if (!handle || !d_input_llrs || !d_output_llrs) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // Per 3GPP TS 38.212 Section 5.4.2.2, interleaving applies for Q_m >= 2
    // (QPSK, 16QAM, 64QAM, 256QAM)
    if (Q_m < 2 || Q_m > 8) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    int block_size = 256;
    dim3 grid((E + block_size - 1) / block_size, num_cbs);

    bit_deinterleave_llr_kernel<<<grid, block_size, 0, stream>>>(
        d_input_llrs, d_output_llrs, E, Q_m, num_cbs);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t rate_matcher_dematch_batch(rate_matcher_handle_t handle,
                                             const float* d_received_llrs,
                                             float* d_output_llrs,
                                             int num_cbs,
                                             cudaStream_t stream) {
    if (!handle || !d_received_llrs || !d_output_llrs || num_cbs <= 0) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    int E = handle->E;
    int N_cb = handle->N;
    int N_full = handle->N_full;
    int N_cols = handle->N_cols;
    int k0 = handle->k0;
    int Z = handle->Z;
    int Kd = handle->Kd;
    int F = handle->F;
    int puncture_offset = handle->puncture_offset;

    // Kd = num_info_bits = K - F, in ENCODER coordinates (includes punctured).
    // Convert to CIRCULAR BUFFER coordinates for the derate_match algorithm.
    // The filler hole is at CB positions [Kd_cb, Kd_cb + F).
    int Kd_cb = Kd - puncture_offset;

    int block_size = 256;

    // Initialize output LLRs to zero using cudaMemsetAsync (faster than kernel)
    // 0.0f in IEEE 754 is all zeros, so cudaMemset works correctly
    int total_llrs = N_full * num_cbs;
    cudaMemsetAsync(d_output_llrs, 0, total_llrs * sizeof(float), stream);

    // Launch de-rate matching kernel for all CBs
    dim3 grid((E + block_size - 1) / block_size, num_cbs);

    rate_match_rx_kernel<<<grid, block_size, 0, stream>>>(
        d_received_llrs, d_output_llrs,
        N_cb, N_full, k0, E, Z, N_cols, puncture_offset, Kd_cb, F, num_cbs);

    // Set punctured positions (first 2*Z) to 0 - uninformative prior
    // The decoder excludes these from min computation and learns values via parity
    {
        int puncture_count = 2 * Z;
        dim3 punct_grid((puncture_count + block_size - 1) / block_size, num_cbs);
        set_punctured_llrs_kernel<<<punct_grid, block_size, 0, stream>>>(
            d_output_llrs, puncture_count, N_full, num_cbs);
    }

    // Set filler bit LLRs to +infinity (known to be 0)
    // Use very high value (10000) to match cuPHY/Aerial reference implementation
    // IMPORTANT: Pass Kd_cb (circular buffer coords) to the kernel, not Kd (encoder coords).
    // The kernel adds puncture_offset internally to convert to encoder coords.
    if (F > 0) {
        dim3 filler_grid((F + block_size - 1) / block_size, num_cbs);
        float filler_llr = 10000.0f;
        set_filler_llrs_kernel<<<filler_grid, block_size, 0, stream>>>(
            d_output_llrs, Kd_cb, F, Z, N_cols, puncture_offset, filler_llr, num_cbs);
    }

    return NR_LDPC_SUCCESS;
}

// ============================================================================
// Half-precision (fp16) Rate Dematching Kernels
// ============================================================================

/**
 * @brief Initialize fp16 LLRs to zero
 */
__global__ void init_llrs_zero_half_kernel(
    __half* __restrict__ d_llrs,
    int N,
    int num_cbs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * num_cbs) return;
    d_llrs[idx] = __float2half(0.0f);
}

/**
 * @brief De-rate matching kernel for fp16 LLRs
 *
 * Note: Uses atomicAdd for __half which requires compute capability 7.0+
 */
__global__ void rate_match_rx_half_kernel(
    const __half* __restrict__ d_received_llrs,
    __half* __restrict__ d_output_llrs,
    int N_cb,
    int N_full,
    int k0,
    int E,
    int Z,
    int N_cols,
    int puncture_offset,
    int Kd,
    int F,
    int num_cbs,
    bool no_repetition
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int recv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (recv_idx >= E) return;

    // Use filler-skipping algorithm
    int cb_pos = derate_match_calc_index(recv_idx, Kd, F, k0, N_cb);
    int dec_pos = cb_pos + puncture_offset;

    int column = dec_pos / Z;
    int offset_in_col = dec_pos % Z;
    int llr_idx = column * Z + offset_in_col;

    __half llr = d_received_llrs[cb_idx * E + recv_idx];
    int output_stride = N_cols * Z;
    __half* out = &d_output_llrs[cb_idx * output_stride + llr_idx];

    // Use saturating add to clamp to [-127, 127] - prevents decoder instability
    // when E > N_cb causes accumulated values to exceed typical LLR range
    if (no_repetition) {
        *out = llr;
    } else {
        atomicAddSaturatingHalf(out, llr);
    }
}

/**
 * @brief Set punctured column fp16 LLRs to uninformative prior
 * The decoder excludes these from min computation and learns values via parity
 */
__global__ void set_punctured_llrs_half_kernel(
    __half* __restrict__ d_llrs,
    int puncture_count,
    int stride,
    int num_cbs
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= puncture_count) return;

    // Punctured positions have no channel info - set to 0 (erasure)
    // The decoder excludes punctured columns from min computation
    int pos = cb_idx * stride + idx;
    d_llrs[pos] = __float2half(0.0f);
}

/**
 * @brief Set filler bit fp16 LLRs
 *
 * Kd is in CB coords (position where filler starts in circular buffer).
 * Add puncture_offset to convert to encoder coordinates.
 * Filler bits are at encoder positions [puncture_offset + Kd, puncture_offset + K).
 */
__global__ void set_filler_llrs_half_kernel(
    __half* __restrict__ d_llrs,
    int Kd,
    int F,
    int Z,
    int N_cols,
    int puncture_offset,
    __half filler_llr,
    int num_cbs
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int filler_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (filler_idx >= F) return;

    // Filler bit position in encoder coordinates.
    // Kd is in CB coords (position where filler starts in circular buffer).
    // Add puncture_offset to convert to encoder coordinates.
    int pos = puncture_offset + Kd + filler_idx;
    int output_stride = N_cols * Z;
    d_llrs[cb_idx * output_stride + pos] = filler_llr;
}

nr_ldpc_status_t rate_matcher_dematch_batch_half(rate_matcher_handle_t handle,
                                                   const void* d_received_llrs_half,
                                                   void* d_output_llrs_half,
                                                   int num_cbs,
                                                   cudaStream_t stream) {
    if (!handle || !d_received_llrs_half || !d_output_llrs_half || num_cbs <= 0) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    const __half* d_in = static_cast<const __half*>(d_received_llrs_half);
    __half* d_out = static_cast<__half*>(d_output_llrs_half);

    int E = handle->E;
    int N_cb = handle->N;
    int N_full = handle->N_full;
    int N_cols = handle->N_cols;
    int k0 = handle->k0;
    int Z = handle->Z;
    int Kd = handle->Kd;
    int F = handle->F;
    int puncture_offset = handle->puncture_offset;
    // Convert Kd from encoder to circular buffer coordinates
    int Kd_cb = Kd - puncture_offset;

    int block_size = 256;

    // Initialize output LLRs to zero using cudaMemsetAsync (faster than kernel)
    // 0.0 in half precision is also all zeros, so cudaMemset works correctly
    int total_llrs = N_full * num_cbs;
    cudaMemsetAsync(d_out, 0, total_llrs * sizeof(__half), stream);

    dim3 grid((E + block_size - 1) / block_size, num_cbs);
    const bool no_repetition = E <= (N_cb - F);
    rate_match_rx_half_kernel<<<grid, block_size, 0, stream>>>(
        d_in, d_out,
        N_cb, N_full, k0, E, Z, N_cols, puncture_offset, Kd_cb, F, num_cbs, no_repetition);

    // Punctured positions (first 2*Z) are already zero from cudaMemsetAsync above.
    // Filler bit LLRs are set inside the LDPC decoder kernel's APP init phase,
    // eliminating 2 kernel launches (~6-10 us saved).

    return NR_LDPC_SUCCESS;
}

// ============================================================================
// Fused Deinterleave + Rate Dematch (FP16) — eliminates intermediate buffer
// ============================================================================

/**
 * @brief Fused deinterleave + de-rate matching kernel for FP16 LLRs
 *
 * For each recv_idx (0..E-1), applies the deinterleave permutation on-the-fly
 * to find the source position in the interleaved input, then maps recv_idx to
 * the output codeword position via derate_match_calc_index.
 */
__global__ void rate_dematch_deinterleave_half_kernel(
    const __half* __restrict__ d_received_llrs,  // Interleaved input [num_cbs * E]
    __half* __restrict__ d_output_llrs,           // Rate-dematched output [num_cbs * N_full]
    int N_cb,
    int N_full,
    int k0,
    int E,
    int Z,
    int N_cols,
    int puncture_offset,
    int Kd,
    int F,
    int Q_m,
    int K,            // = E / Q_m
    int num_cbs,
    bool no_repetition
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int recv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (recv_idx >= E) return;

    // Step 1: Deinterleave — compute source position in interleaved buffer
    // Deinterleave: for deinterleaved position recv_idx,
    //   j = recv_idx / K, i = recv_idx % K
    //   source in interleaved buffer = i * Q_m + j
    int j = recv_idx / K;
    int i = recv_idx % K;
    int src_pos = i * Q_m + j;

    // Step 2: Rate dematch — compute output position in codeword buffer
    int cb_pos = derate_match_calc_index(recv_idx, Kd, F, k0, N_cb);
    int dec_pos = cb_pos + puncture_offset;
    int column = dec_pos / Z;
    int offset_in_col = dec_pos % Z;
    int llr_idx = column * Z + offset_in_col;

    // Read from interleaved input, write to output
    __half llr = d_received_llrs[cb_idx * E + src_pos];
    int output_stride = N_cols * Z;
    __half* out = &d_output_llrs[cb_idx * output_stride + llr_idx];
    if (no_repetition) {
        *out = llr;
    } else {
        atomicAddSaturatingHalf(out, llr);
    }
}

/**
 * @brief Fused descramble + deinterleave + de-rate matching kernel for FP16 LLRs
 *
 * Same as rate_dematch_deinterleave_half_kernel, but also applies descrambling
 * inline (sign flip based on scrambling sequence bit), eliminating a separate
 * descramble kernel launch and saving one full read+write pass over the input.
 */
__global__ void descramble_rate_dematch_deinterleave_half_kernel(
    const __half* __restrict__ d_received_llrs,  // Scrambled interleaved input [num_cbs * E]
    __half* __restrict__ d_output_llrs,           // Rate-dematched output [num_cbs * N_full]
    const uint32_t* __restrict__ d_scramble_seq,  // Packed scrambling sequence
    int scramble_offset,                          // Bit offset into scramble sequence
    int N_cb,
    int N_full,
    int k0,
    int E,
    int Z,
    int N_cols,
    int puncture_offset,
    int Kd,
    int F,
    int Q_m,
    int K,            // = E / Q_m
    int num_cbs,
    bool no_repetition
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int recv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (recv_idx >= E) return;

    // Step 1: Deinterleave — compute source position in interleaved buffer
    int j = recv_idx / K;
    int i = recv_idx % K;
    int src_pos = i * Q_m + j;

    // Step 2: Read input LLR and apply descrambling inline.
    // The scrambling sequence is indexed by the global bit position in the
    // interleaved domain (cb_idx * E + src_pos), offset by scramble_offset.
    int global_pos = scramble_offset + cb_idx * E + src_pos;
    int word_idx = global_pos / 32;
    int bit_pos = 31 - (global_pos % 32);  // MSB-first to match gold sequence generator
    uint32_t seq_bit = (d_scramble_seq[word_idx] >> bit_pos) & 1;

    __half llr = d_received_llrs[cb_idx * E + src_pos];
    if (seq_bit) {
        // Flip sign bit of fp16 value (bit 15)
        llr = __ushort_as_half(__half_as_ushort(llr) ^ 0x8000u);
    }

    // Step 3: Rate dematch — compute output position in codeword buffer
    int cb_pos = derate_match_calc_index(recv_idx, Kd, F, k0, N_cb);
    int dec_pos = cb_pos + puncture_offset;
    int column = dec_pos / Z;
    int offset_in_col = dec_pos % Z;
    int llr_idx = column * Z + offset_in_col;

    int output_stride = N_cols * Z;
    __half* out = &d_output_llrs[cb_idx * output_stride + llr_idx];
    if (no_repetition) {
        *out = llr;
    } else {
        atomicAddSaturatingHalf(out, llr);
    }
}

/**
 * @brief Fused descramble + de-rate matching kernel for FP16 LLRs (no interleaving)
 *
 * For BPSK (Q_m <= 1) where no bit interleaving is used.
 */
__global__ void descramble_rate_dematch_half_kernel(
    const __half* __restrict__ d_received_llrs,
    __half* __restrict__ d_output_llrs,
    const uint32_t* __restrict__ d_scramble_seq,
    int scramble_offset,
    int N_cb,
    int N_full,
    int k0,
    int E,
    int Z,
    int N_cols,
    int puncture_offset,
    int Kd,
    int F,
    int num_cbs,
    bool no_repetition
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int recv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (recv_idx >= E) return;

    // Descramble inline
    int global_pos = scramble_offset + cb_idx * E + recv_idx;
    int word_idx = global_pos / 32;
    int bit_pos = 31 - (global_pos % 32);
    uint32_t seq_bit = (d_scramble_seq[word_idx] >> bit_pos) & 1;

    __half llr = d_received_llrs[cb_idx * E + recv_idx];
    if (seq_bit) {
        llr = __ushort_as_half(__half_as_ushort(llr) ^ 0x8000u);
    }

    // Rate dematch
    int cb_pos = derate_match_calc_index(recv_idx, Kd, F, k0, N_cb);
    int dec_pos = cb_pos + puncture_offset;
    int column = dec_pos / Z;
    int offset_in_col = dec_pos % Z;
    int llr_idx = column * Z + offset_in_col;

    int output_stride = N_cols * Z;
    __half* out = &d_output_llrs[cb_idx * output_stride + llr_idx];
    if (no_repetition) {
        *out = llr;
    } else {
        atomicAddSaturatingHalf(out, llr);
    }
}

nr_ldpc_status_t rate_matcher_deinterleave_and_dematch_batch_half(rate_matcher_handle_t handle,
                                                                    const void* d_received_llrs_half,
                                                                    void* d_output_llrs_half,
                                                                    int Q_m,
                                                                    int num_cbs,
                                                                    cudaStream_t stream,
                                                                    const unsigned int* d_scramble_sequence,
                                                                    int scramble_offset) {
    if (!handle || !d_received_llrs_half || !d_output_llrs_half || num_cbs <= 0) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    if (Q_m < 1 || Q_m > 8) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    const __half* d_in = static_cast<const __half*>(d_received_llrs_half);
    __half* d_out = static_cast<__half*>(d_output_llrs_half);

    int E = handle->E;
    int N_cb = handle->N;
    int N_full = handle->N_full;
    int N_cols = handle->N_cols;
    int k0 = handle->k0;
    int Z = handle->Z;
    int Kd = handle->Kd;
    int F = handle->F;
    int puncture_offset = handle->puncture_offset;
    int Kd_cb = Kd - puncture_offset;
    int K = (Q_m > 1) ? (E / Q_m) : E;
    const bool no_repetition = E <= (N_cb - F);

    int block_size = 256;

    // Initialize output LLRs to zero
    int total_llrs = N_full * num_cbs;
    cudaMemsetAsync(d_out, 0, total_llrs * sizeof(__half), stream);

    if (d_scramble_sequence != nullptr) {
        // Fused descramble + deinterleave + rate dematch (saves a kernel launch + memory pass)
        dim3 grid((E + block_size - 1) / block_size, num_cbs);
        if (Q_m <= 1) {
            descramble_rate_dematch_half_kernel<<<grid, block_size, 0, stream>>>(
                d_in, d_out, d_scramble_sequence, scramble_offset,
                N_cb, N_full, k0, E, Z, N_cols, puncture_offset, Kd_cb, F, num_cbs, no_repetition);
        } else {
            descramble_rate_dematch_deinterleave_half_kernel<<<grid, block_size, 0, stream>>>(
                d_in, d_out, d_scramble_sequence, scramble_offset,
                N_cb, N_full, k0, E, Z, N_cols, puncture_offset, Kd_cb, F,
                Q_m, K, num_cbs, no_repetition);
        }
    } else {
        // No descrambling — use original kernels
        dim3 grid((E + block_size - 1) / block_size, num_cbs);
        if (Q_m <= 1) {
            rate_match_rx_half_kernel<<<grid, block_size, 0, stream>>>(
                d_in, d_out,
                N_cb, N_full, k0, E, Z, N_cols, puncture_offset, Kd_cb, F, num_cbs, no_repetition);
        } else {
            rate_dematch_deinterleave_half_kernel<<<grid, block_size, 0, stream>>>(
                d_in, d_out,
                N_cb, N_full, k0, E, Z, N_cols, puncture_offset, Kd_cb, F,
                Q_m, K, num_cbs, no_repetition);
        }
    }

    // Punctured positions (first 2*Z) are already zero from cudaMemsetAsync above.
    // Filler bit LLRs are set inside the LDPC decoder kernel's APP init phase,
    // eliminating 2 kernel launches (~6-10 us saved).

    return NR_LDPC_SUCCESS;
}

int rate_matcher_get_full_size(rate_matcher_handle_t handle) {
    return handle ? handle->N_full : 0;
}

} // extern "C"
