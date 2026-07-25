/**
 * @file pdsch_fused.cu
 * @brief Fused PDSCH TX kernels implementation
 *
 * High-efficiency TX pipeline kernels that combine:
 * - Scrambling (XOR with Gold sequence)
 * - Modulation (bit-to-symbol mapping)
 * - FP16 output (reduced memory bandwidth)
 */

#include "../include/pdsch_fused.h"
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <cstdio>
#include <mutex>

/* ============================================================================
 * LFSR Jump Tables for on-the-fly Gold sequence generation
 * NOTE: These are defined here (not extern) because CUDA __constant__
 * memory is module-local and cannot be linked across compilation units.
 * ============================================================================ */
#define NC_SKIP 1600
#define LFSR_BITS 31
#define MAX_JUMP_POWER 24

__constant__ uint32_t d_x1_jump_pdsch[MAX_JUMP_POWER + 1][LFSR_BITS];
__constant__ uint32_t d_x2_jump_pdsch[MAX_JUMP_POWER + 1][LFSR_BITS];

static std::once_flag g_pdsch_fused_jump_tables_init_flag;

static void matrix_multiply_gf2_pdsch(const uint32_t A[LFSR_BITS],
                                       const uint32_t B[LFSR_BITS],
                                       uint32_t C[LFSR_BITS]) {
    uint32_t B_T[LFSR_BITS] = {0};
    for (int i = 0; i < LFSR_BITS; i++) {
        for (int j = 0; j < LFSR_BITS; j++) {
            if (B[j] & (1u << i)) {
                B_T[i] |= (1u << j);
            }
        }
    }
    for (int i = 0; i < LFSR_BITS; i++) {
        C[i] = 0;
        for (int j = 0; j < LFSR_BITS; j++) {
            uint32_t dot = A[i] & B_T[j];
            if (__builtin_popcount(dot) & 1) {
                C[i] |= (1u << j);
            }
        }
    }
}

static void init_x1_base_matrix_pdsch(uint32_t M[LFSR_BITS]) {
    for (int i = 0; i < LFSR_BITS; i++) M[i] = 0;
    for (int i = 0; i < LFSR_BITS - 1; i++) M[i] = 1u << (i + 1);
    M[LFSR_BITS - 1] = (1u << 0) | (1u << 3);
}

static void init_x2_base_matrix_pdsch(uint32_t M[LFSR_BITS]) {
    for (int i = 0; i < LFSR_BITS; i++) M[i] = 0;
    for (int i = 0; i < LFSR_BITS - 1; i++) M[i] = 1u << (i + 1);
    M[LFSR_BITS - 1] = (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3);
}

static cudaError_t do_initialize_pdsch_jump_tables() {
    uint32_t h_x1_jump[MAX_JUMP_POWER + 1][LFSR_BITS];
    uint32_t h_x2_jump[MAX_JUMP_POWER + 1][LFSR_BITS];

    init_x1_base_matrix_pdsch(h_x1_jump[0]);
    init_x2_base_matrix_pdsch(h_x2_jump[0]);

    for (int p = 1; p <= MAX_JUMP_POWER; p++) {
        matrix_multiply_gf2_pdsch(h_x1_jump[p-1], h_x1_jump[p-1], h_x1_jump[p]);
        matrix_multiply_gf2_pdsch(h_x2_jump[p-1], h_x2_jump[p-1], h_x2_jump[p]);
    }

    cudaError_t err;
    err = cudaMemcpyToSymbol(d_x1_jump_pdsch, h_x1_jump, sizeof(h_x1_jump));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyToSymbol(d_x2_jump_pdsch, h_x2_jump, sizeof(h_x2_jump));
    return err;
}

static cudaError_t ensure_pdsch_jump_tables_initialized() {
    static cudaError_t init_result = cudaSuccess;
    std::call_once(g_pdsch_fused_jump_tables_init_flag, [&]() {
        init_result = do_initialize_pdsch_jump_tables();
    });
    return init_result;
}

__device__ __forceinline__ uint32_t apply_jump_matrix_pdsch(uint32_t state,
                                                             const uint32_t jump[LFSR_BITS]) {
    uint32_t result = 0;
    #pragma unroll
    for (int i = 0; i < LFSR_BITS; i++) {
        uint32_t masked = state & jump[i];
        if (__popc(masked) & 1) {
            result |= (1u << i);
        }
    }
    return result;
}

__device__ __forceinline__ uint32_t advance_x1_pdsch(uint32_t x1, int n) {
    while (n) {
        int p = __ffs(n) - 1;
        x1 = apply_jump_matrix_pdsch(x1, d_x1_jump_pdsch[p]);
        n &= (n - 1);
    }
    return x1;
}

__device__ __forceinline__ uint32_t advance_x2_pdsch(uint32_t x2, int n) {
    while (n) {
        int p = __ffs(n) - 1;
        x2 = apply_jump_matrix_pdsch(x2, d_x2_jump_pdsch[p]);
        n &= (n - 1);
    }
    return x2;
}

__device__ __forceinline__ uint32_t step_x1_pdsch(uint32_t x1) {
    uint32_t new_bit = ((x1 >> 3) ^ x1) & 1;
    return (x1 >> 1) | (new_bit << 30);
}

__device__ __forceinline__ uint32_t step_x2_pdsch(uint32_t x2) {
    uint32_t new_bit = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
    return (x2 >> 1) | (new_bit << 30);
}

// ============================================================================
// Device Constants - Constellation Points (Gray coded, normalized)
// ============================================================================

// QPSK constellation: 1/sqrt(2)
__device__ __constant__ float PDSCH_QPSK_I[4] = { 0.7071068f, -0.7071068f,  0.7071068f, -0.7071068f};
__device__ __constant__ float PDSCH_QPSK_Q[4] = { 0.7071068f,  0.7071068f, -0.7071068f, -0.7071068f};

// 16QAM (normalized by 1/sqrt(10))
__device__ __constant__ float PDSCH_QAM16_TABLE[8] = {
     0.316227766f, -0.316227766f,  0.316227766f, -0.316227766f,
     0.948683298f, -0.948683298f,  0.948683298f, -0.948683298f
};

// 64QAM (normalized by 1/sqrt(42))
__device__ __constant__ float PDSCH_QAM64_TABLE[8] = {
     0.462910049886276f, -0.462910049886276f,
     0.77151674981046f,  -0.77151674981046f,
     0.154303349962092f, -0.154303349962092f,
     1.08012344973464f,  -1.08012344973464f
};

// 256QAM (normalized by 1/sqrt(170))
__device__ __constant__ float PDSCH_QAM256_TABLE[16] = {
     0.383482494f, -0.383482494f,  0.843661488f, -0.843661488f,
     0.230089497f, -0.230089497f,  0.997054486f, -0.997054486f,
     0.536875492f, -0.536875492f,  0.69026849f,  -0.69026849f,
     0.076696499f, -0.076696499f,  1.150447483f, -1.150447483f
};

// ============================================================================
// Fused Scramble + Modulate Kernels (FP16 output)
// ============================================================================

/**
 * @brief Fused QPSK scramble + modulate with FP16 output
 */
__global__ void __launch_bounds__(256, 4) fused_scramble_qpsk_half_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    __half2* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 2;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    // Read input bits
    uint32_t bits;
    if (bit_pos <= 30) {
        bits = (d_bits[word_idx] >> bit_pos) & 0x3;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0x3;
    }

    // Read scrambling sequence
    uint32_t scr;
    if (bit_pos <= 30) {
        scr = (d_scramble_seq[word_idx] >> bit_pos) & 0x3;
    } else {
        scr = ((d_scramble_seq[word_idx] >> bit_pos) | (d_scramble_seq[word_idx + 1] << (32 - bit_pos))) & 0x3;
    }

    // XOR to scramble
    uint32_t scrambled = bits ^ scr;

    // Modulate to FP16 complex
    d_symbols[idx] = __floats2half2_rn(PDSCH_QPSK_I[scrambled], PDSCH_QPSK_Q[scrambled]);
}

/**
 * @brief Fused 16QAM scramble + modulate with FP16 output
 */
__global__ void __launch_bounds__(256, 4) fused_scramble_16qam_half_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    __half2* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 4;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    // Read input bits
    uint32_t bits;
    if (bit_pos <= 28) {
        bits = (d_bits[word_idx] >> bit_pos) & 0xF;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0xF;
    }

    // Read scrambling sequence
    uint32_t scr;
    if (bit_pos <= 28) {
        scr = (d_scramble_seq[word_idx] >> bit_pos) & 0xF;
    } else {
        scr = ((d_scramble_seq[word_idx] >> bit_pos) | (d_scramble_seq[word_idx + 1] << (32 - bit_pos))) & 0xF;
    }

    // XOR to scramble
    uint32_t scrambled = bits ^ scr;

    // Map to constellation
    int i_idx = scrambled & 0x5;
    int q_idx = (scrambled >> 1) & 0x5;

    d_symbols[idx] = __floats2half2_rn(PDSCH_QAM16_TABLE[i_idx], PDSCH_QAM16_TABLE[q_idx]);
}

/**
 * @brief Fused 64QAM scramble + modulate with FP16 output
 */
__global__ void __launch_bounds__(256, 4) fused_scramble_64qam_half_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    __half2* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 6;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    // Read input bits
    uint32_t bits;
    if (bit_pos <= 26) {
        bits = (d_bits[word_idx] >> bit_pos) & 0x3F;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0x3F;
    }

    // Read scrambling sequence
    uint32_t scr;
    if (bit_pos <= 26) {
        scr = (d_scramble_seq[word_idx] >> bit_pos) & 0x3F;
    } else {
        scr = ((d_scramble_seq[word_idx] >> bit_pos) | (d_scramble_seq[word_idx + 1] << (32 - bit_pos))) & 0x3F;
    }

    // XOR to scramble
    uint32_t scrambled = bits ^ scr;

    // Map to constellation
    int i_idx = (scrambled & 0x1) | ((scrambled & 0x4) >> 1) | ((scrambled & 0x10) >> 2);
    int q_idx = ((scrambled >> 1) & 0x1) | (((scrambled >> 1) & 0x4) >> 1) | (((scrambled >> 1) & 0x10) >> 2);

    d_symbols[idx] = __floats2half2_rn(PDSCH_QAM64_TABLE[i_idx], PDSCH_QAM64_TABLE[q_idx]);
}

/**
 * @brief Fused 256QAM scramble + modulate with FP16 output
 */
__global__ void __launch_bounds__(256, 4) fused_scramble_256qam_half_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    __half2* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 8;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    // Read input bits
    uint32_t bits;
    if (bit_pos <= 24) {
        bits = (d_bits[word_idx] >> bit_pos) & 0xFF;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0xFF;
    }

    // Read scrambling sequence
    uint32_t scr;
    if (bit_pos <= 24) {
        scr = (d_scramble_seq[word_idx] >> bit_pos) & 0xFF;
    } else {
        scr = ((d_scramble_seq[word_idx] >> bit_pos) | (d_scramble_seq[word_idx + 1] << (32 - bit_pos))) & 0xFF;
    }

    // XOR to scramble
    uint32_t scrambled = bits ^ scr;

    // Map to constellation
    int i_idx = (scrambled & 0x1) | ((scrambled & 0x4) >> 1) | ((scrambled & 0x10) >> 2) | ((scrambled & 0x40) >> 3);
    int q_idx = ((scrambled >> 1) & 0x1) | (((scrambled >> 1) & 0x4) >> 1) | (((scrambled >> 1) & 0x10) >> 2) | (((scrambled >> 1) & 0x40) >> 3);

    d_symbols[idx] = __floats2half2_rn(PDSCH_QAM256_TABLE[i_idx], PDSCH_QAM256_TABLE[q_idx]);
}

// ============================================================================
// Fused Scramble + Modulate Kernels (FP32 output)
// ============================================================================

__global__ void __launch_bounds__(256, 4) fused_scramble_qpsk_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    cuFloatComplex* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 2;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    uint32_t bits, scr;
    if (bit_pos <= 30) {
        bits = (d_bits[word_idx] >> bit_pos) & 0x3;
        scr = (d_scramble_seq[word_idx] >> bit_pos) & 0x3;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0x3;
        scr = ((d_scramble_seq[word_idx] >> bit_pos) | (d_scramble_seq[word_idx + 1] << (32 - bit_pos))) & 0x3;
    }

    uint32_t scrambled = bits ^ scr;
    d_symbols[idx] = make_cuFloatComplex(PDSCH_QPSK_I[scrambled], PDSCH_QPSK_Q[scrambled]);
}

__global__ void __launch_bounds__(256, 4) fused_scramble_64qam_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    cuFloatComplex* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 6;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    uint32_t bits, scr;
    if (bit_pos <= 26) {
        bits = (d_bits[word_idx] >> bit_pos) & 0x3F;
        scr = (d_scramble_seq[word_idx] >> bit_pos) & 0x3F;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0x3F;
        scr = ((d_scramble_seq[word_idx] >> bit_pos) | (d_scramble_seq[word_idx + 1] << (32 - bit_pos))) & 0x3F;
    }

    uint32_t scrambled = bits ^ scr;
    int i_idx = (scrambled & 0x1) | ((scrambled & 0x4) >> 1) | ((scrambled & 0x10) >> 2);
    int q_idx = ((scrambled >> 1) & 0x1) | (((scrambled >> 1) & 0x4) >> 1) | (((scrambled >> 1) & 0x10) >> 2);

    d_symbols[idx] = make_cuFloatComplex(PDSCH_QAM64_TABLE[i_idx], PDSCH_QAM64_TABLE[q_idx]);
}

// ============================================================================
// INT8 Constellation Tables (unnormalized, for direct INT8 output)
// ============================================================================

__device__ __constant__ int8_t PDSCH_QPSK_I8[4] = { 1, -1,  1, -1};
__device__ __constant__ int8_t PDSCH_QPSK_Q8[4] = { 1,  1, -1, -1};

__device__ __constant__ int8_t PDSCH_QAM16_I8[4] = { 1, -1,  3, -3};
__device__ __constant__ int8_t PDSCH_QAM16_Q8[4] = { 1, -1,  3, -3};

__device__ __constant__ int8_t PDSCH_QAM64_I8[8] = { 3, -3,  5, -5,  1, -1,  7, -7};

__device__ __constant__ int8_t PDSCH_QAM256_I8[16] = {
     5, -5,  11, -11,   3, -3,  13, -13,
     7, -7,   9,  -9,   1, -1,  15, -15
};

// ============================================================================
// Rate Matching Index Calculation (same as rate_matching.cu)
// ============================================================================

/**
 * @brief Calculate circular buffer position with filler skipping
 *
 * The circular buffer has a "hole" for filler bits at positions [Kd, Kd+F).
 * k0 is given in actual buffer coordinates (from rate_matcher_compute_k0).
 * We convert to conceptual coordinates (excluding filler), advance by outIdx,
 * wrap at the effective buffer length, then convert back to actual coordinates.
 *
 * Conceptual buffer has (Ncb - F) usable bits:
 *   - Positions [0, Kd) = info bits (maps directly)
 *   - Positions [Kd, Ncb-F) = parity bits (maps to [Kd+F, Ncb) in actual buffer)
 *
 * This matches the CPU rate matcher behavior in ldpc_rate_matcher_impl.cpp.
 */
__device__ __forceinline__ int fused_rate_match_index(int outIdx, int Kd, int F, int k0, int Ncb) {
    // Effective buffer length (excluding filler hole)
    int effective_len = Ncb - F;

    // Convert k0 from actual to conceptual coordinates
    // k0 should never be in the filler region [Kd, Kd+F) per 3GPP spec
    int conceptual_k0;
    if (k0 < Kd) {
        conceptual_k0 = k0;
    } else {
        // k0 is past the filler region, subtract F to get conceptual position
        conceptual_k0 = k0 - F;
    }

    // Advance by outIdx and wrap at effective buffer length
    int conceptual_pos = (conceptual_k0 + outIdx) % effective_len;

    // Map conceptual position back to actual buffer position
    // Positions >= Kd need to skip over the F filler bits
    if (conceptual_pos >= Kd) {
        return conceptual_pos + F;
    } else {
        return conceptual_pos;
    }
}

// ============================================================================
// Fused RM + Interleave + Scramble + Modulate → INT8 Kernels
// ============================================================================

/**
 * @brief Fused 64QAM: LDPC encoded → rate match → interleave → scramble → INT8 symbols
 *
 * Eliminates intermediate rate-matched buffer and packed bytes buffer.
 * Each thread outputs one INT8 complex symbol (2 bytes).
 */
__global__ void __launch_bounds__(256, 4) fused_rm_interleave_scramble_64qam_int8_kernel(
    const uint32_t* __restrict__ d_encoded,    // LDPC encoded bits
    uint32_t c_init,
    int8_t* __restrict__ d_symbols_int8,       // Output INT8 symbols (2 bytes each)
    int N_cb,              // Circular buffer size
    int N_full,            // Full codeword size
    int k0,                // Starting position
    int E,                 // Rate-matched bits per CB
    int Kd,                // Payload bits in CB coordinates
    int F,                 // Filler bits
    int puncture_offset,   // Offset for punctured bits (2*Z)
    int num_cbs,           // Number of code blocks
    int encoded_stride,    // Words per CB in encoded buffer
    int nof_short,         // Number of short CBs (E_short)
    int E_short,           // Bits per short CB
    int E_long,            // Bits per long CB
    int total_symbols      // Total output symbols
) {
    int sym_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sym_idx >= total_symbols) return;

    // Calculate global output bit position
    int out_bit_base = sym_idx * 6;

    // Determine which CB and local bit position
    int short_cb_total = nof_short * E_short;
    int cb_idx, local_base, bits_this_cb;

    if (out_bit_base < short_cb_total) {
        cb_idx = out_bit_base / E_short;
        local_base = out_bit_base % E_short;
        bits_this_cb = E_short;
    } else {
        int offset = out_bit_base - short_cb_total;
        cb_idx = nof_short + offset / E_long;
        local_base = offset % E_long;
        bits_this_cb = E_long;
    }

    // On-the-fly scrambling: advance LFSR to this thread's bit position
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;
    x1 = advance_x1_pdsch(x1, NC_SKIP + out_bit_base);
    x2 = advance_x2_pdsch(x2, NC_SKIP + out_bit_base);

    // Read 6 bits with interleaving, rate matching, and scrambling
    uint32_t scrambled_bits = 0;
    int K = bits_this_cb / 6;  // Number of symbols for this CB

    #pragma unroll
    for (int b = 0; b < 6; b++) {
        // De-interleave: output bit position → rate-matched position
        // Per 3GPP: j = bit_in_cb / Q_m, i = bit_in_cb % Q_m
        // rate_matched_pos = i * K + j
        int bit_in_cb = local_base + b;
        int j = bit_in_cb / 6;
        int i = bit_in_cb % 6;
        int rm_bit_in_cb = i * K + j;

        // Rate match: rate-matched position → LDPC encoded position
        int cb_pos = fused_rate_match_index(rm_bit_in_cb, Kd, F, k0, N_cb);
        int enc_pos = cb_pos + puncture_offset;

        // Read from LDPC encoded buffer (MSB-first within each byte)
        int in_word = enc_pos / 32;
        int bit_in_word = enc_pos % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int in_bit_pos = byte_in_word * 8 + bit_in_byte;

        uint32_t encoded_word = d_encoded[cb_idx * encoded_stride + in_word];
        uint32_t bit = (encoded_word >> in_bit_pos) & 1u;

        // Generate scrambling bit on-the-fly
        uint32_t scr_bit = (x1 ^ x2) & 1u;
        x1 = step_x1_pdsch(x1);
        x2 = step_x2_pdsch(x2);

        // XOR to scramble and accumulate
        scrambled_bits |= ((bit ^ scr_bit) << b);
    }

    // Map to INT8 constellation (64QAM)
    int i_idx = (scrambled_bits & 0x1) | ((scrambled_bits & 0x4) >> 1) | ((scrambled_bits & 0x10) >> 2);
    int q_idx = ((scrambled_bits >> 1) & 0x1) | (((scrambled_bits >> 1) & 0x4) >> 1) | (((scrambled_bits >> 1) & 0x10) >> 2);

    d_symbols_int8[sym_idx * 2]     = PDSCH_QAM64_I8[i_idx];
    d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QAM64_I8[q_idx];
}

/**
 * @brief Fused QPSK: LDPC encoded → rate match → interleave → scramble → INT8 symbols
 */
__global__ void __launch_bounds__(256, 4) fused_rm_interleave_scramble_qpsk_int8_kernel(
    const uint32_t* __restrict__ d_encoded,
    uint32_t c_init,
    int8_t* __restrict__ d_symbols_int8,
    int N_cb, int N_full, int k0, int E, int Kd, int F, int puncture_offset,
    int num_cbs, int encoded_stride,
    int nof_short, int E_short, int E_long,
    int total_symbols
) {
    int sym_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sym_idx >= total_symbols) return;

    int out_bit_base = sym_idx * 2;
    int short_cb_total = nof_short * E_short;
    int cb_idx, local_base, bits_this_cb;

    if (out_bit_base < short_cb_total) {
        cb_idx = out_bit_base / E_short;
        local_base = out_bit_base % E_short;
        bits_this_cb = E_short;
    } else {
        int offset = out_bit_base - short_cb_total;
        cb_idx = nof_short + offset / E_long;
        local_base = offset % E_long;
        bits_this_cb = E_long;
    }

    // On-the-fly scrambling: advance LFSR to this thread's bit position
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;
    x1 = advance_x1_pdsch(x1, NC_SKIP + out_bit_base);
    x2 = advance_x2_pdsch(x2, NC_SKIP + out_bit_base);

    uint32_t scrambled_bits = 0;
    int K = bits_this_cb / 2;

    #pragma unroll
    for (int b = 0; b < 2; b++) {
        int bit_in_cb = local_base + b;
        int j = bit_in_cb / 2;
        int i = bit_in_cb % 2;
        int rm_bit_in_cb = i * K + j;

        int cb_pos = fused_rate_match_index(rm_bit_in_cb, Kd, F, k0, N_cb);
        int enc_pos = cb_pos + puncture_offset;

        int in_word = enc_pos / 32;
        int bit_in_word = enc_pos % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int in_bit_pos = byte_in_word * 8 + bit_in_byte;

        uint32_t encoded_word = d_encoded[cb_idx * encoded_stride + in_word];
        uint32_t bit = (encoded_word >> in_bit_pos) & 1u;

        // Generate scrambling bit on-the-fly
        uint32_t scr_bit = (x1 ^ x2) & 1u;
        x1 = step_x1_pdsch(x1);
        x2 = step_x2_pdsch(x2);

        // XOR to scramble and accumulate
        scrambled_bits |= ((bit ^ scr_bit) << b);
    }

    d_symbols_int8[sym_idx * 2]     = PDSCH_QPSK_I8[scrambled_bits];
    d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QPSK_Q8[scrambled_bits];
}

/**
 * @brief Fused 16QAM: LDPC encoded → rate match → interleave → scramble → INT8 symbols
 */
__global__ void __launch_bounds__(256, 4) fused_rm_interleave_scramble_16qam_int8_kernel(
    const uint32_t* __restrict__ d_encoded,
    uint32_t c_init,
    int8_t* __restrict__ d_symbols_int8,
    int N_cb, int N_full, int k0, int E, int Kd, int F, int puncture_offset,
    int num_cbs, int encoded_stride,
    int nof_short, int E_short, int E_long,
    int total_symbols
) {
    int sym_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sym_idx >= total_symbols) return;

    int out_bit_base = sym_idx * 4;
    int short_cb_total = nof_short * E_short;
    int cb_idx, local_base, bits_this_cb;

    if (out_bit_base < short_cb_total) {
        cb_idx = out_bit_base / E_short;
        local_base = out_bit_base % E_short;
        bits_this_cb = E_short;
    } else {
        int offset = out_bit_base - short_cb_total;
        cb_idx = nof_short + offset / E_long;
        local_base = offset % E_long;
        bits_this_cb = E_long;
    }

    // On-the-fly scrambling: advance LFSR to this thread's bit position
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;
    x1 = advance_x1_pdsch(x1, NC_SKIP + out_bit_base);
    x2 = advance_x2_pdsch(x2, NC_SKIP + out_bit_base);

    uint32_t scrambled_bits = 0;
    int K = bits_this_cb / 4;

    #pragma unroll
    for (int b = 0; b < 4; b++) {
        int bit_in_cb = local_base + b;
        int j = bit_in_cb / 4;
        int i = bit_in_cb % 4;
        int rm_bit_in_cb = i * K + j;

        int cb_pos = fused_rate_match_index(rm_bit_in_cb, Kd, F, k0, N_cb);
        int enc_pos = cb_pos + puncture_offset;

        int in_word = enc_pos / 32;
        int bit_in_word = enc_pos % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int in_bit_pos = byte_in_word * 8 + bit_in_byte;

        uint32_t encoded_word = d_encoded[cb_idx * encoded_stride + in_word];
        uint32_t bit = (encoded_word >> in_bit_pos) & 1u;

        // Generate scrambling bit on-the-fly
        uint32_t scr_bit = (x1 ^ x2) & 1u;
        x1 = step_x1_pdsch(x1);
        x2 = step_x2_pdsch(x2);

        bit ^= scr_bit;
        scrambled_bits |= (bit << b);
    }

    // 16QAM mapping: bits b0,b2 → I index, bits b1,b3 → Q index
    int i_idx = (scrambled_bits & 0x1) | ((scrambled_bits & 0x4) >> 1);
    int q_idx = ((scrambled_bits >> 1) & 0x1) | (((scrambled_bits >> 1) & 0x4) >> 1);

    d_symbols_int8[sym_idx * 2]     = PDSCH_QAM16_I8[i_idx];
    d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QAM16_Q8[q_idx];
}

/**
 * @brief Fused 256QAM: LDPC encoded → rate match → interleave → scramble → INT8 symbols
 */
__global__ void __launch_bounds__(256, 4) fused_rm_interleave_scramble_256qam_int8_kernel(
    const uint32_t* __restrict__ d_encoded,
    uint32_t c_init,
    int8_t* __restrict__ d_symbols_int8,
    int N_cb, int N_full, int k0, int E, int Kd, int F, int puncture_offset,
    int num_cbs, int encoded_stride,
    int nof_short, int E_short, int E_long,
    int total_symbols
) {
    int sym_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sym_idx >= total_symbols) return;

    int out_bit_base = sym_idx * 8;
    int short_cb_total = nof_short * E_short;
    int cb_idx, local_base, bits_this_cb;

    if (out_bit_base < short_cb_total) {
        cb_idx = out_bit_base / E_short;
        local_base = out_bit_base % E_short;
        bits_this_cb = E_short;
    } else {
        int offset = out_bit_base - short_cb_total;
        cb_idx = nof_short + offset / E_long;
        local_base = offset % E_long;
        bits_this_cb = E_long;
    }

    // On-the-fly scrambling: advance LFSR to this thread's bit position
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;
    x1 = advance_x1_pdsch(x1, NC_SKIP + out_bit_base);
    x2 = advance_x2_pdsch(x2, NC_SKIP + out_bit_base);

    uint32_t scrambled_bits = 0;
    int K = bits_this_cb / 8;

    #pragma unroll
    for (int b = 0; b < 8; b++) {
        int bit_in_cb = local_base + b;
        int j = bit_in_cb / 8;
        int i = bit_in_cb % 8;
        int rm_bit_in_cb = i * K + j;

        int cb_pos = fused_rate_match_index(rm_bit_in_cb, Kd, F, k0, N_cb);
        int enc_pos = cb_pos + puncture_offset;

        int in_word = enc_pos / 32;
        int bit_in_word = enc_pos % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int in_bit_pos = byte_in_word * 8 + bit_in_byte;

        uint32_t encoded_word = d_encoded[cb_idx * encoded_stride + in_word];
        uint32_t bit = (encoded_word >> in_bit_pos) & 1u;

        // Generate scrambling bit on-the-fly
        uint32_t scr_bit = (x1 ^ x2) & 1u;
        x1 = step_x1_pdsch(x1);
        x2 = step_x2_pdsch(x2);

        bit ^= scr_bit;
        scrambled_bits |= (bit << b);
    }

    // 256QAM mapping: bits b0,b2,b4,b6 → I index, bits b1,b3,b5,b7 → Q index
    int i_idx = (scrambled_bits & 0x1) | ((scrambled_bits & 0x4) >> 1) |
                ((scrambled_bits & 0x10) >> 2) | ((scrambled_bits & 0x40) >> 3);
    int q_idx = ((scrambled_bits >> 1) & 0x1) | (((scrambled_bits >> 1) & 0x4) >> 1) |
                (((scrambled_bits >> 1) & 0x10) >> 2) | (((scrambled_bits >> 1) & 0x40) >> 3);

    d_symbols_int8[sym_idx * 2]     = PDSCH_QAM256_I8[i_idx];
    d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QAM256_I8[q_idx];
}

__device__ __forceinline__ uint32_t pdsch_read_scramble_bit(
    const uint32_t* __restrict__ d_scramble_seq,
    int bit_idx)
{
    int word_idx = bit_idx >> 5;
    int bit_pos = 31 - (bit_idx & 31);
    return (d_scramble_seq[word_idx] >> bit_pos) & 1u;
}

template <int Q_M>
__global__ void __launch_bounds__(256, 4) fused_rm_interleave_scramble_precomputed_int8_kernel(
    const uint32_t* __restrict__ d_encoded,
    const uint32_t* __restrict__ d_scramble_seq,
    int8_t* __restrict__ d_symbols_int8,
    int N_cb,
    int N_full,
    int k0,
    int E,
    int Kd,
    int F,
    int puncture_offset,
    int num_cbs,
    int encoded_stride,
    int nof_short,
    int E_short,
    int E_long,
    int total_symbols
) {
    int sym_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sym_idx >= total_symbols) return;

    int out_bit_base = sym_idx * Q_M;
    int short_cb_total = nof_short * E_short;
    int cb_idx, local_base, bits_this_cb;

    if (out_bit_base < short_cb_total) {
        cb_idx = out_bit_base / E_short;
        local_base = out_bit_base % E_short;
        bits_this_cb = E_short;
    } else {
        int offset = out_bit_base - short_cb_total;
        cb_idx = nof_short + offset / E_long;
        local_base = offset % E_long;
        bits_this_cb = E_long;
    }

    uint32_t scrambled_bits = 0;
    int K = bits_this_cb / Q_M;

    #pragma unroll
    for (int b = 0; b < Q_M; b++) {
        int bit_in_cb = local_base + b;
        int j = bit_in_cb / Q_M;
        int i = bit_in_cb % Q_M;
        int rm_bit_in_cb = i * K + j;

        int cb_pos = fused_rate_match_index(rm_bit_in_cb, Kd, F, k0, N_cb);
        int enc_pos = cb_pos + puncture_offset;

        int in_word = enc_pos / 32;
        int bit_in_word = enc_pos % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int in_bit_pos = byte_in_word * 8 + bit_in_byte;

        uint32_t encoded_word = d_encoded[cb_idx * encoded_stride + in_word];
        uint32_t bit = (encoded_word >> in_bit_pos) & 1u;
        uint32_t scr_bit = pdsch_read_scramble_bit(d_scramble_seq, out_bit_base + b);

        scrambled_bits |= ((bit ^ scr_bit) << b);
    }

    if (Q_M == 2) {
        d_symbols_int8[sym_idx * 2]     = PDSCH_QPSK_I8[scrambled_bits];
        d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QPSK_Q8[scrambled_bits];
    } else if (Q_M == 4) {
        int i_idx = (scrambled_bits & 0x1) | ((scrambled_bits & 0x4) >> 1);
        int q_idx = ((scrambled_bits >> 1) & 0x1) | (((scrambled_bits >> 1) & 0x4) >> 1);
        d_symbols_int8[sym_idx * 2]     = PDSCH_QAM16_I8[i_idx];
        d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QAM16_Q8[q_idx];
    } else if (Q_M == 6) {
        int i_idx = (scrambled_bits & 0x1) | ((scrambled_bits & 0x4) >> 1) | ((scrambled_bits & 0x10) >> 2);
        int q_idx = ((scrambled_bits >> 1) & 0x1) | (((scrambled_bits >> 1) & 0x4) >> 1) | (((scrambled_bits >> 1) & 0x10) >> 2);
        d_symbols_int8[sym_idx * 2]     = PDSCH_QAM64_I8[i_idx];
        d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QAM64_I8[q_idx];
    } else if (Q_M == 8) {
        int i_idx = (scrambled_bits & 0x1) | ((scrambled_bits & 0x4) >> 1) |
                    ((scrambled_bits & 0x10) >> 2) | ((scrambled_bits & 0x40) >> 3);
        int q_idx = ((scrambled_bits >> 1) & 0x1) | (((scrambled_bits >> 1) & 0x4) >> 1) |
                    (((scrambled_bits >> 1) & 0x10) >> 2) | (((scrambled_bits >> 1) & 0x40) >> 3);
        d_symbols_int8[sym_idx * 2]     = PDSCH_QAM256_I8[i_idx];
        d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QAM256_I8[q_idx];
    }
}

// ============================================================================
// API Functions
// ============================================================================

extern "C" {

int pdsch_fused_scramble_modulate_half(
    const uint32_t* d_bits,
    const uint32_t* d_scramble_seq,
    void* d_symbols_half,
    int num_bits,
    int mod_order,
    cudaStream_t stream)
{
    if (!d_bits || !d_scramble_seq || !d_symbols_half || num_bits <= 0) return -1;

    __half2* d_symbols = static_cast<__half2*>(d_symbols_half);
    int num_symbols = (num_bits + mod_order - 1) / mod_order;
    int block_size = 256;
    int num_blocks = (num_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 2:  // QPSK
            fused_scramble_qpsk_half_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, num_symbols);
            break;
        case 4:  // 16QAM
            fused_scramble_16qam_half_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, num_symbols);
            break;
        case 6:  // 64QAM
            fused_scramble_64qam_half_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, num_symbols);
            break;
        case 8:  // 256QAM
            fused_scramble_256qam_half_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, num_symbols);
            break;
        default:
            return -1;
    }

    return 0;
}

int pdsch_fused_scramble_modulate(
    const uint32_t* d_bits,
    const uint32_t* d_scramble_seq,
    void* d_symbols,
    int num_bits,
    int mod_order,
    cudaStream_t stream)
{
    if (!d_bits || !d_scramble_seq || !d_symbols || num_bits <= 0) return -1;

    cuFloatComplex* d_out = static_cast<cuFloatComplex*>(d_symbols);
    int num_symbols = (num_bits + mod_order - 1) / mod_order;
    int block_size = 256;
    int num_blocks = (num_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 2:  // QPSK
            fused_scramble_qpsk_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_out, num_symbols);
            break;
        case 6:  // 64QAM
            fused_scramble_64qam_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_out, num_symbols);
            break;
        default:
            // Fall back to separate scramble + modulate for other orders
            return -1;
    }

    return 0;
}

int pdsch_fused_encode_to_symbols_int8(
    const uint32_t* d_encoded,
    uint32_t c_init,
    int8_t* d_symbols_int8,
    const pdsch_fused_tx_config_t* cfg,
    cudaStream_t stream)
{
    if (!d_encoded || !d_symbols_int8 || !cfg) return -1;
    if (cfg->total_symbols <= 0) return -1;

    cudaError_t err = ensure_pdsch_jump_tables_initialized();
    if (err != cudaSuccess) return -1;

    int block_size = 256;
    int num_blocks = (cfg->total_symbols + block_size - 1) / block_size;

    // Determine E for current CB (using E_long as default for kernel param)
    int E = cfg->E_long;

    switch (cfg->mod_order) {
        case 2:  // QPSK
            fused_rm_interleave_scramble_qpsk_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_encoded, c_init, d_symbols_int8,
                cfg->N_cb, cfg->N_full, cfg->k0, E, cfg->Kd, cfg->F, cfg->puncture_offset,
                cfg->num_cbs, cfg->encoded_stride,
                cfg->nof_short, cfg->E_short, cfg->E_long,
                cfg->total_symbols);
            break;
        case 4:  // 16QAM
            fused_rm_interleave_scramble_16qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_encoded, c_init, d_symbols_int8,
                cfg->N_cb, cfg->N_full, cfg->k0, E, cfg->Kd, cfg->F, cfg->puncture_offset,
                cfg->num_cbs, cfg->encoded_stride,
                cfg->nof_short, cfg->E_short, cfg->E_long,
                cfg->total_symbols);
            break;
        case 6:  // 64QAM
            fused_rm_interleave_scramble_64qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_encoded, c_init, d_symbols_int8,
                cfg->N_cb, cfg->N_full, cfg->k0, E, cfg->Kd, cfg->F, cfg->puncture_offset,
                cfg->num_cbs, cfg->encoded_stride,
                cfg->nof_short, cfg->E_short, cfg->E_long,
                cfg->total_symbols);
            break;
        case 8:  // 256QAM
            fused_rm_interleave_scramble_256qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_encoded, c_init, d_symbols_int8,
                cfg->N_cb, cfg->N_full, cfg->k0, E, cfg->Kd, cfg->F, cfg->puncture_offset,
                cfg->num_cbs, cfg->encoded_stride,
                cfg->nof_short, cfg->E_short, cfg->E_long,
                cfg->total_symbols);
            break;
        default:
            return -1;
    }

    return 0;
}

int pdsch_fused_encode_to_symbols_int8_precomputed(
    const uint32_t* d_encoded,
    const uint32_t* d_scramble_seq,
    int8_t* d_symbols_int8,
    const pdsch_fused_tx_config_t* cfg,
    cudaStream_t stream)
{
    if (!d_encoded || !d_scramble_seq || !d_symbols_int8 || !cfg) return -1;
    if (cfg->total_symbols <= 0) return -1;

    int block_size = 256;
    int num_blocks = (cfg->total_symbols + block_size - 1) / block_size;
    int E = cfg->E_long;

    switch (cfg->mod_order) {
        case 2:
            fused_rm_interleave_scramble_precomputed_int8_kernel<2><<<num_blocks, block_size, 0, stream>>>(
                d_encoded, d_scramble_seq, d_symbols_int8,
                cfg->N_cb, cfg->N_full, cfg->k0, E, cfg->Kd, cfg->F, cfg->puncture_offset,
                cfg->num_cbs, cfg->encoded_stride,
                cfg->nof_short, cfg->E_short, cfg->E_long,
                cfg->total_symbols);
            break;
        case 4:
            fused_rm_interleave_scramble_precomputed_int8_kernel<4><<<num_blocks, block_size, 0, stream>>>(
                d_encoded, d_scramble_seq, d_symbols_int8,
                cfg->N_cb, cfg->N_full, cfg->k0, E, cfg->Kd, cfg->F, cfg->puncture_offset,
                cfg->num_cbs, cfg->encoded_stride,
                cfg->nof_short, cfg->E_short, cfg->E_long,
                cfg->total_symbols);
            break;
        case 6:
            fused_rm_interleave_scramble_precomputed_int8_kernel<6><<<num_blocks, block_size, 0, stream>>>(
                d_encoded, d_scramble_seq, d_symbols_int8,
                cfg->N_cb, cfg->N_full, cfg->k0, E, cfg->Kd, cfg->F, cfg->puncture_offset,
                cfg->num_cbs, cfg->encoded_stride,
                cfg->nof_short, cfg->E_short, cfg->E_long,
                cfg->total_symbols);
            break;
        case 8:
            fused_rm_interleave_scramble_precomputed_int8_kernel<8><<<num_blocks, block_size, 0, stream>>>(
                d_encoded, d_scramble_seq, d_symbols_int8,
                cfg->N_cb, cfg->N_full, cfg->k0, E, cfg->Kd, cfg->F, cfg->puncture_offset,
                cfg->num_cbs, cfg->encoded_stride,
                cfg->nof_short, cfg->E_short, cfg->E_long,
                cfg->total_symbols);
            break;
        default:
            return -1;
    }

    return 0;
}

// ============================================================================
// Rate-Matched Input Kernels (Interleave + Scramble + Modulate only, no RM)
// ============================================================================

/**
 * @brief QPSK: Rate-matched input → interleave → scramble → INT8 symbols
 * Used with fused encoder + rate matcher to skip RM step in this kernel.
 */
__global__ void __launch_bounds__(256, 4) interleave_scramble_qpsk_int8_kernel(
    const uint32_t* __restrict__ d_rate_matched,  // Already rate-matched bits
    uint32_t c_init,
    int8_t* __restrict__ d_symbols_int8,
    int rm_stride,         // Words per CB in rate-matched buffer
    int nof_short, int E_short, int E_long,
    int total_symbols
) {
    int sym_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sym_idx >= total_symbols) return;

    int out_bit_base = sym_idx * 2;
    int short_cb_total = nof_short * E_short;
    int cb_idx, local_base, bits_this_cb;

    if (out_bit_base < short_cb_total) {
        cb_idx = out_bit_base / E_short;
        local_base = out_bit_base % E_short;
        bits_this_cb = E_short;
    } else {
        int offset = out_bit_base - short_cb_total;
        cb_idx = nof_short + offset / E_long;
        local_base = offset % E_long;
        bits_this_cb = E_long;
    }

    // On-the-fly scrambling: advance LFSR to this thread's bit position
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;
    x1 = advance_x1_pdsch(x1, NC_SKIP + out_bit_base);
    x2 = advance_x2_pdsch(x2, NC_SKIP + out_bit_base);

    uint32_t scrambled_bits = 0;
    int K = bits_this_cb / 2;

    #pragma unroll
    for (int b = 0; b < 2; b++) {
        int bit_in_cb = local_base + b;

        // De-interleave: output position → rate-matched position
        int j = bit_in_cb / 2;
        int i = bit_in_cb % 2;
        int rm_bit_in_cb = i * K + j;

        // Read from rate-matched buffer (MSB-first within each byte)
        int in_word = rm_bit_in_cb / 32;
        int bit_in_word = rm_bit_in_cb % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int in_bit_pos = byte_in_word * 8 + bit_in_byte;

        uint32_t rm_word = d_rate_matched[cb_idx * rm_stride + in_word];
        uint32_t bit = (rm_word >> in_bit_pos) & 1u;

        // Generate scrambling bit on-the-fly
        uint32_t scr_bit = (x1 ^ x2) & 1u;
        x1 = step_x1_pdsch(x1);
        x2 = step_x2_pdsch(x2);

        bit ^= scr_bit;
        scrambled_bits |= (bit << b);
    }

    d_symbols_int8[sym_idx * 2]     = PDSCH_QPSK_I8[scrambled_bits];
    d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QPSK_Q8[scrambled_bits];
}

/**
 * @brief 16QAM: Rate-matched input → interleave → scramble → INT8 symbols
 */
__global__ void __launch_bounds__(256, 4) interleave_scramble_16qam_int8_kernel(
    const uint32_t* __restrict__ d_rate_matched,
    uint32_t c_init,
    int8_t* __restrict__ d_symbols_int8,
    int rm_stride,
    int nof_short, int E_short, int E_long,
    int total_symbols
) {
    int sym_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sym_idx >= total_symbols) return;

    int out_bit_base = sym_idx * 4;
    int short_cb_total = nof_short * E_short;
    int cb_idx, local_base, bits_this_cb;

    if (out_bit_base < short_cb_total) {
        cb_idx = out_bit_base / E_short;
        local_base = out_bit_base % E_short;
        bits_this_cb = E_short;
    } else {
        int offset = out_bit_base - short_cb_total;
        cb_idx = nof_short + offset / E_long;
        local_base = offset % E_long;
        bits_this_cb = E_long;
    }

    // On-the-fly scrambling: advance LFSR to this thread's bit position
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;
    x1 = advance_x1_pdsch(x1, NC_SKIP + out_bit_base);
    x2 = advance_x2_pdsch(x2, NC_SKIP + out_bit_base);

    uint32_t scrambled_bits = 0;
    int K = bits_this_cb / 4;

    #pragma unroll
    for (int b = 0; b < 4; b++) {
        int bit_in_cb = local_base + b;
        int j = bit_in_cb / 4;
        int i = bit_in_cb % 4;
        int rm_bit_in_cb = i * K + j;

        int in_word = rm_bit_in_cb / 32;
        int bit_in_word = rm_bit_in_cb % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int in_bit_pos = byte_in_word * 8 + bit_in_byte;

        uint32_t rm_word = d_rate_matched[cb_idx * rm_stride + in_word];
        uint32_t bit = (rm_word >> in_bit_pos) & 1u;

        // Generate scrambling bit on-the-fly
        uint32_t scr_bit = (x1 ^ x2) & 1u;
        x1 = step_x1_pdsch(x1);
        x2 = step_x2_pdsch(x2);

        bit ^= scr_bit;
        scrambled_bits |= (bit << b);
    }

    int i_idx = (scrambled_bits & 0x1) | ((scrambled_bits & 0x4) >> 1);
    int q_idx = ((scrambled_bits >> 1) & 0x1) | (((scrambled_bits >> 1) & 0x4) >> 1);

    d_symbols_int8[sym_idx * 2]     = PDSCH_QAM16_I8[i_idx];
    d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QAM16_Q8[q_idx];
}

/**
 * @brief 64QAM: Rate-matched input → interleave → scramble → INT8 symbols
 */
__global__ void __launch_bounds__(256, 4) interleave_scramble_64qam_int8_kernel(
    const uint32_t* __restrict__ d_rate_matched,
    uint32_t c_init,
    int8_t* __restrict__ d_symbols_int8,
    int rm_stride,
    int nof_short, int E_short, int E_long,
    int total_symbols
) {
    int sym_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sym_idx >= total_symbols) return;

    int out_bit_base = sym_idx * 6;
    int short_cb_total = nof_short * E_short;
    int cb_idx, local_base, bits_this_cb;

    if (out_bit_base < short_cb_total) {
        cb_idx = out_bit_base / E_short;
        local_base = out_bit_base % E_short;
        bits_this_cb = E_short;
    } else {
        int offset = out_bit_base - short_cb_total;
        cb_idx = nof_short + offset / E_long;
        local_base = offset % E_long;
        bits_this_cb = E_long;
    }

    // On-the-fly scrambling: advance LFSR to this thread's bit position
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;
    x1 = advance_x1_pdsch(x1, NC_SKIP + out_bit_base);
    x2 = advance_x2_pdsch(x2, NC_SKIP + out_bit_base);

    uint32_t scrambled_bits = 0;
    int K = bits_this_cb / 6;

    #pragma unroll
    for (int b = 0; b < 6; b++) {
        int bit_in_cb = local_base + b;
        int j = bit_in_cb / 6;
        int i = bit_in_cb % 6;
        int rm_bit_in_cb = i * K + j;

        int in_word = rm_bit_in_cb / 32;
        int bit_in_word = rm_bit_in_cb % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int in_bit_pos = byte_in_word * 8 + bit_in_byte;

        uint32_t rm_word = d_rate_matched[cb_idx * rm_stride + in_word];
        uint32_t bit = (rm_word >> in_bit_pos) & 1u;

        // Generate scrambling bit on-the-fly
        uint32_t scr_bit = (x1 ^ x2) & 1u;
        x1 = step_x1_pdsch(x1);
        x2 = step_x2_pdsch(x2);

        bit ^= scr_bit;
        scrambled_bits |= (bit << b);
    }

    int i_idx = (scrambled_bits & 0x1) | ((scrambled_bits & 0x4) >> 1) | ((scrambled_bits & 0x10) >> 2);
    int q_idx = ((scrambled_bits >> 1) & 0x1) | (((scrambled_bits >> 1) & 0x4) >> 1) | (((scrambled_bits >> 1) & 0x10) >> 2);

    d_symbols_int8[sym_idx * 2]     = PDSCH_QAM64_I8[i_idx];
    d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QAM64_I8[q_idx];
}

/**
 * @brief 256QAM: Rate-matched input → interleave → scramble → INT8 symbols
 */
__global__ void __launch_bounds__(256, 4) interleave_scramble_256qam_int8_kernel(
    const uint32_t* __restrict__ d_rate_matched,
    uint32_t c_init,
    int8_t* __restrict__ d_symbols_int8,
    int rm_stride,
    int nof_short, int E_short, int E_long,
    int total_symbols
) {
    int sym_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sym_idx >= total_symbols) return;

    int out_bit_base = sym_idx * 8;
    int short_cb_total = nof_short * E_short;
    int cb_idx, local_base, bits_this_cb;

    if (out_bit_base < short_cb_total) {
        cb_idx = out_bit_base / E_short;
        local_base = out_bit_base % E_short;
        bits_this_cb = E_short;
    } else {
        int offset = out_bit_base - short_cb_total;
        cb_idx = nof_short + offset / E_long;
        local_base = offset % E_long;
        bits_this_cb = E_long;
    }

    // On-the-fly scrambling: advance LFSR to this thread's bit position
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;
    x1 = advance_x1_pdsch(x1, NC_SKIP + out_bit_base);
    x2 = advance_x2_pdsch(x2, NC_SKIP + out_bit_base);

    uint32_t scrambled_bits = 0;
    int K = bits_this_cb / 8;

    #pragma unroll
    for (int b = 0; b < 8; b++) {
        int bit_in_cb = local_base + b;
        int j = bit_in_cb / 8;
        int i = bit_in_cb % 8;
        int rm_bit_in_cb = i * K + j;

        int in_word = rm_bit_in_cb / 32;
        int bit_in_word = rm_bit_in_cb % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int in_bit_pos = byte_in_word * 8 + bit_in_byte;

        uint32_t rm_word = d_rate_matched[cb_idx * rm_stride + in_word];
        uint32_t bit = (rm_word >> in_bit_pos) & 1u;

        // Generate scrambling bit on-the-fly
        uint32_t scr_bit = (x1 ^ x2) & 1u;
        x1 = step_x1_pdsch(x1);
        x2 = step_x2_pdsch(x2);

        bit ^= scr_bit;
        scrambled_bits |= (bit << b);
    }

    int i_idx = (scrambled_bits & 0x1) | ((scrambled_bits & 0x4) >> 1) | ((scrambled_bits & 0x10) >> 2) | ((scrambled_bits & 0x40) >> 3);
    int q_idx = ((scrambled_bits >> 1) & 0x1) | (((scrambled_bits >> 1) & 0x4) >> 1) | (((scrambled_bits >> 1) & 0x10) >> 2) | (((scrambled_bits >> 1) & 0x40) >> 3);

    d_symbols_int8[sym_idx * 2]     = PDSCH_QAM256_I8[i_idx];
    d_symbols_int8[sym_idx * 2 + 1] = PDSCH_QAM256_I8[q_idx];
}

int pdsch_fused_from_rate_matched_to_symbols_int8(
    const uint32_t* d_rate_matched,
    uint32_t c_init,
    int8_t* d_symbols_int8,
    int rm_stride,         // Words per CB in rate-matched buffer
    int nof_short,
    int E_short,
    int E_long,
    int mod_order,
    int total_symbols,
    cudaStream_t stream)
{
    if (!d_rate_matched || !d_symbols_int8) return -1;
    if (total_symbols <= 0) return -1;

    cudaError_t err = ensure_pdsch_jump_tables_initialized();
    if (err != cudaSuccess) return -1;

    int block_size = 256;
    int num_blocks = (total_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 2:  // QPSK
            interleave_scramble_qpsk_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_rate_matched, c_init, d_symbols_int8,
                rm_stride, nof_short, E_short, E_long, total_symbols);
            break;
        case 4:  // 16QAM
            interleave_scramble_16qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_rate_matched, c_init, d_symbols_int8,
                rm_stride, nof_short, E_short, E_long, total_symbols);
            break;
        case 6:  // 64QAM
            interleave_scramble_64qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_rate_matched, c_init, d_symbols_int8,
                rm_stride, nof_short, E_short, E_long, total_symbols);
            break;
        case 8:  // 256QAM
            interleave_scramble_256qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_rate_matched, c_init, d_symbols_int8,
                rm_stride, nof_short, E_short, E_long, total_symbols);
            break;
        default:
            return -1;
    }

    return 0;
}

} // extern "C"
