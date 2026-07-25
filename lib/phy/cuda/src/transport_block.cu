/**
 * @file transport_block.cu
 * @brief 5G NR Transport Block Processing Implementation
 *
 * Complete transport block encoding and decoding chain including:
 * - CRC attachment/checking
 * - Code block segmentation/de-segmentation
 * - LDPC encoding/decoding
 * - Rate matching/de-rate matching
 * - Bit interleaving/de-interleaving (per 3GPP TS 38.212)
 * - Scrambling/descrambling (per 3GPP TS 38.211)
 */

#include "transport_block.h"
#include "ldpc_encoder.h"
#include "ldpc_decoder.h"
#include "rate_matching.h"
#include "scrambling.h"
#include "pdsch_fused.h"
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <new>
#include <vector>

// Optional PDSCH diagnostics can be enabled through the OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS and
// OCUDU_PHY_CUDA_PDSCH_BIT_DIAGNOSTICS compile definitions.

// External CRC table declarations (defined in crc.cu)
// Used by check_tb_crc_kernel for decoder-side TB CRC validation
extern __device__ __constant__ uint32_t CRC24A_TABLE[256];
extern __device__ __constant__ uint32_t CRC24A_S0[256];
extern __device__ __constant__ uint32_t CRC24A_S1[256];
extern __device__ __constant__ uint32_t CRC24A_S2[256];
extern __device__ __constant__ uint32_t CRC24A_S3[256];
extern __device__ __constant__ uint16_t CRC16_TABLE[256];
extern __device__ __constant__ uint32_t CRC16_S0[256];
extern __device__ __constant__ uint32_t CRC16_S1[256];
extern __device__ __constant__ uint32_t CRC16_S2[256];
extern __device__ __constant__ uint32_t CRC16_S3[256];

// Precomputed x^(8*2^i) powers for O(log n) CRC-24A shifting
extern __device__ __constant__ uint32_t CRC24A_X_POWERS[21];

// CRC-24A polynomial (MSB-first): D^24 + D^23 + D^18 + D^17 + D^14 + D^11 + D^10 + D^7 + D^6 + D^5 + D^4 + D^3 + D + 1
#define TB_CRC24A_POLY 0x864CFB

// ============================================================================
// Fast Parallel CRC-24A Helpers for TB CRC Checking
// ============================================================================

/**
 * @brief GF(2^24) multiplication for forward CRC-24A polynomial (device version)
 *
 * Computes (a * b) mod G(x) using Russian peasant multiplication.
 * MSB-first variant for standard TB CRC processing.
 */
__device__ __forceinline__ uint32_t gf24_multiply_crc24a(uint32_t a, uint32_t b) {
    uint32_t result = 0;

    #pragma unroll
    for (int i = 0; i < 24; i++) {
        // If MSB of b is set, XOR a into result
        result ^= (b & 0x800000) ? a : 0;
        b <<= 1;
        // Shift a left, reduce if MSB was set
        uint32_t msb = a & 0x800000;
        a = (a << 1) & 0xFFFFFF;
        a ^= msb ? TB_CRC24A_POLY : 0;
    }

    return result & 0xFFFFFF;
}

/**
 * @brief Fast CRC-24A shift by n bytes using precomputed x^(8*2^i) powers
 *
 * shift(crc, n*8 bits) = crc * x^(n*8) mod G(x)
 * O(log n) multiplications instead of O(n) byte shifts!
 */
__device__ __forceinline__ uint32_t crc24a_shift_bytes_fast(uint32_t crc, int n_bytes) {
    if (n_bytes == 0 || crc == 0) return crc;

    // Compute x^(n*8) mod G(x) using precomputed powers
    uint32_t x_power = 0;
    bool first = true;

    for (int i = 0; i < 21 && n_bytes > 0; i++) {
        if (n_bytes & 1) {
            if (first) {
                x_power = CRC24A_X_POWERS[i];
                first = false;
            } else {
                x_power = gf24_multiply_crc24a(x_power, CRC24A_X_POWERS[i]);
            }
        }
        n_bytes >>= 1;
    }

    // Multiply crc by x_power
    if (first) return crc;  // n_bytes was 0
    return gf24_multiply_crc24a(crc, x_power);
}

// CRC24A table for TB CRC computation (TX path - fused kernel)
// NOTE: Without separable compilation, extern __device__ doesn't work across .cu files.
// Define and initialize the table directly here for the fully_fused kernel.
// CRC24A polynomial: D^24 + D^23 + D^18 + D^17 + D^14 + D^11 + D^10 + D^7 + D^6 + D^5 + D^4 + D^3 + D + 1
#define LOCAL_CRC24A_POLY 0x864CFB
__device__ __constant__ uint32_t TB_CRC24A_TABLE[256];
static bool tb_crc24a_initialized = false;
static void init_tb_crc24a_table() {
    if (tb_crc24a_initialized) return;
    uint32_t h_table[256];
    for (int i = 0; i < 256; i++) {
        uint32_t crc = (uint32_t)i << 16;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x800000) {
                crc = (crc << 1) ^ LOCAL_CRC24A_POLY;
            } else {
                crc <<= 1;
            }
        }
        h_table[i] = crc & 0xFFFFFF;
    }
    cudaMemcpyToSymbol(TB_CRC24A_TABLE, h_table, sizeof(h_table));
    tb_crc24a_initialized = true;
}

// CRC24B table for CB CRC computation (TX path)
// NOTE: Without separable compilation, extern __device__ doesn't work across .cu files.
// Define and initialize the table directly here.
// CRC24B polynomial: D^24 + D^23 + D^6 + D^5 + D + 1
// In MSB-first notation with leading 1: 0x1800063
// The leading 1 bit (D^24 term) is needed for the table-based algorithm
#define CRC24B_POLY 0x1800063
#define CRC24B_POLY_NO_MSB 0x800063
__device__ __constant__ uint32_t TB_CRC24B_TABLE[256];
__device__ __constant__ uint32_t TB_CRC24B_SLICE0[256];
__device__ __constant__ uint32_t TB_CRC24B_SLICE1[256];
__device__ __constant__ uint32_t TB_CRC24B_SLICE2[256];
__device__ __constant__ uint32_t TB_CRC24B_SLICE3[256];
__device__ __constant__ uint32_t TB_CRC24B_X_POWERS[21];
static bool tb_crc24b_initialized = false;
static void init_tb_crc24b_table() {
    if (tb_crc24b_initialized) return;
    uint32_t h_table[256];
    uint32_t h_slice0[256];
    uint32_t h_slice1[256];
    uint32_t h_slice2[256];
    uint32_t h_slice3[256];
    uint32_t h_x_powers[21];
    for (int i = 0; i < 256; i++) {
        uint32_t crc = (uint32_t)i << 16;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x800000) {
                crc = (crc << 1) ^ CRC24B_POLY;
            } else {
                crc <<= 1;
            }
        }
        h_table[i] = crc & 0xFFFFFF;
    }

    for (int i = 0; i < 256; i++) {
        uint32_t crc = h_table[i];
        h_slice3[i] = crc;
        crc = ((crc << 8) ^ h_table[(crc >> 16) & 0xFF]) & 0xFFFFFF;
        h_slice2[i] = crc;
        crc = ((crc << 8) ^ h_table[(crc >> 16) & 0xFF]) & 0xFFFFFF;
        h_slice1[i] = crc;
        crc = ((crc << 8) ^ h_table[(crc >> 16) & 0xFF]) & 0xFFFFFF;
        h_slice0[i] = crc;
    }

    uint32_t x_8 = 1;
    for (int i = 0; i < 8; i++) {
        if (x_8 & 0x800000) {
            x_8 = ((x_8 << 1) ^ CRC24B_POLY_NO_MSB) & 0xFFFFFF;
        } else {
            x_8 = (x_8 << 1) & 0xFFFFFF;
        }
    }
    h_x_powers[0] = x_8;

    for (int i = 1; i < 21; i++) {
        uint32_t a = h_x_powers[i - 1];
        uint32_t b = h_x_powers[i - 1];
        uint32_t result = 0;
        while (b) {
            if (b & 1) {
                result ^= a;
            }
            b >>= 1;
            if (a & 0x800000) {
                a = ((a << 1) ^ CRC24B_POLY_NO_MSB) & 0xFFFFFF;
            } else {
                a = (a << 1) & 0xFFFFFF;
            }
        }
        h_x_powers[i] = result & 0xFFFFFF;
    }

    cudaMemcpyToSymbol(TB_CRC24B_TABLE, h_table, sizeof(h_table));
    cudaMemcpyToSymbol(TB_CRC24B_SLICE0, h_slice0, sizeof(h_slice0));
    cudaMemcpyToSymbol(TB_CRC24B_SLICE1, h_slice1, sizeof(h_slice1));
    cudaMemcpyToSymbol(TB_CRC24B_SLICE2, h_slice2, sizeof(h_slice2));
    cudaMemcpyToSymbol(TB_CRC24B_SLICE3, h_slice3, sizeof(h_slice3));
    cudaMemcpyToSymbol(TB_CRC24B_X_POWERS, h_x_powers, sizeof(h_x_powers));
    tb_crc24b_initialized = true;
}

__device__ __forceinline__ uint32_t gf24_multiply_crc24b(uint32_t a, uint32_t b) {
    uint32_t result = 0;

    #pragma unroll
    for (int i = 0; i < 24; i++) {
        result ^= (b & 1) ? a : 0;
        b >>= 1;
        if (a & 0x800000) {
            a = ((a << 1) ^ CRC24B_POLY_NO_MSB) & 0xFFFFFF;
        } else {
            a = (a << 1) & 0xFFFFFF;
        }
    }

    return result & 0xFFFFFF;
}

__device__ __forceinline__ uint32_t crc24b_shift_bytes_fast(uint32_t crc, int n_bytes) {
    if (n_bytes == 0 || crc == 0) return crc;

    uint32_t x_power = 0;
    bool first = true;

    for (int i = 0; i < 21 && n_bytes > 0; i++) {
        if (n_bytes & 1) {
            if (first) {
                x_power = TB_CRC24B_X_POWERS[i];
                first = false;
            } else {
                x_power = gf24_multiply_crc24b(x_power, TB_CRC24B_X_POWERS[i]);
            }
        }
        n_bytes >>= 1;
    }

    return first ? crc : gf24_multiply_crc24b(crc, x_power);
}

__device__ __forceinline__ uint32_t crc24b_update_bytes_slicing4(uint32_t crc, const uint8_t* bytes, int num_bytes) {
    int byte_idx = 0;
    for (; byte_idx + 4 <= num_bytes; byte_idx += 4) {
        uint8_t b0 = bytes[byte_idx];
        uint8_t b1 = bytes[byte_idx + 1];
        uint8_t b2 = bytes[byte_idx + 2];
        uint8_t b3 = bytes[byte_idx + 3];
        crc = TB_CRC24B_SLICE0[((crc >> 16) ^ b0) & 0xFF] ^
              TB_CRC24B_SLICE1[((crc >> 8) ^ b1) & 0xFF] ^
              TB_CRC24B_SLICE2[(crc ^ b2) & 0xFF] ^
              TB_CRC24B_SLICE3[b3];
    }

    for (; byte_idx < num_bytes; byte_idx++) {
        uint8_t index = ((crc >> 16) ^ bytes[byte_idx]) & 0xFF;
        crc = ((crc << 8) ^ TB_CRC24B_TABLE[index]) & 0xFFFFFF;
    }

    return crc & 0xFFFFFF;
}

__device__ __forceinline__ uint32_t crc24b_update_bytes_parallel_block(
    const uint8_t* bytes,
    int num_bytes,
    uint32_t* s_warp_crcs,
    int* s_warp_bytes
) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    int bytes_per_thread = ((num_bytes + blockDim.x - 1) / blockDim.x + 3) & ~3;
    int my_start = tid * bytes_per_thread;
    int my_end = min(my_start + bytes_per_thread, num_bytes);
    int my_bytes = max(my_end - my_start, 0);

    uint32_t my_crc = 0;
    if (my_bytes > 0) {
        my_crc = crc24b_update_bytes_slicing4(0, bytes + my_start, my_bytes);
    }

    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        uint32_t right_crc = __shfl_down_sync(0xFFFFFFFF, my_crc, offset);
        int right_bytes = __shfl_down_sync(0xFFFFFFFF, my_bytes, offset);
        if ((lane_id & ((offset << 1) - 1)) == 0) {
            my_crc = crc24b_shift_bytes_fast(my_crc, right_bytes) ^ right_crc;
            my_bytes += right_bytes;
        }
    }

    if (lane_id == 0) {
        s_warp_crcs[warp_id] = my_crc;
        s_warp_bytes[warp_id] = my_bytes;
    }
    __syncthreads();

    constexpr int num_warps = 8;
    if (warp_id == 0 && lane_id < num_warps) {
        my_crc = s_warp_crcs[lane_id];
        my_bytes = s_warp_bytes[lane_id];

        #pragma unroll
        for (int offset = 1; offset < num_warps; offset <<= 1) {
            uint32_t right_crc = __shfl_down_sync(0xFF, my_crc, offset);
            int right_bytes = __shfl_down_sync(0xFF, my_bytes, offset);
            if ((lane_id & ((offset << 1) - 1)) == 0) {
                my_crc = crc24b_shift_bytes_fast(my_crc, right_bytes) ^ right_crc;
                my_bytes += right_bytes;
            }
        }
    }

    return (tid == 0) ? (my_crc & 0xFFFFFF) : 0;
}

// CRC16 table for TB CRC computation (TX path - small TBs <= 3824 bits)
// CRC16 polynomial: D^16 + D^12 + D^5 + 1 = 0x1021
#define LOCAL_CRC16_POLY 0x1021
__device__ __constant__ uint16_t TB_CRC16_TABLE[256];
static bool tb_crc16_initialized = false;
static void init_tb_crc16_table() {
    if (tb_crc16_initialized) return;
    uint16_t h_table[256];
    for (int i = 0; i < 256; i++) {
        uint16_t crc = (uint16_t)i << 8;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ LOCAL_CRC16_POLY;
            } else {
                crc <<= 1;
            }
        }
        h_table[i] = crc;
    }
    cudaMemcpyToSymbol(TB_CRC16_TABLE, h_table, sizeof(h_table));
    tb_crc16_initialized = true;
}

// ============================================================================
// Transport Block Encoder Context
// ============================================================================

struct tb_encoder_ctx {
    tb_encoder_config_t config;
    nr_tb_config_t tb_cfg;

    ldpc_encoder_handle_t ldpc_enc;
    rate_matcher_handle_t rate_matcher;
    scrambler_handle_t scrambler;

    // Device buffers
    uint8_t* d_tb_with_crc;        // TB + CRC
    uint32_t* d_cb_bits;           // Segmented code blocks
    uint32_t* d_encoded_bits;      // LDPC encoded
    uint32_t* d_rate_matched;      // Rate matched output
    uint32_t* d_interleaved;       // Bit interleaved (before scrambling)
    uint32_t* d_scrambled;         // Scrambled output
    uint32_t* d_crc;               // Pre-allocated CRC buffer (avoids malloc in hot path)

    // Secondary stream for parallel scrambler execution (Phase 2 optimization)
    cudaStream_t scrambler_stream;
    cudaEvent_t scrambler_done;    // Event to sync scrambler completion

    // Required buffer sizes for current config
    size_t tb_with_crc_size;
    size_t cb_bits_size;
    size_t encoded_bits_size;
    size_t rate_matched_size;
    size_t output_words;           // Total output words

    // Allocated buffer sizes (for reuse without malloc when size fits)
    size_t alloc_tb_with_crc_size;
    size_t alloc_cb_bits_size;
    size_t alloc_encoded_bits_size;
    size_t alloc_rate_matched_size;
    size_t alloc_output_words;

    // Per-CB rate matching (for variable CB lengths per 3GPP TS 38.212 Section 5.4.2.1)
    int nof_short_segments;        // Number of CBs with short E
    int E_short;                   // Rate-matched length for short CBs (floor)
    int E_long;                    // Rate-matched length for long CBs (ceil)
};

// ============================================================================
// Transport Block Decoder Context
// ============================================================================

struct tb_decoder_ctx {
    tb_decoder_config_t config;
    nr_tb_config_t tb_cfg;

    ldpc_decoder_handle_t ldpc_dec;
    rate_matcher_handle_t rate_matcher;
    scrambler_handle_t scrambler;

    // Device buffers (FP32 path - legacy)
    float* d_descrambled_llrs;     // Descrambled LLRs (after scrambler)
    float* d_deinterleaved_llrs;   // De-interleaved LLRs
    float* d_derate_llrs;          // De-rate matched LLRs
    uint32_t* d_decoded_bits;      // LDPC decoded
    uint8_t* d_tb_output;          // Assembled TB

    // FP16 buffers (optimized path - avoids FP32 overhead)
    __half* d_llrs_half;           // FP16 working buffer for deinterleave
    __half* d_derate_llrs_half;    // FP16 de-rate matched LLRs

    // Pre-allocated CRC buffers (avoid cudaMalloc in hot path!)
    uint32_t* d_computed_crc;      // For CRC computation
    uint32_t* d_received_crc;      // For CRC extraction
    int* d_crc_pass;               // GPU-side CRC comparison result
    int* h_crc_pass;               // Pinned host memory for async copy (avoids sync!)

    size_t input_llrs_size;        // Size of input LLR buffer
    size_t derate_llrs_size;
    size_t decoded_bits_size;
    size_t tb_output_size;

    // Per-CB rate matching (for variable CB lengths per 3GPP TS 38.212 Section 5.4.2.1)
    int nof_short_segments;        // Number of CBs with short E
    int E_short;                   // Rate-matched length for short CBs (floor)
    int E_long;                    // Rate-matched length for long CBs (ceil)
};

// ============================================================================
// CUDA Kernels for TB Processing
// ============================================================================

/**
 * @brief Segment transport block into code blocks
 */
__global__ void segment_tb_kernel(
    const uint8_t* __restrict__ d_tb_with_crc,
    uint32_t* __restrict__ d_cb_bits,
    int tb_bits,
    int cb_size_bits,
    int num_cbs,
    int cb_stride_words,  // LDPC input words per CB (includes filler)
    int cb_crc_bits  // CB CRC bits (0 for single CB, 24 for multiple CBs)
) {
    // OPTIMIZED: Each thread processes one word (32 bits) - NO atomics, NO memset needed!
    int cb_idx = blockIdx.x;
    if (cb_idx >= num_cbs) return;

    // Data bits per CB = ceil(tb_bits / num_cbs) = amount to copy from TB+CRC buffer
    // Info bits per CB = data_per_cb - cb_crc_bits = where CB CRC goes
    // tb_bits here is TB + TB_CRC total
    int data_per_cb = (tb_bits + num_cbs - 1) / num_cbs;
    int info_bits_per_cb = data_per_cb - cb_crc_bits;
    int cb_start_bit = cb_idx * data_per_cb;

    // Process ALL words in the LDPC input buffer (including filler word region)
    for (int word_idx = threadIdx.x; word_idx < cb_stride_words; word_idx += blockDim.x) {
        uint32_t word = 0;

        // Pack 32 bits into this word (data or filler=0)
        // NOTE: Use MSB-first bit ordering within each byte to match srsRAN's bit_buffer format
        int base_bit = word_idx * 32;
        for (int b = 0; b < 32; b++) {
            int bit_idx = base_bit + b;
            if (bit_idx >= cb_size_bits) break;  // Past CB size = filler region

            // Only copy TB data for the data region (0 to data_per_cb-1)
            // Filler region (data_per_cb to cb_size_bits-1) stays 0 for CRC kernel
            int tb_bit_idx = cb_start_bit + bit_idx;
            if (bit_idx < data_per_cb && tb_bit_idx < tb_bits) {
                int byte_idx = tb_bit_idx / 8;
                int bit_pos = tb_bit_idx % 8;
                uint8_t bit = (d_tb_with_crc[byte_idx] >> (7 - bit_pos)) & 1;
                // MSB-first within each byte: bit 0 of byte goes to bit 7, bit 1 to bit 6, etc.
                int byte_in_word = b / 8;
                int bit_in_byte = 7 - (b % 8);
                int out_bit_pos = byte_in_word * 8 + bit_in_byte;
                word |= (uint32_t)bit << out_bit_pos;
            }
            // CB CRC region and filler bits are 0 (implicit in word=0)
        }

        // Direct store - no atomics needed!
        d_cb_bits[cb_idx * cb_stride_words + word_idx] = word;
    }
}

/**
 * @brief Compute and attach CRC24B to each code block (multi-CB case only)
 *
 * Per 3GPP TS 38.212 Section 5.2.2, when TB is segmented into multiple CBs,
 * each CB gets CRC24B attached after the info bits.
 *
 * The CB data is stored as packed uint32_t with sequential bit order:
 * - Bit i is at word[i/32] position (i%32)
 * - Info bits are at positions 0 to (cb_size_bits - 24 - 1)
 * - CRC24B goes immediately after info bits (positions info_bits to info_bits+23)
 * - Filler bits are at positions info_bits+24 to cb_size_bits-1
 *
 * CRC24B is computed MSB-first over the info bits, then stored MSB-first.
 */
__global__ void attach_cb_crc_kernel(
    uint32_t* __restrict__ d_cb_bits,
    int tb_bits,           // Total bits of TB+CRC (for computing info_bits_per_cb)
    int cb_size_bits,      // Total bits per CB (LDPC input size = Kb*Z)
    int num_cbs,
    int cb_stride_words    // Words per CB in buffer
) {
    int cb_idx = blockIdx.x;
    if (cb_idx >= num_cbs) return;

    // Only thread 0 per CB computes CRC (sequential algorithm)
    if (threadIdx.x != 0) return;

    uint32_t* cb_ptr = d_cb_bits + cb_idx * cb_stride_words;
    // Info bits per CB = ceil(tb_bits / num_cbs) - 24 (CB CRC)
    int info_bits = (tb_bits + num_cbs - 1) / num_cbs - 24;
    int info_bytes = (info_bits + 7) / 8;

    // Compute CRC24B over info bits, MSB-first (as per 3GPP)
    // The bits are stored sequentially: bit 0 at position 0, bit 1 at position 1, etc.
    // We need to process them in bytes, MSB-first within each byte
    uint32_t crc = 0;

    // With MSB-first bit ordering within each byte, bytes are already contiguous
    // Just read them directly from memory
    uint8_t* cb_bytes = (uint8_t*)cb_ptr;
    for (int byte_idx = 0; byte_idx < info_bytes; byte_idx++) {
        uint8_t byte_val = cb_bytes[byte_idx];
        // Handle partial last byte
        if (byte_idx == info_bytes - 1 && (info_bits % 8) != 0) {
            // Mask out unused bits (they should be 0 anyway)
            int used_bits = info_bits % 8;
            byte_val &= (0xFF << (8 - used_bits));
        }

        // Update CRC with this byte
        uint8_t index = ((crc >> 16) ^ byte_val) & 0xFF;
        crc = ((crc << 8) ^ TB_CRC24B_TABLE[index]) & 0xFFFFFF;
    }

    // Handle non-byte-aligned input: reverse CRC bits
    // This compensates for the zero-padding of the partial last byte
    int remainder_bits = info_bits % 8;
    if (remainder_bits != 0) {
        int reverse_bits = 8 - remainder_bits;
        for (int m = 0; m < reverse_bits; m++) {
            if (crc & 1) {
                crc = (crc ^ CRC24B_POLY) >> 1;
            } else {
                crc = crc >> 1;
            }
        }
        crc &= 0xFFFFFF;
    }

    // Store CRC24B at the end of info bits (positions info_bits to info_bits+23)
    // With MSB-first byte ordering, we can write CRC bytes directly
    int crc_byte_start = info_bits / 8;
    int crc_bit_offset = info_bits % 8;

    if (crc_bit_offset == 0) {
        // CRC starts at byte boundary - simple case
        cb_bytes[crc_byte_start] = (crc >> 16) & 0xFF;
        cb_bytes[crc_byte_start + 1] = (crc >> 8) & 0xFF;
        cb_bytes[crc_byte_start + 2] = crc & 0xFF;
    } else {
        // CRC starts mid-byte - need to merge
        // This case shouldn't happen for standard CB sizes, but handle it anyway
        for (int i = 0; i < 24; i++) {
            int bit_pos = info_bits + i;
            int byte_idx_out = bit_pos / 8;
            int bit_in_byte = 7 - (bit_pos % 8);  // MSB-first within byte
            uint8_t crc_bit = (crc >> (23 - i)) & 1;
            if (crc_bit) {
                cb_bytes[byte_idx_out] |= (1 << bit_in_byte);
            }
        }
    }
}

// ============================================================================
// FUSED TB Processing Kernel (TB CRC attach + Segment + CB CRC)
// ============================================================================

/**
 * @brief FUSED kernel: Attach TB CRC + Segment TB into CBs + Attach CB CRC
 *
 * This kernel combines 3 separate operations into 1 kernel launch:
 *   1. Attach pre-computed TB CRC to TB buffer
 *   2. Segment TB into code blocks
 *   3. Compute and attach CB CRC24B to each CB
 *
 * Launch: <<<num_cbs, 256>>>
 * Each block handles one CB. Thread 0 also handles CRC computation.
 *
 * @param d_tb_input     Input TB data (without CRC)
 * @param d_tb_crc       Pre-computed TB CRC (24 or 16 bits in uint32_t)
 * @param d_cb_output    Output CB buffer (packed uint32_t per CB)
 * @param tb_size_bits   TB size in bits (A, not including CRC)
 * @param tb_crc_bits    TB CRC bits (24 or 16)
 * @param cb_size_bits   CB size in bits (K, including CB CRC if any)
 * @param cb_crc_bits    CB CRC bits (24 for multi-CB, 0 for single CB)
 * @param num_cbs        Number of code blocks
 * @param cb_stride_words Words per CB in output buffer
 */
__global__ void __launch_bounds__(256, 4) fused_tb_segment_crc_kernel(
    const uint8_t* __restrict__ d_tb_input,
    const uint32_t* __restrict__ d_tb_crc,
    uint32_t* __restrict__ d_cb_output,
    int tb_size_bits,
    int tb_crc_bits,
    int cb_size_bits,
    int cb_crc_bits,
    int num_cbs,
    int cb_stride_words,
    int num_filler_bits  // F - filler bits per CB from LDPC config
) {
    int cb_idx = blockIdx.x;
    if (cb_idx >= num_cbs) return;

    // Shared memory for TB CRC bytes (up to 3 bytes for CRC24)
    __shared__ uint8_t s_tb_crc[4];
    __shared__ uint32_t s_cb_crc_warp_crcs[8];
    __shared__ int s_cb_crc_warp_bytes[8];

    // Thread 0 loads TB CRC into shared memory
    if (threadIdx.x == 0) {
        uint32_t crc_val = *d_tb_crc;
        if (tb_crc_bits == 24) {
            s_tb_crc[0] = (crc_val >> 16) & 0xFF;
            s_tb_crc[1] = (crc_val >> 8) & 0xFF;
            s_tb_crc[2] = crc_val & 0xFF;
        } else {  // 16 bits
            s_tb_crc[0] = (crc_val >> 8) & 0xFF;
            s_tb_crc[1] = crc_val & 0xFF;
        }
    }
    __syncthreads();

    // Calculate this CB's portion of TB
    // Use LDPC-derived segment size (matches CPU segmenter behavior):
    // - K = cb_size_bits (total LDPC input bits = Kb * Z)
    // - K' = K - F (actual info bits per CB including CB CRC)
    // - cb_info_bits = K' - CB_CRC (data bits from TB per CB)
    //
    // For multi-CB last CB: cb_info_bits_last = cb_info_bits - TB_CRC (TB CRC handled separately)
    // For single-CB (cb_crc_bits == 0): include TB CRC in CB data (no subtraction)
    int K_prime = cb_size_bits - num_filler_bits;  // Actual info bits per CB (incl CB CRC)
    int cb_info_bits = K_prime - cb_crc_bits;       // Data bits from TB per CB

    // All CBs (including last) copy cb_info_bits of data
    // The data comes from the combined TB+TB_CRC stream
    int this_cb_data_bits = cb_info_bits;

    // Compute cb_start_bit by summing previous CBs' data bits
    int cb_start_bit = cb_idx * cb_info_bits;


    // Output pointer for this CB
    uint32_t* cb_ptr = d_cb_output + cb_idx * cb_stride_words;

    bool byte_aligned_fast_path = ((cb_start_bit | this_cb_data_bits | tb_size_bits | tb_crc_bits) & 7) == 0;

    // Phase 1: Segment TB data into this CB (parallel across threads).
    // Common PDSCH transport blocks are byte-aligned, so preserve byte order
    // directly instead of rebuilding every output word bit-by-bit.
    if (byte_aligned_fast_path) {
        int cb_start_byte = cb_start_bit >> 3;
        int data_bytes = this_cb_data_bits >> 3;
        int tb_bytes = tb_size_bits >> 3;

        for (int word_idx = threadIdx.x; word_idx < cb_stride_words; word_idx += blockDim.x) {
            int byte_base = word_idx << 2;
            uint32_t word = 0;

            #pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                int local_byte = byte_base + byte;
                uint8_t value = 0;

                if (local_byte < data_bytes) {
                    int tb_byte_idx = cb_start_byte + local_byte;
                    if (tb_byte_idx < tb_bytes) {
                        value = d_tb_input[tb_byte_idx];
                    } else if (tb_byte_idx < tb_bytes + (tb_crc_bits >> 3)) {
                        value = s_tb_crc[tb_byte_idx - tb_bytes];
                    }
                }

                word |= static_cast<uint32_t>(value) << (8 * byte);
            }

            cb_ptr[word_idx] = word;
        }
    } else {
        for (int word_idx = threadIdx.x; word_idx < cb_stride_words; word_idx += blockDim.x) {
            uint32_t word = 0;

            // Pack 32 bits into this word
            int base_bit = word_idx * 32;
            for (int b = 0; b < 32; b++) {
                int bit_idx = base_bit + b;
                if (bit_idx >= cb_size_bits) break;  // Past CB size = filler region

                // Only copy data for data region (0 to this_cb_data_bits-1)
                // CB CRC goes at position this_cb_data_bits
                // Filler region after CB CRC stays 0
                if (bit_idx < this_cb_data_bits) {
                    int tb_bit_idx = cb_start_bit + bit_idx;
                    uint8_t bit_val = 0;

                    // Read from TB data or TB CRC depending on position
                    // TB CRC is distributed across CBs for both single-CB and multi-CB cases
                    if (tb_bit_idx < tb_size_bits) {
                        // Read from TB data
                        int byte_idx = tb_bit_idx / 8;
                        int bit_pos = tb_bit_idx % 8;
                        bit_val = (d_tb_input[byte_idx] >> (7 - bit_pos)) & 1;
                    } else if (tb_bit_idx < tb_size_bits + tb_crc_bits) {
                        // Read from TB CRC (bits after TB data, distributed across CBs)
                        int crc_bit_idx = tb_bit_idx - tb_size_bits;
                        // TB CRC is stored MSB-first in s_tb_crc (byte 0 = MSB)
                        int crc_byte_idx = crc_bit_idx / 8;
                        int crc_bit_pos = crc_bit_idx % 8;
                        bit_val = (s_tb_crc[crc_byte_idx] >> (7 - crc_bit_pos)) & 1;
                    }
                    // else: past TB+CRC = 0 (shouldn't happen for valid segmentation)

                    // MSB-first within each byte
                    int byte_in_word = b / 8;
                    int bit_in_byte = 7 - (b % 8);
                    int out_bit_pos = byte_in_word * 8 + bit_in_byte;
                    word |= (uint32_t)bit_val << out_bit_pos;
                }
                // CB CRC region and filler bits are 0 (implicit)
            }

            cb_ptr[word_idx] = word;
        }
    }

    // Sync before CB CRC computation
    __syncthreads();

    // Phase 2: Compute and attach CB CRC24B. All threads compute the
    // full-byte CRC region; thread 0 handles partial-byte tails and writes.
    if (cb_crc_bits > 0) {
        // Compute CRC over the data bits we actually copied (this_cb_data_bits)
        // CB CRC is placed at position this_cb_data_bits
        int info_bits = this_cb_data_bits;
        int info_bytes = info_bits / 8;

        // Compute CRC24B over info bits
        uint8_t* cb_bytes = (uint8_t*)cb_ptr;
        uint32_t crc = crc24b_update_bytes_parallel_block(cb_bytes, info_bytes, s_cb_crc_warp_crcs, s_cb_crc_warp_bytes);
        __syncthreads();

        int remainder_bits = info_bits % 8;
        if (threadIdx.x == 0) {
            if (remainder_bits != 0) {
                uint8_t byte_val = cb_bytes[info_bytes] & (0xFF << (8 - remainder_bits));
                uint8_t index = ((crc >> 16) ^ byte_val) & 0xFF;
                crc = ((crc << 8) ^ TB_CRC24B_TABLE[index]) & 0xFFFFFF;
            }

            // Handle non-byte-aligned input: reverse CRC bits
            // This compensates for the zero-padding of the partial last byte
            if (remainder_bits != 0) {
                int reverse_bits = 8 - remainder_bits;
                for (int m = 0; m < reverse_bits; m++) {
                    if (crc & 1) {
                        crc = (crc ^ CRC24B_POLY) >> 1;
                    } else {
                        crc = crc >> 1;
                    }
                }
                crc &= 0xFFFFFF;
            }

            // Store CRC24B at the end of info bits
            int crc_byte_start = info_bits / 8;
            int crc_bit_offset = info_bits % 8;

            if (crc_bit_offset == 0) {
                // CRC starts at byte boundary
                cb_bytes[crc_byte_start] = (crc >> 16) & 0xFF;
                cb_bytes[crc_byte_start + 1] = (crc >> 8) & 0xFF;
                cb_bytes[crc_byte_start + 2] = crc & 0xFF;
            } else {
                // CRC starts mid-byte (rare case)
                for (int i = 0; i < 24; i++) {
                    int bit_pos = info_bits + i;
                    int byte_idx_out = bit_pos / 8;
                    int bit_in_byte = 7 - (bit_pos % 8);
                    uint8_t crc_bit = (crc >> (23 - i)) & 1;
                    if (crc_bit) {
                        cb_bytes[byte_idx_out] |= (1 << bit_in_byte);
                    }
                }
            }
        }
    }
}

// ============================================================================
// FULLY-FUSED TB CRC + Segment Kernel (eliminates separate CRC kernel)
// ============================================================================

/**
 * @brief FULLY-FUSED kernel: Compute TB CRC + Segment + CB CRC in ONE launch
 *
 * This kernel eliminates the separate crc24a_compute kernel call by computing
 * the TB CRC directly within block 0, then signaling other blocks to proceed.
 *
 * Optimizations:
 *   1. Block 0 thread 0 computes TB CRC using slicing-by-4 (4x faster than byte-at-a-time)
 *   2. Uses __threadfence() + atomic flag for inter-block synchronization
 *   3. All blocks then segment in parallel
 *
 * Launch: <<<num_cbs, 256>>>
 *
 * @param d_tb_input     Input TB data (without CRC)
 * @param d_crc_out      Output CRC value (uint32_t, also used as sync flag)
 * @param d_cb_output    Output CB buffer (packed uint32_t per CB)
 * @param tb_size_bits   TB size in bits (A, not including CRC)
 * @param tb_crc_bits    TB CRC bits (24 or 16)
 * @param cb_size_bits   CB size in bits (K, including CB CRC if any)
 * @param cb_crc_bits    CB CRC bits (24 for multi-CB, 0 for single CB)
 * @param num_cbs        Number of code blocks
 * @param cb_stride_words Words per CB in output buffer
 */
__global__ void __launch_bounds__(256, 4) fully_fused_tb_crc_segment_kernel(
    const uint8_t* __restrict__ d_tb_input,
    uint32_t* __restrict__ d_crc_out,
    uint32_t* __restrict__ d_cb_output,
    int tb_size_bits,
    int tb_crc_bits,
    int cb_size_bits,
    int cb_crc_bits,
    int num_cbs,
    int cb_stride_words,
    int num_filler_bits  // F - filler bits per CB from LDPC config
) {
    int cb_idx = blockIdx.x;
    if (cb_idx >= num_cbs) return;

    // Shared memory for TB CRC bytes (up to 3 bytes for CRC24) and sync flag
    __shared__ uint8_t s_tb_crc[4];
    __shared__ uint32_t s_crc_ready;
    __shared__ uint32_t s_cb_crc_warp_crcs[8];
    __shared__ int s_cb_crc_warp_bytes[8];

    // Initialize sync flag
    if (threadIdx.x == 0) {
        s_crc_ready = 0;
    }
    __syncthreads();

    // Block 0, thread 0 computes the TB CRC using simple byte-by-byte algorithm
    // Handles both CRC24A (tb_crc_bits=24) and CRC16 (tb_crc_bits=16)
    if (cb_idx == 0 && threadIdx.x == 0) {
        int num_bytes = (tb_size_bits + 7) / 8;
        uint32_t crc = 0;

        // Simple byte-by-byte CRC computation
        for (int i = 0; i < num_bytes; i++) {
            uint8_t byte = d_tb_input[i];
            // Mask partial last byte for non-byte-aligned TBS
            if (i == num_bytes - 1 && tb_size_bits % 8 != 0) {
                int remaining = tb_size_bits % 8;
                uint8_t mask = (0xFF << (8 - remaining)) & 0xFF;
                byte &= mask;
            }

            if (tb_crc_bits == 24) {
                uint8_t index = ((crc >> 16) ^ byte) & 0xFF;
                crc = ((crc << 8) ^ TB_CRC24A_TABLE[index]) & 0xFFFFFF;
            } else {  // CRC16
                uint8_t index = ((crc >> 8) ^ byte) & 0xFF;
                crc = ((crc << 8) ^ TB_CRC16_TABLE[index]) & 0xFFFF;
            }
        }

        // Store CRC to global memory with release semantics
        *d_crc_out = crc | 0x80000000;  // Set high bit as "ready" flag
        __threadfence();  // Ensure all blocks see the write

        // Also store to shared memory for this block (MSB-first)
        if (tb_crc_bits == 24) {
            s_tb_crc[0] = (crc >> 16) & 0xFF;
            s_tb_crc[1] = (crc >> 8) & 0xFF;
            s_tb_crc[2] = crc & 0xFF;
        } else {  // CRC16
            s_tb_crc[0] = (crc >> 8) & 0xFF;
            s_tb_crc[1] = crc & 0xFF;
        }
        s_crc_ready = 1;
    }

    // Non-block-0 blocks wait for CRC to be computed
    if (cb_idx != 0 && threadIdx.x == 0) {
        // Spin-wait for CRC ready (check high bit)
        uint32_t val;
        do {
            val = atomicAdd(d_crc_out, 0);  // Atomic read
        } while ((val & 0x80000000) == 0);

        // Extract CRC and store to shared memory (MSB-first)
        uint32_t crc = (tb_crc_bits == 24) ? (val & 0x00FFFFFF) : (val & 0x0000FFFF);
        if (tb_crc_bits == 24) {
            s_tb_crc[0] = (crc >> 16) & 0xFF;
            s_tb_crc[1] = (crc >> 8) & 0xFF;
            s_tb_crc[2] = crc & 0xFF;
        } else {  // CRC16
            s_tb_crc[0] = (crc >> 8) & 0xFF;
            s_tb_crc[1] = crc & 0xFF;
        }
        s_crc_ready = 1;
    }

    __syncthreads();  // All threads wait for CRC to be in shared memory

    // Calculate this CB's portion of TB using LDPC-derived segment size
    // (matches fused_tb_segment_crc_kernel behavior)
    // - K = cb_size_bits (total LDPC input bits = Kb * Z)
    // - K' = K - F (actual info bits per CB including CB CRC)
    // - cb_info_bits = K' - CB_CRC (data bits from TB per CB)
    int K_prime = cb_size_bits - num_filler_bits;  // Actual info bits per CB (incl CB CRC)
    int cb_info_bits = K_prime - cb_crc_bits;       // Data bits from TB per CB

    // All CBs copy cb_info_bits worth of data from the combined TB+TB_CRC stream
    int this_cb_data_bits = cb_info_bits;
    int cb_start_bit = cb_idx * cb_info_bits;
    int tb_with_crc_bits = tb_size_bits + tb_crc_bits;

    // Output pointer for this CB
    uint32_t* cb_ptr = d_cb_output + cb_idx * cb_stride_words;

    // Phase 1: Segment TB data into this CB (parallel across threads)
    // Uses direct global memory reads - L2 cache provides good hit rate for sequential access
    for (int word_idx = threadIdx.x; word_idx < cb_stride_words; word_idx += blockDim.x) {
        uint32_t word = 0;

        int base_bit = word_idx * 32;
        for (int b = 0; b < 32; b++) {
            int bit_idx = base_bit + b;
            if (bit_idx >= cb_size_bits) break;

            if (bit_idx < this_cb_data_bits) {
                int tb_bit_idx = cb_start_bit + bit_idx;
                uint8_t bit_val = 0;

                if (tb_bit_idx < tb_size_bits) {
                    // Read from TB data (global memory - L2 cached)
                    int byte_idx = tb_bit_idx / 8;
                    int bit_pos = tb_bit_idx % 8;
                    bit_val = (d_tb_input[byte_idx] >> (7 - bit_pos)) & 1;
                } else if (tb_bit_idx < tb_with_crc_bits) {
                    // Read from TB CRC (shared memory)
                    int crc_bit_idx = tb_bit_idx - tb_size_bits;
                    int crc_byte_idx = crc_bit_idx / 8;
                    int crc_bit_pos = crc_bit_idx % 8;
                    bit_val = (s_tb_crc[crc_byte_idx] >> (7 - crc_bit_pos)) & 1;
                }

                int byte_in_word = b / 8;
                int bit_in_byte = 7 - (b % 8);
                int out_bit_pos = byte_in_word * 8 + bit_in_byte;
                word |= (uint32_t)bit_val << out_bit_pos;
            }
        }

        cb_ptr[word_idx] = word;
    }

    __syncthreads();

    // Phase 2: Compute and attach CB CRC24B.
    if (cb_crc_bits > 0) {
        // Compute CRC over the data bits we actually copied (this_cb_data_bits)
        int info_bits = this_cb_data_bits;
        int info_bytes = info_bits / 8;

        uint8_t* cb_bytes = (uint8_t*)cb_ptr;
        uint32_t crc = crc24b_update_bytes_parallel_block(cb_bytes, info_bytes, s_cb_crc_warp_crcs, s_cb_crc_warp_bytes);
        __syncthreads();

        int remainder_bits = info_bits % 8;
        if (threadIdx.x == 0) {
            if (remainder_bits != 0) {
                uint8_t byte_val = cb_bytes[info_bytes] & (0xFF << (8 - remainder_bits));
                uint8_t index = ((crc >> 16) ^ byte_val) & 0xFF;
                crc = ((crc << 8) ^ TB_CRC24B_TABLE[index]) & 0xFFFFFF;
            }

            // Handle non-byte-aligned input: reverse CRC bits
            if (remainder_bits != 0) {
                int reverse_bits = 8 - remainder_bits;
                for (int m = 0; m < reverse_bits; m++) {
                    if (crc & 1) {
                        crc = (crc ^ CRC24B_POLY) >> 1;
                    } else {
                        crc = crc >> 1;
                    }
                }
                crc &= 0xFFFFFF;
            }

            int crc_byte_start = info_bits / 8;
            int crc_bit_offset = info_bits % 8;

            if (crc_bit_offset == 0) {
                cb_bytes[crc_byte_start] = (crc >> 16) & 0xFF;
                cb_bytes[crc_byte_start + 1] = (crc >> 8) & 0xFF;
                cb_bytes[crc_byte_start + 2] = crc & 0xFF;
            } else {
                for (int i = 0; i < 24; i++) {
                    int bit_pos = info_bits + i;
                    int byte_idx_out = bit_pos / 8;
                    int bit_in_byte = 7 - (bit_pos % 8);
                    uint8_t crc_bit = (crc >> (23 - i)) & 1;
                    if (crc_bit) {
                        cb_bytes[byte_idx_out] |= (1 << bit_in_byte);
                    }
                }
            }
        }
    }
}

/**
 * @brief De-segment code blocks back to transport block
 */
__global__ void desegment_cb_kernel(
    const uint32_t* __restrict__ d_decoded_bits,
    uint8_t* __restrict__ d_tb_output,
    int tb_bits,
    int cb_size_bits,
    int num_cbs,
    int cb_stride_words,
    int cb_crc_bits  // CB CRC bits (0 for single CB, 24 for multiple CBs)
) {
    int cb_idx = blockIdx.x;
    if (cb_idx >= num_cbs) return;

    // Info bits per CB = ceil(tb_bits / num_cbs)
    // tb_bits here is TB + TB_CRC, matching CPU segmenter's cb_info_bits
    // NOTE: Do NOT subtract cb_crc_bits here - the ceiling division already gives
    // the correct TB data bits per CB. The CB CRC is NOT part of the TB data stream.
    int info_bits_per_cb = (tb_bits + num_cbs - 1) / num_cbs;
    int cb_start_bit = cb_idx * info_bits_per_cb;

    for (int bit_idx = threadIdx.x; bit_idx < info_bits_per_cb; bit_idx += blockDim.x) {
        int tb_bit_idx = cb_start_bit + bit_idx;
        if (tb_bit_idx >= tb_bits) continue;

        // Read bit from decoded CB (MSB-first per byte, matching decoder output)
        int word_idx = bit_idx / 32;
        int bit_in_word = bit_idx % 32;
        int word_bit = (bit_in_word & ~7) | (7 - (bit_in_word & 7));
        uint32_t word = d_decoded_bits[cb_idx * cb_stride_words + word_idx];
        uint8_t bit = (word >> word_bit) & 1;

        // Write to TB output (byte-packed, MSB-first internal representation)
        int byte_idx = tb_bit_idx / 8;
        int byte_bit = 7 - (tb_bit_idx % 8);  // MSB-first: bit 0 goes to position 7

        atomicOr((uint32_t*)&d_tb_output[byte_idx & ~3],
                 (uint32_t)bit << (byte_bit + 8 * (byte_idx & 3)));
    }
}

/**
 * @brief Convert bytes to packed bits
 */
__global__ void bytes_to_bits_kernel(
    const uint8_t* __restrict__ d_bytes,
    uint32_t* __restrict__ d_bits,
    int num_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bytes) return;

    uint8_t byte = d_bytes[idx];

    // Pack 8 bits from this byte
    for (int b = 0; b < 8; b++) {
        int bit_idx = idx * 8 + b;
        int word_idx = bit_idx / 32;
        int word_bit = bit_idx % 32;
        uint8_t bit = (byte >> (7 - b)) & 1;
        atomicOr(&d_bits[word_idx], (uint32_t)bit << word_bit);
    }
}

/**
 * @brief Convert packed bits to bytes
 */
__global__ void bits_to_bytes_kernel(
    const uint32_t* __restrict__ d_bits,
    uint8_t* __restrict__ d_bytes,
    int num_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bytes) return;

    uint8_t byte = 0;
    for (int b = 0; b < 8; b++) {
        int bit_idx = idx * 8 + b;
        int word_idx = bit_idx / 32;
        int word_bit = bit_idx % 32;
        uint8_t bit = (d_bits[word_idx] >> word_bit) & 1;
        byte |= bit << (7 - b);
    }

    d_bytes[idx] = byte;
}

/**
 * @brief Clear unused bits in the last byte of a buffer (for non-byte-aligned TBS)
 */
__global__ void clear_unused_bits_kernel(
    uint8_t* __restrict__ d_bytes,
    int byte_idx,
    uint8_t mask
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_bytes[byte_idx] &= mask;
    }
}

/**
 * @brief Reorder LLRs from modulator bit order to output bit order
 *
 * The modulator reads bits from uint32_t* with FLAT indexing, but the encoder
 * packs output bytes with MSB-first bit ordering. This creates a bit reversal
 * within each byte:
 *   - Modulator bit m corresponds to encoder output bit (m/8)*8 + 7 - (m%8)
 *
 * This kernel reorders LLRs so that output LLR i represents encoder output bit i.
 * This is needed before de-interleaving which expects LLRs in output bit order.
 *
 * Formula: output[i] = input[(i/8)*8 + 7 - (i%8)]
 */
__global__ void reorder_llrs_byte_bit_reverse_kernel(
    const float* __restrict__ d_input_llrs,
    float* __restrict__ d_output_llrs,
    int num_llrs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_llrs) return;

    // Map output index to input index (reverse bits within each byte)
    int byte_idx = idx / 8;
    int bit_in_byte = idx % 8;
    int src_idx = byte_idx * 8 + (7 - bit_in_byte);

    d_output_llrs[idx] = d_input_llrs[src_idx];
}

/**
 * @brief FP16 bit de-interleaving kernel for TB decoder
 *
 * Same as rate_matching deinterleave but for __half type.
 * Per 3GPP TS 38.212 Section 5.4.2.2 (inverse of interleaving):
 * Output position out_idx = i*K + j maps to input position j*Qm + i
 */
__global__ void bit_deinterleave_llr_half_kernel(
    const __half* __restrict__ d_input_llrs,
    __half* __restrict__ d_output_llrs,
    int E,
    int Q_m,
    int num_cbs
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= E) return;

    int K = E / Q_m;  // Number of symbols
    int i = out_idx / K;   // Row index
    int j = out_idx % K;   // Column index
    int in_idx = j * Q_m + i;

    d_output_llrs[cb_idx * E + out_idx] = d_input_llrs[cb_idx * E + in_idx];
}

/**
 * @brief FUSED Descramble + Deinterleave kernel for FP16 LLRs
 *
 * This kernel combines descrambling and deinterleaving in a single pass:
 * 1. Read LLR from interleaved position (where scrambling was applied on TX)
 * 2. Apply descrambling (flip sign if scramble bit = 1)
 * 3. Write to deinterleaved output position
 *
 * This is more efficient than separate descramble → deinterleave because:
 * - Single memory pass instead of two
 * - Scramble sequence lookup happens at correct (interleaved) index
 *
 * Per 3GPP TS 38.211/38.212: TX order is interleave→scramble→modulate
 * So RX order must be demod→descramble→deinterleave
 *
 * @param scramble_bit_offset Starting bit offset in scramble sequence (for non-uniform E)
 */
__global__ void bit_descramble_deinterleave_llr_half_kernel(
    const __half* __restrict__ d_input_llrs,
    __half* __restrict__ d_output_llrs,
    const uint32_t* __restrict__ d_scramble_seq,
    int E,
    int Q_m,
    int num_cbs,
    int scramble_bit_offset
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= E) return;

    // Deinterleave index calculation (3GPP TS 38.212 Section 5.4.2.2)
    int K = E / Q_m;  // Number of symbols
    int i = out_idx / K;   // Row index
    int j = out_idx % K;   // Column index
    int in_idx = j * Q_m + i;

    // Global bit index for scrambling sequence lookup
    // scramble_bit_offset accounts for CBs processed in earlier calls (non-uniform E case)
    int global_bit_idx = scramble_bit_offset + cb_idx * E + in_idx;

    // Get scramble bit (packed in uint32_t words)
    int word_idx = global_bit_idx / 32;
    int bit_pos = global_bit_idx % 32;
    uint32_t scramble_word = d_scramble_seq[word_idx];
    bool scramble_bit = (scramble_word >> bit_pos) & 1;

    // Load LLR from interleaved position
    __half llr = d_input_llrs[cb_idx * E + in_idx];

    // Descramble: flip sign if scramble bit is 1
    // For FP16, XOR the sign bit (bit 15) with scramble_bit
    if (scramble_bit) {
        unsigned short* llr_bits = reinterpret_cast<unsigned short*>(&llr);
        *llr_bits ^= 0x8000;  // Flip sign bit
    }

    // Write to deinterleaved position
    d_output_llrs[cb_idx * E + out_idx] = llr;
}

/**
 * @brief Bit de-interleaving kernel for INT8 LLRs
 * Per 3GPP TS 38.212 Section 5.4.2.2 (inverse of interleaving):
 * Output position out_idx = i*K + j maps to input position j*Qm + i
 */
__global__ void bit_deinterleave_llr_int8_kernel(
    const int8_t* __restrict__ d_input_llrs,
    int8_t* __restrict__ d_output_llrs,
    int E,
    int Q_m,
    int num_cbs
) {
    int cb_idx = blockIdx.y;
    if (cb_idx >= num_cbs) return;

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= E) return;

    int K = E / Q_m;  // Number of symbols
    int i = out_idx / K;   // Row index
    int j = out_idx % K;   // Column index
    int in_idx = j * Q_m + i;

    d_output_llrs[cb_idx * E + out_idx] = d_input_llrs[cb_idx * E + in_idx];
}

#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
/**
 * @brief Diagnostic kernel to print CB boundary values.
 */
__global__ void pdsch_cb_boundary_diagnostics_kernel(
    const uint32_t* __restrict__ d_rate_matched,
    const uint32_t* __restrict__ d_scramble_seq,
    int nof_short,
    int E_short,
    int E_long,
    int words_per_cb,
    int Q_m,
    int total_bits
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Print values around the CB boundary.
    int short_cb_total_bits = nof_short * E_short;
    printf("[GPU CB Boundary Diagnostics] short_cb_total_bits=%d, words_per_cb=%d\n",
           short_cb_total_bits,
           words_per_cb);

    for (int out_bit_idx = short_cb_total_bits - 5; out_bit_idx < short_cb_total_bits + 5 && out_bit_idx < total_bits; out_bit_idx++) {
        if (out_bit_idx < 0) continue;

        // Step 1: Find CB and bit position
        int cb_idx, bit_in_cb, bits_per_cb;
        if (out_bit_idx < short_cb_total_bits) {
            cb_idx = out_bit_idx / E_short;
            bit_in_cb = out_bit_idx % E_short;
            bits_per_cb = E_short;
        } else {
            int offset = out_bit_idx - short_cb_total_bits;
            cb_idx = nof_short + offset / E_long;
            bit_in_cb = offset % E_long;
            bits_per_cb = E_long;
        }

        // Interleaving
        int K = bits_per_cb / Q_m;
        int j = bit_in_cb / Q_m;
        int i = bit_in_cb % Q_m;
        int in_bit_in_cb = i * K + j;

        // Read from rate matched buffer (MSB-first)
        int in_word = cb_idx * words_per_cb + in_bit_in_cb / 32;
        int in_bit_in_word = in_bit_in_cb % 32;
        int in_byte_in_word = in_bit_in_word / 8;
        int in_bit_in_byte = 7 - (in_bit_in_word % 8);
        int in_bit_pos = in_byte_in_word * 8 + in_bit_in_byte;
        uint32_t rm_bit = (d_rate_matched[in_word] >> in_bit_pos) & 1u;

        // Scrambling bit (MSB-first to match gold_sequence_generate_kernel)
        int scr_word_idx = out_bit_idx / 32;
        int scr_bit_pos = 31 - (out_bit_idx % 32);  // MSB-first: bit 0 at position 31
        uint32_t scr_bit = (d_scramble_seq[scr_word_idx] >> scr_bit_pos) & 1u;

        printf("  out[%d]: cb=%d, bit_in_cb=%d, K=%d, in_bit=%d, in_word=%d, rm=%d, scr=%d, final=%d\n",
               out_bit_idx, cb_idx, bit_in_cb, K, in_bit_in_cb, in_word, rm_bit, scr_bit, rm_bit ^ scr_bit);
    }
}
#endif

/**
 * @brief FUSED kernel: interleave + scramble + pack_to_bytes in ONE pass!
 *
 * This replaces 3 separate kernel launches with 1, eliminating ~30-40µs of
 * kernel launch overhead. Each thread processes 4 output bytes (32 bits = 1 word).
 *
 * Input: d_rate_matched (per-CB word-aligned layout)
 * Output: d_output_bytes (packed scrambled interleaved bytes)
 *
 * Supports variable per-CB E values per 3GPP TS 38.212 Section 5.4.2.1:
 *   - CBs 0 to nof_short-1 have E_short bits
 *   - CBs nof_short to num_cbs-1 have E_long bits
 */
__global__ void interleave_scramble_pack_kernel(
    const uint32_t* __restrict__ d_rate_matched,  // Input: rate matched bits (per-CB layout)
    const uint32_t* __restrict__ d_scramble_seq,  // Scrambling sequence
    uint8_t* __restrict__ d_output_bytes,         // Output: packed bytes
    int nof_short,      // Number of CBs with E_short
    int E_short,        // Rate-matched length for short CBs
    int E_long,         // Rate-matched length for long CBs
    int words_per_cb,   // Words per CB in input (based on E_long for uniform stride)
    int num_cbs,
    int Q_m,            // Modulation order (for interleaving)
    int total_bits,     // Total output bits
    int num_bytes       // Total output bytes
) {
    // Each thread processes 4 bytes (32 bits) for better efficiency
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_byte = word_idx * 4;
    if (base_byte >= num_bytes) return;

    // Precompute boundary between short and long CBs
    int short_cb_total_bits = nof_short * E_short;

    // Process 4 bytes (32 bits)
    uint8_t bytes[4] = {0, 0, 0, 0};

    for (int byte_in_word = 0; byte_in_word < 4 && (base_byte + byte_in_word) < num_bytes; byte_in_word++) {
        uint8_t byte = 0;
        int byte_idx = base_byte + byte_in_word;

        for (int b = 0; b < 8; b++) {
            int out_bit_idx = byte_idx * 8 + b;
            if (out_bit_idx >= total_bits) break;

            // Step 1: Find which CB and bit position (handling variable E per CB)
            int cb_idx;
            int bit_in_cb;
            int bits_per_cb;

            if (out_bit_idx < short_cb_total_bits) {
                // In the short CB region
                cb_idx = out_bit_idx / E_short;
                bit_in_cb = out_bit_idx % E_short;
                bits_per_cb = E_short;
            } else {
                // In the long CB region
                int offset = out_bit_idx - short_cb_total_bits;
                cb_idx = nof_short + offset / E_long;
                bit_in_cb = offset % E_long;
                bits_per_cb = E_long;
            }

            // Apply bit interleaving per 3GPP TS 38.212 Section 5.4.2.2
            int K = bits_per_cb / Q_m;  // Number of symbols for this CB
            int j = bit_in_cb / Q_m;
            int i = bit_in_cb % Q_m;
            int in_bit_in_cb = i * K + j;

            // Read from rate matched buffer (MSB-first within each byte)
            int in_word = cb_idx * words_per_cb + in_bit_in_cb / 32;
            int in_bit_in_word = in_bit_in_cb % 32;
            int in_byte_in_word = in_bit_in_word / 8;
            int in_bit_in_byte = 7 - (in_bit_in_word % 8);  // MSB-first within byte
            int in_bit_pos = in_byte_in_word * 8 + in_bit_in_byte;
            uint32_t bit = (d_rate_matched[in_word] >> in_bit_pos) & 1u;

            // Step 2: Apply scrambling based on OUTPUT position (after interleaving)
            // Per 3GPP, scrambling is applied AFTER bit interleaving.
            // The scrambling sequence is continuous across all CBs.
            // Read scrambling bit (MSB-first to match gold_sequence_generate_kernel)
            int scr_word_idx = out_bit_idx / 32;
            int scr_bit_pos = 31 - (out_bit_idx % 32);  // MSB-first: bit 0 at position 31
            uint32_t scr_bit = (d_scramble_seq[scr_word_idx] >> scr_bit_pos) & 1u;
            bit ^= scr_bit;

            // Step 3: Pack into output byte
            // Use same bit ordering as fallback path (interleave_and_compact + bits_to_bytes)
            // which produces: output bit i at byte bit (7 - i%8) in the intermediate word,
            // then bits_to_bytes reads word bit j and puts at byte bit (7-j%8).
            // Net effect: output bit i → byte bit (i%8) (LSB-first within byte)
            byte |= (uint8_t)bit << b;

#ifdef OCUDU_PHY_CUDA_PDSCH_BIT_DIAGNOSTICS
            // Diagnostics: trace specific bit positions around symbol 104 for QPSK.
            if (out_bit_idx >= 208 && out_bit_idx <= 215) {
                printf("[GPU BIT DIAG] out_bit[%d]: cb=%d, bit_in_cb=%d, K=%d, j=%d, i=%d, in_bit=%d | "
                       "rm_bit=%d (word %d, pos %d), scr_bit=%d, final=%d, byte_in_word=%d, b=%d, byte_pos=%d\n",
                       out_bit_idx, cb_idx, bit_in_cb, K, j, i, in_bit_in_cb,
                       (d_rate_matched[in_word] >> in_bit_pos) & 1u, in_word, in_bit_pos,
                       scr_bit, bit, byte_in_word, b, 7 - b);
            }
#endif
        }
        bytes[byte_in_word] = byte;
    }

    // Write output bytes
    for (int i = 0; i < 4 && (base_byte + i) < num_bytes; i++) {
        d_output_bytes[base_byte + i] = bytes[i];
    }
}

/**
 * @brief Compact per-CB word-aligned bits into a continuous stream
 *
 * Takes bits from multiple CBs with word-aligned storage and packs them
 * into a continuous bit stream.
 *
 * Supports variable per-CB E values per 3GPP TS 38.212 Section 5.4.2.1:
 *   - CBs 0 to nof_short-1 have E_short bits
 *   - CBs nof_short to num_cbs-1 have E_long bits
 *
 * Input layout:
 *   CB 0: words_per_cb words (bits 0 to E_short/E_long-1 valid)
 *   CB 1: words_per_cb words (bits 0 to E_short/E_long-1 valid)
 *   ...
 *
 * Output layout:
 *   Continuous bit stream: CB0[0..E0-1], CB1[0..E1-1], ...
 */
__global__ void compact_cb_bits_kernel(
    const uint32_t* __restrict__ d_input,
    uint32_t* __restrict__ d_output,
    int nof_short,      // Number of CBs with E_short
    int E_short,        // Rate-matched length for short CBs
    int E_long,         // Rate-matched length for long CBs
    int words_per_cb,   // Words per CB in input (includes padding)
    int num_cbs,
    int total_bits      // Total output bits
) {
    int out_bit_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_bit_idx >= total_bits) return;

    // Precompute boundary between short and long CBs
    int short_cb_total_bits = nof_short * E_short;

    // Find which CB and bit position (handling variable E per CB)
    int cb_idx;
    int bit_in_cb;

    if (out_bit_idx < short_cb_total_bits) {
        // In the short CB region
        cb_idx = out_bit_idx / E_short;
        bit_in_cb = out_bit_idx % E_short;
    } else {
        // In the long CB region
        int offset = out_bit_idx - short_cb_total_bits;
        cb_idx = nof_short + offset / E_long;
        bit_in_cb = offset % E_long;
    }

    if (cb_idx >= num_cbs) return;

    // Read from input (word-aligned per CB, MSB-first within each byte)
    int in_word = cb_idx * words_per_cb + bit_in_cb / 32;
    int in_bit_in_word = bit_in_cb % 32;
    int in_byte_in_word = in_bit_in_word / 8;
    int in_bit_in_byte = 7 - (in_bit_in_word % 8);  // MSB-first within byte
    int in_bit = in_byte_in_word * 8 + in_bit_in_byte;
    uint32_t bit = (d_input[in_word] >> in_bit) & 1u;

    // Write to output (continuous stream, MSB-first within each byte)
    int out_word = out_bit_idx / 32;
    int out_bit_in_word = out_bit_idx % 32;
    int out_byte_in_word = out_bit_in_word / 8;
    int out_bit_in_byte = 7 - (out_bit_in_word % 8);  // MSB-first within byte
    int out_bit = out_byte_in_word * 8 + out_bit_in_byte;
    atomicOr(&d_output[out_word], bit << out_bit);
}

/**
 * @brief Interleave and compact per-CB bits in one pass
 *
 * Per 3GPP TS 38.212 Section 5.4.2.2:
 *   e(j*Qm + i) = f(i*K + j)  for i=0..Qm-1, j=0..K-1
 * where K = E/Qm (number of symbols per CB).
 *
 * This kernel reads from per-CB word-aligned rate-matched data,
 * applies the interleaving formula per CB, and writes to a
 * continuous output stream (all CBs concatenated without padding).
 *
 * Supports variable per-CB E values per 3GPP TS 38.212 Section 5.4.2.1:
 *   - CBs 0 to nof_short-1 have E_short bits
 *   - CBs nof_short to num_cbs-1 have E_long bits
 */
__global__ void interleave_and_compact_kernel(
    const uint32_t* __restrict__ d_input,
    uint32_t* __restrict__ d_output,
    int nof_short,      // Number of CBs with E_short
    int E_short,        // Rate-matched length for short CBs
    int E_long,         // Rate-matched length for long CBs
    int words_per_cb,   // Words per CB in input (includes padding)
    int num_cbs,
    int Q_m,            // Modulation order
    int total_bits      // Total output bits
) {
    int out_bit_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_bit_idx >= total_bits) return;

    // Precompute boundary between short and long CBs
    int short_cb_total_bits = nof_short * E_short;

    // Find which CB and bit position (handling variable E per CB)
    int cb_idx;
    int bit_in_cb;
    int bits_per_cb;

    if (out_bit_idx < short_cb_total_bits) {
        // In the short CB region
        cb_idx = out_bit_idx / E_short;
        bit_in_cb = out_bit_idx % E_short;
        bits_per_cb = E_short;
    } else {
        // In the long CB region
        int offset = out_bit_idx - short_cb_total_bits;
        cb_idx = nof_short + offset / E_long;
        bit_in_cb = offset % E_long;
        bits_per_cb = E_long;
    }

    if (cb_idx >= num_cbs) return;

    // Per 3GPP TS 38.212 Section 5.4.2.2:
    // Output position j*Qm + i reads from input position i*K + j
    int K = bits_per_cb / Q_m;  // Number of symbols per CB
    int j = bit_in_cb / Q_m;    // Symbol index
    int i = bit_in_cb % Q_m;    // Bit index within symbol
    int in_bit_in_cb = i * K + j;  // Input position within CB

    // Read from input (word-aligned per CB, MSB-first within each byte)
    int in_word = cb_idx * words_per_cb + in_bit_in_cb / 32;
    int in_bit_in_word = in_bit_in_cb % 32;
    int in_byte_in_word = in_bit_in_word / 8;
    int in_bit_in_byte = 7 - (in_bit_in_word % 8);  // MSB-first within byte
    int in_bit = in_byte_in_word * 8 + in_bit_in_byte;
    uint32_t bit = (d_input[in_word] >> in_bit) & 1u;

    // Write to output (continuous stream, MSB-first within each byte)
    int out_word = out_bit_idx / 32;
    int out_bit_in_word = out_bit_idx % 32;
    int out_byte_in_word = out_bit_in_word / 8;
    int out_bit_in_byte = 7 - (out_bit_in_word % 8);  // MSB-first within byte
    int out_bit = out_byte_in_word * 8 + out_bit_in_byte;
    atomicOr(&d_output[out_word], bit << out_bit);
}

/**
 * @brief Attach CRC bits to transport block data (in byte buffer)
 * Appends CRC bits starting at bit position tb_size_bits
 */
__global__ void attach_crc_kernel(
    uint8_t* __restrict__ d_tb_with_crc,
    uint32_t crc_value,
    int tb_size_bits,
    int crc_bits  // 16 or 24
) {
    int tid = threadIdx.x;
    if (tid >= crc_bits) return;

    // CRC bits are stored MSB first (bit 0 = MSB of CRC)
    int crc_bit_pos = crc_bits - 1 - tid;
    uint8_t bit = (crc_value >> crc_bit_pos) & 1;

    // Position in the output buffer
    int out_bit_idx = tb_size_bits + tid;
    int byte_idx = out_bit_idx / 8;
    int bit_pos = 7 - (out_bit_idx % 8);

    // Use atomic to safely set the bit
    atomicOr((uint32_t*)&d_tb_with_crc[byte_idx & ~3],
             (uint32_t)bit << (bit_pos + 8 * (byte_idx & 3)));
}

/**
 * @brief Clear unused bits in the last data byte when TBS is not byte-aligned
 *
 * For CRC correctness, unused bits must be zeros (not garbage from input).
 * CRC is computed over rounded-up bytes, so bits beyond TBS affect the CRC value.
 */
__global__ void clear_unused_bits_kernel(
    uint8_t* __restrict__ d_data,
    int num_bits
) {
    int remaining_bits = num_bits % 8;
    if (remaining_bits == 0) return;  // Byte-aligned, nothing to do

    int last_byte_idx = num_bits / 8;
    // Keep only the valid bits (MSB-first): mask = 0xFF << (8 - remaining_bits)
    uint8_t mask = (0xFF << (8 - remaining_bits)) & 0xFF;
    d_data[last_byte_idx] &= mask;
}

/**
 * @brief Extract CRC bits from received message (MSB-first)
 *
 * Used for CRC verification by extract-and-compare method, which works correctly
 * for non-byte-aligned TBS values where CRC self-check would fail.
 */
__global__ void extract_crc_kernel(
    const uint8_t* __restrict__ d_data,
    int crc_start_bit,
    int crc_bits,
    uint32_t* __restrict__ d_crc
) {
    uint32_t crc = 0;
#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
    // Diagnostics: print bytes being read for the TBS=500 case.
    if (crc_start_bit == 500) {
        printf("[KERNEL] extract_crc: start_bit=%d, bytes at [62..65]=%02x %02x %02x %02x\n",
               crc_start_bit, d_data[62], d_data[63], d_data[64], d_data[65]);
    }
#endif
    for (int i = 0; i < crc_bits; i++) {
        int bit_pos = crc_start_bit + i;
        int byte_idx = bit_pos / 8;
        int bit_in_byte = 7 - (bit_pos % 8);  // MSB-first
        uint8_t bit = (d_data[byte_idx] >> bit_in_byte) & 1;
        crc = (crc << 1) | bit;
    }
#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
    if (crc_start_bit == 500) {
        printf("[KERNEL] extracted CRC = 0x%x\n", crc);
    }
#endif
    *d_crc = crc;
}

/**
 * @brief GPU-only CRC comparison kernel (avoids D2H sync!)
 * Compares computed CRC with received CRC on GPU and writes result to device memory.
 */
__global__ void compare_crc_kernel(
    const uint32_t* __restrict__ d_computed,
    const uint32_t* __restrict__ d_received,
    int crc_bits,
    int* __restrict__ d_result
) {
    uint32_t mask = (crc_bits == 24) ? 0xFFFFFF : 0xFFFF;
    uint32_t computed = *d_computed & mask;
    uint32_t received = *d_received & mask;
    *d_result = (computed == received) ? 1 : 0;
}

/**
 * @brief Attach CRC bits from device memory (avoids host sync!)
 * Reads CRC from device memory instead of taking as kernel argument.
 * This eliminates the D2H sync required by attach_crc_kernel.
 */
__global__ void attach_crc_from_device_kernel(
    uint8_t* __restrict__ d_tb_with_crc,
    const uint32_t* __restrict__ d_crc,
    int tb_size_bits,
    int crc_bits  // 16 or 24
) {
    int tid = threadIdx.x;

    // Use shared memory to broadcast CRC value (only 1 global read)
    __shared__ uint32_t s_crc;
    if (tid == 0) {
        s_crc = *d_crc;
        if (crc_bits == 16) s_crc &= 0xFFFF;
    }
    __syncthreads();  // Safe: all threads participate before early exit

    if (tid >= crc_bits) return;

    uint32_t crc_value = s_crc;

    // CRC bits are stored MSB first (bit 0 = MSB of CRC)
    int crc_bit_pos = crc_bits - 1 - tid;
    uint8_t bit = (crc_value >> crc_bit_pos) & 1;

    // Position in the output buffer
    int out_bit_idx = tb_size_bits + tid;
    int byte_idx = out_bit_idx / 8;
    int bit_pos = 7 - (out_bit_idx % 8);

    // Use atomic to safely set the bit
    atomicOr((uint32_t*)&d_tb_with_crc[byte_idx & ~3],
             (uint32_t)bit << (bit_pos + 8 * (byte_idx & 3)));
}

/**
 * @brief Normalize LLR magnitudes for BG2 decoder
 *
 * BG2 min-sum decoder is sensitive to LLR magnitude variance.
 * This kernel normalizes LLRs to have uniform magnitude while preserving sign.
 * This is critical for BG2 reliability with real (noisy) channel LLRs.
 */
__global__ void normalize_llr_magnitude_kernel(
    float* __restrict__ d_llrs,
    int num_llrs,
    float target_magnitude
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_llrs) return;

    float llr = d_llrs[idx];
    // Preserve sign, normalize magnitude
    // Only normalize non-zero LLRs (punctured positions stay at 0)
    if (fabsf(llr) > 0.001f) {
        float sign = (llr >= 0.0f) ? 1.0f : -1.0f;
        d_llrs[idx] = sign * target_magnitude;
    }
}

// ============================================================================
// TB Encoder API Implementation
// ============================================================================

extern "C" {

// Maximum sizes for pre-allocation (avoids any runtime cudaMalloc/cudaFree)
// Based on 100MHz 256QAM 4-layer max configuration:
// - Max TBS: ~1.3Mbits = 200KB with CRC
// - Max CBs: 52 (BG1 with max TBS)
// - Max LDPC input: Kb=22 * Z=384 = 8448 bits/CB = 264 words/CB
// - Max LDPC output: 66 * Z = 25344 bits/CB = 792 words/CB
// - Max rate matched: G up to 2Mbits = 62500 words
static constexpr size_t MAX_TB_WITH_CRC_BYTES = 200000;
static constexpr size_t MAX_CB_BITS_SIZE = 52 * 264 * sizeof(uint32_t);        // ~55KB
static constexpr size_t MAX_ENCODED_BITS_SIZE = 52 * 792 * sizeof(uint32_t);   // ~165KB
static constexpr size_t MAX_RATE_MATCHED_SIZE = 52 * 2000 * sizeof(uint32_t);  // ~416KB (E up to 64000 bits/CB)
static constexpr size_t MAX_OUTPUT_WORDS = 65536;                              // ~256KB (2M bits / 32)

nr_ldpc_status_t tb_encoder_create(tb_encoder_handle_t* handle) {
    if (!handle) return NR_LDPC_ERROR_INVALID_CONFIG;

    tb_encoder_ctx* ctx = new (std::nothrow) tb_encoder_ctx;
    if (!ctx) return NR_LDPC_ERROR_ALLOC_FAILED;

    memset(ctx, 0, sizeof(*ctx));

    // Create sub-components
    nr_ldpc_status_t status = ldpc_encoder_create(&ctx->ldpc_enc);
    if (status != NR_LDPC_SUCCESS) {
        delete ctx;
        return status;
    }

    status = rate_matcher_create(&ctx->rate_matcher);
    if (status != NR_LDPC_SUCCESS) {
        ldpc_encoder_destroy(ctx->ldpc_enc);
        delete ctx;
        return status;
    }

    status = scrambler_create(&ctx->scrambler);
    if (status != NR_LDPC_SUCCESS) {
        rate_matcher_destroy(ctx->rate_matcher);
        ldpc_encoder_destroy(ctx->ldpc_enc);
        delete ctx;
        return status;
    }

    // Pre-allocate CRC buffer (avoids malloc/free in hot encode path)
    cudaError_t err = cudaMalloc(&ctx->d_crc, sizeof(uint32_t));
    if (err != cudaSuccess) {
        scrambler_destroy(ctx->scrambler);
        rate_matcher_destroy(ctx->rate_matcher);
        ldpc_encoder_destroy(ctx->ldpc_enc);
        delete ctx;
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }

    // PRE-ALLOCATE ALL BUFFERS TO MAX SIZE
    // This eliminates ~600µs of cudaMalloc/cudaFree overhead per slot!
    err = cudaMalloc(&ctx->d_tb_with_crc, MAX_TB_WITH_CRC_BYTES);
    if (err != cudaSuccess) goto alloc_failed;
    ctx->alloc_tb_with_crc_size = MAX_TB_WITH_CRC_BYTES;

    err = cudaMalloc(&ctx->d_cb_bits, MAX_CB_BITS_SIZE);
    if (err != cudaSuccess) goto alloc_failed;
    ctx->alloc_cb_bits_size = MAX_CB_BITS_SIZE;

    err = cudaMalloc(&ctx->d_encoded_bits, MAX_ENCODED_BITS_SIZE);
    if (err != cudaSuccess) goto alloc_failed;
    ctx->alloc_encoded_bits_size = MAX_ENCODED_BITS_SIZE;

    err = cudaMalloc(&ctx->d_rate_matched, MAX_RATE_MATCHED_SIZE);
    if (err != cudaSuccess) goto alloc_failed;
    ctx->alloc_rate_matched_size = MAX_RATE_MATCHED_SIZE;

    err = cudaMalloc(&ctx->d_interleaved, MAX_OUTPUT_WORDS * sizeof(uint32_t));
    if (err != cudaSuccess) goto alloc_failed;

    err = cudaMalloc(&ctx->d_scrambled, MAX_OUTPUT_WORDS * sizeof(uint32_t));
    if (err != cudaSuccess) goto alloc_failed;
    ctx->alloc_output_words = MAX_OUTPUT_WORDS;

    // Create secondary stream and event for parallel scrambler execution (Phase 2 optimization)
    err = cudaStreamCreate(&ctx->scrambler_stream);
    if (err != cudaSuccess) goto alloc_failed;

    err = cudaEventCreateWithFlags(&ctx->scrambler_done, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        cudaStreamDestroy(ctx->scrambler_stream);
        goto alloc_failed;
    }

    *handle = ctx;
    return NR_LDPC_SUCCESS;

alloc_failed:
    // Clean up on allocation failure
    if (ctx->d_scrambled) cudaFree(ctx->d_scrambled);
    if (ctx->d_interleaved) cudaFree(ctx->d_interleaved);
    if (ctx->d_rate_matched) cudaFree(ctx->d_rate_matched);
    if (ctx->d_encoded_bits) cudaFree(ctx->d_encoded_bits);
    if (ctx->d_cb_bits) cudaFree(ctx->d_cb_bits);
    if (ctx->d_tb_with_crc) cudaFree(ctx->d_tb_with_crc);
    if (ctx->d_crc) cudaFree(ctx->d_crc);
    scrambler_destroy(ctx->scrambler);
    rate_matcher_destroy(ctx->rate_matcher);
    ldpc_encoder_destroy(ctx->ldpc_enc);
    delete ctx;
    return NR_LDPC_ERROR_ALLOC_FAILED;
}

void tb_encoder_destroy(tb_encoder_handle_t handle) {
    if (!handle) return;

    ldpc_encoder_destroy(handle->ldpc_enc);
    rate_matcher_destroy(handle->rate_matcher);
    scrambler_destroy(handle->scrambler);

    // Destroy parallel scrambler stream and event
    if (handle->scrambler_done) cudaEventDestroy(handle->scrambler_done);
    if (handle->scrambler_stream) cudaStreamDestroy(handle->scrambler_stream);

    if (handle->d_tb_with_crc) cudaFree(handle->d_tb_with_crc);
    if (handle->d_cb_bits) cudaFree(handle->d_cb_bits);
    if (handle->d_encoded_bits) cudaFree(handle->d_encoded_bits);
    if (handle->d_rate_matched) cudaFree(handle->d_rate_matched);
    if (handle->d_interleaved) cudaFree(handle->d_interleaved);
    if (handle->d_scrambled) cudaFree(handle->d_scrambled);
    if (handle->d_crc) cudaFree(handle->d_crc);

    delete handle;
}

nr_ldpc_status_t tb_encoder_configure(tb_encoder_handle_t handle,
                                      const tb_encoder_config_t* cfg) {
    if (!handle || !cfg) return NR_LDPC_ERROR_INVALID_CONFIG;

    handle->config = *cfg;

    // Initialize TB configuration
    nr_ldpc_status_t status = nr_tb_init_config(&handle->tb_cfg, cfg->tb_size_bits, cfg->code_rate);
    if (status != NR_LDPC_SUCCESS) return status;

    // Override LDPC and segmentation parameters when provided by the CPU segmenter.
    // This keeps the GPU TB path aligned with the exact BG/Z/F/C shape used by
    // the CPU reference path.
    if (cfg->base_graph > 0) {
        handle->tb_cfg.ldpc_cfg.base_graph = cfg->base_graph;
    }
    if (cfg->num_code_blocks > 0) {
        handle->tb_cfg.num_code_blocks = cfg->num_code_blocks;
        handle->tb_cfg.cb_crc_bits = (cfg->num_code_blocks > 1) ? NR_CB_CRC24B_BITS : 0;
    }
    if (cfg->lifting_size > 0 || cfg->nof_filler_bits > 0) {
        nr_ldpc_config_t* ldpc_cfg = &handle->tb_cfg.ldpc_cfg;
        int old_Z = ldpc_cfg->lifting_size;
        int new_Z = (cfg->lifting_size > 0) ? cfg->lifting_size : old_Z;

        if (old_Z != new_Z || cfg->nof_filler_bits > 0 || cfg->base_graph > 0) {
            ldpc_cfg->lifting_size = new_Z;
            ldpc_cfg->lifting_set_index = nr_ldpc_get_lifting_set_index(new_Z);

            int Kb_full = (ldpc_cfg->base_graph == 1) ? NR_LDPC_BG1_INFO_NODES : NR_LDPC_BG2_INFO_NODES;
            int K_prime = Kb_full * new_Z;
            ldpc_cfg->num_filler_bits = (cfg->nof_filler_bits > 0) ? cfg->nof_filler_bits : ldpc_cfg->num_filler_bits;
            ldpc_cfg->num_info_bits = K_prime - ldpc_cfg->num_filler_bits;
            handle->tb_cfg.cb_size_bits = ldpc_cfg->num_info_bits;
            handle->tb_cfg.filler_bits = ldpc_cfg->num_filler_bits;

            int N = (ldpc_cfg->base_graph == 1) ?
                    NR_LDPC_BG1_UNPUNCTURED_VARS * new_Z :
                    NR_LDPC_BG2_UNPUNCTURED_VARS * new_Z;
            ldpc_cfg->num_codeword_bits = N;
            ldpc_cfg->num_parity_bits = N - K_prime;
            ldpc_cfg->max_parity_nodes = (ldpc_cfg->base_graph == 1) ?
                                          NR_LDPC_BG1_PARITY_NODES : NR_LDPC_BG2_PARITY_NODES;

#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
            printf("[TB Encoder] LDPC override: BG=%d, Z=%d -> %d, K=%d, K'=%d, F=%d, C=%d, N=%d\n",
                   ldpc_cfg->base_graph, old_Z, new_Z, ldpc_cfg->num_info_bits, K_prime,
                   ldpc_cfg->num_filler_bits, handle->tb_cfg.num_code_blocks, N);
#endif
        }
    }

    // Configure LDPC encoder (fast encoder)
    status = ldpc_encoder_configure(handle->ldpc_enc, &handle->tb_cfg.ldpc_cfg);

    // Calculate per-CB rate-matched lengths per 3GPP TS 38.212 Section 5.4.2.1
    // Use CPU-provided values if available for multi-CB consistency
    if (cfg->E_short > 0 && cfg->E_long > 0) {
        // Use values from CPU segmenter for exact consistency
        handle->nof_short_segments = cfg->nof_short_segments;
        handle->E_short = cfg->E_short;
        handle->E_long = cfg->E_long;
    } else {
        // Compute internally (fallback for standalone use)
        int G = cfg->num_allocated_res;
        int Q_m = cfg->modulation_order;
        int nof_layers = cfg->num_layers;
        int C = handle->tb_cfg.num_code_blocks;
        int total_symbols = G / Q_m;
        int symbols_per_layer = total_symbols / nof_layers;

        // nof_short_segments = C - (symbols_per_layer % C)
        // Short CBs use floor, remaining use ceiling
        handle->nof_short_segments = C - (symbols_per_layer % C);
        int symbols_short = symbols_per_layer / C;
        int symbols_long = (symbols_per_layer + C - 1) / C;  // ceiling
        handle->E_short = symbols_short * nof_layers * Q_m;
        handle->E_long = symbols_long * nof_layers * Q_m;
    }

    // Configure rate matcher with the longer E (all CBs must fit in this buffer)
    // We'll handle variable lengths in rate matching and interleaving
    nr_rate_match_config_t rm_cfg = {
        .E = handle->E_long,  // Use max E for buffer sizing
        .Q_m = cfg->modulation_order,
        .rv = cfg->redundancy_version,
        .N_cb = 0,  // Will be set automatically
        .k0 = 0,
        .limited_buffer = false
    };
    status = rate_matcher_configure_tx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg);
    if (status != NR_LDPC_SUCCESS) return status;

    // Configure scrambler (if enabled)
    if (cfg->enable_scrambling) {
        nr_scrambling_config_t scr_cfg = {
            .n_RNTI = cfg->n_RNTI,
            .n_ID = cfg->n_ID,
            .q = cfg->q
        };
        status = scrambler_configure(handle->scrambler, &scr_cfg);
        if (status != NR_LDPC_SUCCESS) return status;
    }

    // Allocate device buffers
    int tb_crc_bits = handle->tb_cfg.tb_crc_bits;
    int tb_with_crc_bytes = (cfg->tb_size_bits + tb_crc_bits + 7) / 8;

    // IMPORTANT: CB buffer must use LDPC encoder's input size (Kb * Z), NOT cb_size_bits
    // cb_size_bits is the actual info bits, but LDPC expects Kb*Z bits (with filler bits = 0)
    int ldpc_input_bits = ldpc_encoder_get_input_bits(handle->ldpc_enc);
    int cb_words = (ldpc_input_bits + 31) / 32;
    int encoded_words = ldpc_encoder_get_output_words(handle->ldpc_enc);
    int rm_words = (rm_cfg.E + 31) / 32;
    int total_output_words = (cfg->num_allocated_res + 31) / 32;

    handle->tb_with_crc_size = tb_with_crc_bytes;
    handle->cb_bits_size = handle->tb_cfg.num_code_blocks * cb_words * sizeof(uint32_t);
    handle->encoded_bits_size = handle->tb_cfg.num_code_blocks * encoded_words * sizeof(uint32_t);
    handle->rate_matched_size = handle->tb_cfg.num_code_blocks * rm_words * sizeof(uint32_t);
    handle->output_words = total_output_words;

    // OPTIMIZATION: Only reallocate when required size exceeds allocated size.
    // This avoids ~600µs of cudaMalloc/cudaFree overhead per slot!
    if (handle->tb_with_crc_size > handle->alloc_tb_with_crc_size) {
        if (handle->d_tb_with_crc) cudaFree(handle->d_tb_with_crc);
        cudaMalloc(&handle->d_tb_with_crc, handle->tb_with_crc_size);
        handle->alloc_tb_with_crc_size = handle->tb_with_crc_size;
    }
    if (handle->cb_bits_size > handle->alloc_cb_bits_size) {
        if (handle->d_cb_bits) cudaFree(handle->d_cb_bits);
        cudaMalloc(&handle->d_cb_bits, handle->cb_bits_size);
        handle->alloc_cb_bits_size = handle->cb_bits_size;
    }
    if (handle->encoded_bits_size > handle->alloc_encoded_bits_size) {
        if (handle->d_encoded_bits) cudaFree(handle->d_encoded_bits);
        cudaMalloc(&handle->d_encoded_bits, handle->encoded_bits_size);
        handle->alloc_encoded_bits_size = handle->encoded_bits_size;
    }
    if (handle->rate_matched_size > handle->alloc_rate_matched_size) {
        if (handle->d_rate_matched) cudaFree(handle->d_rate_matched);
        cudaMalloc(&handle->d_rate_matched, handle->rate_matched_size);
        handle->alloc_rate_matched_size = handle->rate_matched_size;
    }
    if (total_output_words > handle->alloc_output_words) {
        if (handle->d_interleaved) cudaFree(handle->d_interleaved);
        if (handle->d_scrambled) cudaFree(handle->d_scrambled);
        cudaMalloc(&handle->d_interleaved, total_output_words * sizeof(uint32_t));
        cudaMalloc(&handle->d_scrambled, total_output_words * sizeof(uint32_t));
        handle->alloc_output_words = total_output_words;
    }

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t tb_encoder_encode(tb_encoder_handle_t handle,
                                   const uint8_t* d_tb_input,
                                   uint8_t* d_output,
                                   cudaStream_t stream) {
    if (!handle || !d_tb_input || !d_output) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // =========================================================================
    // PHASE 2 OPTIMIZATIONS:
    // 1. FULLY FUSED TB CRC + Segment + CB CRC in single kernel launch
    //    - Computes TB CRC24A in block 0, broadcasts to other blocks via spin-wait
    //    - Segments TB into CBs with coalesced shared memory reads
    //    - Attaches CB CRC24B inline
    //    - Saves 1 kernel launch + better L2 cache utilization
    //
    // 2. PARALLEL SCRAMBLER: Launch scrambler on separate stream, overlap with LDPC
    //    - Scrambler has no dependency on LDPC output
    //    - 5-15% latency reduction by hiding scrambler behind LDPC
    // =========================================================================

    int ldpc_input_bits = ldpc_encoder_get_input_bits(handle->ldpc_enc);
    int cb_words = (ldpc_input_bits + 31) / 32;
    int total_bits = handle->config.num_allocated_res;
    int Q_m = handle->config.modulation_order;

    // Initialize CB CRC24B table (once, on first use)
    if (handle->tb_cfg.cb_crc_bits > 0) {
        init_tb_crc24b_table();
    }

    // =========================================================================
    // PARALLEL SCRAMBLER: Start sequence generation early on separate stream
    // =========================================================================
    // The scrambler sequence only depends on c_init (configured at scrambler_configure),
    // not on the actual TB data or LDPC output. We can generate it in parallel with
    // TB CRC, segmentation, and LDPC encoding.
    const uint32_t* d_scramble_seq = nullptr;
    bool scrambler_launched = false;
    if (handle->config.enable_scrambling && Q_m >= 2) {
        // Launch scrambler on secondary stream - will run in parallel with main pipeline
        nr_ldpc_status_t scr_status = scrambler_generate_sequence(
            handle->scrambler, total_bits, handle->scrambler_stream);
        if (scr_status == NR_LDPC_SUCCESS) {
            d_scramble_seq = scrambler_get_sequence_ptr(handle->scrambler);
            scrambler_launched = true;
            // Record event when scrambler completes
            cudaEventRecord(handle->scrambler_done, handle->scrambler_stream);
        }
    }

    // =========================================================================
    // Step 1: Compute TB CRC separately (more reliable than fused kernel)
    // Step 2: Segment TB + attach CB CRC
    // =========================================================================
    // Compute TB CRC
    if (handle->tb_cfg.tb_crc_bits == 24) {
        crc24a_compute(d_tb_input, handle->config.tb_size_bits, handle->d_crc, stream);
    } else {
        crc16_compute(d_tb_input, handle->config.tb_size_bits, (uint16_t*)handle->d_crc, stream);
    }

    // Initialize CB CRC table if needed
    if (handle->tb_cfg.cb_crc_bits > 0) {
        init_tb_crc24b_table();
    }

    // Segment TB + attach CRCs
    fused_tb_segment_crc_kernel<<<handle->tb_cfg.num_code_blocks, 256, 0, stream>>>(
        d_tb_input,
        handle->d_crc,
        handle->d_cb_bits,
        handle->config.tb_size_bits,
        handle->tb_cfg.tb_crc_bits,
        ldpc_input_bits,  // K = Kb * Z (LDPC block size)
        handle->tb_cfg.cb_crc_bits,
        handle->tb_cfg.num_code_blocks,
        cb_words,
        handle->tb_cfg.ldpc_cfg.num_filler_bits);

#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
    {
        cudaStreamSynchronize(stream);
        uint32_t h_crc = 0;
        cudaMemcpy(&h_crc, handle->d_crc, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[GPU PDSCH] TB CRC = 0x%06x (separate compute)\n", h_crc & 0x00FFFFFF);

        std::vector<uint32_t> cb_raw(cb_words * handle->tb_cfg.num_code_blocks);
        cudaMemcpy(cb_raw.data(), handle->d_cb_bits, cb_raw.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int cb = 0; cb < handle->tb_cfg.num_code_blocks; cb++) {
            fprintf(stderr, "[GPU PDSCH] CB%d data (K*Z=%d bits): first 32 bytes = ", cb, ldpc_input_bits);
            uint8_t* byte_ptr = (uint8_t*)(cb_raw.data() + cb * cb_words);
            for (int b = 0; b < 32; b++) {
                fprintf(stderr, "%02x ", byte_ptr[b]);
            }
            fprintf(stderr, "\n");

            // Print CRC region (TB ends at tb_size_bits, CRC is next 24 bits)
            int crc_byte_start = handle->config.tb_size_bits / 8;
            fprintf(stderr, "[GPU PDSCH] CB%d CRC region (bytes %d-%d): ", cb, crc_byte_start, crc_byte_start + 3);
            for (int b = crc_byte_start; b < crc_byte_start + 4 && b < cb_words * 4; b++) {
                fprintf(stderr, "%02x ", byte_ptr[b]);
            }
            fprintf(stderr, " (expected: %02x %02x %02x)\n",
                    (h_crc >> 16) & 0xFF, (h_crc >> 8) & 0xFF, h_crc & 0xFF);
        }
    }
#endif

    // Step 3: LDPC encoding (fast encoder)
    int encoded_words = ldpc_encoder_get_output_words(handle->ldpc_enc);
    ldpc_encoder_encode_batch(handle->ldpc_enc,
                                   handle->d_cb_bits, handle->d_encoded_bits,
                                   handle->tb_cfg.num_code_blocks, stream);

#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
    // Diagnostics: print first few LDPC encoded bytes per CB before rate matching.
    {
        cudaStreamSynchronize(stream);
        std::vector<uint32_t> enc_diagnostics(handle->tb_cfg.num_code_blocks * encoded_words);
        cudaMemcpy(enc_diagnostics.data(), handle->d_encoded_bits, enc_diagnostics.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int cb = 0; cb < handle->tb_cfg.num_code_blocks; cb++) {
            fprintf(stderr, "[GPU PDSCH] CB%d LDPC encoded (N=%d words): first 8 bytes = ", cb, encoded_words);
            uint8_t* byte_ptr = (uint8_t*)(enc_diagnostics.data() + cb * encoded_words);
            for (int b = 0; b < 8; b++) {
                fprintf(stderr, "%02x ", byte_ptr[b]);
            }
            fprintf(stderr, "\n");
        }
    }
#endif

    // Step 4: Rate matching with per-CB E values (3GPP TS 38.212 Section 5.4.2.1)
    // Short CBs get E_short, long CBs get E_long
    int num_cbs = handle->tb_cfg.num_code_blocks;
    int nof_short = handle->nof_short_segments;
    int E_short = handle->E_short;
    int E_long = handle->E_long;
    int rm_words = (E_long + 31) / 32;  // Uniform stride based on max E

#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
    fprintf(stderr, "[GPU PDSCH] RM: num_cbs=%d, nof_short=%d, E_short=%d, E_long=%d, rm_words=%d\n",
            num_cbs, nof_short, E_short, E_long, rm_words);
#endif

    // Clear output buffer (needed when E_short < E_long)
    cudaMemsetAsync(handle->d_rate_matched, 0,
                    num_cbs * rm_words * sizeof(uint32_t), stream);

    if (nof_short == num_cbs) {
        // All CBs are short - configure with E_short and single batch call
        nr_rate_match_config_t rm_cfg_short = {
            .E = E_short,
            .Q_m = handle->config.modulation_order,
            .rv = handle->config.redundancy_version,
            .N_cb = 0,
            .k0 = 0,
            .limited_buffer = false
        };
        rate_matcher_configure_tx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg_short);
        // Note: for all-short case, E_words == rm_words, so use standard batch call
        rate_matcher_match_batch(handle->rate_matcher,
                                 handle->d_encoded_bits,
                                 handle->d_rate_matched,
                                 num_cbs,
                                 stream);
    } else if (nof_short == 0) {
        // All CBs are long - already configured with E_long, single batch call
        rate_matcher_match_batch(handle->rate_matcher,
                                 handle->d_encoded_bits,
                                 handle->d_rate_matched,
                                 num_cbs,
                                 stream);
    } else {
        // Mixed: rate match short CBs first, then long CBs
        // IMPORTANT: Use rm_words stride for uniform buffer layout
        // Configure for short CBs
        nr_rate_match_config_t rm_cfg_short = {
            .E = E_short,
            .Q_m = handle->config.modulation_order,
            .rv = handle->config.redundancy_version,
            .N_cb = 0,
            .k0 = 0,
            .limited_buffer = false
        };
        rate_matcher_configure_tx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg_short);
        rate_matcher_match_batch_strided(handle->rate_matcher,
                                         handle->d_encoded_bits,
                                         handle->d_rate_matched,
                                         nof_short,
                                         rm_words,  // Use uniform stride for short CBs
                                         stream);

        // Configure for long CBs
        nr_rate_match_config_t rm_cfg_long = {
            .E = E_long,
            .Q_m = handle->config.modulation_order,
            .rv = handle->config.redundancy_version,
            .N_cb = 0,
            .k0 = 0,
            .limited_buffer = false
        };
        rate_matcher_configure_tx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg_long);
        rate_matcher_match_batch_strided(handle->rate_matcher,
                                         handle->d_encoded_bits + nof_short * encoded_words,
                                         handle->d_rate_matched + nof_short * rm_words,
                                         num_cbs - nof_short,
                                         rm_words,  // Use uniform stride for long CBs
                                         stream);
    }

#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
    // Diagnostics: print first few rate-matched bits per CB.
    {
        cudaStreamSynchronize(stream);
        std::vector<uint32_t> rm_diagnostics(num_cbs * rm_words);
        cudaMemcpy(rm_diagnostics.data(), handle->d_rate_matched, rm_diagnostics.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int cb = 0; cb < num_cbs; cb++) {
            int E_cb = (cb < nof_short) ? E_short : E_long;
            fprintf(stderr, "[GPU PDSCH] CB%d rate matched (E=%d): first 8 bytes = ", cb, E_cb);
            uint8_t* byte_ptr = (uint8_t*)(rm_diagnostics.data() + cb * rm_words);
            for (int b = 0; b < 8; b++) {
                fprintf(stderr, "%02x ", byte_ptr[b]);
            }
            fprintf(stderr, "\n");
        }
    }
#endif

    // Steps 5-7: FUSED interleave + scramble + pack_to_bytes
    // This replaces 3-4 separate kernel launches with 1, saving ~30-40µs overhead!
    int output_bytes = (total_bits + 7) / 8;
    int block_size = 256;

    if (handle->config.enable_scrambling && Q_m >= 2) {
        // FAST PATH: Use fused kernel for interleave + scramble + pack (1 kernel!)
        // =====================================================================
        // PHASE 2 OPTIMIZATION: Wait for parallel scrambler to complete
        // =====================================================================
        // The scrambler was launched earlier on scrambler_stream in parallel with
        // TB CRC, segmentation, and LDPC encoding. Now we need to wait for it.
        if (scrambler_launched) {
            // Wait for scrambler stream to complete before using the sequence
            cudaStreamWaitEvent(stream, handle->scrambler_done, 0);
        } else {
            // Fallback: generate scrambling sequence synchronously if parallel launch failed
            nr_ldpc_status_t scr_status = scrambler_generate_sequence(handle->scrambler, total_bits, stream);
            if (scr_status != NR_LDPC_SUCCESS) {
                return scr_status;
            }
            d_scramble_seq = scrambler_get_sequence_ptr(handle->scrambler);
        }

        // Each thread processes 4 bytes (1 word = 32 bits)
        int num_words = (output_bytes + 3) / 4;
        int num_word_blocks = (num_words + block_size - 1) / block_size;
        interleave_scramble_pack_kernel<<<num_word_blocks, block_size, 0, stream>>>(
            handle->d_rate_matched,
            d_scramble_seq,
            d_output,
            nof_short,    // Number of short CBs
            E_short,      // Bits per short CB
            E_long,       // Bits per long CB
            rm_words,     // words per CB in input (uniform stride)
            num_cbs,
            Q_m,
            total_bits,
            output_bytes);

#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
        // Run diagnostic kernel for multi-CB boundary tracing.
        if (num_cbs > 1) {
            cudaStreamSynchronize(stream);
            pdsch_cb_boundary_diagnostics_kernel<<<1, 1, 0, stream>>>(
                handle->d_rate_matched,
                d_scramble_seq,
                nof_short, E_short, E_long, rm_words, Q_m, total_bits);
            cudaStreamSynchronize(stream);
        }
#endif

#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
        // Diagnostics: print scrambling sequence and output bytes.
        {
            cudaStreamSynchronize(stream);
            std::vector<uint32_t> scr_diagnostics(4);
            cudaMemcpy(scr_diagnostics.data(), d_scramble_seq, scr_diagnostics.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[GPU PDSCH] Scramble seq (first 4 words): %08x %08x %08x %08x\n",
                    scr_diagnostics[0], scr_diagnostics[1], scr_diagnostics[2], scr_diagnostics[3]);

            std::vector<uint8_t> out_diagnostics(std::min(32, output_bytes));
            cudaMemcpy(out_diagnostics.data(), d_output, out_diagnostics.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[GPU PDSCH] Output bytes (first %zu): ", out_diagnostics.size());
            for (size_t i = 0; i < out_diagnostics.size(); i++) {
                fprintf(stderr, "%02x ", out_diagnostics[i]);
            }
            fprintf(stderr, "\n");

            // For multi-CB, also print bytes around the CB boundary
            if (num_cbs > 1 && E_short > 0) {
                int cb_boundary_byte = E_short / 8;  // First byte of CB1 region
                if (cb_boundary_byte + 8 <= output_bytes) {
                    std::vector<uint8_t> boundary_diagnostics(16);
                    cudaMemcpy(boundary_diagnostics.data(), d_output + cb_boundary_byte - 8, 16 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                    fprintf(stderr, "[GPU PDSCH] Bytes at CB boundary (byte %d): ", cb_boundary_byte);
                    for (int i = 0; i < 16; i++) {
                        fprintf(stderr, "%02x ", boundary_diagnostics[i]);
                    }
                    fprintf(stderr, "\n");
                }
            }
        }
#endif
    } else {
        // FALLBACK: Separate kernels for edge cases (BPSK or no scrambling)
        int total_words = (total_bits + 31) / 32;
        cudaMemsetAsync(handle->d_interleaved, 0, total_words * sizeof(uint32_t), stream);

        int num_thread_blocks = (total_bits + block_size - 1) / block_size;
        if (Q_m >= 2) {
            interleave_and_compact_kernel<<<num_thread_blocks, block_size, 0, stream>>>(
                handle->d_rate_matched, handle->d_interleaved,
                nof_short, E_short, E_long, rm_words, num_cbs, Q_m, total_bits);
        } else {
            compact_cb_bits_kernel<<<num_thread_blocks, block_size, 0, stream>>>(
                handle->d_rate_matched, handle->d_interleaved,
                nof_short, E_short, E_long, rm_words, num_cbs, total_bits);
        }

        uint32_t* d_final_output = handle->d_interleaved;
        if (handle->config.enable_scrambling) {
            nr_ldpc_status_t scr_status = scrambler_scramble(handle->scrambler,
                handle->d_interleaved, handle->d_scrambled, total_bits, stream);
            if (scr_status != NR_LDPC_SUCCESS) return scr_status;
            d_final_output = handle->d_scrambled;
        }

        int num_byte_blocks = (output_bytes + block_size - 1) / block_size;
        bits_to_bytes_kernel<<<num_byte_blocks, block_size, 0, stream>>>(
            d_final_output, d_output, output_bytes);
    }

    return NR_LDPC_SUCCESS;
}

int tb_encoder_get_output_bits(tb_encoder_handle_t handle) {
    return handle ? handle->config.num_allocated_res : 0;
}

int tb_encoder_get_num_code_blocks(tb_encoder_handle_t handle) {
    return handle ? handle->tb_cfg.num_code_blocks : 0;
}

int tb_encoder_get_base_graph(tb_encoder_handle_t handle) {
    return handle ? handle->tb_cfg.ldpc_cfg.base_graph : 0;
}

int tb_encoder_get_lifting_size(tb_encoder_handle_t handle) {
    return handle ? handle->tb_cfg.ldpc_cfg.lifting_size : 0;
}

// ============================================================================
// TB Encoder Bit-Level Output (for BLER testing without modulation)
// ============================================================================

/**
 * @brief Concatenate rate-matched code blocks and optionally scramble
 *
 * Takes rate-matched bits from multiple CBs and concatenates them into a single
 * bit stream. Each CB may have different E values (E_short vs E_long).
 */
__global__ void concatenate_scramble_bits_kernel(
    const uint32_t* __restrict__ d_rate_matched,
    const uint32_t* __restrict__ d_scramble_seq,
    uint32_t* __restrict__ d_output_bits,
    int nof_short,
    int E_short,
    int E_long,
    int rm_words,
    int num_cbs,
    int total_bits
) {
    int bit_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bit_idx >= total_bits) return;

    // Find which CB this bit belongs to
    int cb_idx = 0;
    int bit_in_cb = bit_idx;

    // Calculate CB index and bit position within CB
    if (nof_short > 0 && bit_idx < nof_short * E_short) {
        // Bit is in short CB region
        cb_idx = bit_idx / E_short;
        bit_in_cb = bit_idx % E_short;
    } else {
        // Bit is in long CB region
        int offset = nof_short * E_short;
        cb_idx = nof_short + (bit_idx - offset) / E_long;
        bit_in_cb = (bit_idx - offset) % E_long;
    }

    // Read bit from rate-matched buffer
    int rm_bit_idx = cb_idx * rm_words * 32 + bit_in_cb;
    int rm_word_idx = rm_bit_idx / 32;
    int rm_bit_pos = rm_bit_idx % 32;
    uint32_t bit = (d_rate_matched[rm_word_idx] >> rm_bit_pos) & 1;

    // Apply scrambling if enabled
    if (d_scramble_seq != nullptr) {
        int scr_word_idx = bit_idx / 32;
        int scr_bit_pos = 31 - (bit_idx % 32);  // MSB-first: bit 0 at position 31
        uint32_t scr_bit = (d_scramble_seq[scr_word_idx] >> scr_bit_pos) & 1;
        bit ^= scr_bit;
    }

    // Write to output
    int out_word_idx = bit_idx / 32;
    int out_bit_pos = bit_idx % 32;
    atomicOr(&d_output_bits[out_word_idx], bit << out_bit_pos);
}

/**
 * @brief Encode transport block to raw scrambled bits (no modulation)
 *
 * Outputs the full rate-matched bit stream (sum of all CB E values) suitable
 * for feeding directly to the TB decoder as LLRs.
 *
 * Pipeline: TB CRC → CB Seg → CB CRC → LDPC → Rate Match → Scramble → Bits
 */
nr_ldpc_status_t tb_encoder_encode_bits(tb_encoder_handle_t handle,
                                        const uint8_t* d_tb_input,
                                        uint8_t* d_output,
                                        cudaStream_t stream) {
    if (!handle || !d_tb_input || !d_output) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // Steps 1-4: Same as regular encoder (CRC, segment, LDPC, rate match)
    int ldpc_input_bits = ldpc_encoder_get_input_bits(handle->ldpc_enc);
    int cb_words = (ldpc_input_bits + 31) / 32;

    // Initialize CB CRC24B table
    if (handle->tb_cfg.cb_crc_bits > 0) {
        init_tb_crc24b_table();
    }

    // Step 1: TB CRC
    if (handle->tb_cfg.tb_crc_bits == 24) {
        crc24a_compute(d_tb_input, handle->config.tb_size_bits, handle->d_crc, stream);
    } else {
        uint16_t* d_crc16 = (uint16_t*)handle->d_crc;
        crc16_compute(d_tb_input, handle->config.tb_size_bits, d_crc16, stream);
    }

    // Step 2: Fused segment + attach CRCs
    fused_tb_segment_crc_kernel<<<handle->tb_cfg.num_code_blocks, 256, 0, stream>>>(
        d_tb_input,
        handle->d_crc,
        handle->d_cb_bits,
        handle->config.tb_size_bits,
        handle->tb_cfg.tb_crc_bits,
        ldpc_input_bits,  // K = Kb * Z (LDPC block size)
        handle->tb_cfg.cb_crc_bits,
        handle->tb_cfg.num_code_blocks,
        cb_words,
        handle->tb_cfg.ldpc_cfg.num_filler_bits);

    // Step 3: LDPC encoding
    int encoded_words = ldpc_encoder_get_output_words(handle->ldpc_enc);
    ldpc_encoder_encode_batch(handle->ldpc_enc,
                              handle->d_cb_bits, handle->d_encoded_bits,
                              handle->tb_cfg.num_code_blocks, stream);

    // Step 4: Rate matching
    int num_cbs = handle->tb_cfg.num_code_blocks;
    int nof_short = handle->nof_short_segments;
    int E_short = handle->E_short;
    int E_long = handle->E_long;
    int rm_words = (E_long + 31) / 32;

    cudaMemsetAsync(handle->d_rate_matched, 0,
                    num_cbs * rm_words * sizeof(uint32_t), stream);

    if (nof_short == num_cbs) {
        nr_rate_match_config_t rm_cfg_short = {
            .E = E_short,
            .Q_m = handle->config.modulation_order,
            .rv = handle->config.redundancy_version,
            .N_cb = 0,
            .k0 = 0,
            .limited_buffer = false
        };
        rate_matcher_configure_tx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg_short);
        rate_matcher_match_batch(handle->rate_matcher,
                                handle->d_encoded_bits,
                                handle->d_rate_matched,
                                num_cbs,
                                stream);
    } else if (nof_short == 0) {
        rate_matcher_match_batch(handle->rate_matcher,
                                handle->d_encoded_bits,
                                handle->d_rate_matched,
                                num_cbs,
                                stream);
    } else {
        // Mixed short/long CBs
        nr_rate_match_config_t rm_cfg_short = {
            .E = E_short,
            .Q_m = handle->config.modulation_order,
            .rv = handle->config.redundancy_version,
            .N_cb = 0,
            .k0 = 0,
            .limited_buffer = false
        };
        rate_matcher_configure_tx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg_short);
        rate_matcher_match_batch_strided(handle->rate_matcher,
                                         handle->d_encoded_bits,
                                         handle->d_rate_matched,
                                         nof_short,
                                         rm_words,
                                         stream);

        nr_rate_match_config_t rm_cfg_long = {
            .E = E_long,
            .Q_m = handle->config.modulation_order,
            .rv = handle->config.redundancy_version,
            .N_cb = 0,
            .k0 = 0,
            .limited_buffer = false
        };
        rate_matcher_configure_tx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg_long);
        rate_matcher_match_batch_strided(handle->rate_matcher,
                                         handle->d_encoded_bits + nof_short * encoded_words,
                                         handle->d_rate_matched + nof_short * rm_words,
                                         num_cbs - nof_short,
                                         rm_words,
                                         stream);
    }

    // Step 5: Interleave, scramble, and pack to bytes
    // Use the SAME kernel as regular encoder to ensure compatibility with decoder
    int Q_m = handle->config.modulation_order;
    // Calculate actual total E bits (sum of all CB rate-matched lengths)
    int total_E_bits = nof_short * E_short + (num_cbs - nof_short) * E_long;
    int output_bytes = (total_E_bits + 7) / 8;

    // Generate scrambling sequence
    const uint32_t* d_scramble_seq = nullptr;
    if (handle->config.enable_scrambling) {
        nr_ldpc_status_t scr_status = scrambler_generate_sequence(handle->scrambler, total_E_bits, stream);
        if (scr_status != NR_LDPC_SUCCESS) {
            return scr_status;
        }
        d_scramble_seq = scrambler_get_sequence_ptr(handle->scrambler);
    }

    // Use the SAME interleave+scramble+pack kernel as regular encoder
    // This ensures bit ordering matches what the decoder expects
    int block_size = 256;
    int num_words = (output_bytes + 3) / 4;
    int num_word_blocks = (num_words + block_size - 1) / block_size;
    interleave_scramble_pack_kernel<<<num_word_blocks, block_size, 0, stream>>>(
        handle->d_rate_matched,
        d_scramble_seq,
        d_output,
        nof_short,
        E_short,
        E_long,
        rm_words,
        num_cbs,
        Q_m,
        total_E_bits,
        output_bytes);

    return NR_LDPC_SUCCESS;
}

/**
 * @brief CUDA kernel: Interleave + Scramble + output UNPACKED bits (WITH interleaving, NO byte packing)
 *
 * This kernel is designed for bits-only interface with QAM modulation preparation.
 * It reads rate-matched bits, applies bit interleaving, applies scrambling, and writes unpacked bits.
 *
 * Pipeline: Rate matched bits → Bit interleaving → Scrambling → Unpacked bits (uint8_t values 0 or 1)
 *
 * The interleaving is per 3GPP TS 38.212 Section 5.4.2.2 (QAM bit interleaving).
 * This ensures the bits are ordered correctly for QAM modulation, matching what the decoder expects.
 *
 * @param d_rate_matched Input rate-matched bits (packed in uint32_t words, per-CB layout)
 * @param d_scramble_seq Scrambling sequence (packed in uint32_t words, MSB-first)
 * @param d_output_bits  Output unpacked bits (uint8_t array, values 0 or 1)
 * @param nof_short      Number of CBs with E_short
 * @param E_short        Rate-matched length for short CBs
 * @param E_long         Rate-matched length for long CBs
 * @param words_per_cb   Words per CB in input (based on E_long)
 * @param Q_m            Modulation order (2=QPSK, 4=16QAM, 6=64QAM)
 * @param total_bits     Total output bits
 */
__global__ void interleave_scramble_to_unpacked_bits_kernel(
    const uint32_t* __restrict__ d_rate_matched,
    const uint32_t* __restrict__ d_scramble_seq,
    uint8_t* __restrict__ d_output_bits,
    int nof_short,
    int E_short,
    int E_long,
    int words_per_cb,
    int Q_m,
    int total_bits,
    bool enable_interleaving
) {
    // Each thread processes one output bit
    int out_bit_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_bit_idx >= total_bits) return;

    // Step 1: Find which CB and bit position in output stream
    int short_cb_total_bits = nof_short * E_short;
    int cb_idx, bit_in_cb, bits_per_cb;

    if (out_bit_idx < short_cb_total_bits) {
        // In the short CB region
        cb_idx = out_bit_idx / E_short;
        bit_in_cb = out_bit_idx % E_short;
        bits_per_cb = E_short;
    } else {
        // In the long CB region
        int offset = out_bit_idx - short_cb_total_bits;
        cb_idx = nof_short + offset / E_long;
        bit_in_cb = offset % E_long;
        bits_per_cb = E_long;
    }

    // Step 2: Apply optional QAM bit interleaving (3GPP TS 38.212 Section 5.4.2.2)
    // Interleaving formula: output_bit[j*Q_m + i] = input_bit[i*K + j]
    // where K = E/Q_m, i = bit_in_cb % Q_m, j = bit_in_cb / Q_m
    // So for output position p = j*Q_m+i, input is at i*K+j
    int in_bit_in_cb;
    if (enable_interleaving && Q_m > 1) {
        int K = bits_per_cb / Q_m;  // Number of symbols
        int i = bit_in_cb % Q_m;     // Bit position within symbol (0 to Q_m-1)
        int j = bit_in_cb / Q_m;     // Symbol index (0 to K-1)
        in_bit_in_cb = i * K + j;    // Correct interleaved position (same as interleave_scramble_pack_kernel)
    } else {
        in_bit_in_cb = bit_in_cb;    // No interleaving
    }

    // Step 3: Read from rate matched buffer (MSB-first within each uint32_t word)
    int in_word = cb_idx * words_per_cb + in_bit_in_cb / 32;
    int in_bit_in_word = in_bit_in_cb % 32;
    int in_byte_in_word = in_bit_in_word / 8;
    int in_bit_in_byte = 7 - (in_bit_in_word % 8);  // MSB-first within byte
    int in_bit_pos = in_byte_in_word * 8 + in_bit_in_byte;
    uint32_t bit = (d_rate_matched[in_word] >> in_bit_pos) & 1u;

    // Step 4: Apply scrambling (MSB-first to match gold_sequence_generate_kernel)
    if (d_scramble_seq != nullptr) {
        int scr_word_idx = out_bit_idx / 32;
        int scr_bit_pos = 31 - (out_bit_idx % 32);  // MSB-first: bit 0 at position 31
        uint32_t scr_bit = (d_scramble_seq[scr_word_idx] >> scr_bit_pos) & 1u;
        bit ^= scr_bit;
    }

    // Step 5: Write unpacked bit (0 or 1)
    d_output_bits[out_bit_idx] = (uint8_t)bit;
}

/**
 * @brief TB encoder with UNPACKED BITS output (optional interleaving, NO byte packing)
 *
 * This is designed for bits-only interface for PyTorch training loops.
 * Input: unpacked TB bits [tbs_bits] uint8_t, values 0 or 1
 * Output: unpacked coded bits [total_E_bits] uint8_t, values 0 or 1
 *
 * Pipeline: TB CRC → CB Seg → CB CRC → LDPC → Rate Match → [Interleave] → Scramble → Unpacked bits
 *
 * NOTE: Optionally includes QAM bit interleaving (per 3GPP TS 38.212 Section 5.4.2.2) but NO byte packing.
 *       The bits are (optionally) interleaved and scrambled, matching the full TX chain, but output as unpacked bits.
 *
 * @param handle TB encoder handle
 * @param d_tb_input Input TB bits (packed in bytes, LSB-first)
 * @param d_output Output coded bits (unpacked, uint8_t array, values 0 or 1)
 * @param enable_interleaving Enable QAM bit interleaving (true/false)
 * @param stream CUDA stream
 * @return Status code
 */
nr_ldpc_status_t tb_encoder_encode_to_unpacked_bits(tb_encoder_handle_t handle,
                                                     const uint8_t* d_tb_input,
                                                     uint8_t* d_output,
                                                     bool enable_interleaving,
                                                     cudaStream_t stream) {
    if (!handle || !d_tb_input || !d_output) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // Steps 1-4: Same as regular encoder (CRC, segment, LDPC, rate match)
    int ldpc_input_bits = ldpc_encoder_get_input_bits(handle->ldpc_enc);
    int cb_words = (ldpc_input_bits + 31) / 32;

    // Initialize CB CRC24B table
    if (handle->tb_cfg.cb_crc_bits > 0) {
        init_tb_crc24b_table();
    }

    // Step 1: TB CRC
    if (handle->tb_cfg.tb_crc_bits == 24) {
        crc24a_compute(d_tb_input, handle->config.tb_size_bits, handle->d_crc, stream);
    } else {
        uint16_t* d_crc16 = (uint16_t*)handle->d_crc;
        crc16_compute(d_tb_input, handle->config.tb_size_bits, d_crc16, stream);
    }

    // Step 2: Fused segment + attach CRCs
    fused_tb_segment_crc_kernel<<<handle->tb_cfg.num_code_blocks, 256, 0, stream>>>(
        d_tb_input,
        handle->d_crc,
        handle->d_cb_bits,
        handle->config.tb_size_bits,
        handle->tb_cfg.tb_crc_bits,
        ldpc_input_bits,  // K = Kb * Z (LDPC block size)
        handle->tb_cfg.cb_crc_bits,
        handle->tb_cfg.num_code_blocks,
        cb_words,
        handle->tb_cfg.ldpc_cfg.num_filler_bits);

    // Step 3: LDPC encoding
    int encoded_words = ldpc_encoder_get_output_words(handle->ldpc_enc);
    ldpc_encoder_encode_batch(handle->ldpc_enc,
                              handle->d_cb_bits, handle->d_encoded_bits,
                              handle->tb_cfg.num_code_blocks, stream);

    // Step 4: Rate matching
    int num_cbs = handle->tb_cfg.num_code_blocks;
    int nof_short = handle->nof_short_segments;
    int E_short = handle->E_short;
    int E_long = handle->E_long;
    int rm_words = (E_long + 31) / 32;

    cudaMemsetAsync(handle->d_rate_matched, 0,
                    num_cbs * rm_words * sizeof(uint32_t), stream);

    if (nof_short == num_cbs) {
        nr_rate_match_config_t rm_cfg_short = {
            .E = E_short,
            .Q_m = handle->config.modulation_order,
            .rv = handle->config.redundancy_version,
            .N_cb = 0,
            .k0 = 0,
            .limited_buffer = false
        };
        rate_matcher_configure_tx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg_short);
        rate_matcher_match_batch(handle->rate_matcher,
                                handle->d_encoded_bits,
                                handle->d_rate_matched,
                                num_cbs,
                                stream);
    } else if (nof_short == 0) {
        rate_matcher_match_batch(handle->rate_matcher,
                                handle->d_encoded_bits,
                                handle->d_rate_matched,
                                num_cbs,
                                stream);
    } else {
        // Mixed short/long CBs
        nr_rate_match_config_t rm_cfg_short = {
            .E = E_short,
            .Q_m = handle->config.modulation_order,
            .rv = handle->config.redundancy_version,
            .N_cb = 0,
            .k0 = 0,
            .limited_buffer = false
        };
        rate_matcher_configure_tx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg_short);
        rate_matcher_match_batch_strided(handle->rate_matcher,
                                         handle->d_encoded_bits,
                                         handle->d_rate_matched,
                                         nof_short,
                                         rm_words,
                                         stream);

        nr_rate_match_config_t rm_cfg_long = {
            .E = E_long,
            .Q_m = handle->config.modulation_order,
            .rv = handle->config.redundancy_version,
            .N_cb = 0,
            .k0 = 0,
            .limited_buffer = false
        };
        rate_matcher_configure_tx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg_long);
        rate_matcher_match_batch_strided(handle->rate_matcher,
                                         handle->d_encoded_bits + nof_short * encoded_words,
                                         handle->d_rate_matched + nof_short * rm_words,
                                         num_cbs - nof_short,
                                         rm_words,
                                         stream);
    }

    // Step 5: Scramble and output unpacked bits
    int total_E_bits = nof_short * E_short + (num_cbs - nof_short) * E_long;

    // Generate scrambling sequence
    const uint32_t* d_scramble_seq = nullptr;
    if (handle->config.enable_scrambling) {
        nr_ldpc_status_t scr_status = scrambler_generate_sequence(handle->scrambler, total_E_bits, stream);
        if (scr_status != NR_LDPC_SUCCESS) {
            return scr_status;
        }
        d_scramble_seq = scrambler_get_sequence_ptr(handle->scrambler);
    }

    // Launch kernel to output unpacked bits (with optional interleaving)
    int block_size = 256;
    int num_blocks = (total_E_bits + block_size - 1) / block_size;
    int Q_m = handle->config.modulation_order;
    interleave_scramble_to_unpacked_bits_kernel<<<num_blocks, block_size, 0, stream>>>(
        handle->d_rate_matched,
        d_scramble_seq,
        d_output,
        nof_short,
        E_short,
        E_long,
        rm_words,
        Q_m,
        total_E_bits,
        enable_interleaving);

    return NR_LDPC_SUCCESS;
}

/**
 * @brief Get total E bits (sum of all CB rate-matched lengths)
 *
 * Returns the actual sum of E values across all code blocks.
 */
int tb_encoder_get_total_E_bits(tb_encoder_handle_t handle) {
    if (!handle) return 0;
    int nof_short = handle->nof_short_segments;
    int E_short = handle->E_short;
    int E_long = handle->E_long;
    int num_cbs = handle->tb_cfg.num_code_blocks;
    return nof_short * E_short + (num_cbs - nof_short) * E_long;
}

nr_ldpc_status_t tb_encoder_encode_to_symbols_int8(tb_encoder_handle_t handle,
                                                    const uint8_t* d_tb_input,
                                                    int8_t* d_symbols_int8,
                                                    cudaStream_t stream) {
    if (!handle || !d_tb_input || !d_symbols_int8) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // =========================================================================
    // ULTRA-FAST PATH: TB → INT8 Symbols with minimal kernel launches
    // =========================================================================
    // Step 1: TB CRC compute
    // Step 2: Fused segment + TB CRC + CB CRC (single kernel)
    // Step 3: LDPC encode (fused or standard)
    // Step 4: Fused Interleave + Scramble + Modulate → INT8 (single kernel!)
    // =========================================================================

    int ldpc_input_bits = ldpc_encoder_get_input_bits(handle->ldpc_enc);
    int cb_words = (ldpc_input_bits + 31) / 32;
    int num_cbs = handle->tb_cfg.num_code_blocks;

    // Initialize CB CRC table for segmentation kernel
    if (handle->tb_cfg.cb_crc_bits > 0) {
        init_tb_crc24b_table();
    }

    // =========================================================================
    // OPTIMIZED PATH: Separate parallel CRC + fused segmentation
    // =========================================================================
    // The fully_fused kernel was doing serial TB CRC in block 0, causing all
    // other blocks to spin-wait. For 400Kb TB this took 1.8ms!
    //
    // Instead, use the optimized parallel CRC kernels which take ~65µs for
    // large TBs, then pass the pre-computed CRC to the segmentation kernel.
    // =========================================================================

    // Step 1: Compute TB CRC using optimized parallel kernel
    if (handle->tb_cfg.tb_crc_bits == 24) {
        crc24a_compute(d_tb_input, handle->config.tb_size_bits, handle->d_crc, stream);
    } else {
        crc16_compute(d_tb_input, handle->config.tb_size_bits, (uint16_t*)handle->d_crc, stream);
    }

    // Step 2: Fused segmentation + CB CRC (uses pre-computed TB CRC)
    fused_tb_segment_crc_kernel<<<num_cbs, 256, 0, stream>>>(
        d_tb_input,
        handle->d_crc,
        handle->d_cb_bits,
        handle->config.tb_size_bits,
        handle->tb_cfg.tb_crc_bits,
        ldpc_input_bits,  // K = Kb * Z (LDPC block size)
        handle->tb_cfg.cb_crc_bits,
        num_cbs,
        cb_words,
        handle->tb_cfg.ldpc_cfg.num_filler_bits);

    // Generate or reuse the packed Gold sequence. The scrambler caches by
    // length/offset, so steady-state slots avoid per-symbol LFSR jumps without
    // adding a recurring sequence generation kernel.
    int total_bits = handle->config.num_allocated_res;
    nr_ldpc_status_t scr_status = scrambler_generate_sequence(handle->scrambler, total_bits, stream);
    if (scr_status != NR_LDPC_SUCCESS) {
        return scr_status;
    }
    const uint32_t* d_scramble_seq = scrambler_get_sequence_ptr(handle->scrambler);
    uint32_t c_init = scrambler_get_c_init(handle->scrambler);

    // Common parameters
    int Q_m = handle->config.modulation_order;
    int nof_short = handle->nof_short_segments;
    int E_short = handle->E_short;
    int E_long = handle->E_long;
    int num_symbols = total_bits / Q_m;

    // LDPC parameters for rate matching
    int Z = handle->tb_cfg.ldpc_cfg.lifting_size;
    int Kb = (handle->tb_cfg.ldpc_cfg.base_graph == 1) ? 22 : 10;
    int N_cols = (handle->tb_cfg.ldpc_cfg.base_graph == 1) ? 68 : 52;
    int N_full = N_cols * Z;
    int N_cb = N_full - 2 * Z;
    int F = handle->tb_cfg.ldpc_cfg.num_filler_bits;
    // Kd = position where filler bits start in circular buffer
    // CPU filler range is [K - F, K) where K = (Kb-2)*Z
    // GPU's fused_rate_match_index skips [Kd, Kd+F), so Kd must be K - F
    int Kd = Kb * Z - 2 * Z - F;
    int puncture_offset = 2 * Z;

    // Compute k0 from RV per 3GPP TS 38.212 Table 5.4.2.1-2
    int bg = handle->tb_cfg.ldpc_cfg.base_graph;
    int rv = handle->config.redundancy_version;
    int k0 = rate_matcher_compute_k0(bg, Z, rv, N_cb);

    // =========================================================================
    // STANDARD PATH: Separate LDPC encode + fused RM/interleave/scramble/modulate
    // =========================================================================
    int encoded_words = ldpc_encoder_get_output_words(handle->ldpc_enc);

    // Step 4: Standard LDPC encode (full codewords)
    // LDPC runs on main stream, in parallel with scrambler on secondary stream
    ldpc_encoder_encode_batch(handle->ldpc_enc,
                              handle->d_cb_bits, handle->d_encoded_bits,
                              num_cbs, stream);

    // Step 5: Fused RM + Interleave + Scramble + Modulate → INT8
    pdsch_fused_tx_config_t fused_cfg = {};
    fused_cfg.N_cb = N_cb;
    fused_cfg.N_full = N_full;
    fused_cfg.k0 = k0;  // RV-dependent start position
    fused_cfg.Kd = Kd;
    fused_cfg.F = F;
    fused_cfg.puncture_offset = puncture_offset;
    fused_cfg.encoded_stride = encoded_words;
    fused_cfg.nof_short = nof_short;
    fused_cfg.E_short = E_short;
    fused_cfg.E_long = E_long;
    fused_cfg.num_cbs = num_cbs;
    fused_cfg.mod_order = Q_m;
    fused_cfg.total_symbols = num_symbols;

    int ret = pdsch_fused_encode_to_symbols_int8_precomputed(
        handle->d_encoded_bits, d_scramble_seq, d_symbols_int8, &fused_cfg, stream);

    if (ret != 0) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t tb_encoder_warmup(tb_encoder_handle_t handle, cudaStream_t stream) {
    if (!handle) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // Use a minimal TB configuration to trigger all kernel JIT compilation.
    // This uses a small TB (1024 bits) with single code block and BG2.
    tb_encoder_config_t warmup_cfg = {};
    warmup_cfg.tb_size_bits = 1024;       // Small TB for fast warmup
    warmup_cfg.num_layers = 1;
    warmup_cfg.modulation_order = 2;      // QPSK
    warmup_cfg.num_allocated_res = 2048;  // G (coded bits)
    warmup_cfg.redundancy_version = 0;
    warmup_cfg.code_rate = 0.5f;
    warmup_cfg.n_RNTI = 0x1234;
    warmup_cfg.n_ID = 0;
    warmup_cfg.q = 0;
    warmup_cfg.enable_scrambling = true;

    nr_ldpc_status_t status = tb_encoder_configure(handle, &warmup_cfg);
    if (status != NR_LDPC_SUCCESS) {
        return status;
    }

    // Allocate temporary input buffer
    int tb_bytes = (warmup_cfg.tb_size_bits + 7) / 8;
    uint8_t* d_dummy_input = nullptr;
    cudaError_t err = cudaMalloc(&d_dummy_input, tb_bytes);
    if (err != cudaSuccess) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }
    cudaMemsetAsync(d_dummy_input, 0, tb_bytes, stream);

    // Allocate temporary output buffer
    int out_bytes = (warmup_cfg.num_allocated_res + 7) / 8;
    uint8_t* d_dummy_output = nullptr;
    err = cudaMalloc(&d_dummy_output, out_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_dummy_input);
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    // Run the full encode pipeline to trigger JIT compilation
    status = tb_encoder_encode(handle, d_dummy_input, d_dummy_output, stream);

    // Sync to ensure kernels complete before freeing
    cudaStreamSynchronize(stream);

    // Free temporary buffers
    cudaFree(d_dummy_input);
    cudaFree(d_dummy_output);

    return status;
}

// ============================================================================
// Batch TB Encoder Implementation
// ============================================================================

/**
 * @brief Batch CRC-24A compute + attach kernel
 *
 * Each block processes one TB: computes CRC and attaches it.
 * Grid: (num_tbs), Block: (32)
 */
__global__ void batch_crc24a_compute_attach_kernel(
    uint8_t* __restrict__ d_tb_data,  // Contiguous TBs with space for CRC
    int tb_size_bits,
    int tb_stride_bytes  // Bytes per TB (including CRC space)
) {
    int tb_idx = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ uint32_t s_crc;

    uint8_t* tb = d_tb_data + tb_idx * tb_stride_bytes;
    int num_bytes = (tb_size_bits + 7) / 8;

    // Thread 0 computes CRC using slicing-by-4
    if (tid == 0) {
        uint32_t crc = 0;
        int i = 0;

        int num_words = num_bytes / 4;
        for (int w = 0; w < num_words; w++) {
            uint8_t b0 = tb[i++];
            uint8_t b1 = tb[i++];
            uint8_t b2 = tb[i++];
            uint8_t b3 = tb[i++];

            uint8_t idx0 = ((crc >> 16) ^ b0) & 0xFF;
            uint8_t idx1 = ((crc >> 8) ^ b1) & 0xFF;
            uint8_t idx2 = (crc ^ b2) & 0xFF;
            uint8_t idx3 = b3;

            crc = CRC24A_S0[idx0] ^ CRC24A_S1[idx1] ^ CRC24A_S2[idx2] ^ CRC24A_S3[idx3];
        }

        while (i < num_bytes) {
            uint8_t byte = tb[i++];
            uint8_t index = ((crc >> 16) ^ byte) & 0xFF;
            crc = (crc << 8) ^ CRC24A_TABLE[index];
        }

        s_crc = crc & 0xFFFFFF;
    }
    __syncthreads();

    // Thread 0 writes all 24 CRC bits (3 bytes) directly - avoids alignment issues
    if (tid == 0) {
        uint32_t crc_value = s_crc;
        int crc_start_byte = (tb_size_bits + 7) / 8;

        // Write CRC as 3 bytes (MSB first)
        tb[crc_start_byte]     = (crc_value >> 16) & 0xFF;
        tb[crc_start_byte + 1] = (crc_value >> 8) & 0xFF;
        tb[crc_start_byte + 2] = crc_value & 0xFF;

        // Handle case where TB bits don't end on byte boundary
        int partial_bits = tb_size_bits % 8;
        if (partial_bits > 0) {
            // Need to merge CRC MSBs with partial last byte of TB data
            int last_tb_byte = tb_size_bits / 8;
            uint8_t tb_byte = tb[last_tb_byte];
            uint8_t mask = (0xFF << (8 - partial_bits)) & 0xFF;
            int crc_bits_in_first = 8 - partial_bits;
            uint8_t crc_top = (crc_value >> (24 - crc_bits_in_first)) & ((1 << crc_bits_in_first) - 1);
            tb[last_tb_byte] = (tb_byte & mask) | crc_top;

            // Shift remaining CRC bits into next bytes
            int remaining_bits = 24 - crc_bits_in_first;
            uint32_t remaining_crc = crc_value << crc_bits_in_first;
            tb[last_tb_byte + 1] = (remaining_crc >> 16) & 0xFF;
            tb[last_tb_byte + 2] = (remaining_crc >> 8) & 0xFF;
            tb[last_tb_byte + 3] = remaining_crc & 0xFF;
        }
    }
}

/**
 * @brief Attach pre-computed CRC-24A values to TB buffers
 *
 * Grid: (num_tbs), Block: (1)
 */
__global__ void batch_crc24a_attach_kernel(
    uint8_t* __restrict__ d_tb_data,
    const uint32_t* __restrict__ d_crc_values,
    int tb_size_bits,
    int tb_stride_bytes
) {
    int tb_idx = blockIdx.x;
    uint8_t* tb = d_tb_data + tb_idx * tb_stride_bytes;
    uint32_t crc_value = d_crc_values[tb_idx] & 0xFFFFFF;

    int partial_bits = tb_size_bits % 8;
    if (partial_bits == 0) {
        int crc_start_byte = tb_size_bits / 8;
        tb[crc_start_byte] = (crc_value >> 16) & 0xFF;
        tb[crc_start_byte + 1] = (crc_value >> 8) & 0xFF;
        tb[crc_start_byte + 2] = crc_value & 0xFF;
        return;
    }

    int last_tb_byte = tb_size_bits / 8;
    uint8_t tb_byte = tb[last_tb_byte];
    uint8_t mask = (0xFF << (8 - partial_bits)) & 0xFF;
    int crc_bits_in_first = 8 - partial_bits;
    uint8_t crc_top = (crc_value >> (24 - crc_bits_in_first)) & ((1 << crc_bits_in_first) - 1);
    tb[last_tb_byte] = (tb_byte & mask) | crc_top;

    uint32_t remaining_crc = crc_value << crc_bits_in_first;
    tb[last_tb_byte + 1] = (remaining_crc >> 16) & 0xFF;
    tb[last_tb_byte + 2] = (remaining_crc >> 8) & 0xFF;
    tb[last_tb_byte + 3] = remaining_crc & 0xFF;
}

/**
 * @brief Batch segmentation kernel for multiple TBs
 *
 * Grid: (total_cbs), Block: (256)
 * Each block processes one CB from any TB.
 */
__global__ void batch_segment_tb_kernel(
    const uint8_t* __restrict__ d_tbs_with_crc,  // All TBs contiguous
    uint32_t* __restrict__ d_cb_bits,             // All CBs contiguous
    int tb_bits,             // Bits per TB (including CRC)
    int tb_stride_bytes,     // Bytes per TB in input
    int cb_size_bits,        // Bits per CB (before CB CRC)
    int cbs_per_tb,          // CBs per TB
    int cb_stride_words,     // LDPC input words per CB
    int cb_crc_bits          // CB CRC bits (0 or 24)
) {
    int global_cb_idx = blockIdx.x;
    int tb_idx = global_cb_idx / cbs_per_tb;
    int cb_idx = global_cb_idx % cbs_per_tb;

    const uint8_t* tb = d_tbs_with_crc + tb_idx * tb_stride_bytes;

    // Data bits per CB = ceil(tb_bits / cbs_per_tb) = amount to copy from TB+CRC buffer
    // Info bits per CB = data_per_cb - cb_crc_bits = where CB CRC goes
    // tb_bits is TB + TB_CRC total
    int data_per_cb = (tb_bits + cbs_per_tb - 1) / cbs_per_tb;
    int info_bits_per_cb = data_per_cb - cb_crc_bits;
    int cb_start_bit = cb_idx * data_per_cb;

    for (int word_idx = threadIdx.x; word_idx < cb_stride_words; word_idx += blockDim.x) {
        uint32_t word = 0;

        int base_bit = word_idx * 32;
        for (int b = 0; b < 32; b++) {
            int bit_idx = base_bit + b;
            if (bit_idx >= cb_size_bits) break;

            // Only copy TB data for the data region (0 to data_per_cb-1)
            // Filler region (data_per_cb to cb_size_bits-1) stays 0
            int tb_bit_idx = cb_start_bit + bit_idx;
            if (bit_idx < data_per_cb && tb_bit_idx < tb_bits) {
                int byte_idx = tb_bit_idx / 8;
                int bit_pos = tb_bit_idx % 8;
                uint8_t bit = (tb[byte_idx] >> (7 - bit_pos)) & 1;
                word |= (uint32_t)bit << b;
            }
        }

        d_cb_bits[global_cb_idx * cb_stride_words + word_idx] = word;
    }
}

/**
 * @brief Batch interleave + scramble + pack for multiple TBs
 *
 * Same as single-TB version but handles multiple TBs worth of output.
 */
__global__ void batch_interleave_scramble_pack_kernel(
    const uint32_t* __restrict__ d_rate_matched,  // All CBs from all TBs
    const uint32_t* __restrict__ d_scramble_seq,
    uint8_t* __restrict__ d_output,               // All TBs output
    int bits_per_cb,
    int words_per_cb,
    int total_cbs,
    int Q_m,
    int total_bits,
    int num_bytes
) {
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_byte = word_idx * 4;
    if (base_byte >= num_bytes) return;

    uint8_t out[4] = {0, 0, 0, 0};

    for (int b = 0; b < 32 && (base_byte * 8 + b) < total_bits; b++) {
        int global_bit = word_idx * 32 + b;
        if (global_bit >= total_bits) break;

        // Deinterleave
        int cb_idx = global_bit / bits_per_cb;
        int bit_in_cb = global_bit % bits_per_cb;
        int row = bit_in_cb / Q_m;
        int col = bit_in_cb % Q_m;
        int cols = bits_per_cb / Q_m;
        int interleaved_idx = col * cols + row;

        // Read from rate matched buffer (MSB-first within each byte)
        int word_in_cb = interleaved_idx / 32;
        int bit_in_word = interleaved_idx % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);  // MSB-first within byte
        int rm_bit_pos = byte_in_word * 8 + bit_in_byte;
        uint32_t rm_word = d_rate_matched[cb_idx * words_per_cb + word_in_cb];
        uint8_t bit = (rm_word >> rm_bit_pos) & 1;

        // Scramble based on OUTPUT position (after interleaving)
        // MSB-first to match gold_sequence_generate_kernel
        int scr_word_idx = global_bit / 32;
        int scr_bit_pos = 31 - (global_bit % 32);  // MSB-first: bit 0 at position 31
        uint8_t scr_bit = (d_scramble_seq[scr_word_idx] >> scr_bit_pos) & 1;
        bit ^= scr_bit;

        // Pack
        int byte_offset = b / 8;
        int bit_pos = 7 - (b % 8);
        out[byte_offset] |= (bit << bit_pos);
    }

    // Write output
    for (int i = 0; i < 4 && base_byte + i < num_bytes; i++) {
        d_output[base_byte + i] = out[i];
    }
}

// Batch encoder context
struct tb_batch_encoder_ctx {
    tb_batch_encoder_config_t config;
    nr_tb_config_t tb_cfg;

    ldpc_encoder_handle_t ldpc_enc;
    rate_matcher_handle_t rate_matcher;
    scrambler_handle_t scrambler;

    // Device buffers for batch processing
    uint8_t* d_tbs_with_crc;    // All TBs with CRC space
    uint32_t* d_cb_bits;         // All CBs segmented
    uint32_t* d_encoded_bits;    // All CBs encoded
    uint32_t* d_rate_matched;    // All CBs rate matched
    uint32_t* d_crc_results;     // One TB CRC result per batched TB

    int max_batch_size;
    int cbs_per_tb;
    int tb_input_bytes;
    int tb_with_crc_bytes;
    int output_bits_per_tb;
    int output_bytes_per_tb;
};

nr_ldpc_status_t tb_batch_encoder_create(tb_batch_encoder_handle_t* handle) {
    if (!handle) return NR_LDPC_ERROR_INVALID_CONFIG;

    tb_batch_encoder_ctx* ctx = new (std::nothrow) tb_batch_encoder_ctx;
    if (!ctx) return NR_LDPC_ERROR_ALLOC_FAILED;

    memset(ctx, 0, sizeof(*ctx));

    nr_ldpc_status_t status = ldpc_encoder_create(&ctx->ldpc_enc);
    if (status != NR_LDPC_SUCCESS) {
        delete ctx;
        return status;
    }

    status = rate_matcher_create(&ctx->rate_matcher);
    if (status != NR_LDPC_SUCCESS) {
        ldpc_encoder_destroy(ctx->ldpc_enc);
        delete ctx;
        return status;
    }

    status = scrambler_create(&ctx->scrambler);
    if (status != NR_LDPC_SUCCESS) {
        rate_matcher_destroy(ctx->rate_matcher);
        ldpc_encoder_destroy(ctx->ldpc_enc);
        delete ctx;
        return status;
    }

    *handle = ctx;
    return NR_LDPC_SUCCESS;
}

void tb_batch_encoder_destroy(tb_batch_encoder_handle_t handle) {
    if (!handle) return;

    ldpc_encoder_destroy(handle->ldpc_enc);
    rate_matcher_destroy(handle->rate_matcher);
    scrambler_destroy(handle->scrambler);

    if (handle->d_tbs_with_crc) cudaFree(handle->d_tbs_with_crc);
    if (handle->d_cb_bits) cudaFree(handle->d_cb_bits);
    if (handle->d_encoded_bits) cudaFree(handle->d_encoded_bits);
    if (handle->d_rate_matched) cudaFree(handle->d_rate_matched);
    if (handle->d_crc_results) cudaFree(handle->d_crc_results);

    delete handle;
}

nr_ldpc_status_t tb_batch_encoder_configure(tb_batch_encoder_handle_t handle,
                                             const tb_batch_encoder_config_t* cfg) {
    if (!handle || !cfg) return NR_LDPC_ERROR_INVALID_CONFIG;

    handle->config = *cfg;
    handle->max_batch_size = cfg->max_batch_size;

    // Initialize TB configuration
    nr_ldpc_status_t status = nr_tb_init_config(&handle->tb_cfg,
                                                 cfg->tb_cfg.tb_size_bits,
                                                 cfg->tb_cfg.code_rate);
    if (status != NR_LDPC_SUCCESS) return status;

    // Configure LDPC encoder
    status = ldpc_encoder_configure(handle->ldpc_enc, &handle->tb_cfg.ldpc_cfg);
    if (status != NR_LDPC_SUCCESS) return status;

    // Configure rate matcher
    int rm_bits = cfg->tb_cfg.num_allocated_res / handle->tb_cfg.num_code_blocks;
    nr_rate_match_config_t rm_cfg = {
        .E = rm_bits,
        .Q_m = cfg->tb_cfg.modulation_order,
        .rv = cfg->tb_cfg.redundancy_version,
        .N_cb = 0,  // Will be set automatically
        .k0 = 0,
        .limited_buffer = false
    };
    status = rate_matcher_configure_tx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg);
    if (status != NR_LDPC_SUCCESS) return status;

    // Configure scrambler (if enabled)
    if (cfg->tb_cfg.enable_scrambling) {
        nr_scrambling_config_t scr_cfg = {
            .n_RNTI = cfg->tb_cfg.n_RNTI,
            .n_ID = cfg->tb_cfg.n_ID,
            .q = cfg->tb_cfg.q
        };
        status = scrambler_configure(handle->scrambler, &scr_cfg);
        if (status != NR_LDPC_SUCCESS) return status;
    }

    // Calculate sizes
    handle->cbs_per_tb = handle->tb_cfg.num_code_blocks;
    handle->tb_input_bytes = (cfg->tb_cfg.tb_size_bits + 7) / 8;
    handle->tb_with_crc_bytes = (cfg->tb_cfg.tb_size_bits + handle->tb_cfg.tb_crc_bits + 7) / 8;
    handle->output_bits_per_tb = cfg->tb_cfg.num_allocated_res;
    handle->output_bytes_per_tb = (handle->output_bits_per_tb + 7) / 8;

    int max_tbs = cfg->max_batch_size;
    int max_total_cbs = max_tbs * handle->cbs_per_tb;

    int ldpc_input_bits = ldpc_encoder_get_input_bits(handle->ldpc_enc);
    int cb_words = (ldpc_input_bits + 31) / 32;
    int encoded_words = ldpc_encoder_get_output_words(handle->ldpc_enc);
    int rm_words = (rm_bits + 31) / 32;

    // Allocate buffers for max batch size
    if (handle->d_tbs_with_crc) cudaFree(handle->d_tbs_with_crc);
    if (handle->d_cb_bits) cudaFree(handle->d_cb_bits);
    if (handle->d_encoded_bits) cudaFree(handle->d_encoded_bits);
    if (handle->d_rate_matched) cudaFree(handle->d_rate_matched);
    if (handle->d_crc_results) cudaFree(handle->d_crc_results);

    cudaMalloc(&handle->d_tbs_with_crc, max_tbs * handle->tb_with_crc_bytes);
    cudaMalloc(&handle->d_cb_bits, max_total_cbs * cb_words * sizeof(uint32_t));
    cudaMalloc(&handle->d_encoded_bits, max_total_cbs * encoded_words * sizeof(uint32_t));
    cudaMalloc(&handle->d_rate_matched, max_total_cbs * rm_words * sizeof(uint32_t));
    cudaMalloc(&handle->d_crc_results, max_tbs * sizeof(uint32_t));

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t tb_batch_encoder_encode(tb_batch_encoder_handle_t handle,
                                          const uint8_t* d_tb_inputs,
                                          uint8_t* d_outputs,
                                          int num_tbs,
                                          cudaStream_t stream) {
    if (!handle || !d_tb_inputs || !d_outputs || num_tbs <= 0) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (num_tbs > handle->max_batch_size) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    int total_cbs = num_tbs * handle->cbs_per_tb;
    int tb_size_bits = handle->config.tb_cfg.tb_size_bits;
    int tb_with_crc_bits = tb_size_bits + handle->tb_cfg.tb_crc_bits;

    // Step 1: Copy all TBs to working buffer (with CRC space)
    // Use cudaMemcpy2D for efficient batched copy instead of per-TB loop
    if (handle->tb_input_bytes == handle->tb_with_crc_bytes) {
        // No CRC padding needed - single contiguous copy
        cudaMemcpyAsync(handle->d_tbs_with_crc, d_tb_inputs,
                        num_tbs * handle->tb_input_bytes,
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        // Use 2D memcpy for strided copy with CRC padding
        // Zero the entire destination first (one call instead of num_tbs calls)
        cudaMemsetAsync(handle->d_tbs_with_crc, 0,
                        num_tbs * handle->tb_with_crc_bytes, stream);
        // Then copy TBs with 2D memcpy
        cudaMemcpy2DAsync(handle->d_tbs_with_crc,              // dst
                          handle->tb_with_crc_bytes,           // dpitch
                          d_tb_inputs,                         // src
                          handle->tb_input_bytes,              // spitch
                          handle->tb_input_bytes,              // width
                          num_tbs,                             // height
                          cudaMemcpyDeviceToDevice, stream);
    }

    // Step 2: Batch parallel CRC compute + attach.
    nr_ldpc_status_t crc_status = crc24a_compute_batch(handle->d_tbs_with_crc,
                                                       handle->d_crc_results,
                                                       num_tbs,
                                                       tb_size_bits,
                                                       handle->tb_with_crc_bytes,
                                                       stream);
    if (crc_status != NR_LDPC_SUCCESS) {
        return crc_status;
    }
    batch_crc24a_attach_kernel<<<num_tbs, 1, 0, stream>>>(
        handle->d_tbs_with_crc,
        handle->d_crc_results,
        tb_size_bits,
        handle->tb_with_crc_bytes);

    // Step 3: Batch segmentation (ALL CBs from ALL TBs in ONE kernel!)
    int ldpc_input_bits = ldpc_encoder_get_input_bits(handle->ldpc_enc);
    int cb_stride_words = (ldpc_input_bits + 31) / 32;
    batch_segment_tb_kernel<<<total_cbs, 256, 0, stream>>>(
        handle->d_tbs_with_crc, handle->d_cb_bits,
        tb_with_crc_bits,
        handle->tb_with_crc_bytes,
        handle->tb_cfg.cb_size_bits,
        handle->cbs_per_tb,
        cb_stride_words,
        handle->tb_cfg.cb_crc_bits);

    // Step 4: LDPC encoding (ALL CBs in ONE call!)
    ldpc_encoder_encode_batch(handle->ldpc_enc,
                               handle->d_cb_bits, handle->d_encoded_bits,
                               total_cbs, stream);

    // Step 5: Rate matching (ALL CBs in ONE call!)
    rate_matcher_match_batch(handle->rate_matcher,
                              handle->d_encoded_bits,
                              handle->d_rate_matched,
                              total_cbs,
                              stream);

    // Step 6: Interleave + scramble + pack (ALL output in ONE kernel!)
    int Q_m = handle->config.tb_cfg.modulation_order;
    int total_bits = num_tbs * handle->output_bits_per_tb;
    int rm_bits = handle->config.tb_cfg.num_allocated_res / handle->cbs_per_tb;
    int rm_words = (rm_bits + 31) / 32;
    int output_bytes = num_tbs * handle->output_bytes_per_tb;

    if (handle->config.tb_cfg.enable_scrambling && Q_m >= 2) {
        // Generate scrambling sequence for all bits
        nr_ldpc_status_t scr_status = scrambler_generate_sequence(handle->scrambler, total_bits, stream);
        if (scr_status != NR_LDPC_SUCCESS) return scr_status;
        const uint32_t* d_scramble_seq = scrambler_get_sequence_ptr(handle->scrambler);

        int num_words = (output_bytes + 3) / 4;
        int block_size = 256;
        int num_blocks = (num_words + block_size - 1) / block_size;

        batch_interleave_scramble_pack_kernel<<<num_blocks, block_size, 0, stream>>>(
            handle->d_rate_matched,
            d_scramble_seq,
            d_outputs,
            rm_bits,
            rm_words,
            total_cbs,
            Q_m,
            total_bits,
            output_bytes);
    }

    return NR_LDPC_SUCCESS;
}

int tb_batch_encoder_get_output_bits(tb_batch_encoder_handle_t handle) {
    return handle ? handle->output_bits_per_tb : 0;
}

int tb_batch_encoder_get_input_bits(tb_batch_encoder_handle_t handle) {
    return handle ? handle->config.tb_cfg.tb_size_bits : 0;
}

// ============================================================================
// TB Decoder API Implementation
// ============================================================================

nr_ldpc_status_t tb_decoder_create(tb_decoder_handle_t* handle) {
    if (!handle) return NR_LDPC_ERROR_INVALID_CONFIG;

    tb_decoder_ctx* ctx = new (std::nothrow) tb_decoder_ctx;
    if (!ctx) return NR_LDPC_ERROR_ALLOC_FAILED;

    memset(ctx, 0, sizeof(*ctx));

    nr_ldpc_status_t status = ldpc_decoder_create(&ctx->ldpc_dec);
    if (status != NR_LDPC_SUCCESS) {
        delete ctx;
        return status;
    }

    status = rate_matcher_create(&ctx->rate_matcher);
    if (status != NR_LDPC_SUCCESS) {
        ldpc_decoder_destroy(ctx->ldpc_dec);
        delete ctx;
        return status;
    }

    status = scrambler_create(&ctx->scrambler);
    if (status != NR_LDPC_SUCCESS) {
        rate_matcher_destroy(ctx->rate_matcher);
        ldpc_decoder_destroy(ctx->ldpc_dec);
        delete ctx;
        return status;
    }

    // Pre-allocate CRC buffers (avoid cudaMalloc in hot path!)
    cudaError_t err = cudaMalloc(&ctx->d_computed_crc, sizeof(uint32_t));
    if (err != cudaSuccess) {
        scrambler_destroy(ctx->scrambler);
        rate_matcher_destroy(ctx->rate_matcher);
        ldpc_decoder_destroy(ctx->ldpc_dec);
        delete ctx;
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }
    err = cudaMalloc(&ctx->d_received_crc, sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(ctx->d_computed_crc);
        scrambler_destroy(ctx->scrambler);
        rate_matcher_destroy(ctx->rate_matcher);
        ldpc_decoder_destroy(ctx->ldpc_dec);
        delete ctx;
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }

    // GPU-side CRC result buffer
    err = cudaMalloc(&ctx->d_crc_pass, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(ctx->d_received_crc);
        cudaFree(ctx->d_computed_crc);
        scrambler_destroy(ctx->scrambler);
        rate_matcher_destroy(ctx->rate_matcher);
        ldpc_decoder_destroy(ctx->ldpc_dec);
        delete ctx;
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }

    // Pinned host memory for async CRC result copy (avoids sync!)
    err = cudaMallocHost(&ctx->h_crc_pass, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(ctx->d_crc_pass);
        cudaFree(ctx->d_received_crc);
        cudaFree(ctx->d_computed_crc);
        scrambler_destroy(ctx->scrambler);
        rate_matcher_destroy(ctx->rate_matcher);
        ldpc_decoder_destroy(ctx->ldpc_dec);
        delete ctx;
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }

    *handle = ctx;
    return NR_LDPC_SUCCESS;
}

void tb_decoder_destroy(tb_decoder_handle_t handle) {
    if (!handle) return;

    ldpc_decoder_destroy(handle->ldpc_dec);
    rate_matcher_destroy(handle->rate_matcher);
    scrambler_destroy(handle->scrambler);

    if (handle->d_descrambled_llrs) cudaFree(handle->d_descrambled_llrs);
    if (handle->d_deinterleaved_llrs) cudaFree(handle->d_deinterleaved_llrs);
    if (handle->d_derate_llrs) cudaFree(handle->d_derate_llrs);
    if (handle->d_decoded_bits) cudaFree(handle->d_decoded_bits);
    if (handle->d_tb_output) cudaFree(handle->d_tb_output);
    if (handle->d_llrs_half) cudaFree(handle->d_llrs_half);
    if (handle->d_derate_llrs_half) cudaFree(handle->d_derate_llrs_half);
    if (handle->d_computed_crc) cudaFree(handle->d_computed_crc);
    if (handle->d_received_crc) cudaFree(handle->d_received_crc);
    if (handle->d_crc_pass) cudaFree(handle->d_crc_pass);
    if (handle->h_crc_pass) cudaFreeHost(handle->h_crc_pass);

    delete handle;
}

nr_ldpc_status_t tb_decoder_configure(tb_decoder_handle_t handle,
                                      const tb_decoder_config_t* cfg) {
    if (!handle || !cfg) return NR_LDPC_ERROR_INVALID_CONFIG;

    handle->config = *cfg;

    // Initialize TB configuration
    nr_ldpc_status_t status = nr_tb_init_config(&handle->tb_cfg, cfg->tb_size_bits, cfg->code_rate);
    if (status != NR_LDPC_SUCCESS) return status;

    // Calculate per-CB E values (same as encoder, per 3GPP TS 38.212 Section 5.4.2.1)
    int G = cfg->num_received_bits;
    int Q_m = cfg->modulation_order;
    int nof_layers = cfg->num_layers;
    int C = handle->tb_cfg.num_code_blocks;

    int total_symbols = G / Q_m;
    int symbols_per_layer = total_symbols / nof_layers;

    // nof_short_segments = C - (symbols_per_layer % C)
    // Short CBs use floor, remaining use ceiling
    handle->nof_short_segments = C - (symbols_per_layer % C);
    int symbols_short = symbols_per_layer / C;
    int symbols_long = (symbols_per_layer + C - 1) / C;  // ceiling
    handle->E_short = symbols_short * nof_layers * Q_m;
    handle->E_long = symbols_long * nof_layers * Q_m;

    // Configure LDPC decoder
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = cfg->max_iterations;
    dec_params.llr_clamp = cfg->llr_clamp;
    dec_params.skip_iteration_stats = false;  // Enable iteration stats for diagnostics.

    // Set CRC type based on TB segmentation for early termination
    // Default is CRC24B (multi-CB), but single-CB uses TB CRC (24A or 16)
    if (handle->tb_cfg.num_code_blocks == 1) {
        dec_params.crc_type = (handle->tb_cfg.tb_crc_bits == 24)
                              ? LDPC_CRC_24A : LDPC_CRC_16;
    } else {
        dec_params.crc_type = LDPC_CRC_24B;
    }

    // BG2 requires special decoder settings for reliable decoding
    // BG2 has fewer systematic bits (10*Z vs 22*Z for BG1) and higher puncturing
    // at moderate code rates, requiring offset min-sum and higher LLR clamp
    bool is_bg2 = (handle->tb_cfg.ldpc_cfg.base_graph == 2);

    // Calculate erasure ratio for adaptive decoder settings
    int E = cfg->num_received_bits / handle->tb_cfg.num_code_blocks;
    int Z = handle->tb_cfg.ldpc_cfg.lifting_size;
    int N_cols = is_bg2 ? 52 : 68;
    float erasure_ratio = 1.0f - (float)(E / Z) / (float)(N_cols - 2);

    // For BG2 or when explicitly requested, use offset min-sum decoder
    // BG2 needs this for reliable convergence at moderate-to-high erasure ratios
    if (is_bg2 || cfg->use_offset_minsum) {
        // For BG2, always use higher LLR clamp if user didn't set a high value
        if (is_bg2 && cfg->llr_clamp < 64.0f) {
            dec_params.llr_clamp = 127.0f;
        }

        // Use rate-adaptive normalization from cuPHY tables for best sensitivity
        // auto_scale=true selects optimal normalization based on base graph and rate
        dec_params.auto_scale = true;
        // Small offset helps with BG2 at high erasure ratios
        if (erasure_ratio > 0.5f) {
            dec_params.min_sum_offset = 0.5f;
        }
    }

    status = ldpc_decoder_configure(handle->ldpc_dec, &handle->tb_cfg.ldpc_cfg, &dec_params);
    if (status != NR_LDPC_SUCCESS) return status;

    // Pre-allocate LDPC workspace to avoid cudaMalloc in hot path
    // This eliminates ~100µs of allocation overhead per decode call
    status = ldpc_decoder_preallocate_workspace(handle->ldpc_dec, handle->tb_cfg.num_code_blocks);
    if (status != NR_LDPC_SUCCESS) return status;

    // Configure rate matcher with E_long (will reconfigure per-CB in decode)
    nr_rate_match_config_t rm_cfg = {
        .E = handle->E_long,  // Use max E for buffer sizing
        .Q_m = cfg->modulation_order,
        .rv = cfg->redundancy_version,
        .N_cb = 0,
        .k0 = 0,
        .limited_buffer = false
    };
    status = rate_matcher_configure_rx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg);
    if (status != NR_LDPC_SUCCESS) return status;

    // Configure scrambler (if enabled)
    if (cfg->enable_scrambling) {
        nr_scrambling_config_t scr_cfg = {
            .n_RNTI = cfg->n_RNTI,
            .n_ID = cfg->n_ID,
            .q = cfg->q
        };
        status = scrambler_configure(handle->scrambler, &scr_cfg);
        if (status != NR_LDPC_SUCCESS) return status;
    }

    // Allocate device buffers - calculate sizes from config
    // IMPORTANT: Decoder expects LLRs for FULL codeword (including punctured positions)
    // For BG2: 52*Z LLRs, for BG1: 68*Z LLRs
    int N_full_nodes = (handle->tb_cfg.ldpc_cfg.base_graph == 1) ? 68 : 52;
    int N_full_llrs = N_full_nodes * handle->tb_cfg.ldpc_cfg.lifting_size;
    int Kb = (handle->tb_cfg.ldpc_cfg.base_graph == 1) ? 22 : 10;
    int K_bits = Kb * handle->tb_cfg.ldpc_cfg.lifting_size;
    int K_words = (K_bits + 31) / 32;

    handle->input_llrs_size = cfg->num_received_bits * sizeof(float);
    handle->derate_llrs_size = handle->tb_cfg.num_code_blocks * N_full_llrs * sizeof(float);
    handle->decoded_bits_size = handle->tb_cfg.num_code_blocks * K_words * sizeof(uint32_t);
    // Include CRC bytes in output for CRC validation
    int tb_with_crc_bits = cfg->tb_size_bits + handle->tb_cfg.tb_crc_bits;
    int tb_with_crc_bytes = (tb_with_crc_bits + 7) / 8;
    handle->tb_output_size = tb_with_crc_bytes;

    cudaFree(handle->d_descrambled_llrs);
    cudaFree(handle->d_deinterleaved_llrs);
    cudaFree(handle->d_derate_llrs);
    cudaFree(handle->d_decoded_bits);
    cudaFree(handle->d_tb_output);
    cudaFree(handle->d_llrs_half);
    cudaFree(handle->d_derate_llrs_half);

    cudaMalloc(&handle->d_descrambled_llrs, handle->input_llrs_size);
    cudaMalloc(&handle->d_deinterleaved_llrs, handle->input_llrs_size);
    cudaMalloc(&handle->d_derate_llrs, handle->derate_llrs_size);
    cudaMalloc(&handle->d_decoded_bits, handle->decoded_bits_size);
    cudaMalloc(&handle->d_tb_output, handle->tb_output_size);

    // FP16 buffers (half the size of FP32)
    size_t input_llrs_size_half = cfg->num_received_bits * sizeof(__half);
    size_t derate_llrs_size_half = handle->tb_cfg.num_code_blocks * N_full_llrs * sizeof(__half);
    cudaMalloc(&handle->d_llrs_half, input_llrs_size_half);
    cudaMalloc(&handle->d_derate_llrs_half, derate_llrs_size_half);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t tb_decoder_decode(tb_decoder_handle_t handle,
                                   const float* d_llr_input,
                                   uint8_t* d_tb_output,
                                   tb_decode_result_t* result,
                                   cudaStream_t stream) {
    if (!handle || !d_llr_input || !d_tb_output) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // Calculate sizes from config
    // IMPORTANT: Decoder expects LLRs for FULL codeword (including punctured positions)
    int N_full_nodes = (handle->tb_cfg.ldpc_cfg.base_graph == 1) ? 68 : 52;
    int N_full_llrs = N_full_nodes * handle->tb_cfg.ldpc_cfg.lifting_size;
    int Kb = (handle->tb_cfg.ldpc_cfg.base_graph == 1) ? 22 : 10;
    int K_bits = Kb * handle->tb_cfg.ldpc_cfg.lifting_size;
    int K_words = (K_bits + 31) / 32;
    int total_llrs = handle->config.num_received_bits;
    int Q_m = handle->config.modulation_order;
    int num_cbs = handle->tb_cfg.num_code_blocks;

    // Per-CB E values (3GPP TS 38.212 Section 5.4.2.1)
    int nof_short = handle->nof_short_segments;
    int E_short = handle->E_short;
    int E_long = handle->E_long;

    // Pointer to current LLR data (will be updated through pipeline)
    const float* d_current_llrs = d_llr_input;

    // Step 1: Descrambling (per 3GPP TS 38.211 Section 7.3.1.1)
    // Flip LLR signs where scrambling sequence bit is 1
    if (handle->config.enable_scrambling) {
#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
        // Diagnostics: print decoder's scramble sequence and compare with expected.
        {
            cudaStreamSynchronize(stream);
            const uint32_t* d_seq = scrambler_get_sequence_ptr(handle->scrambler);
            if (d_seq) {
                std::vector<uint32_t> scr_diagnostics(4);
                cudaMemcpy(scr_diagnostics.data(), d_seq, scr_diagnostics.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[GPU PUSCH RX] Decoder scramble seq (first 4 words): %08x %08x %08x %08x\n",
                        scr_diagnostics[0], scr_diagnostics[1], scr_diagnostics[2], scr_diagnostics[3]);
            }
            fprintf(stderr, "[GPU PUSCH RX] c_init = 0x%08x\n", scrambler_get_c_init(handle->scrambler));
        }
#endif
        nr_ldpc_status_t scr_status = scrambler_descramble_llr(handle->scrambler,
                                                                d_current_llrs,
                                                                handle->d_descrambled_llrs,
                                                                total_llrs, stream);
        if (scr_status != NR_LDPC_SUCCESS) return scr_status;
#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
        // Diagnostics: print decoder's scramble sequence after regeneration.
        {
            cudaStreamSynchronize(stream);
            const uint32_t* d_seq = scrambler_get_sequence_ptr(handle->scrambler);
            if (d_seq) {
                std::vector<uint32_t> scr_diagnostics(4);
                cudaMemcpy(scr_diagnostics.data(), d_seq, scr_diagnostics.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[GPU PUSCH RX] After descramble, seq (first 4 words): %08x %08x %08x %08x\n",
                        scr_diagnostics[0], scr_diagnostics[1], scr_diagnostics[2], scr_diagnostics[3]);
            }
            // Print first few LLRs before and after descramble
            std::vector<float> llr_before(16), llr_after(16);
            cudaMemcpy(llr_before.data(), d_llr_input, 16 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(llr_after.data(), handle->d_descrambled_llrs, 16 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[GPU PUSCH RX] LLRs before descramble (first 16): ");
            for (int i = 0; i < 16; i++) fprintf(stderr, "%.0f ", llr_before[i] > 0 ? 0.0f : 1.0f);
            fprintf(stderr, "\n");
            fprintf(stderr, "[GPU PUSCH RX] LLRs after descramble (first 16): ");
            for (int i = 0; i < 16; i++) fprintf(stderr, "%.0f ", llr_after[i] > 0 ? 0.0f : 1.0f);
            fprintf(stderr, "\n");
        }
#endif
        d_current_llrs = handle->d_descrambled_llrs;
    }

    // Step 1.5: Byte-bit reorder - DISABLED
    // Analysis shows encoder paths have inconsistent bit ordering:
    // - Fast path (interleave_scramble_pack): MSB-first within bytes
    // - Fallback path (interleave_and_compact + bits_to_bytes): LSB-first within bytes
    // The reorder kernel was designed for MSB-first, but the fallback test passes.
    // Disabling to test if the issue is elsewhere.
#if 0
    {
        int block_size = 256;
        int num_blocks = (total_llrs + block_size - 1) / block_size;
        reorder_llrs_byte_bit_reverse_kernel<<<num_blocks, block_size, 0, stream>>>(
            d_current_llrs, handle->d_deinterleaved_llrs, total_llrs);
        d_current_llrs = handle->d_deinterleaved_llrs;
    }
#endif

    // Step 2: De-interleaving (for Q_m >= 2, per 3GPP TS 38.212 Section 5.4.2.2)
    // Interleaving applies to ALL modulation orders >= QPSK (Q_m >= 2)
    // OPTIMIZED: Use batch API (1-2 kernel launches instead of num_cbs launches)
    if (Q_m >= 2) {
        bool uniform_E_deint = (nof_short == 0) || (nof_short == num_cbs) || (E_short == E_long);

        if (uniform_E_deint) {
            // FAST PATH: All CBs have same E → single batch deinterleave
            int E_uniform = (nof_short == num_cbs) ? E_short : E_long;
            nr_ldpc_status_t deint_status = rate_matcher_deinterleave_llr_batch(
                handle->rate_matcher,
                d_current_llrs,
                handle->d_deinterleaved_llrs,
                E_uniform, Q_m, num_cbs, stream);
            if (deint_status != NR_LDPC_SUCCESS) return deint_status;
        } else {
            // 2-BATCH PATH: Short and long CBs have different E
            // Batch 1: Short CBs
            if (nof_short > 0) {
                nr_ldpc_status_t deint_status = rate_matcher_deinterleave_llr_batch(
                    handle->rate_matcher,
                    d_current_llrs,
                    handle->d_deinterleaved_llrs,
                    E_short, Q_m, nof_short, stream);
                if (deint_status != NR_LDPC_SUCCESS) return deint_status;
            }
            // Batch 2: Long CBs
            int nof_long = num_cbs - nof_short;
            if (nof_long > 0) {
                int short_total = nof_short * E_short;
                nr_ldpc_status_t deint_status = rate_matcher_deinterleave_llr_batch(
                    handle->rate_matcher,
                    d_current_llrs + short_total,
                    handle->d_deinterleaved_llrs + short_total,
                    E_long, Q_m, nof_long, stream);
                if (deint_status != NR_LDPC_SUCCESS) return deint_status;
            }
        }
        d_current_llrs = handle->d_deinterleaved_llrs;
    }

    // Step 3: De-rate matching using BATCH API to minimize kernel launches
    // Old approach: 14 separate kernel launches for 14 CBs → ~140 µs overhead!
    // New approach: 1-2 batch kernel launches → ~10-20 µs overhead

    // Calculate circular buffer size N_cb and starting position k0 per 5G NR spec
    // N_cb = (N_cols - 2) * Z  (after puncturing first 2*Z bits)
    // k0 depends on redundancy version (rv) and base graph
    int bg = handle->tb_cfg.ldpc_cfg.base_graph;
    int rv = handle->config.redundancy_version;
    int Z = handle->tb_cfg.ldpc_cfg.lifting_size;
    int N_cols = (bg == 1) ? 68 : 52;
    int N_cb = (N_cols - 2) * Z;  // Circular buffer size

    // Compute k0 using rate matcher helper (handles all rv cases per 3GPP TS 38.212)
    int k0 = rate_matcher_compute_k0(bg, Z, rv, N_cb);

    // Check if all CBs have same E (common case)
    bool uniform_E = (nof_short == 0) || (nof_short == num_cbs) || (E_short == E_long);

    if (uniform_E) {
        // FAST PATH: All CBs have same E → single batch dematch (1 kernel!)
        int E_uniform = (nof_short == num_cbs) ? E_short : E_long;
        nr_rate_match_config_t rm_cfg = {
            .E = E_uniform,
            .Q_m = Q_m,
            .rv = rv,
            .N_cb = N_cb,      // FIXED: Proper circular buffer size for repetition support
            .k0 = k0,          // FIXED: Proper starting position for this rv
            .limited_buffer = false
        };
        rate_matcher_configure_rx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg);
        rate_matcher_dematch_batch(handle->rate_matcher,
                                   d_current_llrs,
                                   handle->d_derate_llrs,
                                   num_cbs, stream);
    } else {
        // 2-BATCH PATH: Short and long CBs have different E → 2 batch calls
        // Still much better than num_cbs separate calls!

        // Batch 1: Short CBs (indices 0 to nof_short-1)
        if (nof_short > 0) {
            nr_rate_match_config_t rm_cfg_short = {
                .E = E_short,
                .Q_m = Q_m,
                .rv = rv,
                .N_cb = N_cb,  // FIXED: Proper circular buffer size for repetition support
                .k0 = k0,      // FIXED: Proper starting position for this rv
                .limited_buffer = false
            };
            rate_matcher_configure_rx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg_short);
            rate_matcher_dematch_batch(handle->rate_matcher,
                                       d_current_llrs,
                                       handle->d_derate_llrs,
                                       nof_short, stream);
        }

        // Batch 2: Long CBs (indices nof_short to num_cbs-1)
        int nof_long = num_cbs - nof_short;
        if (nof_long > 0) {
            int long_input_offset = nof_short * E_short;
            nr_rate_match_config_t rm_cfg_long = {
                .E = E_long,
                .Q_m = Q_m,
                .rv = rv,
                .N_cb = N_cb,  // FIXED: Proper circular buffer size for repetition support
                .k0 = k0,      // FIXED: Proper starting position for this rv
                .limited_buffer = false
            };
            rate_matcher_configure_rx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg_long);
            rate_matcher_dematch_batch(handle->rate_matcher,
                                       d_current_llrs + long_input_offset,
                                       handle->d_derate_llrs + nof_short * N_full_llrs,
                                       nof_long, stream);
        }
    }

    // NOTE: Previous LLR magnitude normalization was removed because it destroys
    // soft information and causes ~13 dB sensitivity loss. The LDPC decoder needs
    // relative LLR magnitudes for proper soft decoding. LLR clamping (via dec_cfg.llr_clamp)
    // handles extreme values without destroying relative magnitudes.

    // Step 4: LDPC decoding
    cudaMemsetAsync(handle->d_decoded_bits, 0, handle->decoded_bits_size, stream);

    ldpc_decoder_decode_batch(handle->ldpc_dec,
                                   handle->d_derate_llrs, handle->d_decoded_bits,
                                   handle->tb_cfg.num_code_blocks, stream);

    // Step 5: De-segmentation
    cudaMemsetAsync(handle->d_tb_output, 0, handle->tb_output_size, stream);

    // Extract TB + CRC bits (not just TB bits)
    int tb_with_crc_bits = handle->config.tb_size_bits + handle->tb_cfg.tb_crc_bits;
    desegment_cb_kernel<<<handle->tb_cfg.num_code_blocks, 256, 0, stream>>>(
        handle->d_decoded_bits, handle->d_tb_output,
        tb_with_crc_bits, handle->tb_cfg.cb_size_bits,
        handle->tb_cfg.num_code_blocks, K_words, handle->tb_cfg.cb_crc_bits);

    // Step 6: Copy TB (without CRC) to user output buffer
    // IMPORTANT: Only copy tb_size bytes to user, not the full tb_output_size which includes CRC
    int tb_bytes = (handle->config.tb_size_bits + 7) / 8;
    cudaMemcpyAsync(d_tb_output, handle->d_tb_output, tb_bytes,
                   cudaMemcpyDeviceToDevice, stream);

    // Step 6a: Clear unused bits in last byte for non-byte-aligned TBS
    // This prevents CRC bits from leaking into user output
    int valid_bits_last_byte = handle->config.tb_size_bits % 8;
    if (valid_bits_last_byte != 0 && tb_bytes > 0) {
        // MSB-first bit ordering (matches desegment_cb_kernel at line 666)
        uint8_t mask = (uint8_t)(0xFF << (8 - valid_bits_last_byte));
        clear_unused_bits_kernel<<<1, 1, 0, stream>>>(d_tb_output, tb_bytes - 1, mask);
    }

    // Fill result
    if (result) {
        result->num_code_blocks = handle->tb_cfg.num_code_blocks;
        result->avg_iterations = ldpc_decoder_get_avg_iterations(handle->ldpc_dec);

        // CRC validation using extract-and-compare method (works for non-byte-aligned TBS)
        // IMPORTANT: Extract received CRC BEFORE clearing unused bits, because for non-byte-aligned
        // TBS, the "unused" bits in the last data byte actually contain the first few CRC bits.
        // 1. Extract received CRC from the message (preserves CRC bits)
        // 2. Clear unused bits in last data byte (needed for correct CRC computation)
        // 3. Compute CRC over just the data bits
        // 4. Compare computed vs received

        // Use pre-allocated CRC buffers (avoids ~150 µs of cudaMalloc/cudaFree overhead!)
        // IMPORTANT: Must zero because crc16_compute only writes 16 bits to a 32-bit location
        uint32_t* d_computed_crc = handle->d_computed_crc;
        uint32_t* d_received_crc = handle->d_received_crc;
        cudaMemsetAsync(d_computed_crc, 0, sizeof(uint32_t), stream);
        cudaMemsetAsync(d_received_crc, 0, sizeof(uint32_t), stream);

        // Step 1: Extract received CRC FIRST (before modifying the buffer)
        extract_crc_kernel<<<1, 1, 0, stream>>>(
            handle->d_tb_output, handle->config.tb_size_bits,
            handle->tb_cfg.tb_crc_bits, d_received_crc);

        // Step 2: Clear unused bits in last data byte for correct CRC computation
        int remaining_bits = handle->config.tb_size_bits % 8;
        if (remaining_bits != 0) {
            clear_unused_bits_kernel<<<1, 1, 0, stream>>>(
                handle->d_tb_output, handle->config.tb_size_bits);
        }

        // Step 3: Compute CRC over data portion
        if (handle->tb_cfg.tb_crc_bits == 24) {
            crc24a_compute(handle->d_tb_output, handle->config.tb_size_bits, d_computed_crc, stream);
        } else {
            crc16_compute(handle->d_tb_output, handle->config.tb_size_bits,
                          (uint16_t*)d_computed_crc, stream);
        }

        // Step 4: GPU-only CRC comparison (avoids sync!)
        compare_crc_kernel<<<1, 1, 0, stream>>>(
            d_computed_crc, d_received_crc,
            handle->tb_cfg.tb_crc_bits, handle->d_crc_pass);

        // Async copy result to pinned host memory
        cudaMemcpyAsync(handle->h_crc_pass, handle->d_crc_pass, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);

        // Read result from pinned memory
        if (!handle->config.skip_crc_sync) {
            // Standard path: sync and return correct CRC result
            cudaStreamSynchronize(stream);
            result->crc_pass = *(handle->h_crc_pass);
        } else {
            // Fast path: return previous result (valid for pipelined decoding)
            // Current result will be available after caller's stream sync
            result->crc_pass = *(handle->h_crc_pass);  // Previous value
        }

#ifdef OCUDU_PHY_CUDA_PDSCH_DIAGNOSTICS
        // Diagnostic path still uses sync for detailed output.
        if (handle->config.tb_size_bits == 500) {
            uint32_t h_computed, h_received;
            cudaMemcpy(&h_computed, d_computed_crc, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_received, d_received_crc, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            uint32_t crc_mask = (handle->tb_cfg.tb_crc_bits == 24) ? 0xFFFFFF : 0xFFFF;
            fprintf(stderr, "[DIAG TBS500] CRC%d: computed=0x%x, received=0x%x, pass=%d\n",
                    handle->tb_cfg.tb_crc_bits, h_computed & crc_mask, h_received & crc_mask,
                    result->crc_pass);
        }
#endif

        // CRC buffers are pre-allocated - no need to free here

        // This decode path reports TB-level CRC status; per-codeblock CRC status is not exposed separately.
        result->num_cb_crc_pass = handle->tb_cfg.num_code_blocks;
    }

    return NR_LDPC_SUCCESS;
}

/**
 * @brief FP16 TB decode path (optimized - no FP32 overhead)
 *
 * Takes FP16 LLRs as input (already descrambled by frontend).
 * Uses FP16 throughout: deinterleave → rate dematch → LDPC decode.
 * ~50% less memory bandwidth than FP32 path.
 */
nr_ldpc_status_t tb_decoder_decode_half(tb_decoder_handle_t handle,
                                         const void* d_llr_input_half,
                                         uint8_t* d_tb_output,
                                         tb_decode_result_t* result,
                                         cudaStream_t stream) {
    if (!handle || !d_llr_input_half || !d_tb_output) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    const __half* d_input = reinterpret_cast<const __half*>(d_llr_input_half);

    // Calculate sizes
    int N_full_nodes = (handle->tb_cfg.ldpc_cfg.base_graph == 1) ? 68 : 52;
    int N_full_llrs = N_full_nodes * handle->tb_cfg.ldpc_cfg.lifting_size;
    int Kb = (handle->tb_cfg.ldpc_cfg.base_graph == 1) ? 22 : 10;
    int K_words = (Kb * handle->tb_cfg.ldpc_cfg.lifting_size + 31) / 32;
    int Q_m = handle->config.modulation_order;
    int num_cbs = handle->tb_cfg.num_code_blocks;
    int nof_short = handle->nof_short_segments;
    int E_short = handle->E_short;
    int E_long = handle->E_long;

    const __half* d_current = d_input;

    // Step 1: De-interleaving (FP16) - skip if Q_m < 2
    if (Q_m >= 2) {
        bool uniform_E = (nof_short == 0) || (nof_short == num_cbs) || (E_short == E_long);
        int block_size = 256;

        if (uniform_E) {
            int E_uniform = (nof_short == num_cbs) ? E_short : E_long;
            dim3 grid((E_uniform + block_size - 1) / block_size, num_cbs);
            bit_deinterleave_llr_half_kernel<<<grid, block_size, 0, stream>>>(
                d_current, handle->d_llrs_half, E_uniform, Q_m, num_cbs);
        } else {
            // Short CBs
            if (nof_short > 0) {
                dim3 grid((E_short + block_size - 1) / block_size, nof_short);
                bit_deinterleave_llr_half_kernel<<<grid, block_size, 0, stream>>>(
                    d_current, handle->d_llrs_half, E_short, Q_m, nof_short);
            }
            // Long CBs
            int nof_long = num_cbs - nof_short;
            if (nof_long > 0) {
                int short_total = nof_short * E_short;
                dim3 grid((E_long + block_size - 1) / block_size, nof_long);
                bit_deinterleave_llr_half_kernel<<<grid, block_size, 0, stream>>>(
                    d_current + short_total, handle->d_llrs_half + short_total,
                    E_long, Q_m, nof_long);
            }
        }
        d_current = handle->d_llrs_half;
    }

    // Step 2: De-rate matching (FP16 batch)
    bool uniform_E = (nof_short == 0) || (nof_short == num_cbs) || (E_short == E_long);
    if (uniform_E) {
        int E_uniform = (nof_short == num_cbs) ? E_short : E_long;
        nr_rate_match_config_t rm_cfg = {
            .E = E_uniform, .Q_m = Q_m, .rv = handle->config.redundancy_version,
            .N_cb = 0, .k0 = 0, .limited_buffer = false
        };
        rate_matcher_configure_rx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg);
        rate_matcher_dematch_batch_half(handle->rate_matcher,
            d_current, handle->d_derate_llrs_half, num_cbs, stream);
    } else {
        // Short CBs
        if (nof_short > 0) {
            nr_rate_match_config_t rm_cfg = {
                .E = E_short, .Q_m = Q_m, .rv = handle->config.redundancy_version,
                .N_cb = 0, .k0 = 0, .limited_buffer = false
            };
            rate_matcher_configure_rx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg);
            rate_matcher_dematch_batch_half(handle->rate_matcher,
                d_current, handle->d_derate_llrs_half, nof_short, stream);
        }
        // Long CBs
        int nof_long = num_cbs - nof_short;
        if (nof_long > 0) {
            int short_input_offset = nof_short * E_short;
            nr_rate_match_config_t rm_cfg = {
                .E = E_long, .Q_m = Q_m, .rv = handle->config.redundancy_version,
                .N_cb = 0, .k0 = 0, .limited_buffer = false
            };
            rate_matcher_configure_rx(handle->rate_matcher, &handle->tb_cfg.ldpc_cfg, &rm_cfg);
            rate_matcher_dematch_batch_half(handle->rate_matcher,
                d_current + short_input_offset,
                handle->d_derate_llrs_half + nof_short * N_full_llrs,
                nof_long, stream);
        }
    }

    // Step 3: LDPC decoding (FP16)
    cudaMemsetAsync(handle->d_decoded_bits, 0, handle->decoded_bits_size, stream);
    ldpc_decoder_decode_batch_half(handle->ldpc_dec,
        handle->d_derate_llrs_half, handle->d_decoded_bits, num_cbs, stream);

    // Step 4: De-segmentation (same as FP32 path)
    cudaMemsetAsync(handle->d_tb_output, 0, handle->tb_output_size, stream);
    int tb_with_crc_bits = handle->config.tb_size_bits + handle->tb_cfg.tb_crc_bits;
    desegment_cb_kernel<<<num_cbs, 256, 0, stream>>>(
        handle->d_decoded_bits, handle->d_tb_output,
        tb_with_crc_bits, handle->tb_cfg.cb_size_bits,
        num_cbs, K_words, handle->tb_cfg.cb_crc_bits);

    // Step 5: Copy TB to output
    int tb_bytes = (handle->config.tb_size_bits + 7) / 8;
    cudaMemcpyAsync(d_tb_output, handle->d_tb_output, tb_bytes,
                   cudaMemcpyDeviceToDevice, stream);

    // Clear unused bits in last byte
    int valid_bits_last_byte = handle->config.tb_size_bits % 8;
    if (valid_bits_last_byte != 0 && tb_bytes > 0) {
        // MSB-first bit ordering (matches desegment_cb_kernel at line 666)
        uint8_t mask = (uint8_t)(0xFF << (8 - valid_bits_last_byte));
        clear_unused_bits_kernel<<<1, 1, 0, stream>>>(d_tb_output, tb_bytes - 1, mask);
    }

    // Step 6: CRC check (same as FP32 path)
    if (result) {
        result->num_code_blocks = num_cbs;
        result->avg_iterations = ldpc_decoder_get_avg_iterations(handle->ldpc_dec);

        uint32_t* d_computed_crc = handle->d_computed_crc;
        uint32_t* d_received_crc = handle->d_received_crc;
        cudaMemsetAsync(d_computed_crc, 0, sizeof(uint32_t), stream);
        cudaMemsetAsync(d_received_crc, 0, sizeof(uint32_t), stream);

        extract_crc_kernel<<<1, 1, 0, stream>>>(
            handle->d_tb_output, handle->config.tb_size_bits,
            handle->tb_cfg.tb_crc_bits, d_received_crc);

        int remaining_bits = handle->config.tb_size_bits % 8;
        if (remaining_bits != 0) {
            clear_unused_bits_kernel<<<1, 1, 0, stream>>>(
                handle->d_tb_output, handle->config.tb_size_bits);
        }

        if (handle->tb_cfg.tb_crc_bits == 24) {
            crc24a_compute(handle->d_tb_output, handle->config.tb_size_bits, d_computed_crc, stream);
        } else {
            crc16_compute(handle->d_tb_output, handle->config.tb_size_bits,
                          (uint16_t*)d_computed_crc, stream);
        }

        compare_crc_kernel<<<1, 1, 0, stream>>>(
            d_computed_crc, d_received_crc,
            handle->tb_cfg.tb_crc_bits, handle->d_crc_pass);

        cudaMemcpyAsync(handle->h_crc_pass, handle->d_crc_pass, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);

        if (!handle->config.skip_crc_sync) {
            cudaStreamSynchronize(stream);
            result->crc_pass = *(handle->h_crc_pass);
        } else {
            result->crc_pass = *(handle->h_crc_pass);
        }

        result->num_cb_crc_pass = num_cbs;
    }

    return NR_LDPC_SUCCESS;
}

/**
 * @brief INT8 TB decode path (maximum efficiency - native INT8 LDPC decode)
 *
 * Takes FP16 LLRs as input, handles descrambling internally if enabled.
 * Uses: Fused FP16 descramble+deinterleave → fused FP16→INT8 rate dematch → native INT8 LDPC decode.
 * This is the most memory-efficient path with 2x less bandwidth than FP16.
 */
int tb_decoder_get_input_llrs(tb_decoder_handle_t handle) {
    return handle ? handle->config.num_received_bits : 0;
}

// ============================================================================
// LLR Gather Kernel for Non-Uniform CB Compaction
// ============================================================================

/**
 * @brief Gather scattered FP16 LLR segments into contiguous memory
 *
 * Each thread copies one element. Grid-stride loop handles arbitrary sizes.
 * Replaces N separate cudaMemcpyAsync D2D calls with a single kernel.
 */
__global__ void gather_llr_segments_kernel(
    const uint16_t* __restrict__ d_src,
    uint16_t* __restrict__ d_dst,
    const unsigned int* __restrict__ d_src_offsets,
    int segment_length,
    int num_segments)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_segments * segment_length;
    for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
        int seg = i / segment_length;
        int elem = i % segment_length;
        d_dst[i] = d_src[d_src_offsets[seg] + elem];
    }
}

// ============================================================================
// TB Desegmentation from Packed Decoder Output
// ============================================================================

/**
 * @brief Desegment code blocks from packed uint32_t format to TB bytes
 *
 * Converts LDPC decoder output (packed uint32_t, LSB first) to transport block
 * bytes (MSB first). Each thread block handles one code block.
 *
 * This kernel writes the data bits (excluding CB CRC) from each CB into
 * the appropriate position in the TB output buffer.
 */
__global__ void desegment_packed_kernel(
    const uint32_t* __restrict__ d_packed_cbs,  // Packed CB decoder output
    uint8_t* __restrict__ d_tb_output,          // Output TB bytes
    int tb_bits,                                // Total TB bits (A + TB_CRC)
    int cb_info_bits,                           // Info bits per CB (K - filler - CB_CRC)
    int num_cbs,                                // Number of code blocks
    int cb_stride_words,                        // Stride between CBs in words
    int cb_crc_bits,                            // CB CRC bits (24 for multi-CB, 0 for single)
    int nof_filler_bits                         // Filler bits (reserved, decoder output already excludes filler)
) {
    int cb_idx = blockIdx.x;
    if (cb_idx >= num_cbs) return;

    int data_bits_per_cb = cb_info_bits;
    int tb_offset = cb_idx * data_bits_per_cb;  // Bit offset in TB

    const uint32_t* cb_data = d_packed_cbs + cb_idx * cb_stride_words;
    int bits_to_copy = min(data_bits_per_cb, tb_bits - tb_offset);
    if (bits_to_copy <= 0) return;

    if ((cb_info_bits & 7) == 0) {
        // FAST PATH: cb_info_bits is byte-aligned → all CBs write non-overlapping
        // byte ranges. Input packed uint32_t and output uint8_t share identical
        // MSB-first byte ordering, so desegmentation is just a byte copy.
        const uint8_t* src = (const uint8_t*)cb_data;
        int start_byte = tb_offset >> 3;
        int full_bytes = bits_to_copy >> 3;
        int remaining = bits_to_copy & 7;

        for (int b = threadIdx.x; b < full_bytes; b += blockDim.x) {
            d_tb_output[start_byte + b] = src[b];
        }

        // Partial last byte (only possible on last CB when tb_bits % 8 != 0)
        if (remaining > 0 && threadIdx.x == 0) {
            uint8_t mask = (uint8_t)(0xFF << (8 - remaining));
            d_tb_output[start_byte + full_bytes] = src[full_bytes] & mask;
        }
    } else {
        // SLOW PATH: non-byte-aligned — per-bit atomicOr (original code)
        for (int bit_idx = threadIdx.x; bit_idx < data_bits_per_cb; bit_idx += blockDim.x) {
            int tb_bit_idx = tb_offset + bit_idx;
            if (tb_bit_idx >= tb_bits) continue;

            int word_idx = bit_idx / 32;
            int bit_pos = (bit_idx & 31) ^ 7;
            uint32_t word = cb_data[word_idx];
            uint8_t bit = (word >> bit_pos) & 1;

            int byte_idx = tb_bit_idx / 8;
            int byte_bit = 7 - (tb_bit_idx % 8);

            atomicOr((uint32_t*)&d_tb_output[byte_idx & ~3],
                     (uint32_t)bit << (byte_bit + 8 * (byte_idx & 3)));
        }
    }
}

/**
 * @brief Check CRC on desegmented TB (single thread) - Legacy version
 * Uses slicing-by-4 for efficiency
 */
__global__ void check_tb_crc_kernel(
    const uint8_t* __restrict__ d_tb_data,
    int total_bytes,  // Including CRC
    int crc_bits,     // 24 for CRC-24A, 16 for CRC-16
    int* __restrict__ d_result
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t crc = 0;
    int i = 0;

    if (crc_bits == 24) {
        // CRC-24A check using slicing-by-4
        int num_words = total_bytes / 4;
        for (int w = 0; w < num_words; w++) {
            uint8_t b0 = d_tb_data[i++];
            uint8_t b1 = d_tb_data[i++];
            uint8_t b2 = d_tb_data[i++];
            uint8_t b3 = d_tb_data[i++];

            uint8_t idx0 = ((crc >> 16) ^ b0) & 0xFF;
            uint8_t idx1 = ((crc >> 8) ^ b1) & 0xFF;
            uint8_t idx2 = (crc ^ b2) & 0xFF;
            uint8_t idx3 = b3;

            crc = CRC24A_S0[idx0] ^ CRC24A_S1[idx1] ^ CRC24A_S2[idx2] ^ CRC24A_S3[idx3];
        }

        // Process remaining bytes
        while (i < total_bytes) {
            uint8_t byte = d_tb_data[i++];
            uint8_t index = ((crc >> 16) ^ byte) & 0xFF;
            crc = (crc << 8) ^ CRC24A_TABLE[index];
        }

        crc &= 0xFFFFFF;
    } else {
        // CRC-16 check using slicing-by-4
        int num_words = total_bytes / 4;
        for (int w = 0; w < num_words; w++) {
            uint8_t b0 = d_tb_data[i++];
            uint8_t b1 = d_tb_data[i++];
            uint8_t b2 = d_tb_data[i++];
            uint8_t b3 = d_tb_data[i++];

            uint8_t idx0 = ((crc >> 8) ^ b0) & 0xFF;
            uint8_t idx1 = (crc ^ b1) & 0xFF;
            uint8_t idx2 = b2;
            uint8_t idx3 = b3;

            crc = CRC16_S0[idx0] ^ CRC16_S1[idx1] ^ CRC16_S2[idx2] ^ CRC16_S3[idx3];
        }

        // Process remaining bytes
        while (i < total_bytes) {
            uint8_t byte = d_tb_data[i++];
            uint8_t index = ((crc >> 8) ^ byte) & 0xFF;
            crc = (crc << 8) ^ CRC16_TABLE[index];
        }

        crc &= 0xFFFF;
    }

    // CRC passes if remainder is 0
    *d_result = (crc == 0) ? 1 : 0;
}

/**
 * @brief Block-parallel TB CRC-24A check kernel with shared-memory data staging
 *
 * 128 threads (4 warps) process the TB in parallel:
 * 1. CRC slicing-by-4 tables staged in shared memory (avoids constant-memory
 *    serialization when lanes hit different addresses).
 * 2. TB data cooperatively loaded into dynamic shared memory via coalesced
 *    uint32_t reads (128 threads × 4B = 512B per iteration).
 * 3. Each thread computes CRC for 1/128 of the TB from shared memory.
 * 4. Intra-warp shuffle reduction (5 rounds per warp) → 4 partial CRCs.
 * 5. Inter-warp reduction via shared memory (2 rounds by warp 0).
 *
 * Using more threads (128 vs 32) means each thread's chunk is ~96 bytes
 * instead of ~375 bytes, so the GF(2^24) shift amounts in the tree
 * reduction are 4× smaller → fewer multiplications → faster combining.
 *
 * Dynamic shared memory: caller passes tb_bytes (rounded to 4) via <<<>>>.
 */
__global__ void __launch_bounds__(128, 1) check_tb_crc_warp_kernel(
    const uint8_t* __restrict__ d_tb_data,
    int total_bytes,  // Including CRC
    int* __restrict__ d_result
) {
    // ---- Stage CRC-24A slicing tables into static shared memory ----
    __shared__ uint32_t s_S0[256], s_S1[256], s_S2[256], s_S3[256];
    __shared__ uint32_t s_TBL[256];
    __shared__ uint32_t s_warp_crcs[4];
    __shared__ int      s_warp_bytes[4];

    const int tid     = threadIdx.x;           // 0..127
    const int warp_id = tid / 32;              // 0..3
    const int lane_id = tid & 31;              // 0..31

    #pragma unroll
    for (int i = tid; i < 256; i += 128) {
        s_S0[i]  = CRC24A_S0[i];
        s_S1[i]  = CRC24A_S1[i];
        s_S2[i]  = CRC24A_S2[i];
        s_S3[i]  = CRC24A_S3[i];
        s_TBL[i] = CRC24A_TABLE[i];
    }

    // ---- Cooperatively load TB data into dynamic shared memory ----
    extern __shared__ uint8_t s_dyn[];
    uint32_t*       s_tb_words = reinterpret_cast<uint32_t*>(s_dyn);
    const uint32_t* d_tb_words = reinterpret_cast<const uint32_t*>(d_tb_data);
    int total_words = (total_bytes + 3) / 4;

    for (int i = tid; i < total_words; i += 128) {
        s_tb_words[i] = d_tb_words[i];
    }
    __syncthreads();

    // ---- Per-thread CRC from shared memory ----
    int bytes_per_thread = ((total_bytes + 127) / 128 + 3) & ~3;
    int my_start = tid * bytes_per_thread;
    int my_end   = min(my_start + bytes_per_thread, total_bytes);
    int my_bytes = max(my_end - my_start, 0);

    uint32_t my_crc = 0;

    if (my_bytes > 0) {
        const uint8_t* my_data = s_dyn + my_start;
        int pos = 0;
        int num_words = my_bytes / 4;

        for (int w = 0; w < num_words; w++) {
            uint8_t b0 = my_data[pos];
            uint8_t b1 = my_data[pos + 1];
            uint8_t b2 = my_data[pos + 2];
            uint8_t b3 = my_data[pos + 3];
            pos += 4;

            my_crc = s_S0[((my_crc >> 16) ^ b0) & 0xFF]
                   ^ s_S1[((my_crc >> 8)  ^ b1) & 0xFF]
                   ^ s_S2[( my_crc         ^ b2) & 0xFF]
                   ^ s_S3[b3];
        }

        while (pos < my_bytes) {
            uint8_t byte = my_data[pos++];
            my_crc = ((my_crc << 8) ^ s_TBL[((my_crc >> 16) ^ byte) & 0xFF]) & 0xFFFFFF;
        }

        my_crc &= 0xFFFFFF;
    }

    // ---- Intra-warp tree reduction (5 rounds per warp) ----
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        uint32_t right_crc   = __shfl_down_sync(0xFFFFFFFF, my_crc, offset);
        int      right_bytes = __shfl_down_sync(0xFFFFFFFF, my_bytes, offset);
        if ((lane_id & (2 * offset - 1)) == 0) {
            my_crc   = crc24a_shift_bytes_fast(my_crc, right_bytes) ^ right_crc;
            my_bytes += right_bytes;
        }
    }

    // ---- Inter-warp reduction via shared memory ----
    // Lane 0 of each warp stores its warp-level result.
    if (lane_id == 0) {
        s_warp_crcs[warp_id]  = my_crc;
        s_warp_bytes[warp_id] = my_bytes;
    }
    __syncthreads();

    // Warp 0 combines the 4 warp-level results (2 rounds of shuffle).
    if (warp_id == 0 && lane_id < 4) {
        my_crc   = s_warp_crcs[lane_id];
        my_bytes = s_warp_bytes[lane_id];

        // Round 1: lanes 0,2 combine with lanes 1,3
        uint32_t right_crc   = __shfl_down_sync(0xF, my_crc, 1);
        int      right_bytes = __shfl_down_sync(0xF, my_bytes, 1);
        if ((lane_id & 1) == 0) {
            my_crc   = crc24a_shift_bytes_fast(my_crc, right_bytes) ^ right_crc;
            my_bytes += right_bytes;
        }

        // Round 2: lane 0 combines with lane 2
        right_crc   = __shfl_down_sync(0xF, my_crc, 2);
        right_bytes = __shfl_down_sync(0xF, my_bytes, 2);
        if (lane_id == 0) {
            my_crc   = crc24a_shift_bytes_fast(my_crc, right_bytes) ^ right_crc;
            *d_result = (my_crc == 0) ? 1 : 0;
        }
    }
}

// ============================================================================
// Fused Deseg + CRC-24A Kernel (last-block completion pattern)
// ============================================================================

/**
 * @brief Fused desegmentation + CRC-24A kernel
 *
 * Combines desegment_packed_kernel and check_tb_crc_warp_kernel into a single
 * kernel launch using the "last block" completion pattern:
 *
 * Phase 1: Each block desegments its CB (identical to desegment_packed_kernel)
 * Phase 2: __threadfence() + atomicAdd to count completed blocks
 * Phase 3: Last block (atomicAdd returned num_cbs-1) computes CRC-24A
 *
 * The __threadfence() before the atomic ensures all deseg global writes are
 * visible to the last block. TB data is in L2 cache (just written), so CRC
 * reads are fast.
 *
 * 256 threads per block = 8 warps for CRC computation.
 * Shared memory: 5KB CRC tables + warp reduction scratch (only in last block).
 */
__global__ void __launch_bounds__(256, 4) desegment_and_crc_fused_kernel(
    const uint32_t* __restrict__ d_packed_cbs,
    uint8_t* __restrict__ d_tb_output,
    int tb_bits,
    int cb_info_bits,
    int num_cbs,
    int cb_stride_words,
    int cb_crc_bits,
    int nof_filler_bits,
    int tb_bytes,           // Total TB bytes including CRC (for CRC check)
    int* __restrict__ d_crc_result  // Output: 1=pass, 0=fail
) {
    // ---- Phase 1: Desegmentation ----
    int cb_idx = blockIdx.x;
    if (cb_idx < num_cbs) {
        int data_bits_per_cb = cb_info_bits;
        int tb_offset = cb_idx * data_bits_per_cb;
        const uint32_t* cb_data = d_packed_cbs + cb_idx * cb_stride_words;
        int bits_to_copy = min(data_bits_per_cb, tb_bits - tb_offset);

        if (bits_to_copy > 0 && (cb_info_bits & 7) == 0) {
            // FAST PATH: cb_info_bits is byte-aligned → all CBs write non-overlapping
            // byte ranges. Input packed uint32_t and output uint8_t share identical
            // MSB-first byte ordering, so desegmentation is just a byte copy.
            const uint8_t* src = (const uint8_t*)cb_data;
            int start_byte = tb_offset >> 3;
            int full_bytes = bits_to_copy >> 3;
            int remaining = bits_to_copy & 7;

            for (int b = threadIdx.x; b < full_bytes; b += blockDim.x) {
                d_tb_output[start_byte + b] = src[b];
            }

            // Partial last byte (only possible on last CB when tb_bits % 8 != 0)
            if (remaining > 0 && threadIdx.x == 0) {
                uint8_t mask = (uint8_t)(0xFF << (8 - remaining));
                d_tb_output[start_byte + full_bytes] = src[full_bytes] & mask;
            }
        } else if (bits_to_copy > 0) {
            // SLOW PATH: non-byte-aligned — per-bit atomicOr (original code)
            for (int bit_idx = threadIdx.x; bit_idx < bits_to_copy; bit_idx += blockDim.x) {
                int tb_bit_idx = tb_offset + bit_idx;
                if (tb_bit_idx >= tb_bits) continue;

                int word_idx = bit_idx / 32;
                int bit_pos = (bit_idx & 31) ^ 7;
                uint32_t word = cb_data[word_idx];
                uint8_t bit = (word >> bit_pos) & 1;

                int byte_idx = tb_bit_idx / 8;
                int byte_bit = 7 - (tb_bit_idx % 8);

                atomicOr((uint32_t*)&d_tb_output[byte_idx & ~3],
                         (uint32_t)bit << (byte_bit + 8 * (byte_idx & 3)));
            }
        }
    }

    // ---- Phase 2: Completion barrier ----
    __syncthreads();
    __threadfence();

    __shared__ int s_is_last;
    if (threadIdx.x == 0) {
        int completed = atomicAdd(d_crc_result, 1);
        s_is_last = (completed == num_cbs - 1) ? 1 : 0;
    }
    __syncthreads();

    if (!s_is_last) return;

    // ---- Phase 3: CRC-24A computation (last block only, 256 threads) ----

    // Stage CRC-24A slicing tables into shared memory
    __shared__ uint32_t s_S0[256], s_S1[256], s_S2[256], s_S3[256];
    __shared__ uint32_t s_TBL[256];
    __shared__ uint32_t s_warp_crcs[8];
    __shared__ int      s_warp_bytes[8];

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;

    // 256 threads load 256-entry tables in one pass
    s_S0[tid]  = CRC24A_S0[tid];
    s_S1[tid]  = CRC24A_S1[tid];
    s_S2[tid]  = CRC24A_S2[tid];
    s_S3[tid]  = CRC24A_S3[tid];
    s_TBL[tid] = CRC24A_TABLE[tid];
    __syncthreads();

    // Per-thread CRC over contiguous byte range from d_tb_output (global, L2-cached)
    int bytes_per_thread = ((tb_bytes + 255) / 256 + 3) & ~3;  // Round up to 4
    int my_start = tid * bytes_per_thread;
    int my_end   = min(my_start + bytes_per_thread, tb_bytes);
    int my_bytes = max(my_end - my_start, 0);

    uint32_t my_crc = 0;

    if (my_bytes > 0) {
        const uint8_t* my_data = d_tb_output + my_start;
        int pos = 0;
        int num_words = my_bytes / 4;

        for (int w = 0; w < num_words; w++) {
            uint8_t b0 = my_data[pos];
            uint8_t b1 = my_data[pos + 1];
            uint8_t b2 = my_data[pos + 2];
            uint8_t b3 = my_data[pos + 3];
            pos += 4;

            my_crc = s_S0[((my_crc >> 16) ^ b0) & 0xFF]
                   ^ s_S1[((my_crc >> 8)  ^ b1) & 0xFF]
                   ^ s_S2[( my_crc         ^ b2) & 0xFF]
                   ^ s_S3[b3];
        }

        while (pos < my_bytes) {
            uint8_t byte = my_data[pos++];
            my_crc = ((my_crc << 8) ^ s_TBL[((my_crc >> 16) ^ byte) & 0xFF]) & 0xFFFFFF;
        }

        my_crc &= 0xFFFFFF;
    }

    // Intra-warp tree reduction (5 rounds)
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        uint32_t right_crc   = __shfl_down_sync(0xFFFFFFFF, my_crc, offset);
        int      right_bytes = __shfl_down_sync(0xFFFFFFFF, my_bytes, offset);
        if ((lane_id & (2 * offset - 1)) == 0) {
            my_crc   = crc24a_shift_bytes_fast(my_crc, right_bytes) ^ right_crc;
            my_bytes += right_bytes;
        }
    }

    // Inter-warp reduction (8 warps → shared memory → warp 0)
    if (lane_id == 0) {
        s_warp_crcs[warp_id]  = my_crc;
        s_warp_bytes[warp_id] = my_bytes;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < 8) {
        my_crc   = s_warp_crcs[lane_id];
        my_bytes = s_warp_bytes[lane_id];

        // Round 1: lanes 0,2,4,6 combine with lanes 1,3,5,7
        uint32_t right_crc   = __shfl_down_sync(0xFF, my_crc, 1);
        int      right_bytes = __shfl_down_sync(0xFF, my_bytes, 1);
        if ((lane_id & 1) == 0) {
            my_crc   = crc24a_shift_bytes_fast(my_crc, right_bytes) ^ right_crc;
            my_bytes += right_bytes;
        }

        // Round 2: lanes 0,4 combine with lanes 2,6
        right_crc   = __shfl_down_sync(0xFF, my_crc, 2);
        right_bytes = __shfl_down_sync(0xFF, my_bytes, 2);
        if ((lane_id & 3) == 0) {
            my_crc   = crc24a_shift_bytes_fast(my_crc, right_bytes) ^ right_crc;
            my_bytes += right_bytes;
        }

        // Round 3: lane 0 combines with lane 4
        right_crc   = __shfl_down_sync(0xFF, my_crc, 4);
        right_bytes = __shfl_down_sync(0xFF, my_bytes, 4);
        if (lane_id == 0) {
            my_crc = crc24a_shift_bytes_fast(my_crc, right_bytes) ^ right_crc;
            // Overwrite the completion counter with the CRC result
            *d_crc_result = (my_crc == 0) ? 1 : 0;
        }
    }
}

nr_ldpc_status_t tb_desegment_and_check_crc(
    const uint32_t* d_packed_cbs,
    uint8_t* d_tb_output,
    const tb_desegment_config_t* cfg,
    tb_desegment_result_t* result,
    cudaStream_t stream)
{
    if (!d_packed_cbs || !d_tb_output || !cfg || !result) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    // Ensure CRC tables are initialized
    crc_init_tables();

    int tb_bytes = (cfg->tb_size_bits + cfg->tb_crc_bits + 7) / 8;
    if ((cfg->cb_info_bits & 7) != 0) {
        // The non-byte-aligned path uses atomicOr into the TB byte stream, so
        // bits not written by the kernel must start at zero.
        cudaMemsetAsync(d_tb_output, 0, tb_bytes, stream);
    }

    // Launch desegmentation kernel
    int num_cbs = cfg->num_code_blocks;
    int tb_total_bits = cfg->tb_size_bits + cfg->tb_crc_bits;

    desegment_packed_kernel<<<num_cbs, 256, 0, stream>>>(
        d_packed_cbs,
        d_tb_output,
        tb_total_bits,
        cfg->cb_info_bits,
        num_cbs,
        cfg->cb_stride_words,
        cfg->cb_crc_bits,
        cfg->nof_filler_bits);

    // Allocate temporary device memory for CRC result
    int* d_crc_result = nullptr;
    cudaMalloc(&d_crc_result, sizeof(int));

    // Launch CRC check kernel - warp-parallel for CRC-24A, single-thread for CRC-16
    if (cfg->tb_crc_bits == 24) {
        // Warp-parallel kernel: 32 threads with shared-memory tables + shuffle reduction
        check_tb_crc_warp_kernel<<<1, 128, ((tb_bytes + 3) / 4) * 4, stream>>>(
            d_tb_output,
            tb_bytes,
            d_crc_result);
    } else {
        // Single-threaded kernel for CRC-16 (small TBs only, A <= 3824 bits)
        check_tb_crc_kernel<<<1, 1, 0, stream>>>(
            d_tb_output,
            tb_bytes,
            cfg->tb_crc_bits,
            d_crc_result);
    }

    // Copy result back
    int h_crc_result = 0;
    cudaMemcpyAsync(&h_crc_result, d_crc_result, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    result->tb_crc_pass = h_crc_result;

    cudaFree(d_crc_result);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t tb_desegment_and_check_crc_async(
    const uint32_t* d_packed_cbs,
    uint8_t* d_tb_output,
    const tb_desegment_config_t* cfg,
    int* d_crc_result,
    cudaStream_t stream)
{
    if (!d_packed_cbs || !d_tb_output || !cfg) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    int tb_bytes = (cfg->tb_size_bits + cfg->tb_crc_bits + 7) / 8;
    int num_cbs = cfg->num_code_blocks;
    int tb_total_bits = cfg->tb_size_bits + cfg->tb_crc_bits;

    if ((cfg->cb_info_bits & 7) != 0) {
        // The byte-aligned path overwrites all produced bytes directly and
        // masks the partial final byte. Only the atomicOr bit path needs a
        // pre-zeroed destination.
        cudaMemsetAsync(d_tb_output, 0, tb_bytes, stream);
    }

    // Use fused deseg+CRC kernel for CRC-24A when CRC result is requested.
    // This saves one kernel launch + full TB re-read from global memory.
    if (d_crc_result && cfg->tb_crc_bits == 24) {
        crc_init_tables();

        // Zero the CRC result — fused kernel uses it as completion counter,
        // then last block overwrites with CRC pass/fail.
        cudaMemsetAsync(d_crc_result, 0, sizeof(int), stream);

        desegment_and_crc_fused_kernel<<<num_cbs, 256, 0, stream>>>(
            d_packed_cbs,
            d_tb_output,
            tb_total_bits,
            cfg->cb_info_bits,
            num_cbs,
            cfg->cb_stride_words,
            cfg->cb_crc_bits,
            cfg->nof_filler_bits,
            tb_bytes,
            d_crc_result);
    } else {
        // Non-fused path: separate deseg + CRC (for CRC-16 or no-CRC cases)
        desegment_packed_kernel<<<num_cbs, 256, 0, stream>>>(
            d_packed_cbs,
            d_tb_output,
            tb_total_bits,
            cfg->cb_info_bits,
            num_cbs,
            cfg->cb_stride_words,
            cfg->cb_crc_bits,
            cfg->nof_filler_bits);

        if (d_crc_result) {
            crc_init_tables();
            if (cfg->tb_crc_bits == 24) {
                check_tb_crc_warp_kernel<<<1, 128, ((tb_bytes + 3) / 4) * 4, stream>>>(
                    d_tb_output,
                    tb_bytes,
                    d_crc_result);
            } else {
                check_tb_crc_kernel<<<1, 1, 0, stream>>>(
                    d_tb_output,
                    tb_bytes,
                    cfg->tb_crc_bits,
                    d_crc_result);
            }
        }
    }

    return NR_LDPC_SUCCESS;
}

// ============================================================================
// LLR Gather API
// ============================================================================

nr_ldpc_status_t gather_llr_segments_half(
    const void* d_src_half,
    void* d_dst_half,
    const unsigned int* h_src_offsets,
    unsigned int* d_src_offsets,
    int segment_length,
    int num_segments,
    cudaStream_t stream)
{
    if (!d_src_half || !d_dst_half || !h_src_offsets || !d_src_offsets) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (num_segments <= 0 || segment_length <= 0) {
        return NR_LDPC_SUCCESS;
    }

    // Upload offsets H2D (small: num_segments * 4 bytes, typically ≤ 648 bytes)
    cudaMemcpyAsync(d_src_offsets, h_src_offsets,
                    num_segments * sizeof(unsigned int),
                    cudaMemcpyHostToDevice, stream);

    // Launch gather kernel
    int total_elements = num_segments * segment_length;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Cap blocks to avoid excessive grid for small workloads
    if (blocks > 1024) blocks = 1024;

    gather_llr_segments_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(d_src_half),
        reinterpret_cast<uint16_t*>(d_dst_half),
        d_src_offsets,
        segment_length,
        num_segments);

    return NR_LDPC_SUCCESS;
}

} // extern "C"
