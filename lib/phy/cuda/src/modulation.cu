/**
 * @file modulation.cu
 * @brief CUDA-accelerated Modulation and Soft Demodulation for 5G NR
 *
 * High-performance implementation of:
 * - BPSK, QPSK, 16QAM, 64QAM, 256QAM modulation
 * - Soft demodulation with exact LLR computation
 * - AWGN channel simulation
 */

#include "../include/modulation.h"
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <new>

// ============================================================================
// Device Constants - Constellation Points (Gray coded, normalized)
// ============================================================================

// QPSK constellation: 1/sqrt(2), Gray coded
// bits = b1b0 (LSB first): b0 determines I sign, b1 determines Q sign
// b0=0 -> I+, b0=1 -> I-, b1=0 -> Q+, b1=1 -> Q-
__device__ __constant__ float QPSK_I[4] = { 0.7071068f, -0.7071068f,  0.7071068f, -0.7071068f};  // bits: 00, 01, 10, 11
__device__ __constant__ float QPSK_Q[4] = { 0.7071068f,  0.7071068f, -0.7071068f, -0.7071068f};

// 16QAM constellation (normalized by 1/sqrt(10))
// Matches cuPHY: I uses bits 0,2; Q uses bits 1,3
// Table indexed by (bit0) | (bit2 << 1) for I
// Table indexed by (bit1) | (bit3 << 1) for Q
__device__ __constant__ float QAM16_TABLE[8] = {
     0.316227766f,   // index 0: +1
    -0.316227766f,   // index 1: -1
     0.316227766f,   // index 2: +1 (same as 0, placeholder)
    -0.316227766f,   // index 3: -1 (same as 1, placeholder)
     0.948683298f,   // index 4: +3
    -0.948683298f,   // index 5: -3
     0.948683298f,   // index 6: +3 (same as 4, placeholder)
    -0.948683298f    // index 7: -3 (same as 5, placeholder)
};

// 64QAM constellation table (normalized by 1/sqrt(42))
// Matches cuPHY: I uses bits 0,2,4; Q uses bits 1,3,5
// Table indexed by (bit0) | (bit2 >> 1) | (bit4 >> 2)
__device__ __constant__ float QAM64_TABLE[8] = {
     0.462910049886276f,   // index 0: +3
    -0.462910049886276f,   // index 1: -3
     0.77151674981046f,    // index 2: +5
    -0.77151674981046f,    // index 3: -5
     0.154303349962092f,   // index 4: +1
    -0.154303349962092f,   // index 5: -1
     1.08012344973464f,    // index 6: +7
    -1.08012344973464f     // index 7: -7
};

// 256QAM constellation table (normalized by 1/sqrt(170))
// Matches cuPHY: I uses bits 0,2,4,6; Q uses bits 1,3,5,7
__device__ __constant__ float QAM256_TABLE[16] = {
     0.383482494f,   // index 0
    -0.383482494f,   // index 1
     0.843661488f,   // index 2
    -0.843661488f,   // index 3
     0.230089497f,   // index 4
    -0.230089497f,   // index 5
     0.997054486f,   // index 6
    -0.997054486f,   // index 7
     0.536875492f,   // index 8
    -0.536875492f,   // index 9
     0.69026849f,    // index 10
    -0.69026849f,    // index 11
     0.076696499f,   // index 12
    -0.076696499f,   // index 13
     1.150447483f,   // index 14
    -1.150447483f    // index 15
};

// ============================================================================
// INT8 Constellation Tables (unnormalized, for direct INT8 output)
// ============================================================================

// QPSK INT8: ±1 (unnormalized)
__device__ __constant__ int8_t QPSK_I8[4] = { 1, -1,  1, -1};
__device__ __constant__ int8_t QPSK_Q8[4] = { 1,  1, -1, -1};

// 16QAM INT8: ±1, ±3 (unnormalized)
__device__ __constant__ int8_t QAM16_I8[4] = { 1, -1,  3, -3};
__device__ __constant__ int8_t QAM16_Q8[4] = { 1, -1,  3, -3};

// 64QAM INT8: ±1, ±3, ±5, ±7 (unnormalized)
// Indexed by (bit0) | (bit2 >> 1) | (bit4 >> 2)
__device__ __constant__ int8_t QAM64_I8[8] = { 3, -3,  5, -5,  1, -1,  7, -7};

// 256QAM INT8: ±1, ±3, ±5, ±7, ±9, ±11, ±13, ±15 (unnormalized)
// Indexed by (bit0) | (bit2 >> 1) | (bit4 >> 2) | (bit6 >> 3)
__device__ __constant__ int8_t QAM256_I8[16] = {
     5, -5,  11, -11,   3, -3,  13, -13,
     7, -7,   9,  -9,   1, -1,  15, -15
};

// ============================================================================
// CPU-Compatible Interval-Based Soft Demodulation Constants
// These match the srsRAN CPU implementation exactly for numerical compatibility
// ============================================================================

// Maximum (absolute) value considered for quantization (matches CPU)
#define RANGE_LIMIT_FLOAT 20.0f

// 16QAM Constants (from demodulation_mapper_qam16.cpp)
// Square root of 1/10
#define M_SQRT1_10 0.316227766f  // 1/sqrt(10)

// 64QAM Constants (from demodulation_mapper_qam64.cpp)
// Square root of 1/42
#define M_SQRT1_42 0.154303350f  // 1/sqrt(42)

// 64QAM Interval parameters
__device__ __constant__ float INTERVAL_WIDTH_64QAM_01 = 2.0f * M_SQRT1_42;
__device__ __constant__ float SLOPE_64QAM_01[8] = {
    16.0f * M_SQRT1_42, 12.0f * M_SQRT1_42, 8.0f * M_SQRT1_42, 4.0f * M_SQRT1_42,
    4.0f * M_SQRT1_42, 8.0f * M_SQRT1_42, 12.0f * M_SQRT1_42, 16.0f * M_SQRT1_42
};
__device__ __constant__ float INTERCEPT_64QAM_01[8] = {
    24.0f / 21.0f, 12.0f / 21.0f, 4.0f / 21.0f, 0.0f,
    0.0f, -4.0f / 21.0f, -12.0f / 21.0f, -24.0f / 21.0f
};

__device__ __constant__ float INTERVAL_WIDTH_64QAM_23 = 2.0f * M_SQRT1_42;
__device__ __constant__ float SLOPE_64QAM_23[8] = {
    8.0f * M_SQRT1_42, 4.0f * M_SQRT1_42, 4.0f * M_SQRT1_42, 8.0f * M_SQRT1_42,
    -8.0f * M_SQRT1_42, -4.0f * M_SQRT1_42, -4.0f * M_SQRT1_42, -8.0f * M_SQRT1_42
};
__device__ __constant__ float INTERCEPT_64QAM_23[8] = {
    20.0f / 21.0f, 8.0f / 21.0f, 8.0f / 21.0f, 12.0f / 21.0f,
    12.0f / 21.0f, 8.0f / 21.0f, 8.0f / 21.0f, 20.0f / 21.0f
};

__device__ __constant__ float INTERVAL_WIDTH_64QAM_45 = 4.0f * M_SQRT1_42;
__device__ __constant__ float SLOPE_64QAM_45[4] = {
    4.0f * M_SQRT1_42, -4.0f * M_SQRT1_42, 4.0f * M_SQRT1_42, -4.0f * M_SQRT1_42
};
__device__ __constant__ float INTERCEPT_64QAM_45[4] = {
    12.0f / 21.0f, -4.0f / 21.0f, -4.0f / 21.0f, 12.0f / 21.0f
};

// 256QAM Constants (from demodulation_mapper_qam256.cpp)
// Square root of 1/170
#define M_SQRT1_170 0.076696499f  // 1/sqrt(170)

// 256QAM Interval parameters for bits 0,1
__device__ __constant__ float INTERVAL_WIDTH_256QAM_01 = 2.0f * M_SQRT1_170;
__device__ __constant__ float SLOPE_256QAM_01[16] = {
    32.0f * M_SQRT1_170, 28.0f * M_SQRT1_170, 24.0f * M_SQRT1_170, 20.0f * M_SQRT1_170,
    16.0f * M_SQRT1_170, 12.0f * M_SQRT1_170, 8.0f * M_SQRT1_170, 4.0f * M_SQRT1_170,
    4.0f * M_SQRT1_170, 8.0f * M_SQRT1_170, 12.0f * M_SQRT1_170, 16.0f * M_SQRT1_170,
    20.0f * M_SQRT1_170, 24.0f * M_SQRT1_170, 28.0f * M_SQRT1_170, 32.0f * M_SQRT1_170
};
__device__ __constant__ float INTERCEPT_256QAM_01[16] = {
    112.0f / 85.0f, 84.0f / 85.0f, 60.0f / 85.0f, 40.0f / 85.0f,
    24.0f / 85.0f, 12.0f / 85.0f, 4.0f / 85.0f, 0.0f,
    0.0f, -4.0f / 85.0f, -12.0f / 85.0f, -24.0f / 85.0f,
    -40.0f / 85.0f, -60.0f / 85.0f, -84.0f / 85.0f, -112.0f / 85.0f
};

// 256QAM Interval parameters for bits 2,3
__device__ __constant__ float INTERVAL_WIDTH_256QAM_23 = 2.0f * M_SQRT1_170;
__device__ __constant__ float SLOPE_256QAM_23[16] = {
    16.0f * M_SQRT1_170, 12.0f * M_SQRT1_170, 8.0f * M_SQRT1_170, 4.0f * M_SQRT1_170,
    4.0f * M_SQRT1_170, 8.0f * M_SQRT1_170, 12.0f * M_SQRT1_170, 16.0f * M_SQRT1_170,
    -16.0f * M_SQRT1_170, -12.0f * M_SQRT1_170, -8.0f * M_SQRT1_170, -4.0f * M_SQRT1_170,
    -4.0f * M_SQRT1_170, -8.0f * M_SQRT1_170, -12.0f * M_SQRT1_170, -16.0f * M_SQRT1_170
};
__device__ __constant__ float INTERCEPT_256QAM_23[16] = {
    88.0f / 85.0f, 60.0f / 85.0f, 36.0f / 85.0f, 16.0f / 85.0f,
    16.0f / 85.0f, 28.0f / 85.0f, 36.0f / 85.0f, 40.0f / 85.0f,
    40.0f / 85.0f, 36.0f / 85.0f, 28.0f / 85.0f, 16.0f / 85.0f,
    16.0f / 85.0f, 36.0f / 85.0f, 60.0f / 85.0f, 88.0f / 85.0f
};

// 256QAM Interval parameters for bits 4,5
__device__ __constant__ float INTERVAL_WIDTH_256QAM_45 = 2.0f * M_SQRT1_170;
__device__ __constant__ float SLOPE_256QAM_45[16] = {
    8.0f * M_SQRT1_170, 4.0f * M_SQRT1_170, 4.0f * M_SQRT1_170, 8.0f * M_SQRT1_170,
    -8.0f * M_SQRT1_170, -4.0f * M_SQRT1_170, -4.0f * M_SQRT1_170, -8.0f * M_SQRT1_170,
    8.0f * M_SQRT1_170, 4.0f * M_SQRT1_170, 4.0f * M_SQRT1_170, 8.0f * M_SQRT1_170,
    -8.0f * M_SQRT1_170, -4.0f * M_SQRT1_170, -4.0f * M_SQRT1_170, -8.0f * M_SQRT1_170
};
__device__ __constant__ float INTERCEPT_256QAM_45[16] = {
    52.0f / 85.0f, 24.0f / 85.0f, 24.0f / 85.0f, 44.0f / 85.0f,
    -20.0f / 85.0f, -8.0f / 85.0f, -8.0f / 85.0f, -12.0f / 85.0f,
    -12.0f / 85.0f, -8.0f / 85.0f, -8.0f / 85.0f, -20.0f / 85.0f,
    44.0f / 85.0f, 24.0f / 85.0f, 24.0f / 85.0f, 52.0f / 85.0f
};

// 256QAM Interval parameters for bits 6,7
__device__ __constant__ float INTERVAL_WIDTH_256QAM_67 = 4.0f * M_SQRT1_170;
__device__ __constant__ float SLOPE_256QAM_67[8] = {
    4.0f * M_SQRT1_170, -4.0f * M_SQRT1_170, 4.0f * M_SQRT1_170, -4.0f * M_SQRT1_170,
    4.0f * M_SQRT1_170, -4.0f * M_SQRT1_170, 4.0f * M_SQRT1_170, -4.0f * M_SQRT1_170
};
__device__ __constant__ float INTERCEPT_256QAM_67[8] = {
    28.0f / 85.0f, -20.0f / 85.0f, 12.0f / 85.0f, -4.0f / 85.0f,
    -4.0f / 85.0f, 12.0f / 85.0f, -20.0f / 85.0f, 28.0f / 85.0f
};

// ============================================================================
// CPU-Compatible Interval Function Helpers
// ============================================================================

/**
 * @brief Compute interval index from value (matches CPU compute_interval_idx)
 */
__device__ __forceinline__ int compute_interval_idx(float value, float interval_width, int nof_intervals) {
    int idx = static_cast<int>(floorf(value / interval_width)) + nof_intervals / 2;
    return max(0, min(nof_intervals - 1, idx));
}

/**
 * @brief Apply interval function (matches CPU interval_function)
 * Returns (slope * value + intercept) * rcp_noise
 */
__device__ __forceinline__ float interval_function_8(
    float value, float rcp_noise, float interval_width, int nof_intervals,
    const float* __restrict__ slopes, const float* __restrict__ intercepts
) {
    int idx = compute_interval_idx(value, interval_width, nof_intervals);
    float l_value = slopes[idx] * value + intercepts[idx];
    return l_value * rcp_noise;
}

/**
 * @brief 16QAM soft demod for bits 0,1 (sign bits) - CPU compatible
 * Matches demod_16QAM_symbol_01 from demodulation_mapper_qam16.cpp
 */
__device__ __forceinline__ float soft_demod_16qam_01_cpu(float x, float rcp_noise) {
    // GAIN_FIRST = 4 * M_SQRT1_10
    const float GAIN_FIRST = 4.0f * M_SQRT1_10;  // 1.264911...
    // THRESHOLD = 2 * M_SQRT1_10
    const float THRESHOLD = 2.0f * M_SQRT1_10;   // 0.632455...
    // CONST_0_8 = 0.8
    const float CONST_0_8 = 0.8f;

    float l_value = GAIN_FIRST * x;
    if (fabsf(x) > THRESHOLD) {
        // l_value = 2 * l_value - copysign(0.8, x)
        l_value = 2.0f * l_value - copysignf(CONST_0_8, x);
    }
    return l_value * rcp_noise;
}

/**
 * @brief 16QAM soft demod for bits 2,3 (magnitude bits) - CPU compatible
 * Matches demod_16QAM_symbol_23 from demodulation_mapper_qam16.cpp
 */
__device__ __forceinline__ float soft_demod_16qam_23_cpu(float x, float rcp_noise) {
    // l_value = 0.8 - 4 * M_SQRT1_10 * |x|
    const float GAIN = 4.0f * M_SQRT1_10;
    const float CONST_0_8 = 0.8f;

    float l_value = CONST_0_8 - GAIN * fabsf(x);
    return l_value * rcp_noise;
}

/**
 * @brief 64QAM soft demod for bits 0,1 - CPU compatible interval function
 */
__device__ __forceinline__ float soft_demod_64qam_01_cpu(float value, float rcp_noise) {
    int idx = compute_interval_idx(value, 2.0f * M_SQRT1_42, 8);
    float l_value = SLOPE_64QAM_01[idx] * value + INTERCEPT_64QAM_01[idx];
    return l_value * rcp_noise;
}

/**
 * @brief 64QAM soft demod for bits 2,3 - CPU compatible interval function
 */
__device__ __forceinline__ float soft_demod_64qam_23_cpu(float value, float rcp_noise) {
    int idx = compute_interval_idx(value, 2.0f * M_SQRT1_42, 8);
    float l_value = SLOPE_64QAM_23[idx] * value + INTERCEPT_64QAM_23[idx];
    return l_value * rcp_noise;
}

/**
 * @brief 64QAM soft demod for bits 4,5 - CPU compatible interval function
 */
__device__ __forceinline__ float soft_demod_64qam_45_cpu(float value, float rcp_noise) {
    int idx = compute_interval_idx(value, 4.0f * M_SQRT1_42, 4);
    float l_value = SLOPE_64QAM_45[idx] * value + INTERCEPT_64QAM_45[idx];
    return l_value * rcp_noise;
}

/**
 * @brief 256QAM soft demod for bits 0,1 - CPU compatible interval function
 */
__device__ __forceinline__ float soft_demod_256qam_01_cpu(float value, float rcp_noise) {
    int idx = compute_interval_idx(value, 2.0f * M_SQRT1_170, 16);
    float l_value = SLOPE_256QAM_01[idx] * value + INTERCEPT_256QAM_01[idx];
    return l_value * rcp_noise;
}

/**
 * @brief 256QAM soft demod for bits 2,3 - CPU compatible interval function
 */
__device__ __forceinline__ float soft_demod_256qam_23_cpu(float value, float rcp_noise) {
    int idx = compute_interval_idx(value, 2.0f * M_SQRT1_170, 16);
    float l_value = SLOPE_256QAM_23[idx] * value + INTERCEPT_256QAM_23[idx];
    return l_value * rcp_noise;
}

/**
 * @brief 256QAM soft demod for bits 4,5 - CPU compatible interval function
 */
__device__ __forceinline__ float soft_demod_256qam_45_cpu(float value, float rcp_noise) {
    int idx = compute_interval_idx(value, 2.0f * M_SQRT1_170, 16);
    float l_value = SLOPE_256QAM_45[idx] * value + INTERCEPT_256QAM_45[idx];
    return l_value * rcp_noise;
}

/**
 * @brief 256QAM soft demod for bits 6,7 - CPU compatible interval function
 */
__device__ __forceinline__ float soft_demod_256qam_67_cpu(float value, float rcp_noise) {
    int idx = compute_interval_idx(value, 4.0f * M_SQRT1_170, 8);
    float l_value = SLOPE_256QAM_67[idx] * value + INTERCEPT_256QAM_67[idx];
    return l_value * rcp_noise;
}

// ============================================================================
// Modulator Context
// ============================================================================

struct modulator_ctx {
    curandState* d_rng_states = nullptr;
    int num_rng_states = 0;
    unsigned long long seed = 12345ULL;
};

// ============================================================================
// CUDA Kernels - Modulation
// ============================================================================

/**
 * @brief BPSK modulation: bit -> {+1, -1}
 */
__global__ void modulate_bpsk_kernel(
    const uint32_t* __restrict__ d_bits,
    cuFloatComplex* __restrict__ d_symbols,
    int num_bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bits) return;

    int word_idx = idx / 32;
    int bit_pos = idx % 32;
    uint32_t bit = (d_bits[word_idx] >> bit_pos) & 1u;

    // BPSK: 0 -> +1, 1 -> -1
    d_symbols[idx] = make_cuFloatComplex(bit ? -1.0f : 1.0f, 0.0f);
}

/**
 * @brief QPSK modulation: 2 bits -> symbol
 */
__global__ void modulate_qpsk_kernel(
    const uint32_t* __restrict__ d_bits,
    cuFloatComplex* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 2;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    // Extract 2 bits
    uint32_t bits;
    if (bit_pos <= 30) {
        bits = (d_bits[word_idx] >> bit_pos) & 0x3;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0x3;
    }

    d_symbols[idx] = make_cuFloatComplex(QPSK_I[bits], QPSK_Q[bits]);
}

/**
 * @brief 16QAM modulation: 4 bits -> symbol
 * Matches cuPHY exactly: I uses bits 0,2; Q uses bits 1,3
 * Index = bits & 0x05 (keeps bits at positions 0 and 2, no shifting)
 */
__global__ void modulate_16qam_kernel(
    const uint32_t* __restrict__ d_bits,
    cuFloatComplex* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 4;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    // Extract 4 bits
    uint32_t bits;
    if (bit_pos <= 28) {
        bits = (d_bits[word_idx] >> bit_pos) & 0xF;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0xF;
    }

    // cuPHY exact convention: index = bits & 0x05 (keeps bits 0 and 2 in place)
    // This gives indices 0, 1, 4, or 5
    int i_idx = bits & 0x5;           // bits 0 and 2 for I
    int q_idx = (bits >> 1) & 0x5;    // bits 1 and 3 for Q (shifted to positions 0 and 2)

    d_symbols[idx] = make_cuFloatComplex(QAM16_TABLE[i_idx], QAM16_TABLE[q_idx]);
}

/**
 * @brief 64QAM modulation: 6 bits -> symbol
 * Matches cuPHY: I uses bits 0,2,4; Q uses bits 1,3,5
 */
__global__ void modulate_64qam_kernel(
    const uint32_t* __restrict__ d_bits,
    cuFloatComplex* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 6;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    // Extract 6 bits
    uint32_t bits;
    if (bit_pos <= 26) {
        bits = (d_bits[word_idx] >> bit_pos) & 0x3F;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0x3F;
    }

    // cuPHY convention: I uses bits 0,2,4; Q uses bits 1,3,5
    // map_index_6bits: (index & 0x1) | ((index & 0x4) >> 1) | ((index & 0x10) >> 2)
    int i_idx = (bits & 0x1) | ((bits & 0x4) >> 1) | ((bits & 0x10) >> 2);
    int q_idx = ((bits >> 1) & 0x1) | (((bits >> 1) & 0x4) >> 1) | (((bits >> 1) & 0x10) >> 2);

    d_symbols[idx] = make_cuFloatComplex(QAM64_TABLE[i_idx], QAM64_TABLE[q_idx]);
}

/**
 * @brief 256QAM modulation: 8 bits -> symbol
 * Matches cuPHY: I uses bits 0,2,4,6; Q uses bits 1,3,5,7
 */
__global__ void modulate_256qam_kernel(
    const uint32_t* __restrict__ d_bits,
    cuFloatComplex* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 8;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    // Extract 8 bits
    uint32_t bits;
    if (bit_pos <= 24) {
        bits = (d_bits[word_idx] >> bit_pos) & 0xFF;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0xFF;
    }

    // cuPHY convention: I uses bits 0,2,4,6; Q uses bits 1,3,5,7
    // map_index_8bits: (index & 0x1) | ((index & 0x4) >> 1) | ((index & 0x10) >> 2) | ((index & 0x40) >> 3)
    int i_idx = (bits & 0x1) | ((bits & 0x4) >> 1) | ((bits & 0x10) >> 2) | ((bits & 0x40) >> 3);
    int q_idx = ((bits >> 1) & 0x1) | (((bits >> 1) & 0x4) >> 1) | (((bits >> 1) & 0x10) >> 2) | (((bits >> 1) & 0x40) >> 3);

    d_symbols[idx] = make_cuFloatComplex(QAM256_TABLE[i_idx], QAM256_TABLE[q_idx]);
}

// ============================================================================
// CUDA Kernels - INT8 Modulation (direct to ci8_t format)
// ============================================================================

/**
 * @brief QPSK modulation to INT8: 2 bits -> ci8_t symbol
 */
__global__ void modulate_qpsk_int8_kernel(
    const uint32_t* __restrict__ d_bits,
    int8_t* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 2;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    uint32_t bits = (d_bits[word_idx] >> bit_pos) & 0x3;
    d_symbols[idx * 2]     = QPSK_I8[bits];
    d_symbols[idx * 2 + 1] = QPSK_Q8[bits];
}

/**
 * @brief 16QAM modulation to INT8: 4 bits -> ci8_t symbol
 */
__global__ void modulate_16qam_int8_kernel(
    const uint32_t* __restrict__ d_bits,
    int8_t* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 4;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    uint32_t bits;
    if (bit_pos <= 28) {
        bits = (d_bits[word_idx] >> bit_pos) & 0xF;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0xF;
    }

    // I uses bits 0,2; Q uses bits 1,3
    int i_idx = (bits & 0x1) | ((bits & 0x4) >> 1);
    int q_idx = ((bits >> 1) & 0x1) | (((bits >> 1) & 0x4) >> 1);

    d_symbols[idx * 2]     = QAM16_I8[i_idx];
    d_symbols[idx * 2 + 1] = QAM16_Q8[q_idx];
}

/**
 * @brief 64QAM modulation to INT8: 6 bits -> ci8_t symbol
 */
__global__ void modulate_64qam_int8_kernel(
    const uint32_t* __restrict__ d_bits,
    int8_t* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 6;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    uint32_t bits;
    if (bit_pos <= 26) {
        bits = (d_bits[word_idx] >> bit_pos) & 0x3F;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0x3F;
    }

    // I uses bits 0,2,4; Q uses bits 1,3,5
    int i_idx = (bits & 0x1) | ((bits & 0x4) >> 1) | ((bits & 0x10) >> 2);
    int q_idx = ((bits >> 1) & 0x1) | (((bits >> 1) & 0x4) >> 1) | (((bits >> 1) & 0x10) >> 2);

    d_symbols[idx * 2]     = QAM64_I8[i_idx];
    d_symbols[idx * 2 + 1] = QAM64_I8[q_idx];
}

/**
 * @brief 256QAM modulation to INT8: 8 bits -> ci8_t symbol
 */
__global__ void modulate_256qam_int8_kernel(
    const uint32_t* __restrict__ d_bits,
    int8_t* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 8;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    uint32_t bits;
    if (bit_pos <= 24) {
        bits = (d_bits[word_idx] >> bit_pos) & 0xFF;
    } else {
        bits = ((d_bits[word_idx] >> bit_pos) | (d_bits[word_idx + 1] << (32 - bit_pos))) & 0xFF;
    }

    // I uses bits 0,2,4,6; Q uses bits 1,3,5,7
    int i_idx = (bits & 0x1) | ((bits & 0x4) >> 1) | ((bits & 0x10) >> 2) | ((bits & 0x40) >> 3);
    int q_idx = ((bits >> 1) & 0x1) | (((bits >> 1) & 0x4) >> 1) | (((bits >> 1) & 0x10) >> 2) | (((bits >> 1) & 0x40) >> 3);

    d_symbols[idx * 2]     = QAM256_I8[i_idx];
    d_symbols[idx * 2 + 1] = QAM256_I8[q_idx];
}

// ============================================================================
// CUDA Kernels - AWGN Noise
// ============================================================================

__global__ void init_rng_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void add_awgn_kernel(
    cuFloatComplex* __restrict__ d_symbols,
    curandState* __restrict__ d_states,
    int num_symbols,
    float noise_std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    curandState local_state = d_states[idx % 65536];

    // Generate complex Gaussian noise
    float ni = curand_normal(&local_state) * noise_std;
    float nq = curand_normal(&local_state) * noise_std;

    d_states[idx % 65536] = local_state;

    cuFloatComplex sym = d_symbols[idx];
    d_symbols[idx] = make_cuFloatComplex(cuCrealf(sym) + ni, cuCimagf(sym) + nq);
}

// ============================================================================
// CUDA Kernels - Soft Demodulation
// ============================================================================

/**
 * @brief BPSK soft demodulation
 * LLR = 2 * y / sigma^2 where y is received symbol (real part)
 */
__global__ void soft_demod_bpsk_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    float* __restrict__ d_llrs,
    int num_symbols,
    float noise_var
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    float y = cuCrealf(d_symbols[idx]);
    // LLR = 2*y/sigma^2 (positive LLR means bit 0 more likely)
    d_llrs[idx] = 2.0f * y / noise_var;
}

/**
 * @brief QPSK soft demodulation (exact LLRs)
 */
__global__ void soft_demod_qpsk_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    float* __restrict__ d_llrs,
    int num_symbols,
    float noise_var
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);

    // For QPSK with Gray coding:
    // bit 0: determined by I component, bit 1: determined by Q component
    // LLR_0 = 2*sqrt(2)*yi/sigma^2
    // LLR_1 = 2*sqrt(2)*yq/sigma^2
    float scale = 2.0f * 1.4142136f / noise_var;
    d_llrs[idx * 2] = scale * yi;
    d_llrs[idx * 2 + 1] = scale * yq;
}

/**
 * @brief 16QAM soft demodulation (CPU-compatible interval approximation)
 * Matches cuPHY: I uses bits 0,2; Q uses bits 1,3 (interleaved)
 *
 * Uses piecewise linear approximation matching srsRAN CPU implementation
 */
__global__ void soft_demod_16qam_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    float* __restrict__ d_llrs,
    int num_symbols,
    float noise_var
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);

    float rcp_noise = 1.0f / noise_var;

    // Use CPU-compatible interval-based soft demodulation
    // Bit 0 (I sign), Bit 1 (Q sign)
    d_llrs[idx * 4 + 0] = soft_demod_16qam_01_cpu(yi, rcp_noise);
    d_llrs[idx * 4 + 1] = soft_demod_16qam_01_cpu(yq, rcp_noise);
    // Bit 2 (I magnitude), Bit 3 (Q magnitude)
    d_llrs[idx * 4 + 2] = soft_demod_16qam_23_cpu(yi, rcp_noise);
    d_llrs[idx * 4 + 3] = soft_demod_16qam_23_cpu(yq, rcp_noise);
}

/**
 * @brief 64QAM soft demodulation (CPU-compatible interval approximation)
 * Matches cuPHY: I uses bits 0,2,4; Q uses bits 1,3,5 (interleaved)
 *
 * Uses interval-based approximation matching srsRAN CPU implementation
 */
__global__ void soft_demod_64qam_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    float* __restrict__ d_llrs,
    int num_symbols,
    float noise_var
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);

    float rcp_noise = 1.0f / noise_var;

    // Use CPU-compatible interval-based soft demodulation
    d_llrs[idx * 6 + 0] = soft_demod_64qam_01_cpu(yi, rcp_noise);
    d_llrs[idx * 6 + 1] = soft_demod_64qam_01_cpu(yq, rcp_noise);
    d_llrs[idx * 6 + 2] = soft_demod_64qam_23_cpu(yi, rcp_noise);
    d_llrs[idx * 6 + 3] = soft_demod_64qam_23_cpu(yq, rcp_noise);
    d_llrs[idx * 6 + 4] = soft_demod_64qam_45_cpu(yi, rcp_noise);
    d_llrs[idx * 6 + 5] = soft_demod_64qam_45_cpu(yq, rcp_noise);
}

/**
 * @brief 256QAM soft demodulation (CPU-compatible interval approximation)
 * Matches cuPHY: I uses bits 0,2,4,6; Q uses bits 1,3,5,7 (interleaved)
 *
 * Uses interval-based approximation matching srsRAN CPU implementation
 */
__global__ void soft_demod_256qam_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    float* __restrict__ d_llrs,
    int num_symbols,
    float noise_var
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);

    float rcp_noise = 1.0f / noise_var;

    // Use CPU-compatible interval-based soft demodulation
    d_llrs[idx * 8 + 0] = soft_demod_256qam_01_cpu(yi, rcp_noise);
    d_llrs[idx * 8 + 1] = soft_demod_256qam_01_cpu(yq, rcp_noise);
    d_llrs[idx * 8 + 2] = soft_demod_256qam_23_cpu(yi, rcp_noise);
    d_llrs[idx * 8 + 3] = soft_demod_256qam_23_cpu(yq, rcp_noise);
    d_llrs[idx * 8 + 4] = soft_demod_256qam_45_cpu(yi, rcp_noise);
    d_llrs[idx * 8 + 5] = soft_demod_256qam_45_cpu(yq, rcp_noise);
    d_llrs[idx * 8 + 6] = soft_demod_256qam_67_cpu(yi, rcp_noise);
    d_llrs[idx * 8 + 7] = soft_demod_256qam_67_cpu(yq, rcp_noise);
}

// ============================================================================
// CUDA Kernels - Soft Demodulation (Per-Symbol Noise Variance)
// ============================================================================

/**
 * @brief BPSK soft demodulation with per-symbol noise variance
 */
__global__ void soft_demod_bpsk_per_symbol_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    float* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    float y = cuCrealf(d_symbols[idx]);
    float noise_var = d_noise_vars[idx];
    // Handle invalid noise variance (zero, negative, or NaN)
    if (!(noise_var > 0.0f)) {
        d_llrs[idx] = 0.0f;
        return;
    }
    d_llrs[idx] = 2.0f * y / noise_var;
}

/**
 * @brief QPSK soft demodulation with per-symbol noise variance
 */
__global__ void soft_demod_qpsk_per_symbol_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    float* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    // Handle invalid noise variance
    if (!(noise_var > 0.0f)) {
        d_llrs[idx * 2] = 0.0f;
        d_llrs[idx * 2 + 1] = 0.0f;
        return;
    }

    float scale = 2.0f * 1.4142136f / noise_var;
    d_llrs[idx * 2] = scale * yi;
    d_llrs[idx * 2 + 1] = scale * yq;
}

/**
 * @brief 16QAM soft demodulation with per-symbol noise variance (CPU-compatible)
 */
__global__ void soft_demod_16qam_per_symbol_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    float* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    // Handle invalid noise variance
    if (!(noise_var > 0.0f)) {
        d_llrs[idx * 4 + 0] = 0.0f;
        d_llrs[idx * 4 + 1] = 0.0f;
        d_llrs[idx * 4 + 2] = 0.0f;
        d_llrs[idx * 4 + 3] = 0.0f;
        return;
    }

    float rcp_noise = 1.0f / noise_var;

    // CPU-compatible interval-based soft demodulation
    d_llrs[idx * 4 + 0] = soft_demod_16qam_01_cpu(yi, rcp_noise);
    d_llrs[idx * 4 + 1] = soft_demod_16qam_01_cpu(yq, rcp_noise);
    d_llrs[idx * 4 + 2] = soft_demod_16qam_23_cpu(yi, rcp_noise);
    d_llrs[idx * 4 + 3] = soft_demod_16qam_23_cpu(yq, rcp_noise);
}

/**
 * @brief 64QAM soft demodulation with per-symbol noise variance (CPU-compatible)
 */
__global__ void soft_demod_64qam_per_symbol_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    float* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    // Handle invalid noise variance
    if (!(noise_var > 0.0f)) {
        for (int i = 0; i < 6; i++) d_llrs[idx * 6 + i] = 0.0f;
        return;
    }

    float rcp_noise = 1.0f / noise_var;

    // CPU-compatible interval-based soft demodulation
    d_llrs[idx * 6 + 0] = soft_demod_64qam_01_cpu(yi, rcp_noise);
    d_llrs[idx * 6 + 1] = soft_demod_64qam_01_cpu(yq, rcp_noise);
    d_llrs[idx * 6 + 2] = soft_demod_64qam_23_cpu(yi, rcp_noise);
    d_llrs[idx * 6 + 3] = soft_demod_64qam_23_cpu(yq, rcp_noise);
    d_llrs[idx * 6 + 4] = soft_demod_64qam_45_cpu(yi, rcp_noise);
    d_llrs[idx * 6 + 5] = soft_demod_64qam_45_cpu(yq, rcp_noise);
}

/**
 * @brief 256QAM soft demodulation with per-symbol noise variance (CPU-compatible)
 */
__global__ void soft_demod_256qam_per_symbol_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    float* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    // Handle invalid noise variance
    if (!(noise_var > 0.0f)) {
        for (int i = 0; i < 8; i++) d_llrs[idx * 8 + i] = 0.0f;
        return;
    }

    float rcp_noise = 1.0f / noise_var;

    // CPU-compatible interval-based soft demodulation
    d_llrs[idx * 8 + 0] = soft_demod_256qam_01_cpu(yi, rcp_noise);
    d_llrs[idx * 8 + 1] = soft_demod_256qam_01_cpu(yq, rcp_noise);
    d_llrs[idx * 8 + 2] = soft_demod_256qam_23_cpu(yi, rcp_noise);
    d_llrs[idx * 8 + 3] = soft_demod_256qam_23_cpu(yq, rcp_noise);
    d_llrs[idx * 8 + 4] = soft_demod_256qam_45_cpu(yi, rcp_noise);
    d_llrs[idx * 8 + 5] = soft_demod_256qam_45_cpu(yq, rcp_noise);
    d_llrs[idx * 8 + 6] = soft_demod_256qam_67_cpu(yi, rcp_noise);
    d_llrs[idx * 8 + 7] = soft_demod_256qam_67_cpu(yq, rcp_noise);
}

// ============================================================================
// CUDA Kernels - Half-precision (fp16) Soft Demodulation
// ============================================================================

/**
 * @brief QPSK soft demodulation outputting fp16 LLRs
 */
__global__ void soft_demod_qpsk_half_per_symbol_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    __half* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    if (!(noise_var > 0.0f)) {
        d_llrs[idx * 2] = __float2half(0.0f);
        d_llrs[idx * 2 + 1] = __float2half(0.0f);
        return;
    }

    float scale = 2.0f * 1.4142136f / noise_var;
    d_llrs[idx * 2] = __float2half(scale * yi);
    d_llrs[idx * 2 + 1] = __float2half(scale * yq);
}

/**
 * @brief 16QAM soft demodulation outputting fp16 LLRs (CPU-compatible)
 */
__global__ void soft_demod_16qam_half_per_symbol_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    __half* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    if (!(noise_var > 0.0f)) {
        for (int i = 0; i < 4; i++) d_llrs[idx * 4 + i] = __float2half(0.0f);
        return;
    }

    float rcp_noise = 1.0f / noise_var;

    // CPU-compatible interval-based soft demodulation
    d_llrs[idx * 4 + 0] = __float2half(soft_demod_16qam_01_cpu(yi, rcp_noise));
    d_llrs[idx * 4 + 1] = __float2half(soft_demod_16qam_01_cpu(yq, rcp_noise));
    d_llrs[idx * 4 + 2] = __float2half(soft_demod_16qam_23_cpu(yi, rcp_noise));
    d_llrs[idx * 4 + 3] = __float2half(soft_demod_16qam_23_cpu(yq, rcp_noise));
}

/**
 * @brief 64QAM soft demodulation outputting fp16 LLRs (CPU-compatible)
 */
__global__ void soft_demod_64qam_half_per_symbol_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    __half* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    if (!(noise_var > 0.0f)) {
        for (int i = 0; i < 6; i++) d_llrs[idx * 6 + i] = __float2half(0.0f);
        return;
    }

    float rcp_noise = 1.0f / noise_var;

    // CPU-compatible interval-based soft demodulation
    d_llrs[idx * 6 + 0] = __float2half(soft_demod_64qam_01_cpu(yi, rcp_noise));
    d_llrs[idx * 6 + 1] = __float2half(soft_demod_64qam_01_cpu(yq, rcp_noise));
    d_llrs[idx * 6 + 2] = __float2half(soft_demod_64qam_23_cpu(yi, rcp_noise));
    d_llrs[idx * 6 + 3] = __float2half(soft_demod_64qam_23_cpu(yq, rcp_noise));
    d_llrs[idx * 6 + 4] = __float2half(soft_demod_64qam_45_cpu(yi, rcp_noise));
    d_llrs[idx * 6 + 5] = __float2half(soft_demod_64qam_45_cpu(yq, rcp_noise));
}

/**
 * @brief 256QAM soft demodulation outputting fp16 LLRs (CPU-compatible)
 */
__global__ void soft_demod_256qam_half_per_symbol_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    __half* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    if (!(noise_var > 0.0f)) {
        for (int i = 0; i < 8; i++) d_llrs[idx * 8 + i] = __float2half(0.0f);
        return;
    }

    float rcp_noise = 1.0f / noise_var;

    // CPU-compatible interval-based soft demodulation
    d_llrs[idx * 8 + 0] = __float2half(soft_demod_256qam_01_cpu(yi, rcp_noise));
    d_llrs[idx * 8 + 1] = __float2half(soft_demod_256qam_01_cpu(yq, rcp_noise));
    d_llrs[idx * 8 + 2] = __float2half(soft_demod_256qam_23_cpu(yi, rcp_noise));
    d_llrs[idx * 8 + 3] = __float2half(soft_demod_256qam_23_cpu(yq, rcp_noise));
    d_llrs[idx * 8 + 4] = __float2half(soft_demod_256qam_45_cpu(yi, rcp_noise));
    d_llrs[idx * 8 + 5] = __float2half(soft_demod_256qam_45_cpu(yq, rcp_noise));
    d_llrs[idx * 8 + 6] = __float2half(soft_demod_256qam_67_cpu(yi, rcp_noise));
    d_llrs[idx * 8 + 7] = __float2half(soft_demod_256qam_67_cpu(yq, rcp_noise));
}

// ============================================================================
// Fused Soft Demod + Descramble + FP16 Output Kernels
// Full FP16 pipeline: demod → descramble → FP16 for rate dematch and decode
// ============================================================================

/**
 * @brief Helper to descramble float LLR and convert to half
 */
__device__ __forceinline__ __half descramble_to_half(float llr, int scramble_bit) {
    // Apply descrambling: negate if scramble_bit == 1
    if (scramble_bit) llr = -llr;
    return __float2half(llr);
}

/**
 * @brief Fused QPSK soft demod + descramble + FP16 output
 */
__global__ void soft_demod_descramble_qpsk_half_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    const uint32_t* __restrict__ d_scramble_seq,
    __half* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    int bit_base = idx * 2;
    int word0 = bit_base / 32;
    int bit0 = 31 - (bit_base % 32);  // MSB-first: bit 0 at position 31
    int word1 = (bit_base + 1) / 32;
    int bit1 = 31 - ((bit_base + 1) % 32);  // MSB-first
    int scr0 = (d_scramble_seq[word0] >> bit0) & 1;
    int scr1 = (d_scramble_seq[word1] >> bit1) & 1;

    if (!(noise_var > 0.0f)) {
        d_llrs[bit_base] = __float2half(0.0f);
        d_llrs[bit_base + 1] = __float2half(0.0f);
        return;
    }

    float scale = 1.0f / sqrtf(2.0f);
    float two_over_var = 2.0f / noise_var;

    float llr0 = two_over_var * yi * 2.0f * scale;
    float llr1 = two_over_var * yq * 2.0f * scale;

    d_llrs[bit_base] = descramble_to_half(llr0, scr0);
    d_llrs[bit_base + 1] = descramble_to_half(llr1, scr1);
}

/**
 * @brief Fused 16QAM soft demod + descramble + FP16 output (CPU-compatible)
 */
__global__ void soft_demod_descramble_16qam_half_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    const uint32_t* __restrict__ d_scramble_seq,
    __half* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    int bit_base = idx * 4;

    if (!(noise_var > 0.0f)) {
        for (int i = 0; i < 4; i++) d_llrs[bit_base + i] = __float2half(0.0f);
        return;
    }

    float rcp_noise = 1.0f / noise_var;

    // CPU-compatible interval-based soft demodulation
    float llrs[4];
    llrs[0] = soft_demod_16qam_01_cpu(yi, rcp_noise);
    llrs[1] = soft_demod_16qam_01_cpu(yq, rcp_noise);
    llrs[2] = soft_demod_16qam_23_cpu(yi, rcp_noise);
    llrs[3] = soft_demod_16qam_23_cpu(yq, rcp_noise);

    for (int i = 0; i < 4; i++) {
        int bit_idx = bit_base + i;
        int word = bit_idx / 32;
        int bit = 31 - (bit_idx % 32);  // MSB-first
        int scr = (d_scramble_seq[word] >> bit) & 1;
        d_llrs[bit_idx] = descramble_to_half(llrs[i], scr);
    }
}

/**
 * @brief Fused 64QAM soft demod + descramble + FP16 output (CPU-compatible)
 */
__global__ void soft_demod_descramble_64qam_half_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    const uint32_t* __restrict__ d_scramble_seq,
    __half* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    int bit_base = idx * 6;

    if (!(noise_var > 0.0f)) {
        for (int i = 0; i < 6; i++) d_llrs[bit_base + i] = __float2half(0.0f);
        return;
    }

    float rcp_noise = 1.0f / noise_var;

    // CPU-compatible interval-based soft demodulation
    float llrs[6];
    llrs[0] = soft_demod_64qam_01_cpu(yi, rcp_noise);
    llrs[1] = soft_demod_64qam_01_cpu(yq, rcp_noise);
    llrs[2] = soft_demod_64qam_23_cpu(yi, rcp_noise);
    llrs[3] = soft_demod_64qam_23_cpu(yq, rcp_noise);
    llrs[4] = soft_demod_64qam_45_cpu(yi, rcp_noise);
    llrs[5] = soft_demod_64qam_45_cpu(yq, rcp_noise);

    for (int i = 0; i < 6; i++) {
        int bit_idx = bit_base + i;
        int word = bit_idx / 32;
        int bit = 31 - (bit_idx % 32);  // MSB-first
        int scr = (d_scramble_seq[word] >> bit) & 1;
        d_llrs[bit_idx] = descramble_to_half(llrs[i], scr);
    }
}

/**
 * @brief Fused 256QAM soft demod + descramble + FP16 output (CPU-compatible)
 */
__global__ void soft_demod_descramble_256qam_half_kernel(
    const cuFloatComplex* __restrict__ d_symbols,
    const float* __restrict__ d_noise_vars,
    const uint32_t* __restrict__ d_scramble_seq,
    __half* __restrict__ d_llrs,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    cuFloatComplex y = d_symbols[idx];
    float yi = cuCrealf(y);
    float yq = cuCimagf(y);
    float noise_var = d_noise_vars[idx];

    int bit_base = idx * 8;

    if (!(noise_var > 0.0f)) {
        for (int i = 0; i < 8; i++) d_llrs[bit_base + i] = __float2half(0.0f);
        return;
    }

    float rcp_noise = 1.0f / noise_var;

    // CPU-compatible interval-based soft demodulation
    float llrs[8];
    llrs[0] = soft_demod_256qam_01_cpu(yi, rcp_noise);
    llrs[1] = soft_demod_256qam_01_cpu(yq, rcp_noise);
    llrs[2] = soft_demod_256qam_23_cpu(yi, rcp_noise);
    llrs[3] = soft_demod_256qam_23_cpu(yq, rcp_noise);
    llrs[4] = soft_demod_256qam_45_cpu(yi, rcp_noise);
    llrs[5] = soft_demod_256qam_45_cpu(yq, rcp_noise);
    llrs[6] = soft_demod_256qam_67_cpu(yi, rcp_noise);
    llrs[7] = soft_demod_256qam_67_cpu(yq, rcp_noise);

    for (int i = 0; i < 8; i++) {
        int bit_idx = bit_base + i;
        int word = bit_idx / 32;
        int bit = 31 - (bit_idx % 32);  // MSB-first
        int scr = (d_scramble_seq[word] >> bit) & 1;
        d_llrs[bit_idx] = descramble_to_half(llrs[i], scr);
    }
}

// ============================================================================
// API Implementation
// ============================================================================

extern "C" {

int modulator_create(modulator_handle_t* handle) {
    if (!handle) return -1;

    modulator_ctx* ctx = new (std::nothrow) modulator_ctx;
    if (!ctx) return -1;

    *handle = ctx;
    return 0;
}

void modulator_destroy(modulator_handle_t handle) {
    if (handle) {
        if (handle->d_rng_states) cudaFree(handle->d_rng_states);
        delete handle;
    }
}

int modulator_modulate(modulator_handle_t handle,
                       const uint32_t* d_bits,
                       cuFloatComplex* d_symbols,
                       int num_bits,
                       int mod_order,
                       cudaStream_t stream) {
    if (!handle || !d_bits || !d_symbols || num_bits <= 0) return -1;

    int num_symbols = (num_bits + mod_order - 1) / mod_order;
    int block_size = 256;
    int num_blocks = (num_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 1:  // BPSK
            modulate_bpsk_kernel<<<num_blocks, block_size, 0, stream>>>(d_bits, d_symbols, num_bits);
            break;
        case 2:  // QPSK
            modulate_qpsk_kernel<<<num_blocks, block_size, 0, stream>>>(d_bits, d_symbols, num_symbols);
            break;
        case 4:  // 16QAM
            modulate_16qam_kernel<<<num_blocks, block_size, 0, stream>>>(d_bits, d_symbols, num_symbols);
            break;
        case 6:  // 64QAM
            modulate_64qam_kernel<<<num_blocks, block_size, 0, stream>>>(d_bits, d_symbols, num_symbols);
            break;
        case 8:  // 256QAM
            modulate_256qam_kernel<<<num_blocks, block_size, 0, stream>>>(d_bits, d_symbols, num_symbols);
            break;
        default:
            return -1;
    }

    return 0;
}

int modulator_modulate_int8(modulator_handle_t handle,
                            const uint32_t* d_bits,
                            int8_t* d_symbols_int8,
                            int num_bits,
                            int mod_order,
                            cudaStream_t stream) {
    if (!handle || !d_bits || !d_symbols_int8 || num_bits <= 0) return -1;

    int num_symbols = (num_bits + mod_order - 1) / mod_order;
    int block_size = 256;
    int num_blocks = (num_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 2:  // QPSK
            modulate_qpsk_int8_kernel<<<num_blocks, block_size, 0, stream>>>(d_bits, d_symbols_int8, num_symbols);
            break;
        case 4:  // 16QAM
            modulate_16qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(d_bits, d_symbols_int8, num_symbols);
            break;
        case 6:  // 64QAM
            modulate_64qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(d_bits, d_symbols_int8, num_symbols);
            break;
        case 8:  // 256QAM
            modulate_256qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(d_bits, d_symbols_int8, num_symbols);
            break;
        default:
            // BPSK not supported for INT8
            return -1;
    }

    return 0;
}

int modulator_add_noise(modulator_handle_t handle,
                        cuFloatComplex* d_symbols,
                        int num_symbols,
                        float noise_std,
                        cudaStream_t stream) {
    if (!handle || !d_symbols || num_symbols <= 0) return -1;

    /* Lazy-init RNG states on first use */
    if (!handle->d_rng_states) {
        handle->num_rng_states = 65536;
        cudaError_t err = cudaMalloc(&handle->d_rng_states,
                                     handle->num_rng_states * sizeof(curandState));
        if (err != cudaSuccess) return -1;
        int bs = 256;
        int nb = (handle->num_rng_states + bs - 1) / bs;
        init_rng_kernel<<<nb, bs, 0, stream>>>(handle->d_rng_states, handle->seed,
                                                handle->num_rng_states);
    }

    int block_size = 256;
    int num_blocks = (num_symbols + block_size - 1) / block_size;

    add_awgn_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_symbols, handle->d_rng_states, num_symbols, noise_std);

    return 0;
}

int modulator_soft_demod(modulator_handle_t handle,
                         const cuFloatComplex* d_symbols,
                         float* d_llrs,
                         int num_symbols,
                         int mod_order,
                         float noise_var,
                         cudaStream_t stream) {
    if (!handle || !d_symbols || !d_llrs || num_symbols <= 0) return -1;

    int block_size = 256;
    int num_blocks = (num_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 1:  // BPSK
            soft_demod_bpsk_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_llrs, num_symbols, noise_var);
            break;
        case 2:  // QPSK
            soft_demod_qpsk_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_llrs, num_symbols, noise_var);
            break;
        case 4:  // 16QAM
            soft_demod_16qam_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_llrs, num_symbols, noise_var);
            break;
        case 6:  // 64QAM
            soft_demod_64qam_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_llrs, num_symbols, noise_var);
            break;
        case 8:  // 256QAM
            soft_demod_256qam_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_llrs, num_symbols, noise_var);
            break;
        default:
            return -1;
    }

    return 0;
}

int modulator_get_num_symbols(int num_bits, int mod_order) {
    return (num_bits + mod_order - 1) / mod_order;
}

int modulator_soft_demod_descramble_half(modulator_handle_t handle,
                                          const cuFloatComplex* d_symbols,
                                          const float* d_noise_vars,
                                          const uint32_t* d_scramble_seq,
                                          void* d_llrs_half,
                                          int num_symbols,
                                          int mod_order,
                                          cudaStream_t stream) {
    if (!handle || !d_symbols || !d_noise_vars || !d_scramble_seq || !d_llrs_half || num_symbols <= 0) return -1;

    __half* d_llrs = static_cast<__half*>(d_llrs_half);
    int block_size = 256;
    int num_blocks = (num_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 2:
            soft_demod_descramble_qpsk_half_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_scramble_seq, d_llrs, num_symbols);
            break;
        case 4:
            soft_demod_descramble_16qam_half_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_scramble_seq, d_llrs, num_symbols);
            break;
        case 6:
            soft_demod_descramble_64qam_half_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_scramble_seq, d_llrs, num_symbols);
            break;
        case 8:
            soft_demod_descramble_256qam_half_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_scramble_seq, d_llrs, num_symbols);
            break;
        default:
            return -1;
    }

    return 0;
}

float snr_to_noise_std(float snr_db, float code_rate, int mod_order) {
    // SNR = Eb/N0
    // Es/N0 = Eb/N0 * bits_per_symbol * code_rate
    // For unit average symbol energy: noise_var = 1 / (Es/N0)
    float snr_linear = powf(10.0f, snr_db / 10.0f);
    float es_n0 = snr_linear * mod_order * code_rate;
    float noise_var = 1.0f / es_n0;
    return sqrtf(noise_var / 2.0f);  // Per-dimension std dev
}

int modulator_soft_demod_per_symbol(modulator_handle_t handle,
                                    const cuFloatComplex* d_symbols,
                                    const float* d_noise_vars,
                                    float* d_llrs,
                                    int num_symbols,
                                    int mod_order,
                                    cudaStream_t stream) {
    if (!handle || !d_symbols || !d_noise_vars || !d_llrs || num_symbols <= 0) return -1;

    int block_size = 256;
    int num_blocks = (num_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 1:  // BPSK
            soft_demod_bpsk_per_symbol_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_llrs, num_symbols);
            break;
        case 2:  // QPSK
            soft_demod_qpsk_per_symbol_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_llrs, num_symbols);
            break;
        case 4:  // 16QAM
            soft_demod_16qam_per_symbol_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_llrs, num_symbols);
            break;
        case 6:  // 64QAM
            soft_demod_64qam_per_symbol_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_llrs, num_symbols);
            break;
        case 8:  // 256QAM
            soft_demod_256qam_per_symbol_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_llrs, num_symbols);
            break;
        default:
            return -1;
    }

    return 0;
}

int modulator_soft_demod_per_symbol_half(modulator_handle_t handle,
                                          const cuFloatComplex* d_symbols,
                                          const float* d_noise_vars,
                                          void* d_llrs_half,
                                          int num_symbols,
                                          int mod_order,
                                          cudaStream_t stream) {
    if (!handle || !d_symbols || !d_noise_vars || !d_llrs_half || num_symbols <= 0) return -1;

    __half* d_llrs = static_cast<__half*>(d_llrs_half);

    int block_size = 256;
    int num_blocks = (num_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 2:  // QPSK
            soft_demod_qpsk_half_per_symbol_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_llrs, num_symbols);
            break;
        case 4:  // 16QAM
            soft_demod_16qam_half_per_symbol_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_llrs, num_symbols);
            break;
        case 6:  // 64QAM
            soft_demod_64qam_half_per_symbol_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_llrs, num_symbols);
            break;
        case 8:  // 256QAM
            soft_demod_256qam_half_per_symbol_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_symbols, d_noise_vars, d_llrs, num_symbols);
            break;
        default:
            // BPSK not implemented for fp16 yet, fall back to fp32 + convert
            return -1;
    }

    return 0;
}

// ============================================================================
// Fused Scramble + Modulate Kernels
// ============================================================================
// These kernels combine scrambling and modulation in a single pass,
// eliminating the intermediate scrambled bits buffer and memory access.

/**
 * @brief Fused scramble + QPSK modulation: XOR bits with scramble seq, map to symbols
 */
__global__ void scramble_modulate_qpsk_kernel(
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

    // Read and XOR with scrambling sequence in one shot
    uint32_t word = d_bits[word_idx] ^ d_scramble_seq[word_idx];
    uint32_t bits;
    if (bit_pos <= 30) {
        bits = (word >> bit_pos) & 0x3;
    } else {
        uint32_t word2 = d_bits[word_idx + 1] ^ d_scramble_seq[word_idx + 1];
        bits = ((word >> bit_pos) | (word2 << (32 - bit_pos))) & 0x3;
    }

    d_symbols[idx] = make_cuFloatComplex(QPSK_I[bits], QPSK_Q[bits]);
}

/**
 * @brief Fused scramble + 16QAM modulation
 */
__global__ void scramble_modulate_16qam_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    cuFloatComplex* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 4;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    uint32_t word = d_bits[word_idx] ^ d_scramble_seq[word_idx];
    uint32_t bits;
    if (bit_pos <= 28) {
        bits = (word >> bit_pos) & 0xF;
    } else {
        uint32_t word2 = d_bits[word_idx + 1] ^ d_scramble_seq[word_idx + 1];
        bits = ((word >> bit_pos) | (word2 << (32 - bit_pos))) & 0xF;
    }

    int i_idx = bits & 0x5;
    int q_idx = (bits >> 1) & 0x5;

    d_symbols[idx] = make_cuFloatComplex(QAM16_TABLE[i_idx], QAM16_TABLE[q_idx]);
}

/**
 * @brief Fused scramble + 64QAM modulation
 */
__global__ void scramble_modulate_64qam_kernel(
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

    uint32_t word = d_bits[word_idx] ^ d_scramble_seq[word_idx];
    uint32_t bits;
    if (bit_pos <= 26) {
        bits = (word >> bit_pos) & 0x3F;
    } else {
        uint32_t word2 = d_bits[word_idx + 1] ^ d_scramble_seq[word_idx + 1];
        bits = ((word >> bit_pos) | (word2 << (32 - bit_pos))) & 0x3F;
    }

    int i_idx = (bits & 0x1) | ((bits & 0x4) >> 1) | ((bits & 0x10) >> 2);
    int q_idx = ((bits >> 1) & 0x1) | (((bits >> 1) & 0x4) >> 1) | (((bits >> 1) & 0x10) >> 2);

    d_symbols[idx] = make_cuFloatComplex(QAM64_TABLE[i_idx], QAM64_TABLE[q_idx]);
}

/**
 * @brief Fused scramble + 256QAM modulation
 */
__global__ void scramble_modulate_256qam_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    cuFloatComplex* __restrict__ d_symbols,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int bit_idx = idx * 8;
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;

    uint32_t word = d_bits[word_idx] ^ d_scramble_seq[word_idx];
    uint32_t bits;
    if (bit_pos <= 24) {
        bits = (word >> bit_pos) & 0xFF;
    } else {
        uint32_t word2 = d_bits[word_idx + 1] ^ d_scramble_seq[word_idx + 1];
        bits = ((word >> bit_pos) | (word2 << (32 - bit_pos))) & 0xFF;
    }

    int i_idx = (bits & 0x1) | ((bits & 0x4) >> 1) | ((bits & 0x10) >> 2) | ((bits & 0x40) >> 3);
    int q_idx = ((bits >> 1) & 0x1) | (((bits >> 1) & 0x4) >> 1) | (((bits >> 1) & 0x10) >> 2) | (((bits >> 1) & 0x40) >> 3);

    d_symbols[idx] = make_cuFloatComplex(QAM256_TABLE[i_idx], QAM256_TABLE[q_idx]);
}

// ============================================================================
// Fused Scramble+Modulate → INT8 Kernels (single kernel for max efficiency)
// ============================================================================

/**
 * @brief Fused scramble + QPSK modulation → INT8 output
 *
 * Fixed bit ordering to match CPU:
 * - Input bits (d_bits): MSB-first within each byte (from tb_encoder)
 * - Scramble sequence: LSB-first within each word, extracted directly
 */
__global__ void scramble_modulate_qpsk_int8_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    int8_t* __restrict__ d_symbols_int8,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int out_bit_base = idx * 2;
    uint32_t scrambled_bits = 0;

    #pragma unroll
    for (int b = 0; b < 2; b++) {
        int out_bit_idx = out_bit_base + b;

        // Read input bit (MSB-first within each byte)
        int in_word = out_bit_idx / 32;
        int in_bit_in_word = out_bit_idx % 32;
        int in_byte_in_word = in_bit_in_word / 8;
        int in_bit_in_byte = 7 - (in_bit_in_word % 8);
        int in_bit_pos = in_byte_in_word * 8 + in_bit_in_byte;
        uint32_t bit = (d_bits[in_word] >> in_bit_pos) & 1u;

        // Read scrambling bit from GPU MSB-first format
        int scr_word_idx = out_bit_idx / 32;
        int scr_bit_pos = 31 - (out_bit_idx % 32);  // MSB-first
        uint32_t scr_bit = (d_scramble_seq[scr_word_idx] >> scr_bit_pos) & 1u;

        bit ^= scr_bit;
        scrambled_bits |= (bit << b);
    }

    d_symbols_int8[idx * 2]     = QPSK_I8[scrambled_bits];
    d_symbols_int8[idx * 2 + 1] = QPSK_Q8[scrambled_bits];
}

/**
 * @brief Fused scramble + 16QAM modulation → INT8 output
 *
 * Fixed bit ordering to match CPU:
 * - Input bits (d_bits): MSB-first within each byte (from tb_encoder)
 * - Scramble sequence: MSB-first within each word
 */
__global__ void scramble_modulate_16qam_int8_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    int8_t* __restrict__ d_symbols_int8,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int out_bit_base = idx * 4;
    uint32_t scrambled_bits = 0;

    #pragma unroll
    for (int b = 0; b < 4; b++) {
        int out_bit_idx = out_bit_base + b;

        // Read input bit (MSB-first within each byte)
        int in_word = out_bit_idx / 32;
        int in_bit_in_word = out_bit_idx % 32;
        int in_byte_in_word = in_bit_in_word / 8;
        int in_bit_in_byte = 7 - (in_bit_in_word % 8);
        int in_bit_pos = in_byte_in_word * 8 + in_bit_in_byte;
        uint32_t bit = (d_bits[in_word] >> in_bit_pos) & 1u;

        // Read scrambling bit from GPU MSB-first format
        int scr_word_idx = out_bit_idx / 32;
        int scr_bit_pos = 31 - (out_bit_idx % 32);  // MSB-first
        uint32_t scr_bit = (d_scramble_seq[scr_word_idx] >> scr_bit_pos) & 1u;

        bit ^= scr_bit;
        scrambled_bits |= (bit << b);
    }

    int i_idx = scrambled_bits & 0x3;
    int q_idx = (scrambled_bits >> 2) & 0x3;

    d_symbols_int8[idx * 2]     = QAM16_I8[i_idx];
    d_symbols_int8[idx * 2 + 1] = QAM16_I8[q_idx];
}

/**
 * @brief Fused scramble + 64QAM modulation → INT8 output
 *
 * Fixed bit ordering to match CPU:
 * - Input bits (d_bits): MSB-first within each byte (from tb_encoder)
 * - Scramble sequence: MSB-first within each word
 */
__global__ void scramble_modulate_64qam_int8_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    int8_t* __restrict__ d_symbols_int8,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int out_bit_base = idx * 6;
    uint32_t scrambled_bits = 0;

    #pragma unroll
    for (int b = 0; b < 6; b++) {
        int out_bit_idx = out_bit_base + b;

        // Read input bit (MSB-first within each byte)
        int in_word = out_bit_idx / 32;
        int in_bit_in_word = out_bit_idx % 32;
        int in_byte_in_word = in_bit_in_word / 8;
        int in_bit_in_byte = 7 - (in_bit_in_word % 8);
        int in_bit_pos = in_byte_in_word * 8 + in_bit_in_byte;
        uint32_t bit = (d_bits[in_word] >> in_bit_pos) & 1u;

        // Read scrambling bit from GPU MSB-first format
        int scr_word_idx = out_bit_idx / 32;
        int scr_bit_pos = 31 - (out_bit_idx % 32);  // MSB-first
        uint32_t scr_bit = (d_scramble_seq[scr_word_idx] >> scr_bit_pos) & 1u;

        bit ^= scr_bit;
        scrambled_bits |= (bit << b);
    }

    int i_idx = (scrambled_bits & 0x1) | ((scrambled_bits & 0x4) >> 1) | ((scrambled_bits & 0x10) >> 2);
    int q_idx = ((scrambled_bits >> 1) & 0x1) | (((scrambled_bits >> 1) & 0x4) >> 1) | (((scrambled_bits >> 1) & 0x10) >> 2);

    d_symbols_int8[idx * 2]     = QAM64_I8[i_idx];
    d_symbols_int8[idx * 2 + 1] = QAM64_I8[q_idx];
}

/**
 * @brief Fused scramble + 256QAM modulation → INT8 output
 *
 * Fixed bit ordering to match CPU:
 * - Input bits (d_bits): MSB-first within each byte (from tb_encoder)
 * - Scramble sequence: MSB-first within each word
 */
__global__ void scramble_modulate_256qam_int8_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    int8_t* __restrict__ d_symbols_int8,
    int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    int out_bit_base = idx * 8;
    uint32_t scrambled_bits = 0;

    #pragma unroll
    for (int b = 0; b < 8; b++) {
        int out_bit_idx = out_bit_base + b;

        // Read input bit (MSB-first within each byte)
        int in_word = out_bit_idx / 32;
        int in_bit_in_word = out_bit_idx % 32;
        int in_byte_in_word = in_bit_in_word / 8;
        int in_bit_in_byte = 7 - (in_bit_in_word % 8);
        int in_bit_pos = in_byte_in_word * 8 + in_bit_in_byte;
        uint32_t bit = (d_bits[in_word] >> in_bit_pos) & 1u;

        // Read scrambling bit from GPU MSB-first format
        int scr_word_idx = out_bit_idx / 32;
        int scr_bit_pos = 31 - (out_bit_idx % 32);  // MSB-first
        uint32_t scr_bit = (d_scramble_seq[scr_word_idx] >> scr_bit_pos) & 1u;

        bit ^= scr_bit;
        scrambled_bits |= (bit << b);
    }

    int i_idx = (scrambled_bits & 0x1) | ((scrambled_bits & 0x4) >> 1) | ((scrambled_bits & 0x10) >> 2) | ((scrambled_bits & 0x40) >> 3);
    int q_idx = ((scrambled_bits >> 1) & 0x1) | (((scrambled_bits >> 1) & 0x4) >> 1) | (((scrambled_bits >> 1) & 0x10) >> 2) | (((scrambled_bits >> 1) & 0x40) >> 3);

    d_symbols_int8[idx * 2]     = QAM256_I8[i_idx];
    d_symbols_int8[idx * 2 + 1] = QAM256_I8[q_idx];
}

// ============================================================================
// Batch Fused Scramble+Modulate Kernels (process all CBs in one launch)
// ============================================================================

/**
 * @brief Batch fused scramble + 64QAM modulation (all CBs in one kernel)
 */
__global__ void batch_scramble_modulate_64qam_kernel(
    const uint32_t* __restrict__ d_bits,      // All CBs, with stride
    const uint32_t* __restrict__ d_scramble_seq,
    cuFloatComplex* __restrict__ d_symbols,
    int symbols_per_cb,
    int words_per_cb,
    int total_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_symbols) return;

    // Determine CB and local symbol index
    int cb_idx = idx / symbols_per_cb;
    int local_sym = idx % symbols_per_cb;

    // Global bit position (for scrambling)
    int global_bit_idx = idx * 6;
    int scr_word_idx = global_bit_idx / 32;
    int scr_bit_pos = global_bit_idx % 32;

    // Local bit position within CB (for reading input)
    int local_bit_idx = local_sym * 6;
    int in_word_idx = local_bit_idx / 32;
    int in_bit_pos = local_bit_idx % 32;

    // Read input bits
    uint32_t word = d_bits[cb_idx * words_per_cb + in_word_idx];
    uint32_t bits;
    if (in_bit_pos <= 26) {
        bits = (word >> in_bit_pos) & 0x3F;
    } else {
        uint32_t word2 = d_bits[cb_idx * words_per_cb + in_word_idx + 1];
        bits = ((word >> in_bit_pos) | (word2 << (32 - in_bit_pos))) & 0x3F;
    }

    // Read scrambling bits (from global position)
    uint32_t scr_word = d_scramble_seq[scr_word_idx];
    uint32_t scr_bits;
    if (scr_bit_pos <= 26) {
        scr_bits = (scr_word >> scr_bit_pos) & 0x3F;
    } else {
        uint32_t scr_word2 = d_scramble_seq[scr_word_idx + 1];
        scr_bits = ((scr_word >> scr_bit_pos) | (scr_word2 << (32 - scr_bit_pos))) & 0x3F;
    }

    bits ^= scr_bits;

    int i_idx = (bits & 0x1) | ((bits & 0x4) >> 1) | ((bits & 0x10) >> 2);
    int q_idx = ((bits >> 1) & 0x1) | (((bits >> 1) & 0x4) >> 1) | (((bits >> 1) & 0x10) >> 2);

    d_symbols[idx] = make_cuFloatComplex(QAM64_TABLE[i_idx], QAM64_TABLE[q_idx]);
}

/**
 * @brief Batch fused scramble + QPSK modulation
 */
__global__ void batch_scramble_modulate_qpsk_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    cuFloatComplex* __restrict__ d_symbols,
    int symbols_per_cb,
    int words_per_cb,
    int total_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_symbols) return;

    int cb_idx = idx / symbols_per_cb;
    int local_sym = idx % symbols_per_cb;

    int global_bit_idx = idx * 2;
    int scr_word_idx = global_bit_idx / 32;
    int scr_bit_pos = global_bit_idx % 32;

    int local_bit_idx = local_sym * 2;
    int in_word_idx = local_bit_idx / 32;
    int in_bit_pos = local_bit_idx % 32;

    uint32_t word = d_bits[cb_idx * words_per_cb + in_word_idx];
    uint32_t bits = (word >> in_bit_pos) & 0x3;

    uint32_t scr_word = d_scramble_seq[scr_word_idx];
    uint32_t scr_bits = (scr_word >> scr_bit_pos) & 0x3;

    bits ^= scr_bits;

    d_symbols[idx] = make_cuFloatComplex(QPSK_I[bits], QPSK_Q[bits]);
}

/**
 * @brief Batch fused scramble + 16QAM modulation
 */
__global__ void batch_scramble_modulate_16qam_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    cuFloatComplex* __restrict__ d_symbols,
    int symbols_per_cb,
    int words_per_cb,
    int total_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_symbols) return;

    int cb_idx = idx / symbols_per_cb;
    int local_sym = idx % symbols_per_cb;

    int global_bit_idx = idx * 4;
    int scr_word_idx = global_bit_idx / 32;
    int scr_bit_pos = global_bit_idx % 32;

    int local_bit_idx = local_sym * 4;
    int in_word_idx = local_bit_idx / 32;
    int in_bit_pos = local_bit_idx % 32;

    uint32_t word = d_bits[cb_idx * words_per_cb + in_word_idx];
    uint32_t bits;
    if (in_bit_pos <= 28) {
        bits = (word >> in_bit_pos) & 0xF;
    } else {
        uint32_t word2 = d_bits[cb_idx * words_per_cb + in_word_idx + 1];
        bits = ((word >> in_bit_pos) | (word2 << (32 - in_bit_pos))) & 0xF;
    }

    uint32_t scr_word = d_scramble_seq[scr_word_idx];
    uint32_t scr_bits;
    if (scr_bit_pos <= 28) {
        scr_bits = (scr_word >> scr_bit_pos) & 0xF;
    } else {
        uint32_t scr_word2 = d_scramble_seq[scr_word_idx + 1];
        scr_bits = ((scr_word >> scr_bit_pos) | (scr_word2 << (32 - scr_bit_pos))) & 0xF;
    }

    bits ^= scr_bits;

    int i_idx = (bits & 0x1) | ((bits & 0x4) >> 1);
    int q_idx = ((bits >> 1) & 0x1) | (((bits >> 1) & 0x4) >> 1);

    d_symbols[idx] = make_cuFloatComplex(QAM16_TABLE[i_idx], QAM16_TABLE[q_idx]);
}

/**
 * @brief Batch fused scramble + 256QAM modulation
 */
__global__ void batch_scramble_modulate_256qam_kernel(
    const uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_scramble_seq,
    cuFloatComplex* __restrict__ d_symbols,
    int symbols_per_cb,
    int words_per_cb,
    int total_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_symbols) return;

    int cb_idx = idx / symbols_per_cb;
    int local_sym = idx % symbols_per_cb;

    int global_bit_idx = idx * 8;
    int scr_word_idx = global_bit_idx / 32;
    int scr_bit_pos = global_bit_idx % 32;

    int local_bit_idx = local_sym * 8;
    int in_word_idx = local_bit_idx / 32;
    int in_bit_pos = local_bit_idx % 32;

    uint32_t word = d_bits[cb_idx * words_per_cb + in_word_idx];
    uint32_t bits;
    if (in_bit_pos <= 24) {
        bits = (word >> in_bit_pos) & 0xFF;
    } else {
        uint32_t word2 = d_bits[cb_idx * words_per_cb + in_word_idx + 1];
        bits = ((word >> in_bit_pos) | (word2 << (32 - in_bit_pos))) & 0xFF;
    }

    uint32_t scr_word = d_scramble_seq[scr_word_idx];
    uint32_t scr_bits;
    if (scr_bit_pos <= 24) {
        scr_bits = (scr_word >> scr_bit_pos) & 0xFF;
    } else {
        uint32_t scr_word2 = d_scramble_seq[scr_word_idx + 1];
        scr_bits = ((scr_word >> scr_bit_pos) | (scr_word2 << (32 - scr_bit_pos))) & 0xFF;
    }

    bits ^= scr_bits;

    int i_idx = (bits & 0x1) | ((bits & 0x4) >> 1) | ((bits & 0x10) >> 2) | ((bits & 0x40) >> 3);
    int q_idx = ((bits >> 1) & 0x1) | (((bits >> 1) & 0x4) >> 1) | (((bits >> 1) & 0x10) >> 2) | (((bits >> 1) & 0x40) >> 3);

    d_symbols[idx] = make_cuFloatComplex(QAM256_TABLE[i_idx], QAM256_TABLE[q_idx]);
}

int modulator_scramble_and_modulate_batch(modulator_handle_t handle,
                                          const uint32_t* d_bits,
                                          const uint32_t* d_scramble_seq,
                                          cuFloatComplex* d_symbols,
                                          int bits_per_cb,
                                          int words_per_cb,
                                          int num_cbs,
                                          int mod_order,
                                          cudaStream_t stream) {
    if (!handle || !d_bits || !d_scramble_seq || !d_symbols || bits_per_cb <= 0 || num_cbs <= 0) return -1;

    int symbols_per_cb = bits_per_cb / mod_order;
    int total_symbols = symbols_per_cb * num_cbs;
    int block_size = 256;
    int num_blocks = (total_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 2:  // QPSK
            batch_scramble_modulate_qpsk_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, symbols_per_cb, words_per_cb, total_symbols);
            break;
        case 4:  // 16QAM
            batch_scramble_modulate_16qam_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, symbols_per_cb, words_per_cb, total_symbols);
            break;
        case 6:  // 64QAM
            batch_scramble_modulate_64qam_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, symbols_per_cb, words_per_cb, total_symbols);
            break;
        case 8:  // 256QAM
            batch_scramble_modulate_256qam_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, symbols_per_cb, words_per_cb, total_symbols);
            break;
        default:
            return -1;
    }

    return 0;
}

int modulator_scramble_and_modulate(modulator_handle_t handle,
                                     const uint32_t* d_bits,
                                     const uint32_t* d_scramble_seq,
                                     cuFloatComplex* d_symbols,
                                     int num_bits,
                                     int mod_order,
                                     cudaStream_t stream) {
    if (!handle || !d_bits || !d_scramble_seq || !d_symbols || num_bits <= 0) return -1;

    int num_symbols = (num_bits + mod_order - 1) / mod_order;
    int block_size = 256;
    int num_blocks = (num_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 2:  // QPSK
            scramble_modulate_qpsk_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, num_symbols);
            break;
        case 4:  // 16QAM
            scramble_modulate_16qam_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, num_symbols);
            break;
        case 6:  // 64QAM
            scramble_modulate_64qam_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, num_symbols);
            break;
        case 8:  // 256QAM
            scramble_modulate_256qam_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols, num_symbols);
            break;
        default:
            // BPSK not supported for fused scramble+modulate yet
            return -1;
    }

    return 0;
}

int modulator_scramble_and_modulate_int8(modulator_handle_t handle,
                                          const uint32_t* d_bits,
                                          const uint32_t* d_scramble_seq,
                                          int8_t* d_symbols_int8,
                                          int num_bits,
                                          int mod_order,
                                          cudaStream_t stream) {
    if (!handle || !d_bits || !d_scramble_seq || !d_symbols_int8 || num_bits <= 0) return -1;

    int num_symbols = (num_bits + mod_order - 1) / mod_order;
    int block_size = 256;
    int num_blocks = (num_symbols + block_size - 1) / block_size;

    switch (mod_order) {
        case 2:  // QPSK
            scramble_modulate_qpsk_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols_int8, num_symbols);
            break;
        case 4:  // 16QAM
            scramble_modulate_16qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols_int8, num_symbols);
            break;
        case 6:  // 64QAM
            scramble_modulate_64qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols_int8, num_symbols);
            break;
        case 8:  // 256QAM
            scramble_modulate_256qam_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_bits, d_scramble_seq, d_symbols_int8, num_symbols);
            break;
        default:
            return -1;
    }

    return 0;
}

// ============================================================================
// Kernels for verification
// ============================================================================

/**
 * @brief Descramble FP32 LLRs in-place kernel
 */
__global__ void descramble_llrs_fp32_kernel(
    float* __restrict__ d_llrs,
    const uint32_t* __restrict__ d_scramble_seq,
    int num_bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bits) return;

    // MSB-first bit indexing to match gold sequence generator
    int word_idx = idx / 32;
    int bit_pos = 31 - (idx % 32);
    uint32_t seq_bit = (d_scramble_seq[word_idx] >> bit_pos) & 1;

    if (seq_bit) {
        d_llrs[idx] = -d_llrs[idx];
    }
}

// ============================================================================
// Verification API functions
// ============================================================================

int modulator_soft_demod_only_fp32(modulator_handle_t handle,
                                    const cuFloatComplex* d_symbols,
                                    const float* d_noise_vars,
                                    float* d_llrs,
                                    int num_symbols,
                                    int mod_order,
                                    cudaStream_t stream) {
    // This is just a wrapper around the existing soft demod per symbol function
    // which already outputs FP32 LLRs without descrambling
    return modulator_soft_demod_per_symbol(handle, d_symbols, d_noise_vars, d_llrs,
                                            num_symbols, mod_order, stream);
}

int modulator_descramble_llrs_fp32(modulator_handle_t handle,
                                    float* d_llrs,
                                    const uint32_t* d_scramble_seq,
                                    int num_bits,
                                    cudaStream_t stream) {
    if (!handle || !d_llrs || !d_scramble_seq || num_bits <= 0) return -1;

    int block_size = 256;
    int num_blocks = (num_bits + block_size - 1) / block_size;

    descramble_llrs_fp32_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_llrs, d_scramble_seq, num_bits);

    return 0;
}

} // extern "C"
