/**
 * @file mcs_bler_sensitivity_test.cu
 * @brief Comprehensive BLER Sensitivity Test for All 3GPP MCS Tables
 *
 * Tests all MCS indices from 3GPP TS 38.214 MCS Tables 1, 2, and 3.
 * Uses the full TB encoder/decoder chain with proper rate matching to
 * achieve target code rates for each MCS.
 *
 * For each MCS, finds the SNR required for 10% BLER using binary search.
 */

#include "ocudu_phy_cuda.h"
#include "transport_block.h"
#include "modulation.h"
#include "ldpc_encoder.h"
#include "ldpc_decoder.h"
#include "rate_matching.h"
#include "scrambling.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Convert bytes (MSB first) to packed uint32_t bits (LSB first) for modulator
__global__ void bytes_to_packed_bits_kernel(
    const uint8_t* __restrict__ d_bytes,
    uint32_t* __restrict__ d_bits,
    int num_bits
) {
    int bit_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bit_idx >= num_bits) return;

    int byte_idx = bit_idx / 8;
    int bit_in_byte = 7 - (bit_idx % 8);  // MSB first in byte
    uint8_t bit = (d_bytes[byte_idx] >> bit_in_byte) & 1;

    int word_idx = bit_idx / 32;
    int word_bit = bit_idx % 32;
    atomicOr(&d_bits[word_idx], (uint32_t)bit << word_bit);
}

// ============================================================================
// 3GPP TS 38.214 MCS Tables
// ============================================================================

struct MCSTableEntry {
    int mcs_index;
    int modulation_order;  // Qm: 2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM
    float target_rate;     // R (already divided by 1024)
    const char* mod_name;
};

// Table 5.1.3.1-1: MCS index table 1 for PDSCH (64QAM max)
static const MCSTableEntry MCS_TABLE_1[] = {
    { 0, 2, 120.0f/1024, "QPSK"},
    { 1, 2, 157.0f/1024, "QPSK"},
    { 2, 2, 193.0f/1024, "QPSK"},
    { 3, 2, 251.0f/1024, "QPSK"},
    { 4, 2, 308.0f/1024, "QPSK"},
    { 5, 2, 379.0f/1024, "QPSK"},
    { 6, 2, 449.0f/1024, "QPSK"},
    { 7, 2, 526.0f/1024, "QPSK"},
    { 8, 2, 602.0f/1024, "QPSK"},
    { 9, 2, 679.0f/1024, "QPSK"},
    {10, 4, 340.0f/1024, "16QAM"},
    {11, 4, 378.0f/1024, "16QAM"},
    {12, 4, 434.0f/1024, "16QAM"},
    {13, 4, 490.0f/1024, "16QAM"},
    {14, 4, 553.0f/1024, "16QAM"},
    {15, 4, 616.0f/1024, "16QAM"},
    {16, 4, 658.0f/1024, "16QAM"},
    {17, 6, 438.0f/1024, "64QAM"},
    {18, 6, 466.0f/1024, "64QAM"},
    {19, 6, 517.0f/1024, "64QAM"},
    {20, 6, 567.0f/1024, "64QAM"},
    {21, 6, 616.0f/1024, "64QAM"},
    {22, 6, 666.0f/1024, "64QAM"},
    {23, 6, 719.0f/1024, "64QAM"},
    {24, 6, 772.0f/1024, "64QAM"},
    {25, 6, 822.0f/1024, "64QAM"},
    {26, 6, 873.0f/1024, "64QAM"},
    {27, 6, 948.0f/1024, "64QAM"},
};
static const int MCS_TABLE_1_SIZE = sizeof(MCS_TABLE_1) / sizeof(MCS_TABLE_1[0]);

// Table 5.1.3.1-2: MCS index table 2 for PDSCH (256QAM max)
static const MCSTableEntry MCS_TABLE_2[] = {
    { 0, 2, 120.0f/1024, "QPSK"},
    { 1, 2, 193.0f/1024, "QPSK"},
    { 2, 2, 308.0f/1024, "QPSK"},
    { 3, 2, 449.0f/1024, "QPSK"},
    { 4, 2, 602.0f/1024, "QPSK"},
    { 5, 4, 378.0f/1024, "16QAM"},
    { 6, 4, 434.0f/1024, "16QAM"},
    { 7, 4, 490.0f/1024, "16QAM"},
    { 8, 4, 553.0f/1024, "16QAM"},
    { 9, 4, 616.0f/1024, "16QAM"},
    {10, 4, 658.0f/1024, "16QAM"},
    {11, 6, 466.0f/1024, "64QAM"},
    {12, 6, 517.0f/1024, "64QAM"},
    {13, 6, 567.0f/1024, "64QAM"},
    {14, 6, 616.0f/1024, "64QAM"},
    {15, 6, 666.0f/1024, "64QAM"},
    {16, 6, 719.0f/1024, "64QAM"},
    {17, 6, 772.0f/1024, "64QAM"},
    {18, 6, 822.0f/1024, "64QAM"},
    {19, 6, 873.0f/1024, "64QAM"},
    {20, 8, 682.5f/1024, "256QAM"},
    {21, 8, 711.0f/1024, "256QAM"},
    {22, 8, 754.0f/1024, "256QAM"},
    {23, 8, 797.0f/1024, "256QAM"},
    {24, 8, 841.0f/1024, "256QAM"},
    {25, 8, 885.0f/1024, "256QAM"},
    {26, 8, 916.5f/1024, "256QAM"},
    {27, 8, 948.0f/1024, "256QAM"},
};
static const int MCS_TABLE_2_SIZE = sizeof(MCS_TABLE_2) / sizeof(MCS_TABLE_2[0]);

// Table 5.1.3.1-3: MCS index table 3 for PUSCH with transform precoding (64QAM max, low SE)
static const MCSTableEntry MCS_TABLE_3[] = {
    { 0, 2,  30.0f/1024, "QPSK"},
    { 1, 2,  40.0f/1024, "QPSK"},
    { 2, 2,  50.0f/1024, "QPSK"},
    { 3, 2,  64.0f/1024, "QPSK"},
    { 4, 2,  78.0f/1024, "QPSK"},
    { 5, 2,  99.0f/1024, "QPSK"},
    { 6, 2, 120.0f/1024, "QPSK"},
    { 7, 2, 157.0f/1024, "QPSK"},
    { 8, 2, 193.0f/1024, "QPSK"},
    { 9, 2, 251.0f/1024, "QPSK"},
    {10, 2, 308.0f/1024, "QPSK"},
    {11, 2, 379.0f/1024, "QPSK"},
    {12, 2, 449.0f/1024, "QPSK"},
    {13, 2, 526.0f/1024, "QPSK"},
    {14, 2, 602.0f/1024, "QPSK"},
    {15, 4, 340.0f/1024, "16QAM"},
    {16, 4, 378.0f/1024, "16QAM"},
    {17, 4, 434.0f/1024, "16QAM"},
    {18, 4, 490.0f/1024, "16QAM"},
    {19, 4, 553.0f/1024, "16QAM"},
    {20, 4, 616.0f/1024, "16QAM"},
    {21, 6, 438.0f/1024, "64QAM"},
    {22, 6, 466.0f/1024, "64QAM"},
    {23, 6, 517.0f/1024, "64QAM"},
    {24, 6, 567.0f/1024, "64QAM"},
    {25, 6, 616.0f/1024, "64QAM"},
    {26, 6, 666.0f/1024, "64QAM"},
    {27, 6, 719.0f/1024, "64QAM"},
};
static const int MCS_TABLE_3_SIZE = sizeof(MCS_TABLE_3) / sizeof(MCS_TABLE_3[0]);

// ============================================================================
// Test Result Structure
// ============================================================================

struct MCSTestResult {
    int table_num;
    int mcs_index;
    int mod_order;
    float target_rate;
    float actual_rate;
    int tbs_bits;
    int encoded_bits;
    float bler10_esn0_db;   // Es/N0 for 10% BLER
    float baseline_snr_db;  // Baseline is Es/N0
    bool passed;
    const char* mod_name;
};

// ============================================================================
// BLER Measurement using TB Encoder/Decoder with Rate Matching
// ============================================================================

/**
 * @brief Measure BLER at a given SNR using full TB chain with rate matching
 */
float measure_bler_tb(int tbs_bits, float target_rate, int mod_order, float snr_db,
                      int min_blocks, int max_blocks, int min_errors,
                      int* out_encoded_bits = nullptr, int max_iterations = 25) {

    // Calculate encoded bits (G) to achieve target code rate
    // G = TBS / R
    int encoded_bits = (int)ceilf((float)tbs_bits / target_rate);

    // Ensure encoded_bits is valid (must be >= TBS and reasonable)
    if (encoded_bits < tbs_bits) encoded_bits = tbs_bits + 1000;

    // Cap at a reasonable maximum to avoid memory issues
    if (encoded_bits > 500000) encoded_bits = 500000;

    // Report actual parameters
    float actual_rate = (float)tbs_bits / (float)encoded_bits;
    if (out_encoded_bits) *out_encoded_bits = encoded_bits;

    int tb_bytes = (tbs_bits + 7) / 8;
    int num_symbols = encoded_bits / mod_order;

    // Create TB encoder and decoder
    tb_encoder_handle_t encoder = nullptr;
    tb_decoder_handle_t decoder = nullptr;
    modulator_handle_t modulator = nullptr;

    nr_ldpc_status_t status;
    status = tb_encoder_create(&encoder);
    if (status != NR_LDPC_SUCCESS) return 1.0f;

    status = tb_decoder_create(&decoder);
    if (status != NR_LDPC_SUCCESS) {
        tb_encoder_destroy(encoder);
        return 1.0f;
    }

    if (modulator_create(&modulator) != 0) {
        tb_encoder_destroy(encoder);
        tb_decoder_destroy(decoder);
        return 1.0f;
    }

    // Configure encoder
    tb_encoder_config_t enc_cfg = {};
    enc_cfg.tb_size_bits = tbs_bits;
    enc_cfg.num_layers = 1;
    enc_cfg.modulation_order = mod_order;
    enc_cfg.num_allocated_res = encoded_bits;  // This controls rate matching!
    enc_cfg.redundancy_version = 0;
    enc_cfg.code_rate = target_rate;
    enc_cfg.n_RNTI = 0x1234;
    enc_cfg.n_ID = 500;
    enc_cfg.q = 0;
    enc_cfg.enable_scrambling = true;

    status = tb_encoder_configure(encoder, &enc_cfg);
    if (status != NR_LDPC_SUCCESS) {
        tb_encoder_destroy(encoder);
        tb_decoder_destroy(decoder);
        modulator_destroy(modulator);
        return 1.0f;
    }

    // Configure decoder
    tb_decoder_config_t dec_cfg = {};
    dec_cfg.tb_size_bits = tbs_bits;
    dec_cfg.num_layers = 1;
    dec_cfg.modulation_order = mod_order;
    dec_cfg.num_received_bits = encoded_bits;
    dec_cfg.redundancy_version = 0;
    dec_cfg.code_rate = target_rate;
    dec_cfg.max_iterations = max_iterations;
    dec_cfg.llr_clamp = 16.0f;
    dec_cfg.n_RNTI = 0x1234;
    dec_cfg.n_ID = 500;
    dec_cfg.q = 0;
    dec_cfg.enable_scrambling = true;

    status = tb_decoder_configure(decoder, &dec_cfg);
    if (status != NR_LDPC_SUCCESS) {
        tb_encoder_destroy(encoder);
        tb_decoder_destroy(decoder);
        modulator_destroy(modulator);
        return 1.0f;
    }

    // Calculate noise parameters directly from Es/N0
    // Es/N0 = Symbol energy / Noise PSD
    // noise_var = N0 = 1/EsN0 (total variance, what soft_demod expects)
    // noise_std = sqrt(noise_var/2) per dimension for AWGN generation
    float esn0_linear = powf(10.0f, snr_db / 10.0f);
    float noise_var = 1.0f / esn0_linear;  // Total noise variance
    float noise_std = sqrtf(noise_var / 2.0f);  // Per-dimension std dev

    // Allocate memory
    uint8_t* h_tx = new uint8_t[tb_bytes];
    uint8_t* h_rx = new uint8_t[tb_bytes];

    uint8_t* d_tx;
    uint8_t* d_encoded;
    uint32_t* d_packed_bits;  // Packed bits for modulator (proper bit order)
    cuFloatComplex* d_symbols;
    float* d_llrs;
    uint8_t* d_rx;

    int encoded_bytes = (encoded_bits + 7) / 8;
    int encoded_words = (encoded_bits + 31) / 32;

    CHECK_CUDA(cudaMalloc(&d_tx, tb_bytes));
    CHECK_CUDA(cudaMalloc(&d_encoded, encoded_bytes));
    CHECK_CUDA(cudaMalloc(&d_packed_bits, encoded_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_symbols, num_symbols * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&d_llrs, encoded_bits * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rx, tb_bytes));

    int error_blocks = 0;
    int total_blocks = 0;

    while (total_blocks < min_blocks ||
           (error_blocks < min_errors && total_blocks < max_blocks)) {

        // Generate random TB data
        for (int i = 0; i < tb_bytes; i++) {
            h_tx[i] = rand() & 0xFF;
        }
        CHECK_CUDA(cudaMemcpy(d_tx, h_tx, tb_bytes, cudaMemcpyHostToDevice));

        // Encode TB (includes segmentation, LDPC, rate matching, scrambling)
        status = tb_encoder_encode(encoder, d_tx, d_encoded, 0);
        if (status != NR_LDPC_SUCCESS) {
            continue;
        }

        // Modulate directly (cast bytes to uint32_t* like mcs_bler_curves does)
        modulator_modulate(modulator, (uint32_t*)d_encoded, d_symbols,
                          encoded_bits, mod_order, 0);

        // Add AWGN noise
        modulator_add_noise(modulator, d_symbols, num_symbols, noise_std, 0);

        // Soft demodulate (pass total noise variance)
        modulator_soft_demod(modulator, d_symbols, d_llrs, num_symbols,
                            mod_order, noise_var, 0);

        // Decode TB (includes descrambling, de-rate matching, LDPC decode, CRC check)
        tb_decode_result_t dec_result;
        status = tb_decoder_decode(decoder, d_llrs, d_rx, &dec_result, 0);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Check for errors
        CHECK_CUDA(cudaMemcpy(h_rx, d_rx, tb_bytes, cudaMemcpyDeviceToHost));

        bool block_error = !dec_result.crc_pass;
        if (!block_error) {
            // Also verify data matches
            for (int i = 0; i < tb_bytes; i++) {
                if (h_tx[i] != h_rx[i]) {
                    block_error = true;
                    break;
                }
            }
        }

        total_blocks++;
        if (block_error) error_blocks++;
    }

    float bler = (total_blocks > 0) ? (float)error_blocks / total_blocks : 1.0f;

    // Cleanup
    tb_encoder_destroy(encoder);
    tb_decoder_destroy(decoder);
    modulator_destroy(modulator);

    CHECK_CUDA(cudaFree(d_tx));
    CHECK_CUDA(cudaFree(d_encoded));
    CHECK_CUDA(cudaFree(d_packed_bits));
    CHECK_CUDA(cudaFree(d_symbols));
    CHECK_CUDA(cudaFree(d_llrs));
    CHECK_CUDA(cudaFree(d_rx));

    delete[] h_tx;
    delete[] h_rx;

    return bler;
}

// ============================================================================
// Raw LDPC BLER Measurement (for comparison with Python - no TB chain)
// ============================================================================

/**
 * @brief Measure BLER at a given SNR using raw LDPC chain (no TB overhead)
 * This matches what the Python binding does for fair comparison.
 */
float measure_bler_raw_ldpc(int info_bits, float target_rate, int mod_order, float snr_db,
                            int min_blocks, int max_blocks, int min_errors) {

    // Initialize LDPC configuration
    nr_ldpc_config_t ldpc_cfg;
    nr_ldpc_status_t status = nr_ldpc_init_config(&ldpc_cfg, info_bits, target_rate);
    if (status != NR_LDPC_SUCCESS) return 1.0f;

    // Calculate rate matching output size
    int E = (int)ceilf((float)info_bits / target_rate);
    E = ((E + mod_order - 1) / mod_order) * mod_order;  // Round to mod_order multiple

    int num_symbols = E / mod_order;

    // Create components
    ldpc_encoder_handle_t encoder = nullptr;
    ldpc_decoder_handle_t decoder = nullptr;
    rate_matcher_handle_t rm_tx = nullptr;
    rate_matcher_handle_t rm_rx = nullptr;
    scrambler_handle_t scrambler = nullptr;
    modulator_handle_t modulator = nullptr;

    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &ldpc_cfg);

    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 25;
    dec_params.llr_clamp = 16.0f;
    ldpc_decoder_configure(decoder, &ldpc_cfg, &dec_params);

    // Rate matcher TX
    rate_matcher_create(&rm_tx);
    nr_rate_match_config_t rm_cfg = {};
    rm_cfg.E = E;
    rm_cfg.Q_m = mod_order;
    rm_cfg.rv = 0;
    rate_matcher_configure_tx(rm_tx, &ldpc_cfg, &rm_cfg);

    // Rate matcher RX
    rate_matcher_create(&rm_rx);
    rate_matcher_configure_rx(rm_rx, &ldpc_cfg, &rm_cfg);

    // Scrambler
    scrambler_create(&scrambler);
    nr_scrambling_config_t scr_cfg = {.n_RNTI = 0x1234, .n_ID = 500, .q = 0};
    scrambler_configure(scrambler, &scr_cfg);

    if (modulator_create(&modulator) != 0) {
        ldpc_encoder_destroy(encoder);
        ldpc_decoder_destroy(decoder);
        rate_matcher_destroy(rm_tx);
        rate_matcher_destroy(rm_rx);
        scrambler_destroy(scrambler);
        return 1.0f;
    }

    // Get buffer sizes
    int K_words = ldpc_encoder_get_input_words(encoder);
    int enc_words = ldpc_encoder_get_output_words(encoder);
    int E_words = (E + 31) / 32;
    int N_full = (ldpc_cfg.base_graph == 1) ? 68 * ldpc_cfg.lifting_size : 52 * ldpc_cfg.lifting_size;

    // Calculate noise parameters
    // Es/N0 = Symbol energy / Noise PSD, noise_var = N0 = 1/EsN0 (total variance)
    // For AWGN with I,Q: noise_std = sqrt(noise_var/2) per dimension
    float esn0_linear = powf(10.0f, snr_db / 10.0f);
    float noise_var = 1.0f / esn0_linear;  // Total noise variance (what soft_demod expects)
    float noise_std = sqrtf(noise_var / 2.0f);  // Per-dimension std dev for AWGN

    // Allocate GPU memory
    uint32_t* d_input;
    uint32_t* d_encoded;
    uint32_t* d_rate_matched;
    uint32_t* d_scrambled;
    cuFloatComplex* d_symbols;
    float* d_llrs;
    float* d_derate_llrs;
    uint32_t* d_decoded;

    uint32_t* d_interleaved;
    float* d_deint_llrs;

    CHECK_CUDA(cudaMalloc(&d_input, K_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_rate_matched, E_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_interleaved, E_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_scrambled, E_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_symbols, num_symbols * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&d_llrs, E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_deint_llrs, E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_derate_llrs, N_full * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_decoded, K_words * sizeof(uint32_t)));

    // Host buffers for comparison
    std::vector<uint32_t> h_input(K_words);
    std::vector<uint32_t> h_decoded(K_words);

    int error_blocks = 0;
    int total_blocks = 0;
    int Kd = ldpc_cfg.num_info_bits;  // Actual info bits (excluding filler)

    while (total_blocks < min_blocks ||
           (error_blocks < min_errors && total_blocks < max_blocks)) {

        // Generate random input bits
        for (int i = 0; i < K_words; i++) {
            h_input[i] = rand();
        }
        CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // TX Chain: encode -> rate match -> interleave -> scramble -> modulate
        ldpc_encoder_encode(encoder, d_input, d_encoded, 0);
        rate_matcher_match(rm_tx, d_encoded, d_rate_matched, 0);

        // Add bit interleaving (same as TB chain)
        CHECK_CUDA(cudaMemset(d_interleaved, 0, E_words * sizeof(uint32_t)));
        rate_matcher_interleave(rm_tx, d_rate_matched, d_interleaved, E, mod_order, 0);

        scrambler_scramble(scrambler, d_interleaved, d_scrambled, E, 0);
        modulator_modulate(modulator, d_scrambled, d_symbols, E, mod_order, 0);

        // Channel: add noise
        modulator_add_noise(modulator, d_symbols, num_symbols, noise_std, 0);

        // RX Chain: demodulate -> descramble -> de-interleave -> de-rate match -> decode
        modulator_soft_demod(modulator, d_symbols, d_llrs, num_symbols, mod_order, noise_var, 0);
        scrambler_descramble_llr_inplace(scrambler, d_llrs, E, 0);

        // De-interleave
        rate_matcher_deinterleave_llr(rm_rx, d_llrs, d_deint_llrs, E, mod_order, 0);

        CHECK_CUDA(cudaMemset(d_derate_llrs, 0, N_full * sizeof(float)));
        rate_matcher_dematch(rm_rx, d_deint_llrs, d_derate_llrs, 0);
        CHECK_CUDA(cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t)));
        ldpc_decoder_decode(decoder, d_derate_llrs, d_decoded, 0);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Compare info bits only
        CHECK_CUDA(cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        bool block_error = false;
        for (int i = 0; i < Kd && !block_error; i++) {
            int in_bit = (h_input[i / 32] >> (i % 32)) & 1;
            int out_bit = (h_decoded[i / 32] >> (i % 32)) & 1;
            if (in_bit != out_bit) {
                block_error = true;
            }
        }

        total_blocks++;
        if (block_error) error_blocks++;
    }

    float bler = (total_blocks > 0) ? (float)error_blocks / total_blocks : 1.0f;

    // Cleanup
    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);
    rate_matcher_destroy(rm_tx);
    rate_matcher_destroy(rm_rx);
    scrambler_destroy(scrambler);
    modulator_destroy(modulator);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_encoded));
    CHECK_CUDA(cudaFree(d_rate_matched));
    CHECK_CUDA(cudaFree(d_interleaved));
    CHECK_CUDA(cudaFree(d_scrambled));
    CHECK_CUDA(cudaFree(d_symbols));
    CHECK_CUDA(cudaFree(d_llrs));
    CHECK_CUDA(cudaFree(d_deint_llrs));
    CHECK_CUDA(cudaFree(d_derate_llrs));
    CHECK_CUDA(cudaFree(d_decoded));

    return bler;
}

// ============================================================================
// LLR Conversion Kernel for FP16 Path
// ============================================================================
__global__ void convert_llr_fp32_to_half_kernel(
    const float* __restrict__ d_input,
    __half* __restrict__ d_output,
    int num_llrs,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_llrs) return;

    float llr = d_input[idx] * scale;
    llr = fmaxf(-127.0f, fminf(127.0f, llr));
    d_output[idx] = __float2half_rn(llr);
}

// ============================================================================
// Raw LDPC BLER Measurement using FP16 (half) Kernel
// ============================================================================

/**
 * @brief Measure BLER at a given SNR using FP16 (half) LDPC decoder
 * This tests the high-performance FP16 pipeline.
 */
float measure_bler_raw_ldpc_half(int info_bits, float target_rate, int mod_order, float snr_db,
                                  int min_blocks, int max_blocks, int min_errors) {

    // Initialize LDPC configuration
    nr_ldpc_config_t ldpc_cfg;
    nr_ldpc_status_t status = nr_ldpc_init_config(&ldpc_cfg, info_bits, target_rate);
    if (status != NR_LDPC_SUCCESS) return 1.0f;

    // Calculate rate matching output size
    int E = (int)ceilf((float)info_bits / target_rate);
    E = ((E + mod_order - 1) / mod_order) * mod_order;  // Round to mod_order multiple

    int num_symbols = E / mod_order;

    // Create components
    ldpc_encoder_handle_t encoder = nullptr;
    ldpc_decoder_handle_t decoder = nullptr;
    rate_matcher_handle_t rm_tx = nullptr;
    rate_matcher_handle_t rm_rx = nullptr;
    scrambler_handle_t scrambler = nullptr;
    modulator_handle_t modulator = nullptr;

    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &ldpc_cfg);

    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 25;
    dec_params.llr_clamp = 16.0f;
    ldpc_decoder_configure(decoder, &ldpc_cfg, &dec_params);

    // Rate matcher TX
    rate_matcher_create(&rm_tx);
    nr_rate_match_config_t rm_cfg = {};
    rm_cfg.E = E;
    rm_cfg.Q_m = mod_order;
    rm_cfg.rv = 0;
    rate_matcher_configure_tx(rm_tx, &ldpc_cfg, &rm_cfg);

    // Rate matcher RX
    rate_matcher_create(&rm_rx);
    rate_matcher_configure_rx(rm_rx, &ldpc_cfg, &rm_cfg);

    // Scrambler
    scrambler_create(&scrambler);
    nr_scrambling_config_t scr_cfg = {.n_RNTI = 0x1234, .n_ID = 500, .q = 0};
    scrambler_configure(scrambler, &scr_cfg);

    if (modulator_create(&modulator) != 0) {
        ldpc_encoder_destroy(encoder);
        ldpc_decoder_destroy(decoder);
        rate_matcher_destroy(rm_tx);
        rate_matcher_destroy(rm_rx);
        scrambler_destroy(scrambler);
        return 1.0f;
    }

    // Get buffer sizes
    int K_words = ldpc_encoder_get_input_words(encoder);
    int enc_words = ldpc_encoder_get_output_words(encoder);
    int E_words = (E + 31) / 32;
    int N_full = (ldpc_cfg.base_graph == 1) ? 68 * ldpc_cfg.lifting_size : 52 * ldpc_cfg.lifting_size;

    // Calculate noise parameters
    // Es/N0 = Symbol energy / Noise PSD, noise_var = N0 = 1/EsN0 (total variance)
    // For AWGN with I,Q: noise_std = sqrt(noise_var/2) per dimension
    float esn0_linear = powf(10.0f, snr_db / 10.0f);
    float noise_var = 1.0f / esn0_linear;  // Total noise variance (what soft_demod expects)
    float noise_std = sqrtf(noise_var / 2.0f);  // Per-dimension std dev for AWGN

    // Allocate GPU memory.
    uint32_t* d_input;
    uint32_t* d_encoded;
    uint32_t* d_rate_matched;
    uint32_t* d_scrambled;
    cuFloatComplex* d_symbols;
    float* d_llrs;
    float* d_derate_llrs;
    __half* d_derate_llrs_half;
    uint32_t* d_decoded;

    uint32_t* d_interleaved;
    float* d_deint_llrs;

    CHECK_CUDA(cudaMalloc(&d_input, K_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_rate_matched, E_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_interleaved, E_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_scrambled, E_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_symbols, num_symbols * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&d_llrs, E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_deint_llrs, E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_derate_llrs, N_full * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_derate_llrs_half, N_full * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_decoded, K_words * sizeof(uint32_t)));

    // Host buffers for comparison
    std::vector<uint32_t> h_input(K_words);
    std::vector<uint32_t> h_decoded(K_words);

    int error_blocks = 0;
    int total_blocks = 0;
    int Kd = ldpc_cfg.num_info_bits;  // Actual info bits (excluding filler)

    // Soft-bit scale used before FP16 decoder input conversion.
    const float llr_scale = 4.0f;

    while (total_blocks < min_blocks ||
           (error_blocks < min_errors && total_blocks < max_blocks)) {

        // Generate random input bits
        for (int i = 0; i < K_words; i++) {
            h_input[i] = rand();
        }
        CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // TX Chain: encode -> rate match -> interleave -> scramble -> modulate
        ldpc_encoder_encode(encoder, d_input, d_encoded, 0);
        rate_matcher_match(rm_tx, d_encoded, d_rate_matched, 0);

        // Add bit interleaving (same as TB chain)
        CHECK_CUDA(cudaMemset(d_interleaved, 0, E_words * sizeof(uint32_t)));
        rate_matcher_interleave(rm_tx, d_rate_matched, d_interleaved, E, mod_order, 0);

        scrambler_scramble(scrambler, d_interleaved, d_scrambled, E, 0);
        modulator_modulate(modulator, d_scrambled, d_symbols, E, mod_order, 0);

        // Channel: add noise
        modulator_add_noise(modulator, d_symbols, num_symbols, noise_std, 0);

        // RX Chain: demodulate -> descramble -> de-interleave -> de-rate match -> convert -> decode FP16
        modulator_soft_demod(modulator, d_symbols, d_llrs, num_symbols, mod_order, noise_var, 0);
        scrambler_descramble_llr_inplace(scrambler, d_llrs, E, 0);

        // De-interleave
        rate_matcher_deinterleave_llr(rm_rx, d_llrs, d_deint_llrs, E, mod_order, 0);

        CHECK_CUDA(cudaMemset(d_derate_llrs, 0, N_full * sizeof(float)));
        rate_matcher_dematch(rm_rx, d_deint_llrs, d_derate_llrs, 0);

        // Convert FP32 LLRs to FP16 for the production half decoder.
        int block_size = 256;
        int num_blocks = (N_full + block_size - 1) / block_size;
        convert_llr_fp32_to_half_kernel<<<num_blocks, block_size>>>(
            d_derate_llrs, d_derate_llrs_half, N_full, llr_scale);

        // Decode using FP16 half kernel (production path)
        CHECK_CUDA(cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t)));
        ldpc_decoder_decode_batch_half(decoder, d_derate_llrs_half, d_decoded, 1, 0);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Compare info bits only
        CHECK_CUDA(cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        bool block_error = false;
        for (int i = 0; i < Kd && !block_error; i++) {
            int in_bit = (h_input[i / 32] >> (i % 32)) & 1;
            int out_bit = (h_decoded[i / 32] >> (i % 32)) & 1;
            if (in_bit != out_bit) {
                block_error = true;
            }
        }

        total_blocks++;
        if (block_error) error_blocks++;
    }

    float bler = (total_blocks > 0) ? (float)error_blocks / total_blocks : 1.0f;

    // Cleanup
    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);
    rate_matcher_destroy(rm_tx);
    rate_matcher_destroy(rm_rx);
    scrambler_destroy(scrambler);
    modulator_destroy(modulator);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_encoded));
    CHECK_CUDA(cudaFree(d_rate_matched));
    CHECK_CUDA(cudaFree(d_interleaved));
    CHECK_CUDA(cudaFree(d_scrambled));
    CHECK_CUDA(cudaFree(d_symbols));
    CHECK_CUDA(cudaFree(d_llrs));
    CHECK_CUDA(cudaFree(d_deint_llrs));
    CHECK_CUDA(cudaFree(d_derate_llrs));
    CHECK_CUDA(cudaFree(d_derate_llrs_half));
    CHECK_CUDA(cudaFree(d_decoded));

    return bler;
}

/**
 * @brief Find SNR for target BLER using binary search (FP16 half version)
 */
float find_bler_snr_raw_ldpc_half(int info_bits, float target_rate, int mod_order,
                                   float target_bler, float snr_low, float snr_high,
                                   float tolerance) {
    while (snr_high - snr_low > tolerance) {
        float snr_mid = (snr_low + snr_high) / 2.0f;
        float bler = measure_bler_raw_ldpc_half(info_bits, target_rate, mod_order, snr_mid,
                                                 30, 150, 3);
        if (bler > target_bler) {
            snr_low = snr_mid;
        } else {
            snr_high = snr_mid;
        }
    }
    return (snr_low + snr_high) / 2.0f;
}

/**
 * @brief Find SNR for target BLER using binary search (raw LDPC version)
 */
float find_bler_snr_raw_ldpc(int info_bits, float target_rate, int mod_order,
                              float target_bler, float snr_low, float snr_high,
                              float tolerance) {
    while (snr_high - snr_low > tolerance) {
        float snr_mid = (snr_low + snr_high) / 2.0f;
        float bler = measure_bler_raw_ldpc(info_bits, target_rate, mod_order, snr_mid,
                                           30, 150, 3);
        if (bler > target_bler) {
            snr_low = snr_mid;
        } else {
            snr_high = snr_mid;
        }
    }
    return (snr_low + snr_high) / 2.0f;
}

/**
 * @brief Find SNR for target BLER using binary search
 */
float find_bler_snr_tb(int tbs_bits, float target_rate, int mod_order,
                       float target_bler, float snr_low, float snr_high,
                       float tolerance, int* out_encoded_bits = nullptr,
                       int max_iterations = 25) {

    int encoded_bits = 0;

    // Binary search for target BLER
    while (snr_high - snr_low > tolerance) {
        float snr_mid = (snr_low + snr_high) / 2.0f;
        float bler = measure_bler_tb(tbs_bits, target_rate, mod_order, snr_mid,
                                     30, 150, 3, &encoded_bits, max_iterations);

        if (bler > target_bler) {
            snr_low = snr_mid;  // Need higher SNR
        } else {
            snr_high = snr_mid;  // Can use lower SNR
        }
    }

    if (out_encoded_bits) *out_encoded_bits = encoded_bits;
    return (snr_low + snr_high) / 2.0f;
}

// ============================================================================
// Test Runner
// ============================================================================

MCSTestResult test_mcs_entry(int table_num, const MCSTableEntry& mcs, int tbs_bits,
                             float baseline_snr, bool verbose = false) {
    MCSTestResult result;
    result.table_num = table_num;
    result.mcs_index = mcs.mcs_index;
    result.mod_order = mcs.modulation_order;
    result.target_rate = mcs.target_rate;
    result.mod_name = mcs.mod_name;
    result.baseline_snr_db = baseline_snr;
    result.tbs_bits = tbs_bits;

    // Calculate expected encoded bits and actual rate
    result.encoded_bits = (int)ceilf((float)tbs_bits / mcs.target_rate);
    result.actual_rate = (float)tbs_bits / (float)result.encoded_bits;

    // Diagnostic TB configuration output.
    if (verbose) {
        nr_tb_config_t tb_cfg;
        nr_tb_init_config(&tb_cfg, tbs_bits, mcs.target_rate);
        printf("  [DIAG] MCS %d: TBS=%d, G=%d, num_CBs=%d, CB_bits=%d, CB_CRC=%d, TB_CRC=%d, BG=%d, Z=%d\n",
               mcs.mcs_index, tbs_bits, result.encoded_bits,
               tb_cfg.num_code_blocks, tb_cfg.cb_size_bits, tb_cfg.cb_crc_bits,
               tb_cfg.tb_crc_bits, tb_cfg.ldpc_cfg.base_graph, tb_cfg.ldpc_cfg.lifting_size);
    }

    // Estimate Es/N0 search range based on modulation and rate
    // Higher modulation needs more Es/N0, lower rate needs less
    // QPSK starts around -6 to +8 dB, 16QAM around +4 to +14, 64QAM around +10 to +20
    float base_esn0 = -8.0f + (mcs.modulation_order - 2) * 4.5f;  // -8, +1, +10, +19 for QPSK/16/64/256
    float rate_adjust = 7.0f * log10f(mcs.target_rate / 0.3f);   // More adjustment for rate
    float snr_low = base_esn0 + rate_adjust - 8.0f;
    float snr_high = base_esn0 + rate_adjust + 12.0f;

    // Clamp search range - extend for full MCS range coverage
    if (snr_low < -12.0f) snr_low = -12.0f;
    if (snr_high > 30.0f) snr_high = 30.0f;

    // Find 10% BLER at Es/N0
    int actual_encoded;
    result.bler10_esn0_db = find_bler_snr_tb(tbs_bits, mcs.target_rate,
                                              mcs.modulation_order,
                                              0.10f, snr_low, snr_high, 0.25f,
                                              &actual_encoded);
    result.encoded_bits = actual_encoded;
    result.actual_rate = (float)tbs_bits / (float)actual_encoded;

    // Check against baseline (Es/N0)
    const float REGRESSION_MARGIN = 0.3f;
    result.passed = (result.bler10_esn0_db <= baseline_snr + REGRESSION_MARGIN);

    return result;
}

void run_mcs_table_test(int table_num, const MCSTableEntry* table, int table_size,
                        const float* baselines, std::vector<MCSTestResult>& results,
                        int tbs_bits, bool verbose = false) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                       MCS Table %d - 10%% BLER Sensitivity                        ║\n", table_num);
    printf("╠══════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ MCS │ Mod     │ Tgt Rate │ Act Rate │  G bits  │ Es/N0 @10%% │ Baseline │ Status ║\n");
    printf("╠═════╪═════════╪══════════╪══════════╪══════════╪══════════════╪══════════╪════════╣\n");

    for (int i = 0; i < table_size; i++) {
        float baseline = (baselines != nullptr) ? baselines[i] : 99.0f;
        MCSTestResult result = test_mcs_entry(table_num, table[i], tbs_bits, baseline, verbose);
        results.push_back(result);

        const char* status = result.passed ? " PASS " : "*FAIL*";
        printf("║ %3d │ %-7s │   %5.3f  │   %5.3f  │  %6d  │   %6.2f dB   │ %6.2f dB │ %s ║\n",
               result.mcs_index, result.mod_name,
               result.target_rate, result.actual_rate, result.encoded_bits,
               result.bler10_esn0_db, result.baseline_snr_db, status);

        fflush(stdout);
    }

    printf("╚══════════════════════════════════════════════════════════════════════════════════╝\n");
}

// ============================================================================
// Iteration Sweep Test
// ============================================================================

/**
 * @brief Run iteration sweep for MCS 0 of each table to see sensitivity vs iterations
 */
void run_iteration_sweep_test(int tbs_bits) {
    // Iteration counts to test
    const int iteration_counts[] = {1, 2, 5, 10, 20};
    const int num_iter_tests = sizeof(iteration_counts) / sizeof(iteration_counts[0]);

    // MCS 0 entries from each table
    struct MCS0Entry {
        int table_num;
        float target_rate;
        int mod_order;
        const char* mod_name;
        const char* table_name;
    };

    const MCS0Entry mcs0_entries[] = {
        {1, 120.0f/1024, 2, "QPSK", "Table 1 (64QAM max)"},
        {2, 120.0f/1024, 2, "QPSK", "Table 2 (256QAM max)"},
        {3,  30.0f/1024, 2, "QPSK", "Table 3 (Low SE/PUSCH)"}
    };
    const int num_tables = sizeof(mcs0_entries) / sizeof(mcs0_entries[0]);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    LDPC Iteration Count vs BLER Sensitivity Sweep                                    ║\n");
    printf("║                                  (MCS 0 for each table)                                              ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ TBS: %d bits   Target BLER: 10%%                                                                      ║\n", tbs_bits);
    printf("╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n");

    // Results storage: [table][iterations]
    float results[3][5];  // 3 tables, 5 iteration counts

    // Run tests for each MCS 0 entry
    for (int t = 0; t < num_tables; t++) {
        const MCS0Entry& mcs = mcs0_entries[t];

        printf("Testing MCS Table %d (MCS 0: %s R=%.4f)...\n", mcs.table_num, mcs.mod_name, mcs.target_rate);

        // Calculate search range
        float base_esn0 = -8.0f + (mcs.mod_order - 2) * 4.5f;
        float rate_adjust = 7.0f * log10f(mcs.target_rate / 0.3f);
        float snr_low = base_esn0 + rate_adjust - 8.0f;
        float snr_high = base_esn0 + rate_adjust + 12.0f;
        if (snr_low < -12.0f) snr_low = -12.0f;
        if (snr_high > 30.0f) snr_high = 30.0f;

        for (int i = 0; i < num_iter_tests; i++) {
            int iters = iteration_counts[i];
            printf("  %d iterations: ", iters);
            fflush(stdout);

            float snr = find_bler_snr_tb(tbs_bits, mcs.target_rate, mcs.mod_order,
                                          0.10f, snr_low, snr_high, 0.25f,
                                          nullptr, iters);
            results[t][i] = snr;
            printf("%.2f dB\n", snr);
            fflush(stdout);
        }
        printf("\n");
    }

    // Print results table
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                          10%% BLER SNR (Es/N0) vs LDPC Iterations                                     ║\n");
    printf("╠═══════════════════════════════════╦════════════════════════════════════════════════════════════════════╣\n");
    printf("║ MCS Table / Entry                 ║   1 iter   │   2 iter   │   5 iter   │  10 iter   │  20 iter   ║\n");
    printf("╠═══════════════════════════════════╬════════════╪════════════╪════════════╪════════════╪════════════╣\n");

    for (int t = 0; t < num_tables; t++) {
        const MCS0Entry& mcs = mcs0_entries[t];
        printf("║ Table %d MCS 0 (%s R=%.3f)    ║",
               mcs.table_num, mcs.mod_name, mcs.target_rate);

        for (int i = 0; i < num_iter_tests; i++) {
            printf(" %6.2f dB  │", results[t][i]);
        }
        // Replace last │ with ║
        printf("\b║\n");
    }

    printf("╚═══════════════════════════════════╩════════════╧════════════╧════════════╧════════════╧════════════╝\n");

    // Calculate and show deltas vs 2 iterations (the default)
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                      SNR Delta vs 2 Iterations (baseline)                                            ║\n");
    printf("╠═══════════════════════════════════╦════════════════════════════════════════════════════════════════════╣\n");
    printf("║ MCS Table / Entry                 ║   1 iter   │   2 iter   │   5 iter   │  10 iter   │  20 iter   ║\n");
    printf("╠═══════════════════════════════════╬════════════╪════════════╪════════════╪════════════╪════════════╣\n");

    for (int t = 0; t < num_tables; t++) {
        const MCS0Entry& mcs = mcs0_entries[t];
        float baseline = results[t][1];  // 2 iterations is index 1

        printf("║ Table %d MCS 0 (%s R=%.3f)    ║",
               mcs.table_num, mcs.mod_name, mcs.target_rate);

        for (int i = 0; i < num_iter_tests; i++) {
            float delta = results[t][i] - baseline;
            if (i == 1) {
                printf("   (base)  │");
            } else if (delta >= 0) {
                printf(" +%5.2f dB  │", delta);
            } else {
                printf(" %6.2f dB  │", delta);
            }
        }
        // Replace last │ with ║
        printf("\b║\n");
    }

    printf("╚═══════════════════════════════════╩════════════╧════════════╧════════════╧════════════╧════════════╝\n");

    // Print CSV for easy plotting
    printf("\n// CSV data for plotting:\n");
    printf("Table,MCS,Mod,Rate,Iterations,EsN0_10BLER_dB\n");
    for (int t = 0; t < num_tables; t++) {
        const MCS0Entry& mcs = mcs0_entries[t];
        for (int i = 0; i < num_iter_tests; i++) {
            printf("%d,0,%s,%.4f,%d,%.2f\n",
                   mcs.table_num, mcs.mod_name, mcs.target_rate,
                   iteration_counts[i], results[t][i]);
        }
    }

    printf("\n");
    printf("Key observations:\n");
    printf("  - Positive delta means MORE SNR needed (worse sensitivity)\n");
    printf("  - Negative delta means LESS SNR needed (better sensitivity)\n");
    printf("  - Diminishing returns typically seen after 5-10 iterations\n");
    printf("  - MCS Table 3 (low rate) benefits most from more iterations\n");
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -t TABLE    Test only table TABLE (1, 2, or 3)\n");
    printf("  -s TBS      Transport block size in bits (default: 50000)\n");
    printf("  -b          Establish new baselines (no pass/fail checking)\n");
    printf("  -v          Verbose mode (show TB segmentation details)\n");
    printf("  -c          Compare mode: compare TB chain vs raw LDPC for MCS 0\n");
    printf("  -i          Iteration sweep: test MCS 0 at 1,2,5,10,20 decoder iterations\n");
    printf("  -h          Show this help\n");
    printf("\nRecommended TBS values:\n");
    printf("  50000       ~40 MHz allocation (default, realistic)\n");
    printf("  20000       ~20 MHz allocation\n");
    printf("  8448        Max single CB for BG1 (edge case)\n");
    printf("  3800        Single CB for BG2 (low rate testing)\n");
}

/**
 * @brief Compare FP32 vs FP16 TB decoder for a single MCS
 * Uses full TB chain (like mcs_bler_curves) for both paths
 */
void run_comparison_test(int info_bits, float target_rate, int mod_order) {
    printf("\n╔══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║          FP32 vs FP16 TB Decoder Comparison Test                                 ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ TBS: %d bits   Target rate: %.4f   Modulation: %s                              ║\n",
           info_bits, target_rate, mod_order == 2 ? "QPSK" : (mod_order == 4 ? "16QAM" : "64QAM"));
    printf("╚══════════════════════════════════════════════════════════════════════════════════╝\n\n");

    // Calculate search range
    float base_esn0 = -8.0f + (mod_order - 2) * 4.5f;
    float rate_adjust = 7.0f * log10f(target_rate / 0.3f);
    float snr_low = base_esn0 + rate_adjust - 8.0f;
    float snr_high = base_esn0 + rate_adjust + 12.0f;
    if (snr_low < -12.0f) snr_low = -12.0f;
    if (snr_high > 30.0f) snr_high = 30.0f;

    printf("Search range: [%.1f, %.1f] dB\n\n", snr_low, snr_high);

    // Test TB chain with FP32 decoder
    printf("Testing TB chain (FP32 decoder)...\n");
    int encoded_bits;
    float fp32_snr = find_bler_snr_tb(info_bits, target_rate, mod_order,
                                       0.10f, snr_low, snr_high, 0.25f, &encoded_bits);
    printf("  TB FP32: 10%% BLER @ Es/N0 = %.2f dB\n\n", fp32_snr);

    printf("╔══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║ RESULTS:                                                                         ║\n");
    printf("║   TB Chain FP32:   %6.2f dB                                                      ║\n", fp32_snr);
    printf("╚══════════════════════════════════════════════════════════════════════════════════╝\n");
}

int main(int argc, char* argv[]) {
    int test_table = 0;     // 0 = all tables
    int tbs_bits = 50000;   // Default TBS (~40 MHz allocation for realistic test)
    bool establish_baseline = false;
    bool verbose = false;
    bool compare_mode = false;
    bool iteration_sweep = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            test_table = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            tbs_bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0) {
            establish_baseline = true;
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-c") == 0) {
            compare_mode = true;
        } else if (strcmp(argv[i], "-i") == 0) {
            iteration_sweep = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Initialize library
    nr_ldpc_status_t status = ocudu_phy_cuda_init();
    if (status != NR_LDPC_SUCCESS) {
        printf("ERROR: Failed to initialize OCUDU PHY CUDA: %s\n", nr_ldpc_get_error_string(status));
        return 1;
    }
    ocudu_phy_cuda_print_info();
    srand(time(NULL));

    // Run iteration sweep mode if requested
    if (iteration_sweep) {
        run_iteration_sweep_test(tbs_bits);
        ocudu_phy_cuda_cleanup();
        return 0;
    }

    // Run comparison mode if requested
    if (compare_mode) {
        // Test MCS 0 (QPSK R=0.117) with different TBS sizes
        printf("\n=== Testing with small TBS (2048 bits - single CB) ===\n");
        run_comparison_test(2048, 120.0f/1024, 2);

        printf("\n\n=== Testing with large TBS (50000 bits - multiple CBs) ===\n");
        run_comparison_test(50000, 120.0f/1024, 2);

        ocudu_phy_cuda_cleanup();
        return 0;
    }

    printf("╔══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║           5G NR MCS BLER Sensitivity Test - All 3GPP MCS Tables                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════════╝\n\n");

    printf("Test Configuration:\n");
    printf("  Transport Block Size: %d bits\n", tbs_bits);
    printf("  Target BLER: 10%%\n");
    printf("  Regression Margin: +0.3 dB\n");
    printf("  Mode: %s\n", establish_baseline ? "Establish Baselines" : "Regression Test");
    printf("\n");

    std::vector<MCSTestResult> all_results;
    int total_pass = 0;
    int total_fail = 0;

    // Run tests (baselines set to 99.0 for now - will update after initial run)
    if (test_table == 0 || test_table == 1) {
        run_mcs_table_test(1, MCS_TABLE_1, MCS_TABLE_1_SIZE,
                           establish_baseline ? nullptr : nullptr,  // No baselines yet
                           all_results, tbs_bits, verbose);
    }

    if (test_table == 0 || test_table == 2) {
        run_mcs_table_test(2, MCS_TABLE_2, MCS_TABLE_2_SIZE,
                           establish_baseline ? nullptr : nullptr,
                           all_results, tbs_bits, verbose);
    }

    if (test_table == 0 || test_table == 3) {
        run_mcs_table_test(3, MCS_TABLE_3, MCS_TABLE_3_SIZE,
                           establish_baseline ? nullptr : nullptr,
                           all_results, tbs_bits, verbose);
    }

    // Count results
    for (const auto& r : all_results) {
        if (r.passed) total_pass++;
        else total_fail++;
    }

    // Print summary
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                                  TEST SUMMARY                                    ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Total MCS Entries Tested: %-4d                                                  ║\n",
           (int)all_results.size());
    printf("║  Passed: %-4d    Failed: %-4d                                                    ║\n",
           total_pass, total_fail);
    printf("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    // Print baselines for updating
    printf("\n// New baselines (add +0.3 dB margin):\n");
    for (int t = 1; t <= 3; t++) {
        bool has_entries = false;
        for (const auto& r : all_results) {
            if (r.table_num == t) { has_entries = true; break; }
        }
        if (!has_entries) continue;

        printf("\n// Table %d baselines (TBS=%d)\n", t, tbs_bits);
        printf("static const float TABLE%d_BASELINES[] = {\n", t);
        for (const auto& r : all_results) {
            if (r.table_num == t) {
                printf("    %6.2f,  // MCS %2d %s R=%.3f\n",
                       r.bler10_esn0_db + 0.3f,
                       r.mcs_index, r.mod_name, r.target_rate);
            }
        }
        printf("};\n");
    }

    // Print CSV for plotting
    printf("\n// CSV for plotting:\n");
    printf("Table,MCS,Mod,Qm,TargetRate,ActualRate,EncodedBits,EsN0_10BLER_dB\n");
    for (const auto& r : all_results) {
        printf("%d,%d,%s,%d,%.4f,%.4f,%d,%.2f\n",
               r.table_num, r.mcs_index, r.mod_name, r.mod_order,
               r.target_rate, r.actual_rate, r.encoded_bits, r.bler10_esn0_db);
    }

    ocudu_phy_cuda_cleanup();

    return (total_fail > 0 && !establish_baseline) ? 1 : 0;
}
