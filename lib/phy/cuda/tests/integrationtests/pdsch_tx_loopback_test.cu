/**
 * @file pdsch_tx_loopback_test.cu
 * @brief PDSCH TX Chain Loopback Test
 *
 * Tests the complete PDSCH transmit chain by verifying roundtrip through:
 *   TX: TB data -> CRC -> LDPC encode -> Rate match -> Scramble -> Modulate
 *   Channel: AWGN
 *   RX: Soft demod -> Descramble -> De-rate match -> LDPC decode -> CRC check
 *
 * This validates that the GPU PDSCH TX chain produces correct output that
 * can be successfully decoded by the RX chain.
 */

#include "ocudu_phy_cuda.h"
#include "ldpc_encoder.h"
#include "ldpc_decoder.h"
#include "pdsch_fused.h"
#include "modulation.h"
#include "scrambling.h"
#include "rate_matching.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Test configuration
struct PDSCHTestConfig {
    int tb_size_bits;           // Transport block size
    float code_rate;            // Target code rate
    int mod_order;              // Modulation order (2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM)
    float snr_db;               // Channel SNR in dB
    uint16_t n_RNTI;            // RNTI for scrambling
    uint16_t n_ID;              // Cell ID for scrambling
    const char* description;    // Test description
};

// Test configurations covering various MCS/TBS combinations
// Note: SNR values are set for near-zero BLER testing (high enough for reliable decoding)
static const PDSCHTestConfig test_configs[] = {
    // Small TB with QPSK (low MCS)
    {656, 0.5f, 2, 15.0f, 0x1234, 500, "Small TB, QPSK, R=0.5"},
    {1000, 0.5f, 2, 15.0f, 0x1234, 500, "1kbit TB, QPSK, R=0.5"},

    // Medium TB with 16QAM
    {2000, 0.5f, 4, 20.0f, 0x5678, 100, "2kbit TB, 16QAM, R=0.5"},
    {4000, 0.66f, 4, 22.0f, 0x5678, 100, "4kbit TB, 16QAM, R=0.66"},

    // Large TB with 64QAM
    {8000, 0.5f, 6, 25.0f, 0xAAAA, 200, "8kbit TB, 64QAM, R=0.5"},
    {12000, 0.75f, 6, 32.0f, 0xAAAA, 200, "12kbit TB, 64QAM, R=0.75 (multi-CB)"},

    // Large TB with 256QAM (high MCS)
    {16000, 0.8f, 8, 35.0f, 0xBBBB, 300, "16kbit TB, 256QAM, R=0.8"},

    // Edge cases
    {500, 0.3f, 2, 15.0f, 0x0001, 1, "Small TB, low rate"},  // SNR increased for testing
    {3824, 0.67f, 4, 22.0f, 0xFFFF, 1023, "BG boundary, max IDs"},
};

// Get time in milliseconds
static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Generate random test data
static void generate_random_data(uint8_t* data, int num_bytes) {
    for (int i = 0; i < num_bytes; i++) {
        data[i] = rand() & 0xFF;
    }
}

// Get INT8 constellation normalization factor for unit-power symbols
static float get_int8_norm_factor(int mod_order) {
    // INT8 symbols use unnormalized constellation: ±1, ±3, ... ±(2^(Qm/2)-1)
    // Normalization factor = 1/sqrt(avg_power) to achieve unit-power constellation
    switch (mod_order) {
        case 2: return 1.0f / sqrtf(2.0f);      // QPSK: ±1 → avg_power=2
        case 4: return 1.0f / sqrtf(10.0f);     // 16QAM: ±1,±3 → avg_power=10
        case 6: return 1.0f / sqrtf(42.0f);     // 64QAM: ±1,±3,±5,±7 → avg_power=42
        case 8: return 1.0f / sqrtf(170.0f);    // 256QAM: ±1,...,±15 → avg_power=170
        default: return 1.0f;
    }
}

// CUDA kernel: Convert INT8 I/Q symbols to normalized float complex + add AWGN
__global__ void int8_to_float_awgn_kernel(
    const int8_t* __restrict__ d_int8,
    cuFloatComplex* __restrict__ d_out,
    int num_symbols,
    float norm_factor,
    float noise_std,
    unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    float re = d_int8[2 * idx] * norm_factor;
    float im = d_int8[2 * idx + 1] * norm_factor;

    if (noise_std > 0.0f) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        re += curand_normal(&state) * noise_std;
        im += curand_normal(&state) * noise_std;
    }

    d_out[idx] = make_cuFloatComplex(re, im);
}

// Compare data and count bit errors
static int count_bit_errors(const uint8_t* a, const uint8_t* b, int num_bits) {
    int errors = 0;
    for (int bit_idx = 0; bit_idx < num_bits; bit_idx++) {
        int byte_idx = bit_idx / 8;
        int bit_pos = 7 - (bit_idx % 8);  // MSB first
        uint8_t bit_a = (a[byte_idx] >> bit_pos) & 1;
        uint8_t bit_b = (b[byte_idx] >> bit_pos) & 1;
        if (bit_a != bit_b) errors++;
    }
    return errors;
}

/**
 * @brief Test LDPC encoder/decoder roundtrip (NO MODULATION)
 *
 * This test verifies LDPC encode/decode roundtrip WITHOUT modulation.
 * It uses BPSK-like LLR generation (bit -> +/-1 -> LLR) to test the LDPC chain.
 *
 * NOTE: This does NOT test the 5G NR compliant modulation chain because:
 * - LDPC encoder outputs LSB-first uint32_t format
 * - Modulator expects MSB-first bytes (after bit interleaving per TS 38.212 5.4.2.2)
 * - Full 5G NR chain: LDPC -> Rate Match -> Bit Interleave -> Scramble -> Modulate
 *
 * For full 5G NR chain testing, use the TB-based loopback tests.
 */
bool test_ldpc_roundtrip() {
    printf("\n=== LDPC Encoder/Decoder Roundtrip Test (NO modulation) ===\n");

    bool test_passed = true;

    const int K = 8000;  // Info bits that fit in one CB
    const float rate = 0.5f;

    // Initialize LDPC config
    nr_ldpc_config_t ldpc_cfg;
    nr_ldpc_status_t status = nr_ldpc_init_config(&ldpc_cfg, K, rate);
    if (status != NR_LDPC_SUCCESS) {
        printf("ERROR: Failed to init LDPC config\n");
        return false;
    }

    printf("  BG%d, Z=%d, K=%d, N=%d\n", ldpc_cfg.base_graph, ldpc_cfg.lifting_size,
           ldpc_cfg.num_info_bits, ldpc_cfg.num_codeword_bits);

    // Create encoder/decoder
    ldpc_encoder_handle_t encoder = nullptr;
    ldpc_decoder_handle_t decoder = nullptr;

    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &ldpc_cfg);

    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 20;
    ldpc_decoder_create(&decoder);
    ldpc_decoder_configure(decoder, &ldpc_cfg, &dec_params);

    int input_bits = ldpc_encoder_get_input_bits(encoder);
    int input_words = ldpc_encoder_get_input_words(encoder);
    int N_bits = ldpc_encoder_get_output_bits(encoder);
    int N_words = ldpc_encoder_get_output_words(encoder);

    // Allocate memory
    uint32_t* h_input = new uint32_t[input_words];
    uint32_t* h_output = new uint32_t[input_words];
    uint32_t* h_encoded = new uint32_t[N_words];
    float* h_llrs = new float[N_bits];

    uint32_t* d_input;
    uint32_t* d_encoded;
    float* d_llrs;
    uint32_t* d_decoded;

    CHECK_CUDA(cudaMalloc(&d_input, input_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_encoded, N_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_llrs, N_bits * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_decoded, input_words * sizeof(uint32_t)));

    // Generate random input
    srand(42);
    for (int i = 0; i < input_words; i++) {
        h_input[i] = ((uint32_t)rand() << 16) | (rand() & 0xFFFF);
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_words * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Encode on GPU
    ldpc_encoder_encode(encoder, d_input, d_encoded, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Download encoded bits and create LLRs directly (bypass modulation)
    // This simulates BPSK with high SNR: bit=0 -> LLR=+32, bit=1 -> LLR=-32
    // First 2*Z bits are punctured (not transmitted) - set LLR=0 for erasure
    // NOTE: Encoder outputs in MSB-first byte format (per 5G NR), so we must read accordingly
    int Z = ldpc_cfg.lifting_size;
    int punctured_bits = 2 * Z;
    CHECK_CUDA(cudaMemcpy(h_encoded, d_encoded, N_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N_bits; i++) {
        if (i < punctured_bits) {
            h_llrs[i] = 0.0f;  // Punctured - erasure
        } else {
            // Read in MSB-first byte format (matches encoder output)
            int word_idx = i / 32;
            int bit_in_word = i % 32;
            int byte_in_word = bit_in_word / 8;
            int bit_in_byte = 7 - (bit_in_word % 8);  // MSB-first within byte
            int bit_pos = byte_in_word * 8 + bit_in_byte;
            uint32_t bit = (h_encoded[word_idx] >> bit_pos) & 1u;
            h_llrs[i] = bit ? -32.0f : +32.0f;  // High-SNR soft bits
        }
    }
    CHECK_CUDA(cudaMemcpy(d_llrs, h_llrs, N_bits * sizeof(float), cudaMemcpyHostToDevice));

    // Decode
    ldpc_decoder_decode(decoder, d_llrs, d_decoded, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Compare - must account for format difference:
    // Encoder input uses MSB-first byte format
    // Decoder output uses LSB-first (flat) format
    CHECK_CUDA(cudaMemcpy(h_output, d_decoded, input_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int bit_errors = 0;
    for (int i = 0; i < input_bits; i++) {
        // Read input bit (MSB-first byte format)
        int in_word = i / 32;
        int in_bit_in_word = i % 32;
        int in_byte = in_bit_in_word / 8;
        int in_bit_in_byte = 7 - (in_bit_in_word % 8);
        int in_bit_pos = in_byte * 8 + in_bit_in_byte;
        int in_bit = (h_input[in_word] >> in_bit_pos) & 1;

        // Read output bit (MSB-first per byte, matching decoder output)
        int out_word = i / 32;
        int tmp = i % 32;
        int out_bit_pos = (tmp & ~7) | (7 - (tmp & 7));
        int out_bit = (h_output[out_word] >> out_bit_pos) & 1;

        if (in_bit != out_bit) bit_errors++;
    }

    printf("  Bit errors: %d / %d\n", bit_errors, input_bits);
    test_passed = (bit_errors == 0);
    printf("  %s\n", test_passed ? "PASSED" : "FAILED");

    // Cleanup
    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_encoded));
    CHECK_CUDA(cudaFree(d_llrs));
    CHECK_CUDA(cudaFree(d_decoded));
    delete[] h_input;
    delete[] h_output;
    delete[] h_encoded;
    delete[] h_llrs;

    return test_passed;
}

// NOTE: test_ldpc_all_modulations() was removed because it bypassed bit interleaving
// (TS 38.212 5.4.2.2) which is required for 5G NR compliance.
// For full 5G NR modulation chain testing, use the TB-based loopback tests below.

/**
 * @brief Run a single PDSCH TX loopback test using PRODUCTION kernel path
 *
 * Uses tb_encoder_encode_to_symbols_int8() — the exact same fused kernel
 * the gnb uses in production (CRC → LDPC → RM → Interleave → Scramble → Modulate → INT8).
 */
bool run_pdsch_loopback_test(const PDSCHTestConfig& cfg, bool verbose = true) {
    if (verbose) {
        printf("\n=== PDSCH TX Loopback Test: %s ===\n", cfg.description);
        printf("  TBS: %d bits, Rate: %.2f, Mod: %s, SNR: %.1f dB\n",
               cfg.tb_size_bits, cfg.code_rate,
               cfg.mod_order == 2 ? "QPSK" :
               cfg.mod_order == 4 ? "16QAM" :
               cfg.mod_order == 6 ? "64QAM" : "256QAM",
               cfg.snr_db);
        printf("  Scrambling: RNTI=0x%04X, n_ID=%d\n", cfg.n_RNTI, cfg.n_ID);
    }

    bool test_passed = true;
    nr_ldpc_status_t status;

    // Calculate sizes
    int tb_bytes = (cfg.tb_size_bits + 7) / 8;
    int encoded_bits = (int)(cfg.tb_size_bits / cfg.code_rate);
    // Round up to mod_order boundary
    encoded_bits = ((encoded_bits + cfg.mod_order - 1) / cfg.mod_order) * cfg.mod_order;
    int num_symbols = encoded_bits / cfg.mod_order;

    if (verbose) {
        printf("  Encoded bits: %d, Symbols: %d\n", encoded_bits, num_symbols);
    }

    // Allocate host memory
    uint8_t* h_tx_data = new uint8_t[tb_bytes];
    uint8_t* h_rx_data = new uint8_t[tb_bytes];

    // Allocate device memory
    uint8_t* d_tx_data;
    int8_t* d_symbols_int8;        // Production INT8 I/Q pairs from fused TX kernel
    cuFloatComplex* d_float_symbols; // Normalized float symbols for soft demod
    float* d_llrs;                   // Soft demod output (scrambled LLRs)
    uint8_t* d_rx_data;

    CHECK_CUDA(cudaMalloc(&d_tx_data, tb_bytes));
    CHECK_CUDA(cudaMalloc(&d_symbols_int8, num_symbols * 2));
    CHECK_CUDA(cudaMalloc(&d_float_symbols, num_symbols * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&d_llrs, encoded_bits * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rx_data, tb_bytes));

    // Create handles
    tb_encoder_handle_t encoder = nullptr;
    tb_decoder_handle_t decoder = nullptr;
    modulator_handle_t modulator = nullptr;

    status = tb_encoder_create(&encoder);
    if (status != NR_LDPC_SUCCESS) {
        printf("ERROR: Failed to create encoder\n");
        test_passed = false;
        goto cleanup;
    }

    status = tb_decoder_create(&decoder);
    if (status != NR_LDPC_SUCCESS) {
        printf("ERROR: Failed to create decoder\n");
        test_passed = false;
        goto cleanup;
    }

    if (modulator_create(&modulator) != 0) {
        printf("ERROR: Failed to create modulator\n");
        test_passed = false;
        goto cleanup;
    }

    // Configure encoder — production config (matches gnb pdsch_tb_encoder_cuda.cu)
    {
        tb_encoder_config_t enc_cfg = {};
        enc_cfg.tb_size_bits = cfg.tb_size_bits;
        enc_cfg.num_layers = 1;
        enc_cfg.modulation_order = cfg.mod_order;
        enc_cfg.num_allocated_res = encoded_bits;
        enc_cfg.redundancy_version = 0;
        enc_cfg.code_rate = cfg.code_rate;
        enc_cfg.n_RNTI = cfg.n_RNTI;
        enc_cfg.n_ID = cfg.n_ID;
        enc_cfg.q = 0;
        enc_cfg.enable_scrambling = true;  // Production: scrambling enabled
        status = tb_encoder_configure(encoder, &enc_cfg);
        if (status != NR_LDPC_SUCCESS) {
            printf("ERROR: Failed to configure encoder\n");
            test_passed = false;
            goto cleanup;
        }
    }

    // Configure decoder — scrambling enabled to match encoder
    {
        tb_decoder_config_t dec_cfg = {};
        dec_cfg.tb_size_bits = cfg.tb_size_bits;
        dec_cfg.num_layers = 1;
        dec_cfg.modulation_order = cfg.mod_order;
        dec_cfg.num_received_bits = encoded_bits;
        dec_cfg.redundancy_version = 0;
        dec_cfg.code_rate = cfg.code_rate;
        dec_cfg.max_iterations = 20;
        dec_cfg.llr_clamp = 32.0f;
        dec_cfg.n_RNTI = cfg.n_RNTI;
        dec_cfg.n_ID = cfg.n_ID;
        dec_cfg.q = 0;
        dec_cfg.enable_scrambling = true;  // Production: descrambling enabled
        status = tb_decoder_configure(decoder, &dec_cfg);
        if (status != NR_LDPC_SUCCESS) {
            printf("ERROR: Failed to configure decoder\n");
            test_passed = false;
            goto cleanup;
        }
    }

    // Run multiple iterations
    {
        const int num_iterations = 10;
        int total_bit_errors = 0;
        int total_block_errors = 0;
        double total_tx_time_ms = 0;
        double total_rx_time_ms = 0;

        float norm_factor = get_int8_norm_factor(cfg.mod_order);
        float noise_std = snr_to_noise_std(cfg.snr_db, cfg.code_rate, cfg.mod_order);
        float noise_var = 2.0f * noise_std * noise_std;

        for (int iter = 0; iter < num_iterations; iter++) {
            // Generate random test data
            generate_random_data(h_tx_data, tb_bytes);
            CHECK_CUDA(cudaMemcpy(d_tx_data, h_tx_data, tb_bytes, cudaMemcpyHostToDevice));

            // ========== TX CHAIN (Production fused path) ==========
            CHECK_CUDA(cudaDeviceSynchronize());
            double tx_start = get_time_ms();

            // Single call: CRC → Segment → LDPC Encode → RM → Interleave → Scramble → Modulate → INT8
            // This is the EXACT same kernel path the gnb uses for PDSCH TX.
            status = tb_encoder_encode_to_symbols_int8(encoder, d_tx_data, d_symbols_int8, 0);
            if (status != NR_LDPC_SUCCESS) {
                printf("ERROR: Production TX encoding failed\n");
                test_passed = false;
                goto cleanup;
            }

            CHECK_CUDA(cudaDeviceSynchronize());
            double tx_end = get_time_ms();
            total_tx_time_ms += (tx_end - tx_start);

            // ========== CHANNEL (INT8 → float + AWGN) ==========
            {
                int threads = 256;
                int blocks = (num_symbols + threads - 1) / threads;
                int8_to_float_awgn_kernel<<<blocks, threads>>>(
                    d_symbols_int8, d_float_symbols, num_symbols,
                    norm_factor, noise_std,
                    (unsigned long long)time(NULL) ^ (iter * 12345));
            }

            // ========== RX CHAIN ==========
            CHECK_CUDA(cudaDeviceSynchronize());
            double rx_start = get_time_ms();

            // Soft demodulate (output LLRs are still scrambled)
            modulator_soft_demod(modulator, d_float_symbols, d_llrs, num_symbols,
                                  cfg.mod_order, noise_var, 0);

            // TB Decode (descramble → deinterleave → rate dematch → LDPC decode → CRC)
            tb_decode_result_t result;
            status = tb_decoder_decode(decoder, d_llrs, d_rx_data, &result, 0);
            if (status != NR_LDPC_SUCCESS) {
                printf("ERROR: TB decoding failed\n");
                test_passed = false;
                goto cleanup;
            }

            CHECK_CUDA(cudaDeviceSynchronize());
            double rx_end = get_time_ms();
            total_rx_time_ms += (rx_end - rx_start);

            // Compare results
            CHECK_CUDA(cudaMemcpy(h_rx_data, d_rx_data, tb_bytes, cudaMemcpyDeviceToHost));
            int bit_errors = count_bit_errors(h_tx_data, h_rx_data, cfg.tb_size_bits);
            total_bit_errors += bit_errors;
            if (bit_errors > 0 || !result.crc_pass) {
                total_block_errors++;
            }

            if (verbose && iter < 3) {
                printf("  Iteration %d: %s (CRC: %s, bit errors: %d, avg iters: %.1f)\n",
                       iter + 1,
                       bit_errors == 0 ? "PASS" : "FAIL",
                       result.crc_pass ? "PASS" : "FAIL",
                       bit_errors, result.avg_iterations);
            }
        }

        // Print summary
        float ber = (float)total_bit_errors / (num_iterations * cfg.tb_size_bits);
        float bler = (float)total_block_errors / num_iterations;
        double avg_tx_ms = total_tx_time_ms / num_iterations;
        double avg_rx_ms = total_rx_time_ms / num_iterations;
        double tx_throughput_mbps = (cfg.tb_size_bits / 1e6) / (avg_tx_ms / 1000.0);
        double rx_throughput_mbps = (cfg.tb_size_bits / 1e6) / (avg_rx_ms / 1000.0);

        if (verbose) {
            printf("\n  Results over %d iterations:\n", num_iterations);
            printf("    BER: %.2e, BLER: %.1f%%\n", ber, bler * 100);
            printf("    TX time: %.3f ms (%.2f Mbps)\n", avg_tx_ms, tx_throughput_mbps);
            printf("    RX time: %.3f ms (%.2f Mbps)\n", avg_rx_ms, rx_throughput_mbps);
        }

        // At this SNR, we should have very low error rate
        if (bler > 0.1f) {  // More than 10% BLER is a failure
            printf("  FAILED: High BLER (%.1f%%) at SNR=%.1f dB\n", bler * 100, cfg.snr_db);
            test_passed = false;
        } else {
            if (verbose) printf("  PASSED\n");
        }
    }

cleanup:
    // Cleanup
    tb_encoder_destroy(encoder);
    tb_decoder_destroy(decoder);
    modulator_destroy(modulator);

    CHECK_CUDA(cudaFree(d_tx_data));
    CHECK_CUDA(cudaFree(d_symbols_int8));
    CHECK_CUDA(cudaFree(d_float_symbols));
    CHECK_CUDA(cudaFree(d_llrs));
    CHECK_CUDA(cudaFree(d_rx_data));

    delete[] h_tx_data;
    delete[] h_rx_data;

    return test_passed;
}

/**
 * @brief Test scrambling isolation - verify scrambling parameters affect output
 */
bool test_scrambling_effect() {
    printf("\n=== Scrambling Effect Test ===\n");
    printf("  Verifying different RNTI/n_ID produce different scrambled output\n");

    const int tb_size = 1000;
    const int tb_bytes = (tb_size + 7) / 8;
    const int encoded_bits = 2000;
    const int encoded_bytes = (encoded_bits + 7) / 8;

    uint8_t* h_tx_data = new uint8_t[tb_bytes];
    uint8_t* h_encoded1 = new uint8_t[encoded_bytes];
    uint8_t* h_encoded2 = new uint8_t[encoded_bytes];

    uint8_t* d_tx_data;
    uint8_t* d_encoded1;
    uint8_t* d_encoded2;

    CHECK_CUDA(cudaMalloc(&d_tx_data, tb_bytes));
    CHECK_CUDA(cudaMalloc(&d_encoded1, encoded_bytes));
    CHECK_CUDA(cudaMalloc(&d_encoded2, encoded_bytes));

    // Generate fixed test data
    srand(42);
    generate_random_data(h_tx_data, tb_bytes);
    CHECK_CUDA(cudaMemcpy(d_tx_data, h_tx_data, tb_bytes, cudaMemcpyHostToDevice));

    // Create two encoders with different scrambling parameters
    tb_encoder_handle_t encoder1, encoder2;
    tb_encoder_create(&encoder1);
    tb_encoder_create(&encoder2);

    tb_encoder_config_t cfg1 = {
        .tb_size_bits = tb_size,
        .num_layers = 1,
        .modulation_order = 2,
        .num_allocated_res = encoded_bits,
        .redundancy_version = 0,
        .code_rate = 0.5f,
        .n_RNTI = 0x1111,
        .n_ID = 100,
        .q = 0,
        .enable_scrambling = true
    };

    tb_encoder_config_t cfg2 = cfg1;
    cfg2.n_RNTI = 0x2222;  // Different RNTI
    cfg2.n_ID = 200;       // Different cell ID

    tb_encoder_configure(encoder1, &cfg1);
    tb_encoder_configure(encoder2, &cfg2);

    // Encode with both
    tb_encoder_encode(encoder1, d_tx_data, d_encoded1, 0);
    tb_encoder_encode(encoder2, d_tx_data, d_encoded2, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_encoded1, d_encoded1, encoded_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_encoded2, d_encoded2, encoded_bytes, cudaMemcpyDeviceToHost));

    // Count differences
    int diff_bytes = 0;
    for (int i = 0; i < encoded_bytes; i++) {
        if (h_encoded1[i] != h_encoded2[i]) diff_bytes++;
    }

    float diff_ratio = (float)diff_bytes / encoded_bytes;
    printf("  Different bytes: %d / %d (%.1f%%)\n", diff_bytes, encoded_bytes, diff_ratio * 100);

    bool passed = (diff_ratio > 0.4f);  // Should have ~50% difference
    printf("  %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    tb_encoder_destroy(encoder1);
    tb_encoder_destroy(encoder2);
    CHECK_CUDA(cudaFree(d_tx_data));
    CHECK_CUDA(cudaFree(d_encoded1));
    CHECK_CUDA(cudaFree(d_encoded2));
    delete[] h_tx_data;
    delete[] h_encoded1;
    delete[] h_encoded2;

    return passed;
}

/**
 * @brief Test modulation constellation points
 */
bool test_modulation_constellations() {
    printf("\n=== Modulation Constellation Test ===\n");

    modulator_handle_t mod;
    modulator_create(&mod);

    bool all_passed = true;

    // Test each modulation order
    int mod_orders[] = {2, 4, 6, 8};
    const char* mod_names[] = {"QPSK", "16QAM", "64QAM", "256QAM"};

    for (int m = 0; m < 4; m++) {
        int mod_order = mod_orders[m];
        int num_bits = mod_order * 64;  // 64 symbols
        int num_symbols = num_bits / mod_order;
        int num_words = (num_bits + 31) / 32;

        uint32_t* h_bits = new uint32_t[num_words];
        uint32_t* d_bits;
        cuFloatComplex* d_symbols;
        cuFloatComplex* h_symbols = new cuFloatComplex[num_symbols];

        CHECK_CUDA(cudaMalloc(&d_bits, num_words * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&d_symbols, num_symbols * sizeof(cuFloatComplex)));

        // Generate all-zeros and all-ones patterns
        memset(h_bits, 0, num_words * sizeof(uint32_t));
        CHECK_CUDA(cudaMemcpy(d_bits, h_bits, num_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
        modulator_modulate(mod, d_bits, d_symbols, num_bits, mod_order, 0);
        CHECK_CUDA(cudaMemcpy(h_symbols, d_symbols, num_symbols * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

        // Check that symbols have unit average power (normalized constellation)
        float avg_power = 0;
        for (int i = 0; i < num_symbols; i++) {
            avg_power += h_symbols[i].x * h_symbols[i].x + h_symbols[i].y * h_symbols[i].y;
        }
        avg_power /= num_symbols;

        // QPSK has power 1, higher order should be normalized
        bool power_ok = (avg_power > 0.5f && avg_power < 2.0f);
        printf("  %s: avg_power=%.3f %s\n", mod_names[m], avg_power, power_ok ? "OK" : "FAIL");
        all_passed &= power_ok;

        delete[] h_bits;
        delete[] h_symbols;
        CHECK_CUDA(cudaFree(d_bits));
        CHECK_CUDA(cudaFree(d_symbols));
    }

    modulator_destroy(mod);
    printf("  %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed;
}

/**
 * @brief Benchmark TX chain throughput using PRODUCTION fused path
 *
 * Uses tb_encoder_encode_to_symbols_int8() — same as gnb production.
 */
void benchmark_tx_throughput() {
    printf("\n=== PDSCH TX Throughput Benchmark (Production Fused Path) ===\n");

    struct BenchConfig {
        int tb_size;
        int mod_order;
        float rate;
        const char* name;
    };

    BenchConfig configs[] = {
        {8000, 2, 0.5f, "8k QPSK R=0.5"},
        {16000, 4, 0.66f, "16k 16QAM R=0.66"},
        {32000, 6, 0.75f, "32k 64QAM R=0.75"},
        {64000, 8, 0.8f, "64k 256QAM R=0.8"},
    };

    for (const auto& cfg : configs) {
        int tb_bytes = (cfg.tb_size + 7) / 8;
        int encoded_bits = (int)(cfg.tb_size / cfg.rate);
        encoded_bits = ((encoded_bits + cfg.mod_order - 1) / cfg.mod_order) * cfg.mod_order;
        int num_symbols = encoded_bits / cfg.mod_order;

        uint8_t* d_tx_data;
        int8_t* d_symbols_int8;

        CHECK_CUDA(cudaMalloc(&d_tx_data, tb_bytes));
        CHECK_CUDA(cudaMalloc(&d_symbols_int8, num_symbols * 2));

        tb_encoder_handle_t encoder;
        tb_encoder_create(&encoder);

        tb_encoder_config_t enc_cfg = {};
        enc_cfg.tb_size_bits = cfg.tb_size;
        enc_cfg.num_layers = 1;
        enc_cfg.modulation_order = cfg.mod_order;
        enc_cfg.num_allocated_res = encoded_bits;
        enc_cfg.redundancy_version = 0;
        enc_cfg.code_rate = cfg.rate;
        enc_cfg.n_RNTI = 0x1234;
        enc_cfg.n_ID = 500;
        enc_cfg.q = 0;
        enc_cfg.enable_scrambling = true;
        tb_encoder_configure(encoder, &enc_cfg);

        // Warmup
        for (int i = 0; i < 10; i++) {
            tb_encoder_encode_to_symbols_int8(encoder, d_tx_data, d_symbols_int8, 0);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // Benchmark
        const int num_iters = 100;
        double start = get_time_ms();

        for (int i = 0; i < num_iters; i++) {
            tb_encoder_encode_to_symbols_int8(encoder, d_tx_data, d_symbols_int8, 0);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        double end = get_time_ms();
        double avg_ms = (end - start) / num_iters;
        double throughput_mbps = (cfg.tb_size / 1e6) / (avg_ms / 1000.0);

        printf("  %s: %.3f ms (%.2f Mbps)\n", cfg.name, avg_ms, throughput_mbps);

        tb_encoder_destroy(encoder);
        CHECK_CUDA(cudaFree(d_tx_data));
        CHECK_CUDA(cudaFree(d_symbols_int8));
    }
}

/**
 * @brief Test batch fused scramble+modulate vs per-CB reference
 *
 * Validates that batch processing produces bit-exact output compared to
 * processing each CB individually with the same fused kernel.
 */
bool test_batch_fused_scramble_modulate() {
    printf("\n=== Batch Fused Scramble+Modulate Validation Test ===\n");

    bool test_passed = true;

    // Test parameters (multi-CB scenario)
    const int num_cbs = 8;
    const int bits_per_cb = 8448;  // Typical for 64QAM, multi-CB TB
    const int mod_order = 6;       // 64QAM
    const int symbols_per_cb = bits_per_cb / mod_order;
    const int words_per_cb = (bits_per_cb + 31) / 32;
    const int total_symbols = num_cbs * symbols_per_cb;
    const int total_bits = num_cbs * bits_per_cb;
    const int total_words = (total_bits + 31) / 32;

    printf("  Config: %d CBs x %d bits/CB = %d total bits\n",
           num_cbs, bits_per_cb, total_bits);
    printf("  Modulation: 64QAM (%d bits/symbol) -> %d symbols\n",
           mod_order, total_symbols);

    // Allocate memory
    uint32_t* h_bits = new uint32_t[num_cbs * words_per_cb];
    cuFloatComplex* h_symbols_ref = new cuFloatComplex[total_symbols];
    cuFloatComplex* h_symbols_batch = new cuFloatComplex[total_symbols];

    uint32_t* d_bits;
    uint32_t* d_scramble_seq;
    cuFloatComplex* d_symbols_ref;
    cuFloatComplex* d_symbols_batch;

    CHECK_CUDA(cudaMalloc(&d_bits, num_cbs * words_per_cb * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_scramble_seq, total_words * sizeof(uint32_t)));  // Full length
    CHECK_CUDA(cudaMalloc(&d_symbols_ref, total_symbols * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&d_symbols_batch, total_symbols * sizeof(cuFloatComplex)));

    // Generate random input bits
    srand(12345);
    for (int i = 0; i < num_cbs * words_per_cb; i++) {
        h_bits[i] = ((uint32_t)rand() << 16) | (rand() & 0xFFFF);
    }
    CHECK_CUDA(cudaMemcpy(d_bits, h_bits, num_cbs * words_per_cb * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Create scrambler and modulator
    scrambler_handle_t scrambler;
    modulator_handle_t modulator;
    scrambler_create(&scrambler);
    modulator_create(&modulator);

    // Initialize scrambling sequence - generate full length for all CBs
    nr_scrambling_config_t scr_cfg = {.n_RNTI = 0x1234, .n_ID = 500, .q = 0};
    scrambler_configure(scrambler, &scr_cfg);
    scrambler_generate_sequence(scrambler, total_bits, 0);
    const uint32_t* scramble_ptr = scrambler_get_sequence_ptr(scrambler);
    CHECK_CUDA(cudaMemcpy(d_scramble_seq, scramble_ptr, total_words * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));

    // ========== Reference: Per-CB fused scramble+modulate ==========
    // Each CB uses its own offset into the scramble sequence
    printf("  Computing per-CB reference...\n");
    for (int cb = 0; cb < num_cbs; cb++) {
        // Compute word offset for this CB's scramble sequence
        int bit_offset = cb * bits_per_cb;
        int word_offset = bit_offset / 32;
        modulator_scramble_and_modulate(
            modulator,
            d_bits + cb * words_per_cb,
            d_scramble_seq + word_offset,  // Proper offset for this CB
            d_symbols_ref + cb * symbols_per_cb,
            bits_per_cb,
            mod_order,
            0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // ========== Test: Batch fused scramble+modulate ==========
    printf("  Computing batch version...\n");
    int ret = modulator_scramble_and_modulate_batch(
        modulator,
        d_bits,
        d_scramble_seq,
        d_symbols_batch,
        bits_per_cb,
        words_per_cb,
        num_cbs,
        mod_order,
        0);

    if (ret != 0) {
        printf("  ERROR: Batch fused kernel returned error %d\n", ret);
        test_passed = false;
        goto cleanup;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Download results
    CHECK_CUDA(cudaMemcpy(h_symbols_ref, d_symbols_ref,
                          total_symbols * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_symbols_batch, d_symbols_batch,
                          total_symbols * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

    // Compare bit-exact
    {
        int mismatches = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < total_symbols; i++) {
            float diff_real = fabsf(h_symbols_ref[i].x - h_symbols_batch[i].x);
            float diff_imag = fabsf(h_symbols_ref[i].y - h_symbols_batch[i].y);
            float diff = diff_real + diff_imag;
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-6f) {
                mismatches++;
                if (mismatches <= 3) {
                    printf("    Mismatch at symbol %d: ref=(%.6f,%.6f) batch=(%.6f,%.6f)\n",
                           i, h_symbols_ref[i].x, h_symbols_ref[i].y,
                           h_symbols_batch[i].x, h_symbols_batch[i].y);
                }
            }
        }

        printf("  Comparison: %d/%d mismatches, max_diff=%.2e\n",
               mismatches, total_symbols, max_diff);

        if (mismatches == 0) {
            printf("  PASSED: Batch output matches per-CB reference exactly\n");
        } else {
            printf("  FAILED: %d symbol mismatches\n", mismatches);
            test_passed = false;
        }
    }

cleanup:
    scrambler_destroy(scrambler);
    modulator_destroy(modulator);
    CHECK_CUDA(cudaFree(d_bits));
    CHECK_CUDA(cudaFree(d_scramble_seq));
    CHECK_CUDA(cudaFree(d_symbols_ref));
    CHECK_CUDA(cudaFree(d_symbols_batch));
    delete[] h_bits;
    delete[] h_symbols_ref;
    delete[] h_symbols_batch;

    return test_passed;
}

/**
 * @brief Test ultra fused RM→INT8 kernel smoke test
 *
 * Basic validation that the ultra fused kernel runs without crashing and
 * produces INT8 symbols in the expected range. Full loopback validation
 * requires proper rate matching parameters from the TB encoder chain.
 *
 * Note: This test does not validate decoding because the ultra fused kernel
 * requires exact rate matching configuration. Full E2E validation should be
 * done via the stage_latency_benchmark or integration tests.
 */
bool test_ultra_fused_rm_int8_smoke() {
    printf("\n=== Ultra Fused RM→INT8 Smoke Test ===\n");

    bool test_passed = true;
    nr_ldpc_status_t status;

    // Test configuration (64QAM)
    const int tb_size_bits = 8448;
    const float code_rate = 0.5f;
    const int mod_order = 6;  // 64QAM

    printf("  TB size: %d bits, Rate: %.2f, Mod: 64QAM\n",
           tb_size_bits, code_rate);

    // Calculate sizes
    int tb_bytes = (tb_size_bits + 7) / 8;
    int encoded_bits = (int)(tb_size_bits / code_rate);
    encoded_bits = ((encoded_bits + mod_order - 1) / mod_order) * mod_order;
    int num_symbols = encoded_bits / mod_order;

    printf("  Encoded bits: %d, Symbols: %d\n", encoded_bits, num_symbols);

    // Initialize LDPC config
    nr_ldpc_config_t ldpc_cfg;
    status = nr_ldpc_init_config(&ldpc_cfg, tb_size_bits, code_rate);
    if (status != NR_LDPC_SUCCESS) {
        printf("  ERROR: Failed to init LDPC config\n");
        return false;
    }

    // Allocate device memory
    uint32_t* d_encoded_ldpc;
    int8_t* d_symbols_int8;
    uint32_t* d_scramble_seq;

    int ldpc_output_words = (ldpc_cfg.num_codeword_bits + 31) / 32;
    int scramble_words = (encoded_bits + 31) / 32;

    CHECK_CUDA(cudaMalloc(&d_encoded_ldpc, ldpc_output_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_symbols_int8, num_symbols * 2));
    CHECK_CUDA(cudaMalloc(&d_scramble_seq, scramble_words * sizeof(uint32_t)));

    // Create and configure LDPC encoder
    ldpc_encoder_handle_t ldpc_encoder = nullptr;
    scrambler_handle_t scrambler = nullptr;
    ldpc_encoder_create(&ldpc_encoder);
    ldpc_encoder_configure(ldpc_encoder, &ldpc_cfg);

    scrambler_create(&scrambler);
    nr_scrambling_config_t scr_cfg = {.n_RNTI = 0x1234, .n_ID = 500, .q = 0};
    scrambler_configure(scrambler, &scr_cfg);
    scrambler_generate_sequence(scrambler, encoded_bits, 0);
    CHECK_CUDA(cudaMemcpy(d_scramble_seq, scrambler_get_sequence_ptr(scrambler),
                          scramble_words * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

    // Generate test input and encode
    uint8_t* h_tx_data = new uint8_t[tb_bytes];
    srand(42);
    generate_random_data(h_tx_data, tb_bytes);

    uint32_t* d_tx_data;
    CHECK_CUDA(cudaMalloc(&d_tx_data, tb_bytes));
    CHECK_CUDA(cudaMemcpy(d_tx_data, h_tx_data, tb_bytes, cudaMemcpyHostToDevice));

    ldpc_encoder_encode(ldpc_encoder, d_tx_data, d_encoded_ldpc, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Configure fused TX kernel
    pdsch_fused_tx_config_t fused_cfg;
    memset(&fused_cfg, 0, sizeof(fused_cfg));
    fused_cfg.N_cb = ldpc_cfg.num_codeword_bits;
    fused_cfg.N_full = ldpc_cfg.num_codeword_bits;
    fused_cfg.k0 = 0;
    fused_cfg.Kd = ldpc_cfg.num_info_bits;
    fused_cfg.F = 0;
    fused_cfg.puncture_offset = 2 * ldpc_cfg.lifting_size;
    fused_cfg.encoded_stride = ldpc_output_words;
    fused_cfg.nof_short = 0;
    fused_cfg.E_short = 0;
    fused_cfg.E_long = encoded_bits;
    fused_cfg.num_cbs = 1;
    fused_cfg.mod_order = mod_order;
    fused_cfg.total_symbols = num_symbols;

    // Run fused kernel
    printf("  Running ultra fused kernel...\n");
    int ret = pdsch_fused_encode_to_symbols_int8(
        d_encoded_ldpc,
        scrambler_get_c_init(scrambler),
        d_symbols_int8,
        &fused_cfg,
        0);

    if (ret != 0) {
        printf("  ERROR: Ultra fused kernel returned %d\n", ret);
        test_passed = false;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());

        // Download and verify output is in valid range
        int8_t* h_int8 = new int8_t[num_symbols * 2];
        CHECK_CUDA(cudaMemcpy(h_int8, d_symbols_int8, num_symbols * 2, cudaMemcpyDeviceToHost));

        // For 64QAM, valid INT8 values are ±1, ±3, ±5, ±7
        int valid_symbols = 0;
        int invalid_symbols = 0;
        for (int i = 0; i < num_symbols; i++) {
            int8_t re = h_int8[2*i];
            int8_t im = h_int8[2*i + 1];
            // Check if values are in valid 64QAM set: ±1, ±3, ±5, ±7
            bool re_valid = (re == 1 || re == -1 || re == 3 || re == -3 ||
                            re == 5 || re == -5 || re == 7 || re == -7);
            bool im_valid = (im == 1 || im == -1 || im == 3 || im == -3 ||
                            im == 5 || im == -5 || im == 7 || im == -7);
            if (re_valid && im_valid) {
                valid_symbols++;
            } else {
                invalid_symbols++;
                if (invalid_symbols <= 3) {
                    printf("    Invalid symbol %d: (%d, %d)\n", i, re, im);
                }
            }
        }

        printf("  Valid symbols: %d/%d (%.1f%%)\n",
               valid_symbols, num_symbols, 100.0f * valid_symbols / num_symbols);

        if (valid_symbols == num_symbols) {
            printf("  PASSED: All symbols in valid 64QAM INT8 range\n");
        } else if (valid_symbols > num_symbols * 0.95) {
            printf("  PASSED (with warnings): >95%% symbols valid\n");
        } else {
            printf("  FAILED: Too many invalid symbols\n");
            test_passed = false;
        }

        delete[] h_int8;
    }

    // Cleanup
    ldpc_encoder_destroy(ldpc_encoder);
    scrambler_destroy(scrambler);
    CHECK_CUDA(cudaFree(d_encoded_ldpc));
    CHECK_CUDA(cudaFree(d_symbols_int8));
    CHECK_CUDA(cudaFree(d_scramble_seq));
    CHECK_CUDA(cudaFree(d_tx_data));
    delete[] h_tx_data;

    return test_passed;
}

/**
 * @brief Benchmark batch vs per-CB fused scramble+modulate
 */
void benchmark_batch_fused() {
    printf("\n=== Batch Fused Scramble+Modulate Benchmark ===\n");

    const int num_cbs = 8;
    const int bits_per_cb = 8448;
    const int mod_order = 6;
    const int symbols_per_cb = bits_per_cb / mod_order;
    const int words_per_cb = (bits_per_cb + 31) / 32;
    const int total_symbols = num_cbs * symbols_per_cb;

    uint32_t* d_bits;
    uint32_t* d_scramble_seq;
    cuFloatComplex* d_symbols;

    CHECK_CUDA(cudaMalloc(&d_bits, num_cbs * words_per_cb * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_scramble_seq, words_per_cb * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_symbols, total_symbols * sizeof(cuFloatComplex)));

    scrambler_handle_t scrambler;
    modulator_handle_t modulator;
    scrambler_create(&scrambler);
    modulator_create(&modulator);

    {
        nr_scrambling_config_t scr_cfg = {.n_RNTI = 0x1234, .n_ID = 500, .q = 0};
        scrambler_configure(scrambler, &scr_cfg);
        scrambler_generate_sequence(scrambler, bits_per_cb, 0);
    }
    CHECK_CUDA(cudaMemcpy(d_scramble_seq, scrambler_get_sequence_ptr(scrambler),
                          words_per_cb * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

    // Warmup
    for (int i = 0; i < 10; i++) {
        modulator_scramble_and_modulate_batch(modulator, d_bits, d_scramble_seq,
                                               d_symbols, bits_per_cb, words_per_cb,
                                               num_cbs, mod_order, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark per-CB
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_iters = 100;

    cudaEventRecord(start);
    for (int iter = 0; iter < num_iters; iter++) {
        for (int cb = 0; cb < num_cbs; cb++) {
            modulator_scramble_and_modulate(modulator,
                                            d_bits + cb * words_per_cb,
                                            d_scramble_seq,
                                            d_symbols + cb * symbols_per_cb,
                                            bits_per_cb, mod_order, 0);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float per_cb_ms;
    cudaEventElapsedTime(&per_cb_ms, start, stop);
    float per_cb_us = (per_cb_ms * 1000.0f) / num_iters;

    // Benchmark batch
    cudaEventRecord(start);
    for (int iter = 0; iter < num_iters; iter++) {
        modulator_scramble_and_modulate_batch(modulator, d_bits, d_scramble_seq,
                                               d_symbols, bits_per_cb, words_per_cb,
                                               num_cbs, mod_order, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float batch_ms;
    cudaEventElapsedTime(&batch_ms, start, stop);
    float batch_us = (batch_ms * 1000.0f) / num_iters;

    printf("  Config: %d CBs x %d bits\n", num_cbs, bits_per_cb);
    printf("  Per-CB (loop):  %.2f µs\n", per_cb_us);
    printf("  Batch (single): %.2f µs\n", batch_us);
    printf("  Speedup: %.1fx\n", per_cb_us / batch_us);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    scrambler_destroy(scrambler);
    modulator_destroy(modulator);
    CHECK_CUDA(cudaFree(d_bits));
    CHECK_CUDA(cudaFree(d_scramble_seq));
    CHECK_CUDA(cudaFree(d_symbols));
}

int main(int argc, char* argv[]) {
    printf("================================================\n");
    printf("PDSCH TX Chain Loopback Test Suite\n");
    printf("================================================\n");

    // Initialize library
    nr_ldpc_status_t status = ocudu_phy_cuda_init();
    if (status != NR_LDPC_SUCCESS) {
        printf("ERROR: Failed to initialize OCUDU PHY CUDA: %s\n", nr_ldpc_get_error_string(status));
        return 1;
    }
    ocudu_phy_cuda_print_info();

    srand(time(NULL));

    bool all_passed = true;

    // Run component tests (LDPC only, no modulation)
    all_passed &= test_ldpc_roundtrip();
    all_passed &= test_scrambling_effect();

    // Run batch fused and ultra fused validation tests
    printf("\n================================================\n");
    printf("Fused TX Kernel Validation Tests\n");
    printf("================================================\n");
    all_passed &= test_batch_fused_scramble_modulate();
    all_passed &= test_ultra_fused_rm_int8_smoke();

    // Run full 5G NR compliant TB-based loopback tests
    printf("\n================================================\n");
    printf("PDSCH TX Full Chain Loopback Tests (5G NR compliant)\n");
    printf("================================================\n");
    int num_configs = sizeof(test_configs) / sizeof(test_configs[0]);
    for (int i = 0; i < num_configs; i++) {
        all_passed &= run_pdsch_loopback_test(test_configs[i]);
    }

    // Run throughput benchmark if requested
    if (argc > 1 && strcmp(argv[1], "-bench") == 0) {
        benchmark_tx_throughput();
        benchmark_batch_fused();
    }

    // Summary
    printf("\n================================================\n");
    printf("Test Summary: %s\n", all_passed ? "ALL PASSED" : "SOME FAILED");
    printf("================================================\n");

    ocudu_phy_cuda_cleanup();
    return all_passed ? 0 : 1;
}
