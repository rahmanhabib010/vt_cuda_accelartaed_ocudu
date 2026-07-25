/**
 * Minimal test: BG1 LDPC decode with punctured first 2 columns.
 * Uses the all-zeros codeword (always valid: H * 0 = 0).
 * Sets first 2*Z LLRs to 0 (erasure), rest to +127 (known zero bits).
 * Tests if the decoder can recover the punctured columns.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "ldpc_decoder.h"
#include "ldpc_encoder.h"
#include "nr_ldpc_defs.h"

int test_bg1_allzero_punctured(int Z) {
    printf("=== BG1 All-Zero Punctured Test (Z=%d) ===\n", Z);

    // BG1 parameters
    const int bg = 1;
    const int Kb = 22;
    const int N_cols = 68;
    const int M = 46;
    const int K = Kb * Z;
    const int N_full = N_cols * Z;
    const int punctured = 2 * Z;
    const int K_words = (K + 31) / 32;

    printf("  K=%d, N=%d, punctured=%d, K_words=%d\n", K, N_full, punctured, K_words);

    // Create decoder
    ldpc_decoder_handle_t decoder;
    ldpc_decoder_create(&decoder);

    nr_ldpc_config_t cfg = {};
    cfg.base_graph = bg;
    cfg.lifting_size = Z;
    cfg.num_info_bits = K;  // No filler for all-zero test
    cfg.num_parity_bits = M * Z;
    cfg.num_codeword_bits = N_full;
    cfg.num_filler_bits = 0;
    cfg.max_parity_nodes = M;

    ldpc_decoder_params_t params;
    ldpc_decoder_params_init(&params);
    params.max_iterations = 20;
    params.early_termination = true;
    params.auto_scale = true;
    params.llr_clamp = 127.0f;
    params.crc_early_termination = false;
    params.skip_iteration_stats = false;

    ldpc_decoder_configure(decoder, &cfg, &params);

    // Allocate device memory
    float* d_llrs;
    uint32_t* d_output;
    cudaMalloc(&d_llrs, N_full * sizeof(float));
    cudaMalloc(&d_output, K_words * sizeof(uint32_t));

    // Create host LLRs
    float* h_llrs = (float*)malloc(N_full * sizeof(float));

    // All-zero codeword: all LLRs = +127 (bit=0 with high confidence)
    for (int i = 0; i < N_full; i++) {
        h_llrs[i] = 127.0f;
    }
    // Puncture first 2*Z (cols 0 and 1)
    for (int i = 0; i < punctured; i++) {
        h_llrs[i] = 0.0f;
    }

    // Copy to device
    cudaMemcpy(d_llrs, h_llrs, N_full * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, K_words * sizeof(uint32_t));

    // Decode
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nr_ldpc_status_t status = ldpc_decoder_decode_batch(decoder, d_llrs, d_output, 1, stream);
    cudaStreamSynchronize(stream);

    if (status != NR_LDPC_SUCCESS) {
        printf("  DECODE FAILED: %d\n", status);
        return 1;
    }

    float avg_iters = ldpc_decoder_get_avg_iterations(decoder);
    printf("  Iterations: %.1f\n", avg_iters);

    // Copy output
    uint32_t* h_output = (uint32_t*)malloc(K_words * sizeof(uint32_t));
    cudaMemcpy(h_output, d_output, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Check: should be all zeros
    int bit_errors = 0;
    for (int i = 0; i < K; i++) {
        int word = i / 32;
        int bit = i % 32;
        int decoded_bit = (h_output[word] >> bit) & 1;
        if (decoded_bit != 0) {
            bit_errors++;
            if (bit_errors <= 10) {
                printf("  Error at bit %d (col=%d, z=%d): expected 0, got 1\n",
                       i, i / Z, i % Z);
            }
        }
    }
    printf("  Bit errors: %d / %d\n", bit_errors, K);
    printf("  Result: %s\n", bit_errors == 0 ? "PASS" : "FAIL");

    // Cleanup
    free(h_llrs);
    free(h_output);
    cudaFree(d_llrs);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    ldpc_decoder_destroy(decoder);

    return bit_errors;
}

int test_bg1_encoded_punctured(int Z) {
    printf("\n=== BG1 Encoded+Punctured Test (Z=%d) ===\n", Z);

    const int bg = 1;
    const int Kb = 22;
    const int N_cols = 68;
    const int M = 46;
    const int K = Kb * Z;
    const int N_full = N_cols * Z;
    const int punctured = 2 * Z;
    const int K_words = (K + 31) / 32;
    const int N_words = (N_full + 31) / 32;

    printf("  K=%d, N=%d, punctured=%d\n", K, N_full, punctured);

    // Create encoder and decoder
    ldpc_encoder_handle_t encoder;
    ldpc_encoder_create(&encoder);

    nr_ldpc_config_t enc_cfg = {};
    enc_cfg.base_graph = bg;
    enc_cfg.lifting_size = Z;
    enc_cfg.num_info_bits = K;
    enc_cfg.num_parity_bits = M * Z;
    enc_cfg.num_codeword_bits = N_full;
    enc_cfg.num_filler_bits = 0;
    ldpc_encoder_configure(encoder, &enc_cfg);

    ldpc_decoder_handle_t decoder;
    ldpc_decoder_create(&decoder);

    nr_ldpc_config_t dec_cfg = {};
    dec_cfg.base_graph = bg;
    dec_cfg.lifting_size = Z;
    dec_cfg.num_info_bits = K;
    dec_cfg.num_parity_bits = M * Z;
    dec_cfg.num_codeword_bits = N_full;
    dec_cfg.num_filler_bits = 0;
    dec_cfg.max_parity_nodes = M;

    ldpc_decoder_params_t params;
    ldpc_decoder_params_init(&params);
    params.max_iterations = 20;
    params.early_termination = true;
    params.auto_scale = true;
    params.llr_clamp = 127.0f;
    params.crc_early_termination = false;
    params.skip_iteration_stats = false;

    ldpc_decoder_configure(decoder, &dec_cfg, &params);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create random input
    uint32_t* h_input = (uint32_t*)calloc(K_words, sizeof(uint32_t));
    srand(42);
    for (int i = 0; i < K_words; i++) {
        h_input[i] = rand();
    }
    // Clear bits beyond K
    if (K % 32 != 0) {
        h_input[K_words - 1] &= (1u << (K % 32)) - 1;
    }

    // GPU buffers
    uint32_t* d_input;
    uint32_t* d_encoded;
    float* d_llrs;
    uint32_t* d_decoded;
    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, N_words * sizeof(uint32_t));
    cudaMalloc(&d_llrs, N_full * sizeof(float));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    // Copy input and encode
    cudaMemcpy(d_input, h_input, K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_encoded, 0, N_words * sizeof(uint32_t));

    nr_ldpc_status_t status = ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);
    if (status != NR_LDPC_SUCCESS) {
        printf("  ENCODE FAILED: %d\n", status);
        return 1;
    }

    // Copy encoded codeword to host
    uint32_t* h_encoded = (uint32_t*)calloc(N_words, sizeof(uint32_t));
    cudaMemcpy(h_encoded, d_encoded, N_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Create LLRs from encoded bits
    // Encoder output uses MSB-first byte ordering within uint32_t words
    // Match the BG1 puncturing convention used by the production decoder path.
    float* h_llrs = (float*)malloc(N_full * sizeof(float));
    for (int i = 0; i < N_full; i++) {
        int word_idx = i / 32;
        int bit_in_word = i % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);  // MSB-first within byte
        int bit_pos = byte_in_word * 8 + bit_in_byte;
        int encoded_bit = (h_encoded[word_idx] >> bit_pos) & 1;
        h_llrs[i] = encoded_bit ? -127.0f : 127.0f;
    }

    // Test 1: Without puncturing (should converge immediately)
    printf("  Test 1: No puncturing (full codeword)\n");
    cudaMemcpy(d_llrs, h_llrs, N_full * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    status = ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
    cudaStreamSynchronize(stream);
    float iters1 = ldpc_decoder_get_avg_iterations(decoder);

    uint32_t* h_decoded = (uint32_t*)calloc(K_words, sizeof(uint32_t));
    cudaMemcpy(h_decoded, d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Decoder output uses LSB-first flat format: bit i at word[i/32] >> (i%32)
    // But encoder input uses MSB-first byte format
    // Compare in the decoder's output format
    int errors1 = 0;
    for (int i = 0; i < K; i++) {
        // Input was in MSB-first byte format
        int in_word = i / 32;
        int in_bit_in_word = i % 32;
        int in_byte = in_bit_in_word / 8;
        int in_bit_in_byte = 7 - (in_bit_in_word % 8);
        int in_bit_pos = in_byte * 8 + in_bit_in_byte;
        int exp_bit = (h_input[in_word] >> in_bit_pos) & 1;

        // Decoder output in LSB-first flat format
        int dec_bit = (h_decoded[i/32] >> (i%32)) & 1;
        if (exp_bit != dec_bit) errors1++;
    }
    printf("    Iterations: %.1f, Errors: %d/%d, %s\n", iters1, errors1, K,
           errors1 == 0 ? "PASS" : "FAIL");

    // Test 2: With puncturing (first 2*Z = 0)
    printf("  Test 2: With puncturing (first 2*Z LLRs = 0)\n");
    for (int i = 0; i < punctured; i++) {
        h_llrs[i] = 0.0f;
    }
    cudaMemcpy(d_llrs, h_llrs, N_full * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    status = ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
    cudaStreamSynchronize(stream);
    float iters2 = ldpc_decoder_get_avg_iterations(decoder);

    cudaMemcpy(h_decoded, d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int errors2 = 0;
    for (int i = 0; i < K; i++) {
        int in_word = i / 32;
        int in_bit_in_word = i % 32;
        int in_byte = in_bit_in_word / 8;
        int in_bit_in_byte = 7 - (in_bit_in_word % 8);
        int in_bit_pos = in_byte * 8 + in_bit_in_byte;
        int exp_bit = (h_input[in_word] >> in_bit_pos) & 1;
        int dec_bit = (h_decoded[i/32] >> (i%32)) & 1;
        if (exp_bit != dec_bit) {
            errors2++;
            if (errors2 <= 10) {
                printf("    Error at bit %d (col=%d, z=%d): expected %d, got %d\n",
                       i, i / Z, i % Z, exp_bit, dec_bit);
            }
        }
    }
    printf("    Iterations: %.1f, Errors: %d/%d, %s\n", iters2, errors2, K,
           errors2 == 0 ? "PASS" : "FAIL");

    // Cleanup
    free(h_input);
    free(h_encoded);
    free(h_llrs);
    free(h_decoded);
    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaFree(d_llrs);
    cudaFree(d_decoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);

    return errors1 + errors2;
}

int test_bg2_encoded_punctured(int Z) {
    printf("\n=== BG2 Encoded+Punctured Test (Z=%d) ===\n", Z);

    const int bg = 2;
    const int Kb = 10;
    const int N_cols = 52;
    const int M = 42;
    const int K = Kb * Z;
    const int N_full = N_cols * Z;
    const int punctured = 2 * Z;
    const int K_words = (K + 31) / 32;
    const int N_words = (N_full + 31) / 32;

    printf("  K=%d, N=%d, punctured=%d\n", K, N_full, punctured);

    ldpc_encoder_handle_t encoder;
    ldpc_encoder_create(&encoder);

    nr_ldpc_config_t enc_cfg = {};
    enc_cfg.base_graph = bg;
    enc_cfg.lifting_size = Z;
    enc_cfg.num_info_bits = K;
    enc_cfg.num_parity_bits = M * Z;
    enc_cfg.num_codeword_bits = N_full;
    enc_cfg.num_filler_bits = 0;
    ldpc_encoder_configure(encoder, &enc_cfg);

    ldpc_decoder_handle_t decoder;
    ldpc_decoder_create(&decoder);

    nr_ldpc_config_t dec_cfg = {};
    dec_cfg.base_graph = bg;
    dec_cfg.lifting_size = Z;
    dec_cfg.num_info_bits = K;
    dec_cfg.num_parity_bits = M * Z;
    dec_cfg.num_codeword_bits = N_full;
    dec_cfg.num_filler_bits = 0;
    dec_cfg.max_parity_nodes = M;

    ldpc_decoder_params_t params;
    ldpc_decoder_params_init(&params);
    params.max_iterations = 20;
    params.early_termination = true;
    params.auto_scale = true;
    params.llr_clamp = 127.0f;
    params.crc_early_termination = false;
    params.skip_iteration_stats = false;

    ldpc_decoder_configure(decoder, &dec_cfg, &params);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint32_t* h_input = (uint32_t*)calloc(K_words, sizeof(uint32_t));
    srand(42);
    for (int i = 0; i < K_words; i++) h_input[i] = rand();
    if (K % 32 != 0) h_input[K_words - 1] &= (1u << (K % 32)) - 1;

    uint32_t* d_input, *d_encoded, *d_decoded;
    float* d_llrs;
    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, N_words * sizeof(uint32_t));
    cudaMalloc(&d_llrs, N_full * sizeof(float));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    cudaMemcpy(d_input, h_input, K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_encoded, 0, N_words * sizeof(uint32_t));

    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    uint32_t* h_encoded = (uint32_t*)calloc(N_words, sizeof(uint32_t));
    cudaMemcpy(h_encoded, d_encoded, N_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    float* h_llrs = (float*)malloc(N_full * sizeof(float));
    for (int i = 0; i < N_full; i++) {
        int word_idx = i / 32;
        int bit_in_word = i % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int bit_pos = byte_in_word * 8 + bit_in_byte;
        int encoded_bit = (h_encoded[word_idx] >> bit_pos) & 1;
        h_llrs[i] = encoded_bit ? -127.0f : 127.0f;
    }

    // With puncturing
    for (int i = 0; i < punctured; i++) h_llrs[i] = 0.0f;

    cudaMemcpy(d_llrs, h_llrs, N_full * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
    cudaStreamSynchronize(stream);
    float iters = ldpc_decoder_get_avg_iterations(decoder);

    uint32_t* h_decoded = (uint32_t*)calloc(K_words, sizeof(uint32_t));
    cudaMemcpy(h_decoded, d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < K; i++) {
        int in_word = i / 32;
        int in_bit_in_word = i % 32;
        int in_byte = in_bit_in_word / 8;
        int in_bit_in_byte = 7 - (in_bit_in_word % 8);
        int in_bit_pos = in_byte * 8 + in_bit_in_byte;
        int exp_bit = (h_input[in_word] >> in_bit_pos) & 1;
        int dec_bit = (h_decoded[i/32] >> (i%32)) & 1;
        if (exp_bit != dec_bit) errors++;
    }
    printf("  With puncturing: Iterations: %.1f, Errors: %d/%d, %s\n",
           iters, errors, K, errors == 0 ? "PASS" : "FAIL");

    free(h_input); free(h_encoded); free(h_llrs); free(h_decoded);
    cudaFree(d_input); cudaFree(d_encoded); cudaFree(d_llrs); cudaFree(d_decoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);

    return errors;
}

// Test with noisy LLRs: encode, add AWGN, decode
int test_bg1_noisy(int Z, float snr_db) {
    printf("\n=== BG1 Noisy Test (Z=%d, SNR=%.0f dB) ===\n", Z, snr_db);

    const int bg = 1;
    const int Kb = 22;
    const int N_cols = 68;
    const int M = 46;
    const int K = Kb * Z;
    const int N_full = N_cols * Z;
    const int punctured = 2 * Z;
    const int K_words = (K + 31) / 32;
    const int N_words = (N_full + 31) / 32;

    // Create encoder and decoder
    ldpc_encoder_handle_t encoder;
    ldpc_encoder_create(&encoder);

    nr_ldpc_config_t enc_cfg = {};
    enc_cfg.base_graph = bg;
    enc_cfg.lifting_size = Z;
    enc_cfg.num_info_bits = K;
    enc_cfg.num_parity_bits = M * Z;
    enc_cfg.num_codeword_bits = N_full;
    enc_cfg.num_filler_bits = 0;
    ldpc_encoder_configure(encoder, &enc_cfg);

    ldpc_decoder_handle_t decoder;
    ldpc_decoder_create(&decoder);

    nr_ldpc_config_t dec_cfg = {};
    dec_cfg.base_graph = bg;
    dec_cfg.lifting_size = Z;
    dec_cfg.num_info_bits = K;
    dec_cfg.num_parity_bits = M * Z;
    dec_cfg.num_codeword_bits = N_full;
    dec_cfg.num_filler_bits = 0;
    dec_cfg.max_parity_nodes = M;

    ldpc_decoder_params_t params;
    ldpc_decoder_params_init(&params);
    params.max_iterations = 20;
    params.early_termination = true;
    params.auto_scale = true;
    params.llr_clamp = 127.0f;
    params.crc_early_termination = false;
    params.skip_iteration_stats = false;

    ldpc_decoder_configure(decoder, &dec_cfg, &params);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create random input
    uint32_t* h_input = (uint32_t*)calloc(K_words, sizeof(uint32_t));
    srand(123);
    for (int i = 0; i < K_words; i++) h_input[i] = rand();
    if (K % 32 != 0) h_input[K_words - 1] &= (1u << (K % 32)) - 1;

    uint32_t* d_input, *d_encoded, *d_decoded;
    float* d_llrs;
    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, N_words * sizeof(uint32_t));
    cudaMalloc(&d_llrs, N_full * sizeof(float));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    cudaMemcpy(d_input, h_input, K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_encoded, 0, N_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    uint32_t* h_encoded = (uint32_t*)calloc(N_words, sizeof(uint32_t));
    cudaMemcpy(h_encoded, d_encoded, N_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Create LLRs from encoded bits with AWGN noise
    // SNR per bit: snr_linear = 10^(snr_db/10)
    // Noise std on LLR scale: use soft LLR = 2*y/sigma^2 model
    // For BPSK-like: signal = ±1, noise_var = 1/snr_linear
    float snr_linear = powf(10.0f, snr_db / 10.0f);
    float llr_signal = 127.0f;  // Perfect LLR magnitude for bit=0/1
    // Scale noise relative to signal: noise_std = llr_signal / sqrt(snr_linear)
    float noise_std = llr_signal / sqrtf(snr_linear);

    printf("  SNR=%.0f dB, noise_std=%.1f, llr_signal=%.0f\n", snr_db, noise_std, llr_signal);

    float* h_llrs = (float*)malloc(N_full * sizeof(float));
    srand(456);
    for (int i = 0; i < N_full; i++) {
        int word_idx = i / 32;
        int bit_in_word = i % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int bit_pos = byte_in_word * 8 + bit_in_byte;
        int encoded_bit = (h_encoded[word_idx] >> bit_pos) & 1;
        float signal = encoded_bit ? -llr_signal : llr_signal;

        // Box-Muller for Gaussian noise
        float u1 = ((float)(rand() % 10000) + 1) / 10001.0f;
        float u2 = ((float)(rand() % 10000)) / 10000.0f;
        float noise = noise_std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);

        h_llrs[i] = signal + noise;
    }

    // Puncture first 2*Z
    for (int i = 0; i < punctured; i++) h_llrs[i] = 0.0f;

    cudaMemcpy(d_llrs, h_llrs, N_full * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    nr_ldpc_status_t status = ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
    cudaStreamSynchronize(stream);
    float iters = ldpc_decoder_get_avg_iterations(decoder);

    uint32_t* h_decoded = (uint32_t*)calloc(K_words, sizeof(uint32_t));
    cudaMemcpy(h_decoded, d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < K; i++) {
        int in_word = i / 32;
        int in_bit_in_word = i % 32;
        int in_byte = in_bit_in_word / 8;
        int in_bit_in_byte = 7 - (in_bit_in_word % 8);
        int in_bit_pos = in_byte * 8 + in_bit_in_byte;
        int exp_bit = (h_input[in_word] >> in_bit_pos) & 1;
        int dec_bit = (h_decoded[i/32] >> (i%32)) & 1;
        if (exp_bit != dec_bit) errors++;
    }
    printf("  Iterations: %.1f, Errors: %d/%d, %s\n",
           iters, errors, K, errors == 0 ? "PASS" : "FAIL");

    free(h_input); free(h_encoded); free(h_llrs); free(h_decoded);
    cudaFree(d_input); cudaFree(d_encoded); cudaFree(d_llrs); cudaFree(d_decoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);

    return errors;
}

int main() {
    // Test BG1 all-zero with puncturing
    int r1 = test_bg1_allzero_punctured(224);

    // Test BG1 encoded data with puncturing
    int r2 = test_bg1_encoded_punctured(224);

    // Test BG2 encoded data with puncturing (should work - baseline)
    int r3 = test_bg2_encoded_punctured(18);

    // Also test BG1 with different Z values
    int r4 = test_bg1_encoded_punctured(176);
    int r5 = test_bg1_encoded_punctured(112);

    // KEY: Test Z=384 (the lifting size used in the failing E2E test)
    int r6 = test_bg1_allzero_punctured(384);
    int r7 = test_bg1_encoded_punctured(384);

    // Noisy tests at Z=384
    int r8 = test_bg1_noisy(384, 25.0f);
    int r9 = test_bg1_noisy(384, 15.0f);
    int r10 = test_bg1_noisy(224, 25.0f);

    printf("\n=== OVERALL RESULTS ===\n");
    printf("BG1 all-zero punctured (Z=224): %s\n", r1 == 0 ? "PASS" : "FAIL");
    printf("BG1 encoded punctured (Z=224):  %s\n", r2 == 0 ? "PASS" : "FAIL");
    printf("BG2 encoded punctured (Z=18):   %s\n", r3 == 0 ? "PASS" : "FAIL");
    printf("BG1 encoded punctured (Z=176):  %s\n", r4 == 0 ? "PASS" : "FAIL");
    printf("BG1 encoded punctured (Z=112):  %s\n", r5 == 0 ? "PASS" : "FAIL");
    printf("BG1 all-zero punctured (Z=384): %s\n", r6 == 0 ? "PASS" : "FAIL");
    printf("BG1 encoded punctured (Z=384):  %s\n", r7 == 0 ? "PASS" : "FAIL");
    printf("BG1 noisy (Z=384, 25dB):        %s\n", r8 == 0 ? "PASS" : "FAIL");
    printf("BG1 noisy (Z=384, 15dB):        %s\n", r9 == 0 ? "PASS" : "FAIL");
    printf("BG1 noisy (Z=224, 25dB):        %s\n", r10 == 0 ? "PASS" : "FAIL");

    return (r1 || r2 || r3 || r4 || r5 || r6 || r7 || r8 || r9 || r10) ? 1 : 0;
}
