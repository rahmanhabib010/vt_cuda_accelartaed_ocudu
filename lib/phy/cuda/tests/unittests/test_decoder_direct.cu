// Direct decoder test - no rate matching, test decoder with known codeword
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Test decoder with all-zeros codeword (should always decode to zeros)
int test_all_zeros(int Z) {
    const int bg = 2;
    const int Kb = 10;
    const int K = Kb * Z;
    const int K_words = (K + 31) / 32;
    const int N_cols = 52;
    const int N_full = N_cols * Z;  // Full LLR length

    // Config
    nr_ldpc_config_t cfg = {};
    cfg.base_graph = bg;
    cfg.lifting_size = Z;
    cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(Z);
    cfg.num_info_bits = K;
    cfg.num_filler_bits = 0;
    cfg.num_parity_bits = 42 * Z;
    cfg.num_codeword_bits = K + cfg.num_parity_bits;
    cfg.puncture = true;
    cfg.redundancy_version = 0;

    // Create decoder
    ldpc_decoder_handle_t decoder;
    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 25;
    ldpc_decoder_configure(decoder, &cfg, &dec_params);

    // Allocate memory
    float *d_llrs;
    uint32_t *d_decoded;
    cudaMalloc(&d_llrs, N_full * sizeof(float));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // All-zeros codeword: all LLRs are positive (bit=0)
    // First 2*Z are punctured (LLR=0), rest are +127
    std::vector<float> h_llrs(N_full, 127.0f);
    // Punctured region (first 2*Z) should have LLR=0
    for (int i = 0; i < 2 * Z; i++) {
        h_llrs[i] = 0.0f;
    }
    cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);

    // Decode
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode(decoder, d_llrs, d_decoded, stream);
    cudaStreamSynchronize(stream);

    // Check output - should be all zeros
    std::vector<uint32_t> h_decoded(K_words);
    cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < K; i++) {
        int out = (h_decoded[i/32] >> (i%32)) & 1;
        if (out != 0) errors++;
    }

    // Cleanup
    cudaFree(d_llrs);
    cudaFree(d_decoded);
    cudaStreamDestroy(stream);
    ldpc_decoder_destroy(decoder);

    return errors;
}

// Test with specific LLR pattern where we know the answer
int test_with_encoder(int Z) {
    const int bg = 2;
    const int Kb = 10;
    const int K = Kb * Z;
    const int K_words = (K + 31) / 32;
    const int N_cols = 52;
    const int N_full = N_cols * Z;

    // Config
    nr_ldpc_config_t cfg = {};
    cfg.base_graph = bg;
    cfg.lifting_size = Z;
    cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(Z);
    cfg.num_info_bits = K;
    cfg.num_filler_bits = 0;
    cfg.num_parity_bits = 42 * Z;
    cfg.num_codeword_bits = K + cfg.num_parity_bits;
    cfg.puncture = true;
    cfg.redundancy_version = 0;

    // Create encoder and decoder
    ldpc_encoder_handle_t encoder;
    ldpc_decoder_handle_t decoder;

    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);

    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 25;
    ldpc_decoder_configure(decoder, &cfg, &dec_params);

    int enc_words = ldpc_encoder_get_output_words(encoder);

    // Allocate memory
    uint32_t *d_input, *d_encoded;
    float *d_llrs;
    uint32_t *d_decoded;

    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
    cudaMalloc(&d_llrs, N_full * sizeof(float));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Test pattern: all 0xAA
    std::vector<uint32_t> h_input(K_words, 0xAAAAAAAA);
    for (int i = K; i < K_words * 32; i++) {
        h_input[i/32] &= ~(1u << (i%32));
    }
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    // Create LLRs directly from encoded bits (no rate matching)
    // This gives PERFECT LLRs without puncturing
    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<float> h_llrs(N_full);
    for (int i = 0; i < N_full; i++) {
        int bit = (h_encoded[i/32] >> (i%32)) & 1;
        h_llrs[i] = bit ? -127.0f : 127.0f;
    }
    // Puncture first 2*Z bits
    for (int i = 0; i < 2 * Z; i++) {
        h_llrs[i] = 0.0f;
    }
    cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);

    // Print the first few LLRs for diagnostics.
    printf("Z=%d: First 20 LLRs: ", Z);
    for (int i = 0; i < 20 && i < N_full; i++) {
        printf("%.0f ", h_llrs[i]);
    }
    printf("...\n");

    // Decode
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode(decoder, d_llrs, d_decoded, stream);
    cudaStreamSynchronize(stream);

    // Compare
    std::vector<uint32_t> h_decoded(K_words);
    cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < K; i++) {
        int in = (h_input[i/32] >> (i%32)) & 1;
        int out = (h_decoded[i/32] >> (i%32)) & 1;
        if (in != out) errors++;
    }

    printf("Z=%d: Input[0]=0x%08X, Decoded[0]=0x%08X, Errors=%d/%d\n",
           Z, h_input[0], h_decoded[0], errors, K);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaFree(d_llrs);
    cudaFree(d_decoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);

    return errors;
}

int main() {
    ocudu_phy_cuda_init();

    printf("=== Test 1: All-zeros codeword (no encoder) ===\n");
    int test_values[] = {9, 18, 32, 36, 72};
    for (int z : test_values) {
        int errors = test_all_zeros(z);
        printf("Z=%-3d: %d errors %s\n", z, errors, errors == 0 ? "PASS" : "FAIL");
    }

    printf("\n=== Test 2: Encode -> LLRs -> Decode (with puncturing) ===\n");
    for (int z : test_values) {
        test_with_encoder(z);
    }

    return 0;
}
