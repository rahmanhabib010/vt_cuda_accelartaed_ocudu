// Diagnostic encoder output for several lifting sizes.
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

void test_encoder(int Z) {
    const int bg = 2;
    const int Kb = 10;
    const int K = Kb * Z;
    const int K_words = (K + 31) / 32;
    const int N_cols = 52;
    const int N_full = N_cols * Z;
    const int N_words = (N_full + 31) / 32;

    printf("\n=== Z=%d, K=%d, N=%d ===\n", Z, K, N_full);

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

    printf("Lifting set index: %d\n", cfg.lifting_set_index);

    ldpc_encoder_handle_t encoder;
    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);

    int enc_words = ldpc_encoder_get_output_words(encoder);
    printf("Encoder output words: %d\n", enc_words);

    uint32_t *d_input, *d_encoded;
    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Simple test pattern: just bit 0 set
    std::vector<uint32_t> h_input(K_words, 0);
    h_input[0] = 0x00000001;  // Only bit 0 set
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    // Check output
    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Print first few encoded bits
    printf("Input bits (first 32): ");
    for (int i = 0; i < 32 && i < K; i++) {
        int bit = (h_input[i/32] >> (i%32)) & 1;
        printf("%d", bit);
    }
    printf("\n");

    printf("Encoded bits (systematic, col 0-1, first 32): ");
    for (int i = 0; i < 32 && i < 2*Z; i++) {
        int bit = (h_encoded[i/32] >> (i%32)) & 1;
        printf("%d", bit);
    }
    printf("\n");

    printf("Encoded bits (systematic, col 2, first 32): ");
    for (int i = 2*Z; i < 2*Z + 32 && i < 3*Z; i++) {
        int bit = (h_encoded[i/32] >> (i%32)) & 1;
        printf("%d", bit);
    }
    printf("\n");

    // Count ones in encoded output
    int total_ones = 0;
    for (int i = 0; i < enc_words * 32 && i < N_full; i++) {
        total_ones += (h_encoded[i/32] >> (i%32)) & 1;
    }
    printf("Total ones in codeword: %d / %d\n", total_ones, N_full);

    // Verify parity check: encode all zeros should give all zeros
    cudaMemset(d_input, 0, K_words * sizeof(uint32_t));
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int zeros_ones = 0;
    for (int i = 0; i < enc_words * 32 && i < N_full; i++) {
        zeros_ones += (h_encoded[i/32] >> (i%32)) & 1;
    }
    printf("All-zeros input produces %d ones (should be 0)\n", zeros_ones);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
}

int main() {
    ocudu_phy_cuda_init();

    test_encoder(9);
    test_encoder(18);
    test_encoder(32);
    test_encoder(36);

    return 0;
}
