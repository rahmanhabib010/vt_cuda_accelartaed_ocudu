// Trace encoder output words to find spurious bits
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

void test_single_bit_words(int Z, int input_bit) {
    const int bg = 2;
    const int Kb = 10;
    const int K = Kb * Z;
    const int K_words = (K + 31) / 32;
    const int N_cols = 52;
    const int N_full = N_cols * Z;

    printf("\n=== Z=%d, input_bit=%d ===\n", Z, input_bit);

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

    ldpc_encoder_handle_t encoder;
    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);

    int enc_words = ldpc_encoder_get_output_words(encoder);
    int K_enc_words = (K + 31) / 32;  // Words for systematic portion

    uint32_t *d_input, *d_encoded;
    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Single bit input
    std::vector<uint32_t> h_input(K_words, 0);
    if (input_bit < K) {
        h_input[input_bit / 32] |= (1u << (input_bit % 32));
    }
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    printf("Input words: ");
    for (int i = 0; i < K_words; i++) {
        printf("[%d]=0x%08X ", i, h_input[i]);
    }
    printf("\n");

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Print all non-zero words in systematic portion
    printf("Encoder output (systematic portion, %d bits, %d words):\n", K, K_enc_words);
    for (int i = 0; i < K_enc_words; i++) {
        if (h_encoded[i] != 0) {
            printf("  word[%d] = 0x%08X (bits %d-%d)\n", i, h_encoded[i], i*32, (i+1)*32-1);
            // Print individual bits
            for (int b = 0; b < 32; b++) {
                if ((h_encoded[i] >> b) & 1) {
                    int bit_idx = i * 32 + b;
                    if (bit_idx < K) {
                        int col = bit_idx / Z;
                        int pos = bit_idx % Z;
                        printf("    bit %d (col=%d, z=%d) SET\n", bit_idx, col, pos);
                    }
                }
            }
        }
    }

    // Find all set bits in systematic portion using linear scan
    printf("All set bits in systematic (linear scan):\n  ");
    int count = 0;
    for (int i = 0; i < K; i++) {
        if ((h_encoded[i/32] >> (i%32)) & 1) {
            printf("%d ", i);
            count++;
        }
    }
    printf("\n  Total: %d bits set\n", count);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
}

int main() {
    ocudu_phy_cuda_init();

    printf("Encoder word-level trace:\n");

    // Z=18 tests
    test_single_bit_words(18, 0);   // Should only set bit 0 (after MSB reorder: logical 7)
    test_single_bit_words(18, 7);   // Should only set bit 7 (after MSB reorder: logical 0)

    // Z=32 tests for comparison
    test_single_bit_words(32, 0);
    test_single_bit_words(32, 7);

    return 0;
}
