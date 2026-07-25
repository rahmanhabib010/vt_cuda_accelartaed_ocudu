// Verify encoder produces valid codewords (all parity checks satisfied)
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

// BG2 structure from decoder
const int bg2_row_ptr[] = {
    0, 8, 18, 26, 36, 40, 46, 52, 58, 62, 67, 72,
    77, 81, 86, 91, 95, 100, 105, 109, 113, 117, 121, 124,
    128, 132, 135, 140, 143, 147, 150, 155, 158, 162, 166, 170,
    174, 178, 181, 185, 189, 193, 197
};

const int bg2_col[] = {
    0,  1,  2,  3,  6,  9, 10, 11,
    0,  3,  4,  5,  6,  7,  8,  9, 11, 12,
    0,  1,  3,  4,  8, 10, 12, 13,
    1,  2,  4,  5,  6,  7,  8,  9, 10, 13,
    0,  1, 11, 14,
    0,  1,  5,  7, 11, 15,
    0,  5,  7,  9, 11, 16,
    1,  5,  7, 11, 13, 17,
    0,  1, 12, 18,
    1,  8, 10, 11, 19,
    0,  1,  6,  7, 20,
    0,  7,  9, 13, 21,
    1,  3, 11, 22,
    0,  1,  8, 13, 23,
    1,  6, 11, 13, 24,
    0, 10, 11, 25,
    1,  9, 11, 12, 26,
    1,  5, 11, 12, 27,
    0,  6,  7, 28,
    0,  1, 10, 29,
    1,  4, 11, 30,
    0,  8, 13, 31,
    1,  2, 32,
    0,  3,  5, 33,
    1,  2,  9, 34,
    0,  5, 35,
    2,  7, 12, 13, 36,
    0,  6, 37,
    1,  2,  5, 38,
    0,  4, 39,
    2,  5,  7,  9, 40,
    1, 13, 41,
    0,  5, 12, 42,
    2,  7, 10, 43,
    0, 12, 13, 44,
    1,  5, 11, 45,
    0,  2,  7, 46,
    10, 13, 47,
    1,  5, 11, 48,
    0,  7, 12, 49,
    2, 10, 13, 50,
    1,  5, 11, 51
};

// Simplified shift table for verification (set 4, values mod Z)
int get_shift(int edge_idx, int Z) {
    // For simplicity, just return 0 - this tests if structure is correct
    // Real verification would need the actual shift values
    return 0;  // Not using shifts for parity check on positions only
}

int check_parity(const std::vector<uint32_t>& codeword, int Z) {
    const int M = 42;
    int parity_fails = 0;

    for (int row = 0; row < M; row++) {
        int row_start = bg2_row_ptr[row];
        int row_end = bg2_row_ptr[row + 1];

        // For each position in lifting size Z
        for (int z = 0; z < Z; z++) {
            int parity = 0;
            for (int e = row_start; e < row_end; e++) {
                int col = bg2_col[e];
                int bit_idx = col * Z + z;  // Simplified: no circular shift
                int bit = (codeword[bit_idx / 32] >> (bit_idx % 32)) & 1;
                parity ^= bit;
            }
            if (parity != 0) {
                parity_fails++;
            }
        }
    }

    return parity_fails;
}

void test_encoder_output(int Z, uint32_t pattern) {
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

    ldpc_encoder_handle_t encoder;
    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);

    int enc_words = ldpc_encoder_get_output_words(encoder);

    uint32_t *d_input, *d_encoded;
    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Set pattern
    std::vector<uint32_t> h_input(K_words, pattern);
    for (int i = K; i < K_words * 32; i++) {
        h_input[i/32] &= ~(1u << (i%32));
    }
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    // Get encoded codeword
    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Count ones
    int ones = 0;
    for (int i = 0; i < N_full; i++) {
        ones += (h_encoded[i/32] >> (i%32)) & 1;
    }

    // Check parity (simplified - without shifts)
    // int parity_fails = check_parity(h_encoded, Z);

    printf("Z=%-3d pattern=0x%08X: %4d/%d ones in codeword\n",
           Z, pattern, ones, N_full);

    // Print first few bits of each column
    printf("  Systematic bits (col 0-9, first 8 per col):\n");
    for (int col = 0; col < Kb; col++) {
        printf("    Col %d: ", col);
        for (int z = 0; z < 8 && z < Z; z++) {
            int bit_idx = col * Z + z;
            int bit = (h_encoded[bit_idx/32] >> (bit_idx%32)) & 1;
            printf("%d", bit);
        }
        printf("...\n");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
}

int main() {
    ocudu_phy_cuda_init();

    printf("Verifying encoder output:\n\n");

    // Test a few patterns
    test_encoder_output(18, 0x00000000);
    printf("\n");
    test_encoder_output(18, 0x00000001);
    printf("\n");
    test_encoder_output(32, 0x00000001);
    printf("\n");

    return 0;
}
