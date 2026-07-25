// Detailed trace of encoder output and LLR creation
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Get bit from packed data using MSB-first byte ordering
int get_bit_msb(const uint32_t* data, int bit_idx) {
    int word_idx = bit_idx / 32;
    int bit_in_word = bit_idx % 32;
    int byte_in_word = bit_in_word / 8;
    int bit_in_byte = 7 - (bit_in_word % 8);
    int bit_pos = byte_in_word * 8 + bit_in_byte;
    return (data[word_idx] >> bit_pos) & 1;
}

// Get bit from packed data using LSB-first (linear) ordering
int get_bit_lsb(const uint32_t* data, int bit_idx) {
    return (data[bit_idx / 32] >> (bit_idx % 32)) & 1;
}

void test_encode_decode_detail(int Z) {
    const int bg = 2;
    const int Kb = 10;
    const int K = Kb * Z;
    const int K_words = (K + 31) / 32;
    const int N_cols = 52;
    const int N_full = N_cols * Z;

    printf("\n=== Detailed trace Z=%d ===\n", Z);

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
    ldpc_decoder_handle_t decoder;

    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);

    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 25;
    ldpc_decoder_configure(decoder, &cfg, &dec_params);

    int enc_words = ldpc_encoder_get_output_words(encoder);

    uint32_t *d_input, *d_encoded;
    float *d_llrs;
    uint32_t *d_decoded;

    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
    cudaMalloc(&d_llrs, N_full * sizeof(float));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // All ones input
    std::vector<uint32_t> h_input(K_words, 0xFFFFFFFF);
    for (int i = K; i < K_words * 32; i++) {
        h_input[i/32] &= ~(1u << (i%32));
    }
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Print first few columns of encoder output in both orderings
    printf("\nEncoder output for first 3 columns (column = %d bits each):\n", Z);
    for (int col = 0; col < 3 && col < Kb; col++) {
        printf("  Column %d (bits %d-%d):\n", col, col*Z, (col+1)*Z-1);
        printf("    LSB-first: ");
        for (int z = 0; z < Z; z++) {
            printf("%d", get_bit_lsb(h_encoded.data(), col*Z + z));
            if ((z+1) % 8 == 0) printf(" ");
        }
        printf("\n");
        printf("    MSB-first: ");
        for (int z = 0; z < Z; z++) {
            printf("%d", get_bit_msb(h_encoded.data(), col*Z + z));
            if ((z+1) % 8 == 0) printf(" ");
        }
        printf("\n");
    }

    // Test 1: Create LLRs using LSB-first ordering (current behavior)
    printf("\nTest with LSB-first LLRs:\n");
    {
        std::vector<float> h_llrs(N_full);
        for (int i = 0; i < N_full; i++) {
            int bit = get_bit_lsb(h_encoded.data(), i);
            h_llrs[i] = bit ? -127.0f : 127.0f;
        }
        cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
        ldpc_decoder_decode(decoder, d_llrs, d_decoded, stream);
        cudaStreamSynchronize(stream);

        std::vector<uint32_t> h_decoded(K_words);
        cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        int errors = 0;
        printf("  Error positions: ");
        for (int i = 0; i < K; i++) {
            int in = get_bit_lsb(h_input.data(), i);
            int out = get_bit_lsb(h_decoded.data(), i);
            if (in != out) {
                if (errors < 20) printf("%d ", i);
                errors++;
            }
        }
        printf("%s\n", errors > 20 ? "..." : "");
        printf("  Total errors: %d/%d %s\n", errors, K, errors == 0 ? "PASS" : "FAIL");
    }

    // Test 2: Create LLRs using MSB-first ordering
    printf("\nTest with MSB-first LLRs:\n");
    {
        std::vector<float> h_llrs(N_full);
        for (int i = 0; i < N_full; i++) {
            int bit = get_bit_msb(h_encoded.data(), i);
            h_llrs[i] = bit ? -127.0f : 127.0f;
        }
        cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
        ldpc_decoder_decode(decoder, d_llrs, d_decoded, stream);
        cudaStreamSynchronize(stream);

        std::vector<uint32_t> h_decoded(K_words);
        cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Compare using MSB-first for both (if encoder used MSB-first, we should compare same way)
        int errors = 0;
        printf("  Error positions: ");
        for (int i = 0; i < K; i++) {
            int in = get_bit_msb(h_input.data(), i);
            int out = get_bit_lsb(h_decoded.data(), i);  // Decoder still outputs LSB-first
            if (in != out) {
                if (errors < 20) printf("%d ", i);
                errors++;
            }
        }
        printf("%s\n", errors > 20 ? "..." : "");
        printf("  Total errors (MSB input vs LSB output): %d/%d\n", errors, K);

        // Also compare LSB vs LSB
        errors = 0;
        for (int i = 0; i < K; i++) {
            int in = get_bit_lsb(h_input.data(), i);
            int out = get_bit_lsb(h_decoded.data(), i);
            if (in != out) errors++;
        }
        printf("  Total errors (LSB vs LSB): %d/%d %s\n", errors, K, errors == 0 ? "PASS" : "FAIL");
    }

    // Test 3: Check if encoder is actually using MSB-first for input
    printf("\nChecking encoder input interpretation:\n");
    {
        // Input a single '1' at bit position 0 (LSB of first word)
        std::vector<uint32_t> h_single_input(K_words, 0);
        h_single_input[0] = 1;  // Bit 0 set

        cudaMemcpy(d_input, h_single_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
        ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
        cudaStreamSynchronize(stream);

        std::vector<uint32_t> h_single_encoded(enc_words);
        cudaMemcpy(h_single_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        printf("  Input: bit 0 set (word[0] = 0x%08X)\n", h_single_input[0]);
        printf("  Encoder output word[0] = 0x%08X\n", h_single_encoded[0]);

        // Find which bits are set in systematic portion
        printf("  Set bits in systematic (LSB-first): ");
        for (int i = 0; i < K; i++) {
            if (get_bit_lsb(h_single_encoded.data(), i)) printf("%d ", i);
        }
        printf("\n");

        printf("  Set bits in systematic (MSB-first): ");
        for (int i = 0; i < K; i++) {
            if (get_bit_msb(h_single_encoded.data(), i)) printf("%d ", i);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaFree(d_llrs);
    cudaFree(d_decoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);
}

int main() {
    ocudu_phy_cuda_init();

    printf("Encoder/Decoder detailed trace:\n");

    test_encode_decode_detail(18);
    test_encode_decode_detail(32);

    return 0;
}
