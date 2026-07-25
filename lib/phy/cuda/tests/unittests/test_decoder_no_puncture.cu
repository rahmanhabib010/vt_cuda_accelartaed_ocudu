// Simple decoder test - NO PUNCTURING (provide all LLRs)
#include "ocudu_phy_cuda.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Helper: Get/set bits with MSB-first byte ordering (encoder format)
inline int get_bit_msb(const uint32_t* data, int bit_idx) {
    int word_idx = bit_idx / 32;
    int bit_in_word = bit_idx % 32;
    int byte_in_word = bit_in_word / 8;
    int bit_in_byte = 7 - (bit_in_word % 8);
    int bit_pos = byte_in_word * 8 + bit_in_byte;
    return (data[word_idx] >> bit_pos) & 1;
}

// Helper: Get bit with LSB-first ordering (decoder format)
inline int get_bit_lsb(const uint32_t* data, int bit_idx) {
    return (data[bit_idx / 32] >> (bit_idx % 32)) & 1;
}

// Test encode -> decode with perfect LLRs (no puncturing)
int test_encode_decode_no_puncture(int Z, uint32_t pattern) {
    const int bg = 2;
    const int Kb = 10;
    const int K = Kb * Z;
    const int K_words = (K + 31) / 32;
    const int N_cols = 52;
    const int N_full = N_cols * Z;

    printf("  Testing Z=%d, pattern=0x%08X\n", Z, pattern);

    // Config - NO PUNCTURING
    nr_ldpc_config_t cfg = {};
    cfg.base_graph = bg;
    cfg.lifting_size = Z;
    cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(Z);
    cfg.num_info_bits = K;
    cfg.num_filler_bits = 0;
    cfg.num_parity_bits = 42 * Z;
    cfg.num_codeword_bits = K + cfg.num_parity_bits;
    cfg.puncture = false;  // NO PUNCTURING
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
    __half *d_llrs_half;

    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
    cudaMalloc(&d_llrs, N_full * sizeof(float));
    cudaMalloc(&d_llrs_half, N_full * sizeof(__half));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create input with pattern - simple LSB-first packing
    std::vector<uint32_t> h_input(K_words, 0);
    for (int i = 0; i < K_words; i++) {
        h_input[i] = pattern;
    }
    // Clear bits beyond K
    int excess_bits = K_words * 32 - K;
    if (excess_bits > 0) {
        h_input[K_words - 1] &= (1u << (32 - excess_bits)) - 1;
    }
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    printf("    Input words: ");
    for (int i = 0; i < K_words && i < 4; i++) {
        printf("0x%08X ", h_input[i]);
    }
    printf("\n");

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    // Get encoded codeword
    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("    Encoded first 4 words: ");
    for (int i = 0; i < 4 && i < enc_words; i++) {
        printf("0x%08X ", h_encoded[i]);
    }
    printf("\n");

    // Create LLRs from ALL encoded bits (no puncturing)
    // Encoder output uses MSB-first byte ordering
    // LLRs are indexed linearly 0 to N-1, matching the decoder expectation
    std::vector<float> h_llrs(N_full);
    std::vector<__half> h_llrs_half(N_full);
    std::vector<int8_t> h_llrs_int8(N_full);

    for (int i = 0; i < N_full; i++) {
        // Read encoder output using MSB-first ordering
        int bit_val = get_bit_msb(h_encoded.data(), i);
        h_llrs[i] = bit_val ? -127.0f : 127.0f;
        h_llrs_half[i] = __float2half(bit_val ? -127.0f : 127.0f);
        h_llrs_int8[i] = bit_val ? -127 : 127;
    }

    cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_llrs_half, h_llrs_half.data(), N_full * sizeof(__half), cudaMemcpyHostToDevice);

    printf("    First 8 LLRs: ");
    for (int i = 0; i < 8; i++) {
        printf("%d ", h_llrs_int8[i]);
    }
    printf("\n");

    // Test float decoder
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode(decoder, d_llrs, d_decoded, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_decoded_float(K_words);
    cudaMemcpy(h_decoded_float.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("    Float decoded: ");
    for (int i = 0; i < K_words && i < 4; i++) {
        printf("0x%08X ", h_decoded_float[i]);
    }
    printf("\n");

    // Test FP16 half decoder
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode_batch_half(decoder, d_llrs_half, d_decoded, 1, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_decoded_half(K_words);
    cudaMemcpy(h_decoded_half.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("    FP16 decoded:  ");
    for (int i = 0; i < K_words && i < 4; i++) {
        printf("0x%08X ", h_decoded_half[i]);
    }
    printf("\n");

    // Compare decoder output with encoder input
    // Encoder input uses MSB-first, decoder output uses LSB-first
    // For systematic bits, logical bit i should match
    int float_errors = 0;
    int half_errors = 0;
    for (int i = 0; i < K; i++) {
        // Input uses MSB-first ordering
        int input_bit = get_bit_msb(h_input.data(), i);
        // Decoder outputs LSB-first
        int float_bit = get_bit_lsb(h_decoded_float.data(), i);
        int half_bit = get_bit_lsb(h_decoded_half.data(), i);

        if (input_bit != float_bit) float_errors++;
        if (input_bit != half_bit) half_errors++;
    }

    printf("    Float errors: %d/%d, FP16 errors: %d/%d\n", float_errors, K, half_errors, K);

    if (float_errors > 0 || half_errors > 0) {
        printf("    FAILED! ");
        if (float_errors > 0) printf("Float errors at: ");
        int count = 0;
        for (int i = 0; i < K && count < 5; i++) {
            int input_bit = get_bit_msb(h_input.data(), i);
            int float_bit = get_bit_lsb(h_decoded_float.data(), i);
            if (input_bit != float_bit) {
                printf("%d ", i);
                count++;
            }
        }
        if (half_errors > 0) printf("| FP16 errors at: ");
        count = 0;
        for (int i = 0; i < K && count < 5; i++) {
            int input_bit = get_bit_msb(h_input.data(), i);
            int half_bit = get_bit_lsb(h_decoded_half.data(), i);
            if (input_bit != half_bit) {
                printf("%d ", i);
                count++;
            }
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaFree(d_llrs);
    cudaFree(d_llrs_half);
    cudaFree(d_decoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);

    return (float_errors == 0 && half_errors == 0) ? 0 : -1;
}

int main() {
    ocudu_phy_cuda_init();

    printf("Decoder Test WITHOUT Puncturing (perfect channel)\n");
    printf("==================================================\n\n");

    int total_tests = 0;
    int passed_tests = 0;

    int test_z[] = {18, 32, 64};
    uint32_t test_patterns[] = {0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555, 0x12345678};

    for (int Z : test_z) {
        printf("\nZ=%d:\n", Z);
        for (uint32_t pattern : test_patterns) {
            total_tests++;
            if (test_encode_decode_no_puncture(Z, pattern) == 0) {
                passed_tests++;
            }
        }
    }

    printf("\n==================================================\n");
    printf("SUMMARY: %d/%d tests passed\n", passed_tests, total_tests);
    printf("==================================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
