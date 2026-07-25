// Test decoder with filler bits (Kd < K, filler LLRs = +127)
// This isolates whether filler bit handling causes FP16 decoder issues
#include "ocudu_phy_cuda.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Helper: Get bit with MSB-first byte ordering (encoder format)
inline int get_bit_msb(const uint32_t* data, int bit_idx) {
    int word_idx = bit_idx / 32;
    int bit_in_word = bit_idx % 32;
    int byte_in_word = bit_in_word / 8;
    int bit_in_byte = 7 - (bit_in_word % 8);
    int bit_pos = byte_in_word * 8 + bit_in_byte;
    return (data[word_idx] >> bit_pos) & 1;
}

// Helper: Set bit with MSB-first byte ordering
inline void set_bit_msb(uint32_t* data, int bit_idx, int value) {
    int word_idx = bit_idx / 32;
    int bit_in_word = bit_idx % 32;
    int byte_in_word = bit_in_word / 8;
    int bit_in_byte = 7 - (bit_in_word % 8);
    int bit_pos = byte_in_word * 8 + bit_in_byte;
    if (value) {
        data[word_idx] |= (1u << bit_pos);
    } else {
        data[word_idx] &= ~(1u << bit_pos);
    }
}

// Helper: Get bit with LSB-first ordering (decoder format)
inline int get_bit_lsb(const uint32_t* data, int bit_idx) {
    return (data[bit_idx / 32] >> (bit_idx % 32)) & 1;
}

int test_encode_decode_with_filler(int Z, uint32_t pattern) {
    const int bg = 2;
    const int Kb = 10;
    const int K = Kb * Z;
    const int K_words = (K + 31) / 32;
    const int N_cols = 52;
    const int N_full = N_cols * Z;
    const int punctured = 2 * Z;  // First 2 columns punctured

    // Msg3-like parameters: actual info bits < K
    const int Kd = 104;  // Actual info bits (like TBS + CRC)
    const int F = K - Kd;  // Filler bits

    printf("  Testing Z=%d, K=%d, Kd=%d, F=%d, pattern=0x%08X\n", Z, K, Kd, F, pattern);

    // Config with filler
    nr_ldpc_config_t cfg = {};
    cfg.base_graph = bg;
    cfg.lifting_size = Z;
    cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(Z);
    cfg.num_info_bits = Kd;  // Actual info bits (like full pipeline)
    cfg.num_filler_bits = F;
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
    __half *d_llrs_half;

    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
    cudaMalloc(&d_llrs, N_full * sizeof(float));
    cudaMalloc(&d_llrs_half, N_full * sizeof(__half));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create input: info bits with pattern, filler bits as 0
    std::vector<uint32_t> h_input(K_words, 0);
    // Set info bits (0 to Kd-1) with pattern
    for (int i = 0; i < Kd; i++) {
        int pattern_bit = (pattern >> (i % 32)) & 1;
        set_bit_msb(h_input.data(), i, pattern_bit);
    }
    // Filler bits (Kd to K-1) are already 0

    printf("    Input words: ");
    for (int i = 0; i < K_words; i++) {
        printf("0x%08X ", h_input[i]);
    }
    printf("\n");

    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("    Encoded first 6 words: ");
    for (int i = 0; i < 6 && i < enc_words; i++) {
        printf("0x%08X ", h_encoded[i]);
    }
    printf("\n");

    // Create LLRs: punctured=0, data from encoder, filler=+127
    // Also simulate rate matcher behavior: positions beyond E are set to 0
    const int E = 792;  // Representative rate-matched length for this filler scenario.
    const int N_cb = N_full - punctured;  // 900 = circular buffer size

    std::vector<float> h_llrs(N_full);
    std::vector<__half> h_llrs_half(N_full);
    std::vector<int8_t> h_llrs_int8(N_full);

    for (int i = 0; i < N_full; i++) {
        if (i < punctured) {
            // Punctured bits: LLR = 0 (erasure)
            h_llrs[i] = 0.0f;
            h_llrs_half[i] = __float2half(0.0f);
            h_llrs_int8[i] = 0;
        } else if (i >= Kd && i < K) {
            // Filler bits: LLR = +127 (known 0)
            h_llrs[i] = 127.0f;
            h_llrs_half[i] = __float2half(127.0f);
            h_llrs_int8[i] = 127;
        // } else if (i >= punctured + E) {
        //     // Beyond rate-matched length: LLR = 0 (like rate dematcher does)
        //     // This simulates the last 32 positions (904-935) being 0
        //     h_llrs[i] = 0.0f;
        //     h_llrs_int8[i] = 0;
        // }
        } else {
            // Data/parity bits: Read from encoder output using MSB-first
            int bit_val = get_bit_msb(h_encoded.data(), i);
            h_llrs[i] = bit_val ? -127.0f : 127.0f;
            h_llrs_half[i] = __float2half(bit_val ? -127.0f : 127.0f);
            h_llrs_int8[i] = bit_val ? -127 : 127;
        }
    }

    printf("    Positions beyond E (>=%d): ", punctured + E);
    for (int i = punctured + E; i < punctured + E + 8 && i < N_full; i++) {
        printf("%d ", h_llrs_int8[i]);
    }
    printf("\n");

    cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_llrs_half, h_llrs_half.data(), N_full * sizeof(__half), cudaMemcpyHostToDevice);

    printf("    Punctured LLRs (0-7): ");
    for (int i = 0; i < 8; i++) printf("%d ", h_llrs_int8[i]);
    printf("\n");

    printf("    Data LLRs (%d-%d): ", punctured, punctured + 7);
    for (int i = punctured; i < punctured + 8; i++) printf("%d ", h_llrs_int8[i]);
    printf("\n");

    printf("    Filler LLRs (%d-%d): ", Kd, Kd + 7);
    for (int i = Kd; i < Kd + 8 && i < K; i++) printf("%d ", h_llrs_int8[i]);
    printf("\n");

    printf("    Parity LLRs (904-911): ");
    for (int i = 904; i < 912 && i < N_full; i++) printf("%d ", h_llrs_int8[i]);
    printf("\n");

    // Test float decoder
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode(decoder, d_llrs, d_decoded, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_decoded_float(K_words);
    cudaMemcpy(h_decoded_float.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("    Float decoded: ");
    for (int i = 0; i < K_words; i++) printf("0x%08X ", h_decoded_float[i]);
    printf("\n");

    // Test FP16 batch decoder
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode_batch_half(decoder, d_llrs_half, d_decoded, 1, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_decoded_half(K_words);
    cudaMemcpy(h_decoded_half.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("    FP16 decoded:  ");
    for (int i = 0; i < K_words; i++) printf("0x%08X ", h_decoded_half[i]);
    printf("\n");

    // Compare using proper orderings (only check Kd bits, not filler)
    int float_errors = 0;
    int half_errors = 0;
    for (int i = 0; i < Kd; i++) {
        int input_bit = get_bit_msb(h_input.data(), i);
        int float_bit = get_bit_lsb(h_decoded_float.data(), i);
        int half_bit = get_bit_lsb(h_decoded_half.data(), i);

        if (input_bit != float_bit) float_errors++;
        if (input_bit != half_bit) half_errors++;
    }

    printf("    Float errors: %d/%d, FP16 errors: %d/%d\n", float_errors, Kd, half_errors, Kd);

    if (half_errors > 0) {
        printf("    FP16 error positions: ");
        int count = 0;
        for (int i = 0; i < Kd && count < 15; i++) {
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

    printf("Decoder Test WITH Filler Bits (Kd=104, F=76, punctured=36)\n");
    printf("============================================================\n\n");

    int total_tests = 0;
    int passed_tests = 0;

    // Only test Z=18 (Msg3 case)
    int Z = 18;
    uint32_t test_patterns[] = {0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555};

    printf("Z=%d:\n", Z);
    for (uint32_t pattern : test_patterns) {
        total_tests++;
        if (test_encode_decode_with_filler(Z, pattern) == 0) {
            passed_tests++;
        }
    }

    printf("\n============================================================\n");
    printf("SUMMARY: %d/%d tests passed\n", passed_tests, total_tests);
    printf("============================================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
