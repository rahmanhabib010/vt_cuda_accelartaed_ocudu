// Test encoder/decoder with proper MSB-first byte ordering
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Helper: Get MSB-first bit position within 32-bit word
inline int msb_first_pos(int linear_pos) {
    int bit_in_word = linear_pos % 32;
    int byte_in_word = bit_in_word / 8;
    int bit_in_byte = 7 - (bit_in_word % 8);
    return byte_in_word * 8 + bit_in_byte;
}

// Helper: Set bit at linear position with MSB-first byte ordering
inline void set_bit_msb(uint32_t* data, int linear_pos, int value) {
    int word_idx = linear_pos / 32;
    int bit_pos = msb_first_pos(linear_pos);
    if (value) {
        data[word_idx] |= (1u << bit_pos);
    } else {
        data[word_idx] &= ~(1u << bit_pos);
    }
}

// Helper: Get bit at linear position with MSB-first byte ordering
inline int get_bit_msb(const uint32_t* data, int linear_pos) {
    int word_idx = linear_pos / 32;
    int bit_pos = msb_first_pos(linear_pos);
    return (data[word_idx] >> bit_pos) & 1;
}

int test_encode_decode(int Z, uint32_t pattern, const char* name) {
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

    // Create input with pattern using MSB-first byte ordering
    // (matching what GPU encoder expects)
    std::vector<uint32_t> h_input(K_words, 0);
    for (int i = 0; i < K; i++) {
        int bit = (pattern >> (i % 32)) & 1;
        set_bit_msb(h_input.data(), i, bit);
    }
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    // Get encoded codeword
    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Create LLRs from encoded bits using MSB-first byte ordering
    // (matching how GPU encoder produced the output)
    std::vector<float> h_llrs(N_full);
    for (int i = 0; i < N_full; i++) {
        int bit = get_bit_msb(h_encoded.data(), i);
        h_llrs[i] = bit ? -127.0f : 127.0f;
    }
    cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);

    // Decode
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode(decoder, d_llrs, d_decoded, stream);
    cudaStreamSynchronize(stream);

    // Get decoded output
    std::vector<uint32_t> h_decoded(K_words);
    cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Compare using same bit ordering as input
    // Decoder outputs LSB-first, so we need to compare accordingly
    int errors = 0;
    for (int i = 0; i < K; i++) {
        // Get input bit (we stored with MSB-first)
        int in_bit = get_bit_msb(h_input.data(), i);

        // Get decoded bit - decoder outputs LSB-first
        int out_bit = (h_decoded[i/32] >> (i%32)) & 1;

        if (in_bit != out_bit) {
            errors++;
        }
    }

    printf("  %-12s (Z=%d): Errors=%3d/%d %s\n", name, Z, errors, K, errors == 0 ? "PASS" : "FAIL");

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

    printf("Testing encoder→decoder with MSB-first bit ordering:\n\n");

    int test_z[] = {9, 18, 32, 36};

    for (int Z : test_z) {
        printf("Z=%d:\n", Z);
        test_encode_decode(Z, 0x00000000, "All zeros");
        test_encode_decode(Z, 0xFFFFFFFF, "All ones");
        test_encode_decode(Z, 0x00000001, "Single bit");
        test_encode_decode(Z, 0x12345678, "Random");
        test_encode_decode(Z, 0xAAAAAAAA, "Alternating");
        printf("\n");
    }

    return 0;
}
