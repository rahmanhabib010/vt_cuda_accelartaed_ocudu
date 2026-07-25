// Test to verify encoder/decoder bit ordering consistency
// Uses MSB-first byte ordering throughout (matching 3GPP convention)
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Helper: Get bit at linear index with MSB-first byte ordering
// Maps linear bit index to MSB-first position within 32-bit words
inline int get_bit_msb_first(const uint32_t* data, int bit_idx) {
    int word_idx = bit_idx / 32;
    int bit_in_word = bit_idx % 32;
    // MSB-first within each byte: reverse bits 0-2 (within byte)
    int byte_in_word = bit_in_word / 8;
    int bit_in_byte = 7 - (bit_in_word % 8);
    int bit_pos = byte_in_word * 8 + bit_in_byte;
    return (data[word_idx] >> bit_pos) & 1;
}

// Helper: Set bit at linear index with MSB-first byte ordering
inline void set_bit_msb_first(uint32_t* data, int bit_idx, int value) {
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

// Helper: Get bit at linear index with LSB-first ordering (naive)
inline int get_bit_lsb_first(const uint32_t* data, int bit_idx) {
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;
    return (data[word_idx] >> bit_pos) & 1;
}

int test_ordering(int Z, uint32_t pattern, const char* name, bool use_msb_first) {
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

    // Create input data
    // For MSB-first test: store pattern using MSB-first convention
    // For LSB-first test: store pattern directly
    std::vector<uint32_t> h_input(K_words, 0);
    for (int i = 0; i < K; i++) {
        int bit = (pattern >> (i % 32)) & 1;
        if (use_msb_first) {
            set_bit_msb_first(h_input.data(), i, bit);
        } else {
            if (bit) h_input[i/32] |= (1u << (i%32));
        }
    }
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    // Get encoded codeword
    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Create LLRs from encoded bits (using same bit ordering)
    std::vector<float> h_llrs(N_full);
    for (int i = 0; i < N_full; i++) {
        int bit;
        if (use_msb_first) {
            bit = get_bit_msb_first(h_encoded.data(), i);
        } else {
            bit = get_bit_lsb_first(h_encoded.data(), i);
        }
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
    int errors = 0;
    for (int i = 0; i < K; i++) {
        int in_bit, out_bit;
        if (use_msb_first) {
            in_bit = get_bit_msb_first(h_input.data(), i);
            out_bit = get_bit_msb_first(h_decoded.data(), i);
        } else {
            in_bit = get_bit_lsb_first(h_input.data(), i);
            out_bit = get_bit_lsb_first(h_decoded.data(), i);
        }
        if (in_bit != out_bit) errors++;
    }

    const char* ordering = use_msb_first ? "MSB-first" : "LSB-first";
    printf("  %-12s (%s): Errors=%3d/%d %s\n", name, ordering, errors, K, errors == 0 ? "PASS" : "FAIL");

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

    printf("Testing bit ordering consistency:\n\n");

    int test_z[] = {9, 18, 32, 36};

    for (int Z : test_z) {
        printf("=== Z=%d ===\n", Z);
        // Test with LSB-first (current test behavior)
        test_ordering(Z, 0x00000000, "All zeros", false);
        test_ordering(Z, 0xFFFFFFFF, "All ones", false);
        test_ordering(Z, 0x00000001, "Single bit", false);
        test_ordering(Z, 0xAAAAAAAA, "Alternating", false);

        printf("\n");

        // Test with MSB-first (3GPP convention)
        test_ordering(Z, 0x00000000, "All zeros", true);
        test_ordering(Z, 0xFFFFFFFF, "All ones", true);
        test_ordering(Z, 0x00000001, "Single bit", true);
        test_ordering(Z, 0xAAAAAAAA, "Alternating", true);

        printf("\n");
    }

    return 0;
}
