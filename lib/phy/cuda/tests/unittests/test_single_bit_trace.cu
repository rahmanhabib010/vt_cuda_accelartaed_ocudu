// Trace a single bit through encoder and decoder to understand bit ordering
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

void print_bits(const uint32_t* data, int num_bits, const char* label) {
    printf("%s: ", label);
    for (int i = 0; i < num_bits && i < 64; i++) {
        int bit = (data[i/32] >> (i%32)) & 1;
        printf("%d", bit);
        if ((i+1) % 8 == 0) printf(" ");
    }
    printf("\n");
}

int test_single_bit(int Z, int bit_position) {
    const int bg = 2;
    const int Kb = 10;
    const int K = Kb * Z;
    const int K_words = (K + 31) / 32;
    const int N_cols = 52;
    const int N_full = N_cols * Z;

    printf("\n=== Z=%d, bit_position=%d ===\n", Z, bit_position);

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

    // Create input with single bit set at specified position
    std::vector<uint32_t> h_input(K_words, 0);
    if (bit_position < K) {
        h_input[bit_position / 32] |= (1u << (bit_position % 32));
    }
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    printf("Input (first 32 bits):\n");
    print_bits(h_input.data(), 32, "  Raw bits");
    printf("  Bit %d set in word %d at position %d\n", bit_position, bit_position/32, bit_position%32);

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    // Get encoded codeword
    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("\nEncoder output (first 64 bits = column 0 + partial column 1 for Z=%d):\n", Z);
    print_bits(h_encoded.data(), 64, "  Raw bits");

    // Count set bits in systematic portion (first Kb*Z bits)
    int sys_ones = 0;
    for (int i = 0; i < K; i++) {
        sys_ones += (h_encoded[i/32] >> (i%32)) & 1;
    }
    printf("  Systematic portion: %d ones out of %d bits\n", sys_ones, K);

    // Find where the 1s are in systematic portion
    printf("  Set bit positions in systematic: ");
    for (int i = 0; i < K; i++) {
        if ((h_encoded[i/32] >> (i%32)) & 1) {
            printf("%d ", i);
        }
    }
    printf("\n");

    // Create LLRs from encoded bits (linear ordering)
    std::vector<float> h_llrs(N_full);
    for (int i = 0; i < N_full; i++) {
        int bit = (h_encoded[i/32] >> (i%32)) & 1;
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

    printf("\nDecoder output (first 32 bits):\n");
    print_bits(h_decoded.data(), 32, "  Raw bits");

    // Find where the 1s are in decoded output
    printf("  Set bit positions in decoded: ");
    for (int i = 0; i < K; i++) {
        if ((h_decoded[i/32] >> (i%32)) & 1) {
            printf("%d ", i);
        }
    }
    printf("\n");

    // Compare
    int errors = 0;
    for (int i = 0; i < K; i++) {
        int in = (h_input[i/32] >> (i%32)) & 1;
        int out = (h_decoded[i/32] >> (i%32)) & 1;
        if (in != out) errors++;
    }
    printf("\nResult: Errors=%d/%d %s\n", errors, K, errors == 0 ? "PASS" : "FAIL");

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

    printf("Tracing single bit through encoder/decoder:\n");

    // Test with Z=18 (Msg3 case) and various bit positions
    test_single_bit(18, 0);   // First bit
    test_single_bit(18, 7);   // Bit 7 (end of first byte)
    test_single_bit(18, 8);   // Bit 8 (start of second byte)
    test_single_bit(18, 17);  // Bit 17 (last bit of first column for Z=18)
    test_single_bit(18, 18);  // Bit 18 (first bit of second column)

    // Also test with Z=32 for comparison
    test_single_bit(32, 0);
    test_single_bit(32, 7);
    test_single_bit(32, 31);
    test_single_bit(32, 32);

    return 0;
}
