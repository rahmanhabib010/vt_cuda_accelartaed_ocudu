// Diagnostic decoder behavior checks.
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

void test_decoder_direct_llrs(int Z) {
    const int bg = 2;
    const int Kb = 10;
    const int K = Kb * Z;
    const int K_words = (K + 31) / 32;
    const int N_cols = 52;
    const int N_full = N_cols * Z;

    printf("\n=== Direct LLR test Z=%d ===\n", Z);

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

    printf("Config: K=%d, N=%d, lifting_set=%d\n", K, N_full, cfg.lifting_set_index);

    ldpc_decoder_handle_t decoder;
    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 25;
    ldpc_decoder_configure(decoder, &cfg, &dec_params);

    float *d_llrs;
    uint32_t *d_decoded;

    cudaMalloc(&d_llrs, N_full * sizeof(float));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Test 1: All zeros codeword (LLRs all positive = 0 bits)
    printf("\nTest 1: All-zeros codeword (LLRs all +127)\n");
    {
        std::vector<float> h_llrs(N_full, 127.0f);
        cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
        ldpc_decoder_decode(decoder, d_llrs, d_decoded, stream);
        cudaStreamSynchronize(stream);

        std::vector<uint32_t> h_decoded(K_words);
        cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        int ones = 0;
        for (int i = 0; i < K; i++) {
            ones += (h_decoded[i/32] >> (i%32)) & 1;
        }
        printf("  Decoded %d ones (expected 0): %s\n", ones, ones == 0 ? "PASS" : "FAIL");
    }

    // Test 2: All ones in systematic (LLRs negative for systematic, positive for parity)
    printf("\nTest 2: All-ones systematic (LLRs -127 for systematic, +127 for parity)\n");
    {
        std::vector<float> h_llrs(N_full);
        for (int i = 0; i < N_full; i++) {
            if (i < K) {
                h_llrs[i] = -127.0f;  // Systematic bits = 1
            } else {
                h_llrs[i] = 127.0f;   // Parity bits = 0 (will be computed by decoder)
            }
        }
        cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
        ldpc_decoder_decode(decoder, d_llrs, d_decoded, stream);
        cudaStreamSynchronize(stream);

        std::vector<uint32_t> h_decoded(K_words);
        cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        int ones = 0;
        for (int i = 0; i < K; i++) {
            ones += (h_decoded[i/32] >> (i%32)) & 1;
        }
        printf("  Decoded %d ones (expected %d): %s\n", ones, K, ones == K ? "PASS" : "FAIL");
    }

    // Test 3: Use encoder to generate valid codeword, then decode with perfect LLRs
    printf("\nTest 3: Encode all-ones, then decode with perfect LLRs\n");
    {
        ldpc_encoder_handle_t encoder;
        ldpc_encoder_create(&encoder);
        ldpc_encoder_configure(encoder, &cfg);

        int enc_words = ldpc_encoder_get_output_words(encoder);

        uint32_t *d_input, *d_encoded;
        cudaMalloc(&d_input, K_words * sizeof(uint32_t));
        cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));

        // All ones input
        std::vector<uint32_t> h_input(K_words, 0xFFFFFFFF);
        // Clear bits beyond K
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

        // Count bits in encoder output
        int enc_ones = 0;
        for (int i = 0; i < N_full; i++) {
            enc_ones += (h_encoded[i/32] >> (i%32)) & 1;
        }
        printf("  Encoder produced %d ones in %d bits\n", enc_ones, N_full);

        // Create LLRs from encoder output
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

        std::vector<uint32_t> h_decoded(K_words);
        cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Compare
        int errors = 0;
        for (int i = 0; i < K; i++) {
            int in = (h_input[i/32] >> (i%32)) & 1;
            int out = (h_decoded[i/32] >> (i%32)) & 1;
            if (in != out) errors++;
        }
        printf("  Decode errors: %d/%d %s\n", errors, K, errors == 0 ? "PASS" : "FAIL");

        // Print first 32 bits of each
        printf("  Input first 32 bits:  ");
        for (int i = 0; i < 32; i++) printf("%d", (h_input[0] >> i) & 1);
        printf("\n");
        printf("  Decoded first 32 bits:");
        for (int i = 0; i < 32; i++) printf("%d", (h_decoded[0] >> i) & 1);
        printf("\n");

        cudaFree(d_input);
        cudaFree(d_encoded);
        ldpc_encoder_destroy(encoder);
    }

    // Cleanup
    cudaFree(d_llrs);
    cudaFree(d_decoded);
    cudaStreamDestroy(stream);
    ldpc_decoder_destroy(decoder);
}

int main() {
    ocudu_phy_cuda_init();

    printf("Decoder diagnostic tests:\n");

    test_decoder_direct_llrs(18);
    test_decoder_direct_llrs(32);

    return 0;
}
