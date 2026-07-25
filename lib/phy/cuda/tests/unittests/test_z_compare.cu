// Compare Z=18 vs Z=32 decoding step by step
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

// Test decode at given Z value with specified iterations
void test_decode(int Z, int max_iter) {
    const int bg = 2;
    const int Kb = 10;
    const int K = Kb * Z;
    const int K_words = (K + 31) / 32;
    const int N_cols = 52;
    const int N_full = N_cols * Z;
    int E = 50 * Z;
    int E_words = (E + 31) / 32;

    printf("\n=== Z=%d, iterations=%d ===\n", Z, max_iter);

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
    rate_matcher_handle_t rm_tx, rm_rx;

    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);

    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = max_iter;
    ldpc_decoder_configure(decoder, &cfg, &dec_params);

    int enc_words = ldpc_encoder_get_output_words(encoder);

    rate_matcher_create(&rm_tx);
    rate_matcher_create(&rm_rx);
    nr_rate_match_config_t rm_cfg = {.E = E, .Q_m = 2, .rv = 0, .N_cb = E, .k0 = 0, .limited_buffer = false};
    rate_matcher_configure_tx(rm_tx, &cfg, &rm_cfg);
    rate_matcher_configure_rx(rm_rx, &cfg, &rm_cfg);

    uint32_t *d_input, *d_encoded, *d_rate_matched, *d_decoded;
    float *d_llrs, *d_derate_llrs;

    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
    cudaMalloc(&d_rate_matched, E_words * sizeof(uint32_t));
    cudaMalloc(&d_llrs, E * sizeof(float));
    cudaMalloc(&d_derate_llrs, N_full * sizeof(float));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Test pattern: simple pattern with few 1s
    std::vector<uint32_t> h_input(K_words, 0x00000000);
    h_input[0] = 0x00000001;  // Just one bit set

    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);

    // Rate match
    cudaMemset(d_rate_matched, 0, E_words * sizeof(uint32_t));
    rate_matcher_match(rm_tx, d_encoded, d_rate_matched, stream);
    cudaStreamSynchronize(stream);

    // Create LLRs
    std::vector<uint32_t> h_rm(E_words);
    cudaMemcpy(h_rm.data(), d_rate_matched, E_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<float> h_llrs(E);
    for (int i = 0; i < E; i++) {
        int bit = (h_rm[i/32] >> (i%32)) & 1;
        h_llrs[i] = bit ? -127.0f : 127.0f;
    }
    cudaMemcpy(d_llrs, h_llrs.data(), E * sizeof(float), cudaMemcpyHostToDevice);

    // Rate dematch
    cudaMemset(d_derate_llrs, 0, N_full * sizeof(float));
    rate_matcher_dematch(rm_rx, d_llrs, d_derate_llrs, stream);
    cudaStreamSynchronize(stream);

    // Check dematched LLRs
    std::vector<float> h_derate(N_full);
    cudaMemcpy(h_derate.data(), d_derate_llrs, N_full * sizeof(float), cudaMemcpyDeviceToHost);

    // Find where bit 0 ends up (should be at position 2*Z due to puncturing)
    int bit0_pos = 2 * Z;
    printf("Bit 0 LLR at position %d: %.0f (expect -127 for bit=1)\n", bit0_pos, h_derate[bit0_pos]);

    // Decode
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode(decoder, d_derate_llrs, d_decoded, stream);
    cudaStreamSynchronize(stream);

    // Compare
    std::vector<uint32_t> h_decoded(K_words);
    cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Input:   0x%08X\n", h_input[0]);
    printf("Decoded: 0x%08X\n", h_decoded[0]);

    int errors = 0;
    for (int i = 0; i < K; i++) {
        int in = (h_input[i/32] >> (i%32)) & 1;
        int out = (h_decoded[i/32] >> (i%32)) & 1;
        if (in != out) errors++;
    }
    printf("Bit errors: %d / %d\n", errors, K);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaFree(d_rate_matched);
    cudaFree(d_llrs);
    cudaFree(d_derate_llrs);
    cudaFree(d_decoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);
    rate_matcher_destroy(rm_tx);
    rate_matcher_destroy(rm_rx);
}

int main() {
    ocudu_phy_cuda_init();

    printf("Testing decoder with simple pattern (just bit 0 set)\n");

    // Test with 1 iteration
    test_decode(18, 1);
    test_decode(32, 1);

    // Test with 5 iterations
    test_decode(18, 5);
    test_decode(32, 5);

    // Test with 25 iterations
    test_decode(18, 25);
    test_decode(32, 25);

    return 0;
}
