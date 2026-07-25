// Simple test for Z=18 decoding without filler bits
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    ocudu_phy_cuda_init();

    // BG2, Z=18: K = 10 * 18 = 180 bits (no filler)
    const int bg = 2;
    const int Z = 18;
    const int Kb = 10;
    const int K = Kb * Z;  // 180
    const int K_words = (K + 31) / 32;  // 6
    const int N_cols = 52;
    const int N_full = N_cols * Z;  // 936
    const int N_cb = 50 * Z;  // 900
    const int E = N_cb;  // Use full buffer (no rate limiting)

    printf("Z=%d, K=%d, N=%d, E=%d\n", Z, K, N_full, E);

    // Config: no filler bits (num_info_bits = K)
    nr_ldpc_config_t cfg = {};
    cfg.base_graph = bg;
    cfg.lifting_size = Z;
    cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(Z);
    cfg.num_info_bits = K;  // Full 180 bits
    cfg.num_filler_bits = 0;
    cfg.num_parity_bits = 42 * Z;
    cfg.num_codeword_bits = K + cfg.num_parity_bits;
    cfg.puncture = true;
    cfg.redundancy_version = 0;

    printf("iLS=%d, num_info_bits=%d, num_filler_bits=%d\n",
           cfg.lifting_set_index, cfg.num_info_bits, cfg.num_filler_bits);

    // Create encoder/decoder
    ldpc_encoder_handle_t encoder;
    ldpc_decoder_handle_t decoder;
    rate_matcher_handle_t rm_tx, rm_rx;

    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);

    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 20;
    ldpc_decoder_configure(decoder, &cfg, &dec_params);

    // Now test Z=32 with the same decoder to compare
    printf("\nComparing Z=18 vs Z=32 decode...\n");

    rate_matcher_create(&rm_tx);
    rate_matcher_create(&rm_rx);
    nr_rate_match_config_t rm_cfg = {.E = E, .Q_m = 2, .rv = 0, .N_cb = N_cb, .k0 = 0, .limited_buffer = false};
    rate_matcher_configure_tx(rm_tx, &cfg, &rm_cfg);
    rate_matcher_configure_rx(rm_rx, &cfg, &rm_cfg);

    // Allocate memory
    int enc_words = ldpc_encoder_get_output_words(encoder);
    int E_words = (E + 31) / 32;

    uint32_t *d_input, *d_encoded, *d_rate_matched;
    float *d_llrs, *d_derate_llrs;
    uint32_t *d_decoded;

    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
    cudaMalloc(&d_rate_matched, E_words * sizeof(uint32_t));
    cudaMalloc(&d_llrs, E * sizeof(float));
    cudaMalloc(&d_derate_llrs, N_full * sizeof(float));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Test pattern: all 0xAA
    std::vector<uint32_t> h_input(K_words, 0xAAAAAAAA);
    // Mask bits beyond K
    for (int i = K; i < K_words * 32; i++) {
        h_input[i/32] &= ~(1u << (i%32));
    }

    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Encode
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);

    // Rate match
    cudaMemset(d_rate_matched, 0, E_words * sizeof(uint32_t));
    rate_matcher_match(rm_tx, d_encoded, d_rate_matched, stream);
    cudaStreamSynchronize(stream);

    // Convert to LLRs
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

    // Decode
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode(decoder, d_derate_llrs, d_decoded, stream);
    cudaStreamSynchronize(stream);

    // Compare
    std::vector<uint32_t> h_decoded(K_words);
    cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Input:   ");
    for (int i = 0; i < K_words; i++) printf("0x%08X ", h_input[i]);
    printf("\n");

    printf("Decoded: ");
    for (int i = 0; i < K_words; i++) printf("0x%08X ", h_decoded[i]);
    printf("\n");

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

    return errors > 0 ? 1 : 0;
}
