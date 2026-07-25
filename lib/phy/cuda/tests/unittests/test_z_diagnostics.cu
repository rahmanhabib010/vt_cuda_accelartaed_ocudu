// Diagnostic test for Z=18 decoding - check APP values after 1 iteration.
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    ocudu_phy_cuda_init();

    const int Z = 18;
    const int bg = 2;
    const int Kb = 10;
    const int K = Kb * Z;  // 180
    const int K_words = (K + 31) / 32;
    const int N_cols = 52;
    const int N_full = N_cols * Z;  // 936

    printf("=== Z=%d Diagnostic Test ===\n", Z);

    // Config: no filler bits
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

    // Create encoder/decoder
    ldpc_encoder_handle_t encoder;
    ldpc_decoder_handle_t decoder;
    rate_matcher_handle_t rm_tx, rm_rx;

    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);

    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 1;  // Keep one iteration to expose intermediate APP values.
    ldpc_decoder_configure(decoder, &cfg, &dec_params);

    int enc_words = ldpc_encoder_get_output_words(encoder);
    int E = 50 * Z;  // N_cb
    int E_words = (E + 31) / 32;

    rate_matcher_create(&rm_tx);
    rate_matcher_create(&rm_rx);
    nr_rate_match_config_t rm_cfg = {.E = E, .Q_m = 2, .rv = 0, .N_cb = E, .k0 = 0, .limited_buffer = false};
    rate_matcher_configure_tx(rm_tx, &cfg, &rm_cfg);
    rate_matcher_configure_rx(rm_rx, &cfg, &rm_cfg);

    // Allocate memory
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

    // Test pattern: alternating 0xAA
    std::vector<uint32_t> h_input(K_words, 0xAAAAAAAA);
    // Mask bits beyond K
    for (int i = K; i < K_words * 32; i++) {
        h_input[i/32] &= ~(1u << (i%32));
    }
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Encode (all zeros)
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);

    // Rate match
    cudaMemset(d_rate_matched, 0, E_words * sizeof(uint32_t));
    rate_matcher_match(rm_tx, d_encoded, d_rate_matched, stream);
    cudaStreamSynchronize(stream);

    // Verify encoded bits are all zeros (since input is all zeros)
    std::vector<uint32_t> h_rm(E_words);
    cudaMemcpy(h_rm.data(), d_rate_matched, E_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    int nonzero = 0;
    for (int i = 0; i < E; i++) {
        if ((h_rm[i/32] >> (i%32)) & 1) nonzero++;
    }
    printf("Rate matched bits: %d/%d non-zero (should be 0 for all-zero codeword)\n", nonzero, E);

    // Create perfect LLRs based on actual encoded bits
    std::vector<float> h_llrs(E);
    for (int i = 0; i < E; i++) {
        int bit = (h_rm[i/32] >> (i%32)) & 1;
        h_llrs[i] = bit ? -127.0f : 127.0f;  // positive = 0, negative = 1
    }
    cudaMemcpy(d_llrs, h_llrs.data(), E * sizeof(float), cudaMemcpyHostToDevice);

    // Rate dematch
    cudaMemset(d_derate_llrs, 0, N_full * sizeof(float));
    rate_matcher_dematch(rm_rx, d_llrs, d_derate_llrs, stream);
    cudaStreamSynchronize(stream);

    // Check dematched LLRs
    std::vector<float> h_derate(N_full);
    cudaMemcpy(h_derate.data(), d_derate_llrs, N_full * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Dematched LLRs (first 36 = punctured): ");
    for (int i = 0; i < 12; i++) printf("%.0f ", h_derate[i]);
    printf("...\n");
    printf("Dematched LLRs (36-72 = systematic): ");
    for (int i = 36; i < 48; i++) printf("%.0f ", h_derate[i]);
    printf("...\n");

    // Decode with 1 iteration
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode(decoder, d_derate_llrs, d_decoded, stream);
    cudaStreamSynchronize(stream);

    // Compare
    std::vector<uint32_t> h_decoded(K_words);
    cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Expected: ");
    for (int i = 0; i < K_words; i++) printf("0x%08X ", h_input[i]);
    printf("\n");

    printf("Decoded:  ");
    for (int i = 0; i < K_words; i++) printf("0x%08X ", h_decoded[i]);
    printf("\n");

    int errors = 0;
    for (int i = 0; i < K; i++) {
        int in = (h_input[i/32] >> (i%32)) & 1;
        int out = (h_decoded[i/32] >> (i%32)) & 1;
        if (in != out) errors++;
    }
    printf("Bit errors after 1 iteration: %d / %d\n", errors, K);

    // Now test with more iterations
    dec_params.max_iterations = 25;
    ldpc_decoder_configure(decoder, &cfg, &dec_params);

    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
    ldpc_decoder_decode(decoder, d_derate_llrs, d_decoded, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Decoded (25 iter): ");
    for (int i = 0; i < K_words; i++) printf("0x%08X ", h_decoded[i]);
    printf("\n");

    errors = 0;
    for (int i = 0; i < K; i++) {
        int in = (h_input[i/32] >> (i%32)) & 1;
        int out = (h_decoded[i/32] >> (i%32)) & 1;
        if (in != out) errors++;
    }
    printf("Bit errors after 25 iterations: %d / %d\n", errors, K);

    // Compare with Z=32 (which we know works)
    printf("\n=== Now testing Z=32 for comparison ===\n");
    const int Z32 = 32;
    const int K32 = Kb * Z32;
    const int K_words32 = (K32 + 31) / 32;
    const int N_full32 = N_cols * Z32;
    int E32 = 50 * Z32;
    int E_words32 = (E32 + 31) / 32;

    nr_ldpc_config_t cfg32 = {};
    cfg32.base_graph = bg;
    cfg32.lifting_size = Z32;
    cfg32.lifting_set_index = nr_ldpc_get_lifting_set_index(Z32);
    cfg32.num_info_bits = K32;
    cfg32.num_filler_bits = 0;
    cfg32.num_parity_bits = 42 * Z32;
    cfg32.num_codeword_bits = K32 + cfg32.num_parity_bits;
    cfg32.puncture = true;
    cfg32.redundancy_version = 0;

    ldpc_encoder_handle_t encoder32;
    ldpc_decoder_handle_t decoder32;
    rate_matcher_handle_t rm_tx32, rm_rx32;

    ldpc_encoder_create(&encoder32);
    ldpc_encoder_configure(encoder32, &cfg32);

    ldpc_decoder_create(&decoder32);
    ldpc_decoder_params_t dec_params32;
    ldpc_decoder_params_init(&dec_params32);
    dec_params32.max_iterations = 25;
    ldpc_decoder_configure(decoder32, &cfg32, &dec_params32);

    int enc_words32 = ldpc_encoder_get_output_words(encoder32);

    rate_matcher_create(&rm_tx32);
    rate_matcher_create(&rm_rx32);
    nr_rate_match_config_t rm_cfg32 = {.E = E32, .Q_m = 2, .rv = 0, .N_cb = E32, .k0 = 0, .limited_buffer = false};
    rate_matcher_configure_tx(rm_tx32, &cfg32, &rm_cfg32);
    rate_matcher_configure_rx(rm_rx32, &cfg32, &rm_cfg32);

    uint32_t *d_input32, *d_encoded32, *d_rate_matched32, *d_decoded32;
    float *d_llrs32, *d_derate_llrs32;

    cudaMalloc(&d_input32, K_words32 * sizeof(uint32_t));
    cudaMalloc(&d_encoded32, enc_words32 * sizeof(uint32_t));
    cudaMalloc(&d_rate_matched32, E_words32 * sizeof(uint32_t));
    cudaMalloc(&d_llrs32, E32 * sizeof(float));
    cudaMalloc(&d_derate_llrs32, N_full32 * sizeof(float));
    cudaMalloc(&d_decoded32, K_words32 * sizeof(uint32_t));

    std::vector<uint32_t> h_input32(K_words32, 0xAAAAAAAA);
    for (int i = K32; i < K_words32 * 32; i++) {
        h_input32[i/32] &= ~(1u << (i%32));
    }
    cudaMemcpy(d_input32, h_input32.data(), K_words32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMemset(d_encoded32, 0, enc_words32 * sizeof(uint32_t));
    ldpc_encoder_encode(encoder32, d_input32, d_encoded32, stream);

    cudaMemset(d_rate_matched32, 0, E_words32 * sizeof(uint32_t));
    rate_matcher_match(rm_tx32, d_encoded32, d_rate_matched32, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_rm32(E_words32);
    cudaMemcpy(h_rm32.data(), d_rate_matched32, E_words32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<float> h_llrs32(E32);
    for (int i = 0; i < E32; i++) {
        int bit = (h_rm32[i/32] >> (i%32)) & 1;
        h_llrs32[i] = bit ? -127.0f : 127.0f;
    }
    cudaMemcpy(d_llrs32, h_llrs32.data(), E32 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_derate_llrs32, 0, N_full32 * sizeof(float));
    rate_matcher_dematch(rm_rx32, d_llrs32, d_derate_llrs32, stream);

    cudaMemset(d_decoded32, 0, K_words32 * sizeof(uint32_t));
    ldpc_decoder_decode(decoder32, d_derate_llrs32, d_decoded32, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_decoded32(K_words32);
    cudaMemcpy(h_decoded32.data(), d_decoded32, K_words32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int errors32 = 0;
    for (int i = 0; i < K32; i++) {
        int in = (h_input32[i/32] >> (i%32)) & 1;
        int out = (h_decoded32[i/32] >> (i%32)) & 1;
        if (in != out) errors32++;
    }
    printf("Z=32: Bit errors = %d / %d\n", errors32, K32);

    cudaFree(d_input32);
    cudaFree(d_encoded32);
    cudaFree(d_rate_matched32);
    cudaFree(d_llrs32);
    cudaFree(d_derate_llrs32);
    cudaFree(d_decoded32);
    ldpc_encoder_destroy(encoder32);
    ldpc_decoder_destroy(decoder32);
    rate_matcher_destroy(rm_tx32);
    rate_matcher_destroy(rm_rx32);

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
