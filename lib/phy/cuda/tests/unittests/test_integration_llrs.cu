/**
 * Test: Load BG1 LLRs dumped from srsRAN integration and decode standalone.
 * This isolates whether the issue is in the data or the decoder configuration.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "ldpc_decoder.h"
#include "ldpc_encoder.h"
#include "nr_ldpc_defs.h"

int main(int argc, char* argv[]) {
    const char* llr_file = (argc > 1) ? argv[1] : "/tmp/bg1_integration_llrs.bin";

    printf("=== Test: Decode integration-dumped LLRs ===\n");
    printf("Loading: %s\n", llr_file);

    FILE* fp = fopen(llr_file, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", llr_file); return 1; }

    int bg, Z, N_total, F, Kd;
    fread(&bg, sizeof(int), 1, fp);
    fread(&Z, sizeof(int), 1, fp);
    fread(&N_total, sizeof(int), 1, fp);
    fread(&F, sizeof(int), 1, fp);
    fread(&Kd, sizeof(int), 1, fp);

    float* h_llrs = (float*)malloc(N_total * sizeof(float));
    fread(h_llrs, sizeof(float), N_total, fp);
    fclose(fp);

    int Kb = (bg == 1) ? 22 : 10;
    int N_cols = (bg == 1) ? 68 : 52;
    int M = (bg == 1) ? 46 : 42;
    int K = Kb * Z;
    int punctured = 2 * Z;

    printf("bg=%d Z=%d N=%d F=%d Kd=%d K=%d\n", bg, Z, N_total, F, Kd, K);

    // LLR statistics
    int n_zero = 0, n_pos = 0, n_neg = 0;
    float min_v = 1e9f, max_v = -1e9f;
    for (int i = 0; i < N_total; i++) {
        if (h_llrs[i] == 0.0f) n_zero++;
        else if (h_llrs[i] > 0) n_pos++;
        else n_neg++;
        if (h_llrs[i] < min_v) min_v = h_llrs[i];
        if (h_llrs[i] > max_v) max_v = h_llrs[i];
    }
    printf("LLR stats: zero=%d pos=%d neg=%d min=%.1f max=%.1f\n", n_zero, n_pos, n_neg, min_v, max_v);

    // Check punctured region
    int punct_nonzero = 0;
    for (int i = 0; i < punctured; i++)
        if (h_llrs[i] != 0.0f) punct_nonzero++;
    printf("Punctured [0,%d): %d non-zero (should be 0)\n", punctured, punct_nonzero);

    // Check filler region
    int filler_wrong = 0;
    for (int i = Kd; i < K; i++)
        if (h_llrs[i] != 127.0f) filler_wrong++;
    printf("Filler [%d,%d): %d not +127 (should be 0)\n", Kd, K, filler_wrong);

    // Count non-zero in info region
    int info_zero = 0;
    for (int i = punctured; i < Kd; i++)
        if (h_llrs[i] == 0.0f) info_zero++;
    printf("Info [%d,%d): %d zeros out of %d\n", punctured, Kd, info_zero, Kd - punctured);

    // Count non-zero in parity region
    int parity_nonzero = 0;
    for (int i = K; i < N_total; i++)
        if (h_llrs[i] != 0.0f) parity_nonzero++;
    printf("Parity [%d,%d): %d non-zero out of %d\n", K, N_total, parity_nonzero, N_total - K);

    // Try decoding with SAME config as integration (llr_clamp=32, auto_scale=true)
    printf("\n=== Decode with integration config (clamp=32, scale=auto) ===\n");
    {
        ldpc_decoder_handle_t decoder;
        ldpc_decoder_create(&decoder);

        nr_ldpc_config_t cfg = {};
        cfg.base_graph = bg;
        cfg.lifting_size = Z;
        cfg.num_info_bits = Kd;
        cfg.num_parity_bits = M * Z;
        cfg.num_codeword_bits = N_cols * Z;
        cfg.num_filler_bits = F;
        cfg.max_parity_nodes = M;

        ldpc_decoder_params_t params;
        ldpc_decoder_params_init(&params);
        params.max_iterations = 20;
        params.early_termination = true;
        params.auto_scale = true;
        params.llr_clamp = 32.0f;
        params.crc_early_termination = false;
        params.skip_iteration_stats = false;

        ldpc_decoder_configure(decoder, &cfg, &params);

        int K_words = (K + 31) / 32;
        float* d_llrs; uint32_t* d_output;
        cudaMalloc(&d_llrs, N_total * sizeof(float));
        cudaMalloc(&d_output, K_words * sizeof(uint32_t));
        cudaMemcpy(d_llrs, h_llrs, N_total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, K_words * sizeof(uint32_t));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        ldpc_decoder_decode_batch(decoder, d_llrs, d_output, 1, stream);
        cudaStreamSynchronize(stream);

        float iters = ldpc_decoder_get_avg_iterations(decoder);
        printf("  Iterations: %.1f (max=20)\n", iters);
        printf("  Result: %s\n", iters < 20 ? "CONVERGED" : "FAILED TO CONVERGE");

        cudaFree(d_llrs); cudaFree(d_output);
        cudaStreamDestroy(stream);
        ldpc_decoder_destroy(decoder);
    }

    // KEY TEST: Compute effective nof_layers like the CPU decoder does,
    // then decode with reduced max_parity_nodes.
    // CPU formula: nof_layers = (max(input_size + 2*Z, min_cb_length) - K) / Z
    // where input_size = number of non-punctured LLRs from rate dematching.
    {
        // Count actual transmitted LLRs (non-zero, excluding filler which is +127)
        int n_transmitted = 0;
        for (int i = punctured; i < N_total; i++) {
            if (i >= Kd && i < K) continue; // Skip filler
            if (h_llrs[i] != 0.0f) n_transmitted++;
        }
        int input_size = n_transmitted;
        int min_cb_length = K + Z;
        int cb_length = (input_size + punctured > min_cb_length) ? (input_size + punctured) : min_cb_length;
        int nof_layers = (cb_length - K) / Z;
        printf("\n=== Decode with CPU-matched layers (M=%d instead of %d) ===\n", nof_layers, M);
        printf("  (input_size=%d, cb_length=%d, K=%d, Z=%d)\n", input_size, cb_length, K, Z);

        ldpc_decoder_handle_t decoder;
        ldpc_decoder_create(&decoder);

        nr_ldpc_config_t cfg = {};
        cfg.base_graph = bg;
        cfg.lifting_size = Z;
        cfg.num_info_bits = Kd;
        cfg.num_parity_bits = nof_layers * Z;  // Only transmitted parity
        cfg.num_codeword_bits = K + nof_layers * Z;
        cfg.num_filler_bits = F;
        cfg.max_parity_nodes = nof_layers;

        ldpc_decoder_params_t params;
        ldpc_decoder_params_init(&params);
        params.max_iterations = 20;
        params.early_termination = true;
        params.auto_scale = true;
        params.llr_clamp = 32.0f;
        params.crc_early_termination = false;
        params.skip_iteration_stats = false;

        ldpc_decoder_configure(decoder, &cfg, &params);

        int K_words = (K + 31) / 32;
        float* d_llrs; uint32_t* d_output;
        cudaMalloc(&d_llrs, N_total * sizeof(float));
        cudaMalloc(&d_output, K_words * sizeof(uint32_t));
        cudaMemcpy(d_llrs, h_llrs, N_total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, K_words * sizeof(uint32_t));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        ldpc_decoder_decode_batch(decoder, d_llrs, d_output, 1, stream);
        cudaStreamSynchronize(stream);

        float iters = ldpc_decoder_get_avg_iterations(decoder);
        printf("  Iterations: %.1f (max=20)\n", iters);
        printf("  Result: %s\n", iters < 20 ? "CONVERGED" : "FAILED TO CONVERGE");

        cudaFree(d_llrs); cudaFree(d_output);
        cudaStreamDestroy(stream);
        ldpc_decoder_destroy(decoder);
    }

    // Try with default config (llr_clamp=default=32, auto_scale=true)
    printf("\n=== Decode with default config (clamp=127, scale=auto) ===\n");
    {
        ldpc_decoder_handle_t decoder;
        ldpc_decoder_create(&decoder);

        nr_ldpc_config_t cfg = {};
        cfg.base_graph = bg;
        cfg.lifting_size = Z;
        cfg.num_info_bits = Kd;
        cfg.num_parity_bits = M * Z;
        cfg.num_codeword_bits = N_cols * Z;
        cfg.num_filler_bits = F;
        cfg.max_parity_nodes = M;

        ldpc_decoder_params_t params;
        ldpc_decoder_params_init(&params);
        params.max_iterations = 20;
        params.early_termination = true;
        params.auto_scale = true;
        params.llr_clamp = 127.0f;
        params.crc_early_termination = false;
        params.skip_iteration_stats = false;

        ldpc_decoder_configure(decoder, &cfg, &params);

        int K_words = (K + 31) / 32;
        float* d_llrs; uint32_t* d_output;
        cudaMalloc(&d_llrs, N_total * sizeof(float));
        cudaMalloc(&d_output, K_words * sizeof(uint32_t));
        cudaMemcpy(d_llrs, h_llrs, N_total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, K_words * sizeof(uint32_t));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        ldpc_decoder_decode_batch(decoder, d_llrs, d_output, 1, stream);
        cudaStreamSynchronize(stream);

        float iters = ldpc_decoder_get_avg_iterations(decoder);
        printf("  Iterations: %.1f (max=20)\n", iters);
        printf("  Result: %s\n", iters < 20 ? "CONVERGED" : "FAILED TO CONVERGE");

        cudaFree(d_llrs); cudaFree(d_output);
        cudaStreamDestroy(stream);
        ldpc_decoder_destroy(decoder);
    }

    // KEY TEST: Encode the info bits with OCUDU PHY CUDA encoder, compare parity
    printf("\n=== Compare: srsRAN parity vs OCUDU PHY CUDA encoder parity ===\n");
    {
        ldpc_encoder_handle_t encoder;
        ldpc_encoder_create(&encoder);

        nr_ldpc_config_t enc_cfg = {};
        enc_cfg.base_graph = bg;
        enc_cfg.lifting_size = Z;
        enc_cfg.num_info_bits = Kd;
        enc_cfg.num_parity_bits = M * Z;
        enc_cfg.num_codeword_bits = N_cols * Z;
        enc_cfg.num_filler_bits = F;
        ldpc_encoder_configure(encoder, &enc_cfg);

        // Extract info bits from LLR signs and pack in MSB-first byte format
        int K_words = (K + 31) / 32;
        uint32_t* h_info = (uint32_t*)calloc(K_words, sizeof(uint32_t));
        for (int i = 0; i < K; i++) {
            int bit_val;
            if (i < punctured) {
                bit_val = 0;  // Punctured - use 0 (will be overwritten by encoder)
            } else if (i >= Kd) {
                bit_val = 0;  // Filler - known zero
            } else {
                bit_val = (h_llrs[i] < 0.0f) ? 1 : 0;
            }
            // Pack MSB-first byte format
            int word_idx = i / 32;
            int bit_in_word = i % 32;
            int byte_in_word = bit_in_word / 8;
            int bit_in_byte = 7 - (bit_in_word % 8);
            int bit_pos = byte_in_word * 8 + bit_in_byte;
            if (bit_val) h_info[word_idx] |= (1u << bit_pos);
        }

        int N_words = (N_cols * Z + 31) / 32;
        uint32_t *d_info, *d_encoded;
        cudaMalloc(&d_info, K_words * sizeof(uint32_t));
        cudaMalloc(&d_encoded, N_words * sizeof(uint32_t));
        cudaMemcpy(d_info, h_info, K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_encoded, 0, N_words * sizeof(uint32_t));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        ldpc_encoder_encode(encoder, d_info, d_encoded, stream);
        cudaStreamSynchronize(stream);

        uint32_t* h_encoded = (uint32_t*)malloc(N_words * sizeof(uint32_t));
        cudaMemcpy(h_encoded, d_encoded, N_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Extract bit from MSB-first format
        auto get_bit = [](const uint32_t* w, int i) -> int {
            int word = i / 32, biw = i % 32;
            int by = biw / 8, bib = 7 - (biw % 8);
            return (w[word] >> (by * 8 + bib)) & 1;
        };

        // Compare systematic bits (cols 2..Kb-1)
        int sys_mm = 0;
        for (int i = punctured; i < Kd; i++) {
            int enc_bit = get_bit(h_encoded, i);
            int llr_bit = (h_llrs[i] < 0.0f) ? 1 : 0;
            if (enc_bit != llr_bit) sys_mm++;
        }
        printf("  Systematic [%d,%d): %d mismatches\n", punctured, Kd, sys_mm);

        // Compare parity bits
        int par_mm = 0, par_total = 0;
        for (int i = K; i < N_total; i++) {
            if (h_llrs[i] == 0.0f) continue;  // Skip erased
            par_total++;
            int enc_bit = get_bit(h_encoded, i);
            int llr_bit = (h_llrs[i] < 0.0f) ? 1 : 0;
            if (enc_bit != llr_bit) {
                par_mm++;
                if (par_mm <= 5) {
                    printf("    Parity mismatch at pos %d (col=%d, z=%d): OCUDU PHY CUDA=%d srsRAN=%d\n",
                           i, i / Z, i % Z, enc_bit, llr_bit);
                }
            }
        }
        printf("  Parity: %d mismatches out of %d non-zero\n", par_mm, par_total);

        if (sys_mm == 0 && par_mm == 0) {
            printf("  *** ENCODERS MATCH ***\n");
        } else if (sys_mm == 0 && par_mm > 0) {
            printf("  *** ENCODER PARITY MISMATCH! ***\n");

            // Create LLRs from OCUDU PHY CUDA-encoded data and try to decode
            printf("\n=== Decode OCUDU PHY CUDA-encoded LLRs (should converge) ===\n");
            float* h_ocudu_phy_cuda_llrs = (float*)calloc(N_total, sizeof(float));
            for (int i = 0; i < N_total; i++) {
                if (i < punctured) {
                    h_ocudu_phy_cuda_llrs[i] = 0.0f;
                } else if (i >= Kd && i < K) {
                    h_ocudu_phy_cuda_llrs[i] = 127.0f;  // Filler = known zero
                } else if (h_llrs[i] == 0.0f) {
                    h_ocudu_phy_cuda_llrs[i] = 0.0f;  // Erased (not transmitted)
                } else {
                    // Use OCUDU PHY CUDA-encoded bit sign, but srsRAN magnitude
                    int enc_bit = get_bit(h_encoded, i);
                    float mag = fabsf(h_llrs[i]);
                    h_ocudu_phy_cuda_llrs[i] = enc_bit ? -mag : mag;
                }
            }

            ldpc_decoder_handle_t decoder;
            ldpc_decoder_create(&decoder);

            nr_ldpc_config_t dec_cfg = {};
            dec_cfg.base_graph = bg;
            dec_cfg.lifting_size = Z;
            dec_cfg.num_info_bits = Kd;
            dec_cfg.num_parity_bits = M * Z;
            dec_cfg.num_codeword_bits = N_cols * Z;
            dec_cfg.num_filler_bits = F;
            dec_cfg.max_parity_nodes = M;

            ldpc_decoder_params_t params;
            ldpc_decoder_params_init(&params);
            params.max_iterations = 20;
            params.early_termination = true;
            params.auto_scale = true;
            params.llr_clamp = 32.0f;
            params.crc_early_termination = false;
            params.skip_iteration_stats = false;

            ldpc_decoder_configure(decoder, &dec_cfg, &params);

            int K_w = (K + 31) / 32;
            float* d_llrs; uint32_t* d_output;
            cudaMalloc(&d_llrs, N_total * sizeof(float));
            cudaMalloc(&d_output, K_w * sizeof(uint32_t));
            cudaMemcpy(d_llrs, h_ocudu_phy_cuda_llrs, N_total * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_output, 0, K_w * sizeof(uint32_t));

            ldpc_decoder_decode_batch(decoder, d_llrs, d_output, 1, stream);
            cudaStreamSynchronize(stream);
            float iters = ldpc_decoder_get_avg_iterations(decoder);
            printf("  OCUDU PHY CUDA-encoded: iterations=%.1f → %s\n",
                   iters, iters < 20 ? "CONVERGED" : "FAILED");

            free(h_ocudu_phy_cuda_llrs);
            cudaFree(d_llrs); cudaFree(d_output);
            ldpc_decoder_destroy(decoder);
        }

        free(h_info); free(h_encoded);
        cudaFree(d_info); cudaFree(d_encoded);
        cudaStreamDestroy(stream);
        ldpc_encoder_destroy(encoder);
    }

    free(h_llrs);
    return 0;
}
