/**
 * Test: Load BG1 LLRs dumped from srsRAN pipeline and decode with OCUDU PHY CUDA.
 * This isolates whether the srsRAN-encoded data is decodable by OCUDU PHY CUDA.
 *
 * Usage: test_srsran_llrs [path_to_llrs.bin]
 * Default: /tmp/bg1_llrs.bin
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "ldpc_decoder.h"
#include "ldpc_encoder.h"
#include "nr_ldpc_defs.h"

int main(int argc, char* argv[]) {
    const char* llr_file = (argc > 1) ? argv[1] : "/tmp/bg1_llrs.bin";

    printf("=== Test: Decode srsRAN-dumped LLRs with OCUDU PHY CUDA ===\n");
    printf("Loading LLRs from: %s\n", llr_file);

    // Read binary file: header (bg, Z, N_total, F) + LLR data
    FILE* fp = fopen(llr_file, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open %s\n", llr_file);
        return 1;
    }

    int bg, Z, N_total, F;
    fread(&bg, sizeof(int), 1, fp);
    fread(&Z, sizeof(int), 1, fp);
    fread(&N_total, sizeof(int), 1, fp);
    fread(&F, sizeof(int), 1, fp);

    printf("Header: bg=%d, Z=%d, N_total=%d, F=%d\n", bg, Z, N_total, F);

    float* h_llrs = (float*)malloc(N_total * sizeof(float));
    size_t nread = fread(h_llrs, sizeof(float), N_total, fp);
    fclose(fp);

    if ((int)nread != N_total) {
        fprintf(stderr, "ERROR: Expected %d floats, got %zu\n", N_total, nread);
        return 1;
    }

    // Print LLR statistics
    int n_zero = 0, n_pos127 = 0, n_neg127 = 0, n_other = 0;
    for (int i = 0; i < N_total; i++) {
        if (h_llrs[i] == 0.0f) n_zero++;
        else if (h_llrs[i] == 127.0f) n_pos127++;
        else if (h_llrs[i] == -127.0f) n_neg127++;
        else n_other++;
    }
    printf("LLR stats: zero=%d, +127=%d, -127=%d, other=%d\n", n_zero, n_pos127, n_neg127, n_other);

    // Check punctured region
    int Kb = (bg == 1) ? 22 : 10;
    int N_cols = (bg == 1) ? 68 : 52;
    int M = (bg == 1) ? 46 : 42;
    int K = Kb * Z;
    int punctured = 2 * Z;

    printf("Kb=%d, K=%d, N_cols=%d, M=%d, punctured=%d\n", Kb, K, N_cols, M, punctured);

    // Verify punctured region is zero
    bool punct_ok = true;
    for (int i = 0; i < punctured; i++) {
        if (h_llrs[i] != 0.0f) {
            printf("  WARN: punctured position %d has LLR=%.1f (expected 0.0)\n", i, h_llrs[i]);
            punct_ok = false;
        }
    }
    printf("Punctured [0, %d): %s\n", punctured, punct_ok ? "OK (all zero)" : "MISMATCH");

    // Verify filler region
    int Kd = K - F;  // info bits
    bool filler_ok = true;
    for (int i = Kd; i < K; i++) {
        if (h_llrs[i] != 127.0f) {
            printf("  WARN: filler position %d has LLR=%.1f (expected +127)\n", i, h_llrs[i]);
            filler_ok = false;
            if (i - Kd > 5) break;
        }
    }
    printf("Filler [%d, %d): %s\n", Kd, K, filler_ok ? "OK (all +127)" : "MISMATCH");

    // Decode with OCUDU PHY CUDA
    printf("\n=== OCUDU PHY CUDA Decode ===\n");

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
    cfg.puncture = true;

    ldpc_decoder_params_t params;
    ldpc_decoder_params_init(&params);
    params.max_iterations = 20;
    params.early_termination = true;
    params.auto_scale = true;
    params.llr_clamp = 127.0f;
    params.crc_early_termination = false;
    params.skip_iteration_stats = false;

    printf("Config: bg=%d Z=%d Kd=%d F=%d N=%d M=%d\n",
           cfg.base_graph, cfg.lifting_size, cfg.num_info_bits,
           cfg.num_filler_bits, cfg.num_codeword_bits, cfg.max_parity_nodes);

    ldpc_decoder_configure(decoder, &cfg, &params);

    // Allocate device memory
    int K_words = (K + 31) / 32;
    float* d_llrs;
    uint32_t* d_output;
    cudaMalloc(&d_llrs, N_total * sizeof(float));
    cudaMalloc(&d_output, K_words * sizeof(uint32_t));

    // Copy LLRs to device
    cudaMemcpy(d_llrs, h_llrs, N_total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, K_words * sizeof(uint32_t));

    // Decode
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nr_ldpc_status_t status = ldpc_decoder_decode_batch(decoder, d_llrs, d_output, 1, stream);
    cudaStreamSynchronize(stream);

    if (status != NR_LDPC_SUCCESS) {
        fprintf(stderr, "ERROR: decode failed with status %d\n", status);
        return 1;
    }

    float avg_iters = ldpc_decoder_get_avg_iterations(decoder);
    printf("Decode result: avg_iterations=%.1f\n", avg_iters);

    // Copy output back
    uint32_t* h_output = (uint32_t*)malloc(K_words * sizeof(uint32_t));
    cudaMemcpy(h_output, d_output, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Count non-zero output bits
    int n_ones = 0;
    for (int i = 0; i < K; i++) {
        int word = i / 32;
        int bit = i % 32;
        if ((h_output[word] >> bit) & 1) n_ones++;
    }
    printf("Output: %d ones out of %d systematic bits\n", n_ones, K);

    // ===================================================================
    // KEY TEST: Extract information bits from LLR signs, encode with OCUDU PHY CUDA,
    // and compare PARITY BITS against LLR signs.
    // If they differ, srsRAN and OCUDU PHY CUDA encoders produce different codewords.
    // ===================================================================
    printf("\n=== Verify: Extract info bits from LLRs, encode with OCUDU PHY CUDA, compare parity ===\n");

    // Extract systematic bits from LLR signs (positions 0..K-1 in GPU coordinates)
    // Information bits: columns 0..Kb-1, i.e., positions 0..K-1
    // In the LLR array: position i has the bit for column i/Z, z-position i%Z
    // Pack into uint32 in MSB-first byte format (OCUDU PHY CUDA encoder input format)
    int K_words_enc = (K + 31) / 32;
    uint32_t* h_info_packed = (uint32_t*)calloc(K_words_enc, sizeof(uint32_t));

    for (int i = 0; i < K; i++) {
        int bit_val;
        if (i < punctured) {
            // Punctured columns - unknown, set to 0
            bit_val = 0;
        } else if (i >= Kd && i < K) {
            // Filler bits - known zeros
            bit_val = 0;
        } else {
            // Information bits - from LLR sign
            bit_val = (h_llrs[i] < 0.0f) ? 1 : 0;
        }

        // Pack in MSB-first byte format within uint32
        int word_idx = i / 32;
        int bit_in_word = i % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int bit_pos = byte_in_word * 8 + bit_in_byte;
        if (bit_val) {
            h_info_packed[word_idx] |= (1u << bit_pos);
        }
    }

    // Copy info bits to device
    uint32_t* d_info;
    cudaMalloc(&d_info, K_words_enc * sizeof(uint32_t));
    cudaMemcpy(d_info, h_info_packed, K_words_enc * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Create encoder and encode
    ldpc_encoder_handle_t encoder;
    ldpc_encoder_create(&encoder);

    nr_ldpc_config_t enc_cfg = cfg;
    enc_cfg.redundancy_version = 0;
    ldpc_encoder_configure(encoder, &enc_cfg);

    uint32_t* d_encoded;
    int encoded_words = (N_cols * Z + 31) / 32;
    cudaMalloc(&d_encoded, encoded_words * sizeof(uint32_t));

    ldpc_encoder_encode(encoder, d_info, d_encoded, stream);
    cudaStreamSynchronize(stream);

    uint32_t* h_encoded = (uint32_t*)malloc(encoded_words * sizeof(uint32_t));
    cudaMemcpy(h_encoded, d_encoded, encoded_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Helper: extract bit from MSB-first byte format packed word
    auto get_msb_bit = [](const uint32_t* words, int bit_idx) -> int {
        int word_idx = bit_idx / 32;
        int bit_in_word = bit_idx % 32;
        int byte_in_word = bit_in_word / 8;
        int bit_in_byte = 7 - (bit_in_word % 8);
        int bit_pos = byte_in_word * 8 + bit_in_byte;
        return (words[word_idx] >> bit_pos) & 1;
    };

    // Compare systematic bits (cols 2..Kb-1, positions 2Z..K-1)
    int sys_mismatches = 0;
    for (int i = punctured; i < Kd; i++) {
        int encoded_bit = get_msb_bit(h_encoded, i);
        int llr_bit = (h_llrs[i] < 0.0f) ? 1 : 0;
        if (encoded_bit != llr_bit) sys_mismatches++;
    }
    printf("Systematic bits [%d, %d): %d mismatches (should be 0 for systematic code)\n",
           punctured, Kd, sys_mismatches);

    // Compare parity bits (cols Kb..N-1, positions K..N*Z-1)
    int par_mismatches = 0;
    int par_total = 0;
    for (int i = K; i < N_total; i++) {
        if (h_llrs[i] == 0.0f) continue;
        int encoded_bit = get_msb_bit(h_encoded, i);
        int llr_bit = (h_llrs[i] < 0.0f) ? 1 : 0;
        if (encoded_bit != llr_bit) {
            par_mismatches++;
            if (par_mismatches <= 10) {
                printf("  Parity mismatch at pos %d (col=%d, z=%d): OCUDU PHY CUDA=%d, srsRAN=%d\n",
                       i, i / Z, i % Z, encoded_bit, llr_bit);
            }
        }
        par_total++;
    }
    printf("Parity bits [%d, %d): %d mismatches out of %d\n", K, N_total, par_mismatches, par_total);

    if (sys_mismatches == 0 && par_mismatches == 0) {
        printf("\n*** Codewords MATCH: srsRAN and OCUDU PHY CUDA encoders agree ***\n");
        printf("The bug must be in the decoder or data path, not the H matrix.\n");
    } else if (sys_mismatches == 0 && par_mismatches > 0) {
        printf("\n*** ENCODER MISMATCH: Same info bits produce DIFFERENT parity ***\n");
        printf("The H matrices may differ, or the encoding algorithms differ.\n");
        printf("par_mismatches=%d out of %d (%.1f%%)\n",
               par_mismatches, par_total, 100.0 * par_mismatches / par_total);
    } else {
        printf("\n*** Systematic mismatch suggests bit extraction/packing error ***\n");
    }

    // Also try: does the OCUDU PHY CUDA-encoded codeword decode successfully?
    printf("\n=== Verify: Encode with OCUDU PHY CUDA, create LLRs, decode ===\n");
    float* h_ocudu_phy_cuda_llrs = (float*)calloc(N_total, sizeof(float));
    for (int i = 0; i < N_total; i++) {
        if (i < punctured) {
            h_ocudu_phy_cuda_llrs[i] = 0.0f;
        } else {
            int encoded_bit = get_msb_bit(h_encoded, i);
            h_ocudu_phy_cuda_llrs[i] = encoded_bit ? -127.0f : 127.0f;
        }
    }
    // Set filler to +127
    for (int i = Kd; i < K; i++) {
        h_ocudu_phy_cuda_llrs[i] = 127.0f;
    }

    cudaMemcpy(d_llrs, h_ocudu_phy_cuda_llrs, N_total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, K_words * sizeof(uint32_t));
    status = ldpc_decoder_decode_batch(decoder, d_llrs, d_output, 1, stream);
    cudaStreamSynchronize(stream);
    float avg_iters2 = ldpc_decoder_get_avg_iterations(decoder);
    printf("OCUDU PHY CUDA encode->decode: avg_iterations=%.1f\n", avg_iters2);
    if (avg_iters2 < params.max_iterations) {
        printf("*** OCUDU PHY CUDA's own codeword decodes fine (iterations=%.1f) ***\n", avg_iters2);
    }

    // Cleanup
    free(h_llrs);
    free(h_output);
    free(h_encoded);
    free(h_info_packed);
    free(h_ocudu_phy_cuda_llrs);
    cudaFree(d_llrs);
    cudaFree(d_output);
    cudaFree(d_encoded);
    cudaFree(d_info);
    cudaStreamDestroy(stream);
    ldpc_decoder_destroy(decoder);
    ldpc_encoder_destroy(encoder);

    return (par_mismatches == 0 && avg_iters < params.max_iterations) ? 0 : 1;
}
