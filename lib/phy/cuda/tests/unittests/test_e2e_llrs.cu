// Standalone test: load E2E-dumped LLRs and attempt decode
// This isolates whether the LLR values are the problem or the decoder config is.

#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    const char* filename = "/tmp/e2e_llrs_bg1.bin";
    if (argc > 1) filename = argv[1];

    FILE* fp = fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return 1; }

    // Read header: bg, Z, nof_cbs, N_full, Kd, F
    int hdr[6];
    fread(hdr, sizeof(int), 6, fp);
    int bg = hdr[0], Z = hdr[1], nof_cbs = hdr[2], N_full = hdr[3], Kd = hdr[4], F = hdr[5];
    int Kb = (bg == 1) ? 22 : 10;
    int K = Kb * Z;
    int parity_nodes = (bg == 1) ? 46 : 42;
    int K_words = (K + 31) / 32;

    printf("Loaded: bg=%d Z=%d nof_cbs=%d N_full=%d Kd=%d F=%d K=%d\n",
           bg, Z, nof_cbs, N_full, Kd, F, K);

    // Read LLR data
    std::vector<float> llrs(nof_cbs * N_full);
    fread(llrs.data(), sizeof(float), nof_cbs * N_full, fp);
    fclose(fp);

    // Print LLR statistics
    for (int cb = 0; cb < nof_cbs; cb++) {
        float* cb_llrs = llrs.data() + cb * N_full;
        int punct_nz=0, sys_nz=0, filler_nz=0, parity_nz=0;
        float sys_abs=0, par_abs=0;
        int punct_end = 2*Z;
        for (int i = 0; i < punct_end; i++) if (cb_llrs[i] != 0.0f) punct_nz++;
        for (int i = punct_end; i < Kd; i++) { if (cb_llrs[i] != 0.0f) { sys_nz++; sys_abs += fabsf(cb_llrs[i]); } }
        for (int i = Kd; i < K; i++) if (cb_llrs[i] != 0.0f) filler_nz++;
        for (int i = K; i < N_full; i++) { if (cb_llrs[i] != 0.0f) { parity_nz++; par_abs += fabsf(cb_llrs[i]); } }

        printf("CB%d: punct=%d/%d sys=%d/%d(avg=%.1f) filler=%d/%d parity=%d/%d(avg=%.1f)\n",
               cb, punct_nz, punct_end, sys_nz, Kd-punct_end,
               sys_nz > 0 ? sys_abs/sys_nz : 0,
               filler_nz, F, parity_nz, N_full-K,
               parity_nz > 0 ? par_abs/parity_nz : 0);
        printf("  first10 sys: ");
        for (int i = punct_end; i < punct_end+10; i++) printf("%.1f ", cb_llrs[i]);
        printf("\n  first10 parity: ");
        for (int i = K; i < K+10; i++) printf("%.1f ", cb_llrs[i]);
        printf("\n  filler[0..2]: %.1f %.1f %.1f\n", cb_llrs[Kd], cb_llrs[Kd+1], cb_llrs[Kd+2]);
    }

    // Initialize OCUDU PHY CUDA
    ocudu_phy_cuda_init();

    // Configure decoder
    nr_ldpc_config_t cfg = {};
    cfg.base_graph = bg;
    cfg.lifting_size = Z;
    cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(Z);
    cfg.num_info_bits = Kd;
    cfg.num_filler_bits = F;
    cfg.num_parity_bits = parity_nodes * Z;
    cfg.num_codeword_bits = K + cfg.num_parity_bits;
    cfg.puncture = true;
    cfg.redundancy_version = 0;

    ldpc_decoder_handle_t decoder;
    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 10;
    dec_params.early_termination = true;
    dec_params.auto_scale = true;
    dec_params.llr_clamp = 127.0f;
    dec_params.crc_early_termination = false;
    dec_params.skip_iteration_stats = false;
    dec_params.deferred_iteration_stats = false;
    ldpc_decoder_configure(decoder, &cfg, &dec_params);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Test 1: Decode E2E LLRs (CB0 only)
    printf("\n=== Test 1: Decode E2E LLRs (CB0) ===\n");
    {
        float* d_llrs;
        uint32_t* d_decoded;
        cudaMalloc(&d_llrs, N_full * sizeof(float));
        cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

        cudaMemcpy(d_llrs, llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));

        ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
        cudaStreamSynchronize(stream);

        float avg_iters = ldpc_decoder_get_avg_iterations(decoder);
        printf("  avg_iterations=%.1f\n", avg_iters);

        std::vector<uint32_t> h_decoded(K_words);
        cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        printf("  first4: %08x %08x %08x %08x\n",
               h_decoded[0], h_decoded[1], h_decoded[2], h_decoded[3]);

        cudaFree(d_llrs);
        cudaFree(d_decoded);
    }

    // Test 2: Decode E2E LLRs with larger clamp
    printf("\n=== Test 2: Decode E2E LLRs (CB0) with llr_clamp=16000 ===\n");
    {
        dec_params.llr_clamp = 16000.0f;
        ldpc_decoder_configure(decoder, &cfg, &dec_params);

        float* d_llrs;
        uint32_t* d_decoded;
        cudaMalloc(&d_llrs, N_full * sizeof(float));
        cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

        cudaMemcpy(d_llrs, llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));

        ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
        cudaStreamSynchronize(stream);

        float avg_iters = ldpc_decoder_get_avg_iterations(decoder);
        printf("  avg_iterations=%.1f\n", avg_iters);

        cudaFree(d_llrs);
        cudaFree(d_decoded);
    }

    // Test 3: Decode with "idealized" LLRs - same signs but ±127 magnitudes
    printf("\n=== Test 3: Idealized LLRs (sign from E2E, magnitude=127) ===\n");
    {
        dec_params.llr_clamp = 127.0f;
        ldpc_decoder_configure(decoder, &cfg, &dec_params);

        std::vector<float> ideal_llrs(N_full);
        float* cb_llrs = llrs.data();  // CB0
        for (int i = 0; i < N_full; i++) {
            if (i < 2*Z) {
                ideal_llrs[i] = 0.0f;  // Punctured
            } else if (i >= Kd && i < K) {
                ideal_llrs[i] = 127.0f;  // Filler
            } else if (cb_llrs[i] == 0.0f) {
                ideal_llrs[i] = 0.0f;
            } else {
                ideal_llrs[i] = (cb_llrs[i] > 0) ? 127.0f : -127.0f;
            }
        }

        float* d_llrs;
        uint32_t* d_decoded;
        cudaMalloc(&d_llrs, N_full * sizeof(float));
        cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

        cudaMemcpy(d_llrs, ideal_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));

        ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
        cudaStreamSynchronize(stream);

        float avg_iters = ldpc_decoder_get_avg_iterations(decoder);
        printf("  avg_iterations=%.1f\n", avg_iters);

        // Count how many LLR signs differ from punctured positions
        int sign_pos = 0, sign_neg = 0, sign_zero = 0;
        for (int i = 2*Z; i < N_full; i++) {
            if (i >= Kd && i < K) continue; // Skip filler
            if (cb_llrs[i] > 0) sign_pos++;
            else if (cb_llrs[i] < 0) sign_neg++;
            else sign_zero++;
        }
        printf("  LLR sign distribution: pos=%d neg=%d zero=%d\n", sign_pos, sign_neg, sign_zero);

        cudaFree(d_llrs);
        cudaFree(d_decoded);
    }

    // Test 4: Decode all-positive (zero codeword) - should always converge
    printf("\n=== Test 4: All-positive LLRs (zero codeword) ===\n");
    {
        std::vector<float> zero_llrs(N_full);
        for (int i = 0; i < N_full; i++) {
            if (i < 2*Z) zero_llrs[i] = 0.0f;
            else zero_llrs[i] = 127.0f;
        }

        float* d_llrs;
        uint32_t* d_decoded;
        cudaMalloc(&d_llrs, N_full * sizeof(float));
        cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

        cudaMemcpy(d_llrs, zero_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));

        ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
        cudaStreamSynchronize(stream);

        float avg_iters = ldpc_decoder_get_avg_iterations(decoder);
        printf("  avg_iterations=%.1f\n", avg_iters);

        std::vector<uint32_t> h_decoded(K_words);
        cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        int nz = 0;
        for (int i = 0; i < K_words; i++) if (h_decoded[i] != 0) nz++;
        printf("  non-zero output words: %d/%d (should be 0 for zero codeword)\n", nz, K_words);

        cudaFree(d_llrs);
        cudaFree(d_decoded);
    }

    // Test 5: Encode known data, create ideal LLRs, decode
    printf("\n=== Test 5: Encode+Decode (known-good LLRs) ===\n");
    {
        ldpc_encoder_handle_t encoder;
        ldpc_encoder_create(&encoder);
        ldpc_encoder_configure(encoder, &cfg);
        int enc_words = ldpc_encoder_get_output_words(encoder);

        // Random data
        std::vector<uint32_t> h_input(K_words, 0);
        srand(42);
        for (int i = 0; i < K_words; i++) h_input[i] = rand();
        // Clear filler bits (positions Kd to K-1 in MSB-first format)
        // Actually, the encoder expects MSB-first packed bits
        for (int bit = Kd; bit < K; bit++) {
            int word_idx = bit / 32;
            int bit_in_word = bit % 32;
            int byte_in_word = bit_in_word / 8;
            int bit_in_byte = 7 - (bit_in_word % 8);
            int bit_pos = byte_in_word * 8 + bit_in_byte;
            h_input[word_idx] &= ~(1u << bit_pos);
        }

        uint32_t *d_input, *d_encoded;
        cudaMalloc(&d_input, K_words * sizeof(uint32_t));
        cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
        cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
        ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
        cudaStreamSynchronize(stream);

        std::vector<uint32_t> h_encoded(enc_words);
        cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Create ideal LLRs from encoded bits (same as standalone test)
        std::vector<float> h_llrs(N_full);
        auto get_bit_msb = [](const uint32_t* data, int bit_idx) -> int {
            int word_idx = bit_idx / 32;
            int bit_in_word = bit_idx % 32;
            int byte_in_word = bit_in_word / 8;
            int bit_in_byte = 7 - (bit_in_word % 8);
            int bit_pos = byte_in_word * 8 + bit_in_byte;
            return (data[word_idx] >> bit_pos) & 1;
        };
        for (int i = 0; i < N_full; i++) {
            if (i < 2*Z) h_llrs[i] = 0.0f;
            else if (i >= Kd && i < K) h_llrs[i] = 127.0f;
            else {
                int bit_val = get_bit_msb(h_encoded.data(), i);
                h_llrs[i] = bit_val ? -127.0f : 127.0f;
            }
        }

        float* d_llrs;
        uint32_t* d_decoded;
        cudaMalloc(&d_llrs, N_full * sizeof(float));
        cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));
        cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));

        ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
        cudaStreamSynchronize(stream);

        float avg_iters = ldpc_decoder_get_avg_iterations(decoder);

        std::vector<uint32_t> h_decoded(K_words);
        cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Compare decoded vs input
        auto get_bit_lsb = [](const uint32_t* data, int bit_idx) -> int {
            return (data[bit_idx / 32] >> (bit_idx % 32)) & 1;
        };
        int errors = 0;
        for (int i = 0; i < Kd; i++) {
            if (get_bit_msb(h_input.data(), i) != get_bit_lsb(h_decoded.data(), i))
                errors++;
        }
        printf("  avg_iterations=%.1f errors=%d/%d %s\n",
               avg_iters, errors, Kd, errors == 0 ? "PASS" : "FAIL");

        cudaFree(d_input);
        cudaFree(d_encoded);
        cudaFree(d_llrs);
        cudaFree(d_decoded);
        ldpc_encoder_destroy(encoder);
    }

    // Test 6: Re-encode E2E systematic bits with OCUDU PHY CUDA and compare parity
    printf("\n=== Test 6: Re-encode E2E systematic bits with OCUDU PHY CUDA ===\n");
    {
        ldpc_encoder_handle_t encoder;
        ldpc_encoder_create(&encoder);
        ldpc_encoder_configure(encoder, &cfg);
        int enc_words = ldpc_encoder_get_output_words(encoder);

        float* cb_llrs = llrs.data();  // CB0

        auto set_bit_msb = [](uint32_t* data, int bit_idx, int val) {
            int word_idx = bit_idx / 32;
            int bit_in_word = bit_idx % 32;
            int byte_in_word = bit_in_word / 8;
            int bit_in_byte = 7 - (bit_in_word % 8);
            int bit_pos = byte_in_word * 8 + bit_in_byte;
            if (val)
                data[word_idx] |= (1u << bit_pos);
            else
                data[word_idx] &= ~(1u << bit_pos);
        };
        auto get_bit_msb = [](const uint32_t* data, int bit_idx) -> int {
            int word_idx = bit_idx / 32;
            int bit_in_word = bit_idx % 32;
            int byte_in_word = bit_in_word / 8;
            int bit_in_byte = 7 - (bit_in_word % 8);
            int bit_pos = byte_in_word * 8 + bit_in_byte;
            return (data[word_idx] >> bit_pos) & 1;
        };

        // Hard-decide E2E LLRs to bits and pack as MSB-first
        std::vector<uint32_t> h_sys_bits(K_words, 0);
        for (int i = 0; i < K; i++) {
            if (i >= Kd && i < K) {
                set_bit_msb(h_sys_bits.data(), i, 0);  // Filler = 0
            } else if (i < 2*Z) {
                set_bit_msb(h_sys_bits.data(), i, 0);  // Punctured = unknown, guess 0
            } else {
                int bit = (cb_llrs[i] < 0.0f) ? 1 : 0;
                set_bit_msb(h_sys_bits.data(), i, bit);
            }
        }

        // Encode with OCUDU PHY CUDA
        uint32_t *d_input, *d_encoded;
        cudaMalloc(&d_input, K_words * sizeof(uint32_t));
        cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
        cudaMemcpy(d_input, h_sys_bits.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
        ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
        cudaStreamSynchronize(stream);

        std::vector<uint32_t> h_reencoded(enc_words);
        cudaMemcpy(h_reencoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Compare parity bits (positions K to N_full-1)
        int parity_total = N_full - K;
        int parity_mismatches = 0;
        int parity_mismatches_received = 0;  // Only where LLR != 0
        int parity_received = 0;
        int first_mismatch = -1;
        for (int i = K; i < N_full; i++) {
            int e2e_bit = (cb_llrs[i] < 0.0f) ? 1 : ((cb_llrs[i] > 0.0f) ? 0 : -1);
            int reenc_bit = get_bit_msb(h_reencoded.data(), i);

            if (e2e_bit >= 0) {  // Only compare where E2E has a definite sign
                parity_received++;
                if (e2e_bit != reenc_bit) {
                    parity_mismatches_received++;
                    if (first_mismatch < 0) first_mismatch = i;
                }
            }
            // Compare ALL (including guessed punctured effect)
            int e2e_hard = (cb_llrs[i] < 0.0f) ? 1 : 0;
            if (e2e_hard != reenc_bit) parity_mismatches++;
        }

        printf("  Parity mismatches (all): %d/%d (%.1f%%)\n",
               parity_mismatches, parity_total, 100.0f * parity_mismatches / parity_total);
        printf("  Parity mismatches (received only): %d/%d (%.1f%%)\n",
               parity_mismatches_received, parity_received,
               parity_received > 0 ? 100.0f * parity_mismatches_received / parity_received : 0.0f);
        if (first_mismatch >= 0) {
            printf("  First received parity mismatch at position %d (col %d, z %d)\n",
                   first_mismatch, first_mismatch / Z, first_mismatch % Z);
        }

        // Also compare systematic bits (2Z to Kd-1)
        int sys_mismatches = 0;
        for (int i = 2*Z; i < Kd; i++) {
            int e2e_bit = (cb_llrs[i] < 0.0f) ? 1 : 0;
            int reenc_bit = get_bit_msb(h_reencoded.data(), i);
            if (e2e_bit != reenc_bit) sys_mismatches++;
        }
        printf("  Systematic mismatches (2Z..Kd-1): %d/%d (should be 0)\n",
               sys_mismatches, Kd - 2*Z);

        // Print first 32 parity bits from each
        printf("  E2E    parity[K..K+32]: ");
        for (int i = K; i < K+32; i++) {
            int bit = (cb_llrs[i] < 0.0f) ? 1 : ((cb_llrs[i] > 0.0f) ? 0 : 2);
            printf("%d", bit);
        }
        printf("\n");
        printf("  OCUDU PHY CUDA parity[K..K+32]: ");
        for (int i = K; i < K+32; i++) printf("%d", get_bit_msb(h_reencoded.data(), i));
        printf("\n");

        cudaFree(d_input);
        cudaFree(d_encoded);
        ldpc_encoder_destroy(encoder);
    }

    // Test 7: Decode with NEGATED LLR signs (test opposite sign convention)
    printf("\n=== Test 7: Decode with negated LLR signs ===\n");
    {
        dec_params.llr_clamp = 127.0f;
        dec_params.max_iterations = 10;
        ldpc_decoder_configure(decoder, &cfg, &dec_params);

        std::vector<float> neg_llrs(N_full);
        float* cb_llrs = llrs.data();
        for (int i = 0; i < N_full; i++) {
            if (i < 2*Z) {
                neg_llrs[i] = 0.0f;  // Punctured stays 0
            } else if (i >= Kd && i < K) {
                neg_llrs[i] = 127.0f;  // Filler stays positive
            } else {
                neg_llrs[i] = -cb_llrs[i];  // Negate sign
            }
        }

        float* d_llrs;
        uint32_t* d_decoded;
        cudaMalloc(&d_llrs, N_full * sizeof(float));
        cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));
        cudaMemcpy(d_llrs, neg_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));

        ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
        cudaStreamSynchronize(stream);

        float avg_iters = ldpc_decoder_get_avg_iterations(decoder);
        printf("  avg_iterations=%.1f %s\n", avg_iters,
               avg_iters < 10.0f ? "CONVERGED (sign convention is WRONG!)" : "did not converge");

        cudaFree(d_llrs);
        cudaFree(d_decoded);
    }

    // Test 8: Decode with 50 iterations (check if just needs more iterations)
    printf("\n=== Test 8: Decode E2E LLRs with max_iterations=50 ===\n");
    {
        dec_params.max_iterations = 50;
        ldpc_decoder_configure(decoder, &cfg, &dec_params);

        float* d_llrs;
        uint32_t* d_decoded;
        cudaMalloc(&d_llrs, N_full * sizeof(float));
        cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));
        cudaMemcpy(d_llrs, llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));

        ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
        cudaStreamSynchronize(stream);

        float avg_iters = ldpc_decoder_get_avg_iterations(decoder);
        printf("  avg_iterations=%.1f %s\n", avg_iters,
               avg_iters < 50.0f ? "CONVERGED" : "did not converge (codeword violates H matrix)");

        cudaFree(d_llrs);
        cudaFree(d_decoded);
    }

    // Test 9: Decode E2E LLRs clamped to ±127 (test if large magnitudes cause issues)
    printf("\n=== Test 9: Decode E2E LLRs clamped to ±127 ===\n");
    {
        dec_params.max_iterations = 10;
        ldpc_decoder_configure(decoder, &cfg, &dec_params);

        std::vector<float> clamped_llrs(N_full);
        float* cb_llrs = llrs.data();
        for (int i = 0; i < N_full; i++) {
            float v = cb_llrs[i];
            if (i >= Kd && i < K) {
                clamped_llrs[i] = 127.0f;  // Filler: use standard 127
            } else if (v > 127.0f) {
                clamped_llrs[i] = 127.0f;
            } else if (v < -127.0f) {
                clamped_llrs[i] = -127.0f;
            } else {
                clamped_llrs[i] = v;
            }
        }

        float* d_llrs;
        uint32_t* d_decoded;
        cudaMalloc(&d_llrs, N_full * sizeof(float));
        cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));
        cudaMemcpy(d_llrs, clamped_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));

        ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, 1, stream);
        cudaStreamSynchronize(stream);

        float avg_iters = ldpc_decoder_get_avg_iterations(decoder);
        printf("  avg_iterations=%.1f\n", avg_iters);

        cudaFree(d_llrs);
        cudaFree(d_decoded);
    }

    // Test 10: CPU Syndrome Check
    // Verify whether the E2E hard-decided bits satisfy H*c = 0 using the OCUDU PHY CUDA H matrix.
    // Also verify the OCUDU PHY CUDA-encoded codeword (from Test 5) as a control.
    printf("\n=== Test 10: CPU Syndrome Check ===\n");
    {
        // BG1 H matrix in CSR format (from ldpc_decoder_flexible.cu)
        static const int16_t bg1_row_ptr[47] = {
            0, 19, 38, 57, 76, 79, 87, 96, 103, 113, 122, 129, 137, 144, 150, 157, 164,
            170, 176, 182, 188, 194, 200, 205, 210, 216, 221, 226, 230, 235, 240, 245,
            250, 255, 260, 265, 270, 275, 279, 284, 289, 293, 298, 302, 307, 312, 316
        };
        static const int8_t bg1_col[316] = {
           0,  1,  2,  3,  5,  6,  9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23,
           0,  2,  3,  4,  5,  7,  8,  9, 11, 12, 14, 15, 16, 17, 19, 21, 22, 23, 24,
           0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 13, 14, 15, 17, 18, 19, 20, 24, 25,
           0,  1,  3,  4,  6,  7,  8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 25,
           0,  1, 26,
           0,  1,  3, 12, 16, 21, 22, 27,
           0,  6, 10, 11, 13, 17, 18, 20, 28,
           0,  1,  4,  7,  8, 14, 29,
           0,  1,  3, 12, 16, 19, 21, 22, 24, 30,
           0,  1, 10, 11, 13, 17, 18, 20, 31,
           1,  2,  4,  7,  8, 14, 32,
           0,  1, 12, 16, 21, 22, 23, 33,
           0,  1, 10, 11, 13, 18, 34,
           0,  3,  7, 20, 23, 35,
           0, 12, 15, 16, 17, 21, 36,
           0,  1, 10, 13, 18, 25, 37,
           1,  3, 11, 20, 22, 38,
           0, 14, 16, 17, 21, 39,
           1, 12, 13, 18, 19, 40,
           0,  1,  7,  8, 10, 41,
           0,  3,  9, 11, 22, 42,
           1,  5, 16, 20, 21, 43,
           0, 12, 13, 17, 44,
           1,  2, 10, 18, 45,
           0,  3,  4, 11, 22, 46,
           1,  6,  7, 14, 47,
           0,  2,  4, 15, 48,
           1,  6,  8, 49,
           0,  4, 19, 21, 50,
           1, 14, 18, 25, 51,
           0, 10, 13, 24, 52,
           1,  7, 22, 25, 53,
           0, 12, 14, 24, 54,
           1,  2, 11, 21, 55,
           0,  7, 15, 17, 56,
           1,  6, 12, 22, 57,
           0, 14, 15, 18, 58,
           1, 13, 23, 59,
           0,  9, 10, 12, 60,
           1,  3,  7, 19, 61,
           0,  8, 17, 62,
           1,  3,  9, 18, 63,
           0,  4, 24, 64,
           1, 16, 18, 25, 65,
           0,  7,  9, 22, 66,
           1,  6, 10, 67
        };
        // Lifting set 7 reference shifts (Z_ref=240)
        static const int16_t bg1_shifts_ls7[316] = {
            135, 227, 126, 134,  84,  83,  53, 225, 205, 128,  75, 135, 217, 220,  90, 105,
            137,   1,   0,  96, 236, 136, 221, 128,  92, 172,  56,  11, 189,  95,  85, 153,
             87, 163, 216,   0,   0,   0, 189,   4, 225, 151, 236, 117, 179,  92,  24,  68,
              6, 101,  33,  96, 125,  67, 230,   0,   0, 128,  23, 162, 220,  43, 186,  96,
              1, 216,  22,  24, 167, 200,  32, 235, 172, 219,   1,   0,  64, 211,   0,   2,
            171,  47, 143, 210, 180, 180,   0, 199,  22,  23, 100,  92, 207,  52,  13,   0,
             77, 146, 209,  32, 166,  18,   0, 181, 105, 141, 223, 177, 145, 199, 153,  38,
              0, 169,  12, 206, 221,  17, 212,  92, 205,   0, 116, 151,  70, 230, 115,  84,
              0,  45, 115, 134,   1, 152, 165, 107,   0, 186, 215, 124, 180,  98,  80,   0,
            220, 185, 154, 178, 150,   0, 124, 144, 182,  95,  72,  76,   0,  39, 138, 220,
            173, 142,  49,   0,  78, 152,  84,   5, 205,   0, 183, 112, 106, 219, 129,   0,
            183, 215, 180, 143,  14,   0, 179, 108, 159, 138, 196,   0,  77, 187, 203, 167,
            130,   0, 197, 122, 215,  65, 216,   0,  25,  47, 126, 178,   0, 185, 127, 117,
            199,   0,  32, 178,   2, 156,  58,   0,  27, 141,  11, 181,   0, 163, 131, 169,
             98,   0, 165, 232,   9,   0,  32,  43, 200, 205,   0, 232,  32, 118, 103,   0,
            170, 199,  26, 105,   0,  73, 149, 175, 108,   0, 103, 110, 151, 211,   0, 199,
            132, 172,  65,   0, 161, 237, 142, 180,   0, 231, 174, 145, 100,   0,  11, 207,
             42, 100,   0,  59, 204, 161,   0, 121,  90,  26, 140,   0, 115, 188, 168,  52,
              0,   4, 103,  30,   0,  53, 189, 215,  24,   0, 222, 170,  71,   0,  22, 127,
             49, 125,   0, 191, 211, 187, 148,   0, 177, 114,  93,   0
        };

        // Helper to get bit from MSB-first packed data
        auto get_bit_msb = [](const uint32_t* data, int bit_idx) -> int {
            int word_idx = bit_idx / 32;
            int bit_in_word = bit_idx % 32;
            int byte_in_word = bit_in_word / 8;
            int bit_in_byte = 7 - (bit_in_word % 8);
            int bit_pos = byte_in_word * 8 + bit_in_byte;
            return (data[word_idx] >> bit_pos) & 1;
        };

        // Compute the actual shifts for this Z (shift_ref % Z)
        std::vector<int16_t> actual_shifts(316);
        for (int i = 0; i < 316; i++) {
            actual_shifts[i] = (bg1_shifts_ls7[i] == 0) ? 0 : (bg1_shifts_ls7[i] % Z);
        }

        int total_rows = 46;
        int M = (bg == 1) ? 46 : 42;

        // === Part A: Encode with OCUDU PHY CUDA and verify syndrome is zero ===
        printf("  Part A: OCUDU PHY CUDA-encoded codeword syndrome check\n");
        {
            ldpc_encoder_handle_t enc;
            ldpc_encoder_create(&enc);
            ldpc_encoder_configure(enc, &cfg);
            int enc_words = ldpc_encoder_get_output_words(enc);

            // Random data (same seed as Test 5)
            std::vector<uint32_t> h_input(K_words, 0);
            srand(42);
            for (int i = 0; i < K_words; i++) h_input[i] = rand();
            for (int bit = Kd; bit < K; bit++) {
                int word_idx = bit / 32;
                int bit_in_word = bit % 32;
                int byte_in_word = bit_in_word / 8;
                int bit_in_byte = 7 - (bit_in_word % 8);
                int bit_pos = byte_in_word * 8 + bit_in_byte;
                h_input[word_idx] &= ~(1u << bit_pos);
            }

            uint32_t *d_input, *d_encoded;
            cudaMalloc(&d_input, K_words * sizeof(uint32_t));
            cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
            cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
            ldpc_encoder_encode(enc, d_input, d_encoded, stream);
            cudaStreamSynchronize(stream);

            std::vector<uint32_t> h_encoded(enc_words);
            cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            // CPU syndrome check: for each row, for each z, XOR contributions
            int total_syn_errors = 0;
            int rows_with_errors = 0;
            for (int row = 0; row < M; row++) {
                int syn_errors_this_row = 0;
                for (int z = 0; z < Z; z++) {
                    int syndrome = 0;
                    for (int e = bg1_row_ptr[row]; e < bg1_row_ptr[row + 1]; e++) {
                        int col = bg1_col[e];
                        int shift = actual_shifts[e];
                        int shifted_z = (z + shift) % Z;
                        int bit_pos = col * Z + shifted_z;
                        syndrome ^= get_bit_msb(h_encoded.data(), bit_pos);
                    }
                    if (syndrome != 0) {
                        syn_errors_this_row++;
                    }
                }
                if (syn_errors_this_row > 0) {
                    total_syn_errors += syn_errors_this_row;
                    rows_with_errors++;
                    if (rows_with_errors <= 5) {
                        printf("    Row %d: %d/%d syndrome errors\n", row, syn_errors_this_row, Z);
                    }
                }
            }
            printf("    Total: %d syndrome errors in %d/%d rows %s\n",
                   total_syn_errors, rows_with_errors, M,
                   total_syn_errors == 0 ? "PASS (valid codeword)" : "FAIL");

            cudaFree(d_input);
            cudaFree(d_encoded);
            ldpc_encoder_destroy(enc);
        }

        // === Part B: E2E hard-decided codeword syndrome check ===
        printf("  Part B: E2E codeword syndrome check (punctured bits = 0)\n");
        {
            // Hard-decide E2E LLRs
            float* cb_llrs = llrs.data();  // CB0
            std::vector<uint8_t> hard_bits(N_full, 0);
            for (int i = 0; i < N_full; i++) {
                hard_bits[i] = (cb_llrs[i] < 0.0f) ? 1 : 0;
            }

            int total_syn_errors = 0;
            int rows_with_errors = 0;
            int rows_col01_only_errors = 0;  // Rows where errors could be from cols 0-1 only

            for (int row = 0; row < M; row++) {
                int syn_errors_this_row = 0;
                // Check which cols are in this row
                bool has_col0 = false, has_col1 = false;
                for (int e = bg1_row_ptr[row]; e < bg1_row_ptr[row + 1]; e++) {
                    if (bg1_col[e] == 0) has_col0 = true;
                    if (bg1_col[e] == 1) has_col1 = true;
                }

                for (int z = 0; z < Z; z++) {
                    int syndrome = 0;
                    for (int e = bg1_row_ptr[row]; e < bg1_row_ptr[row + 1]; e++) {
                        int col = bg1_col[e];
                        int shift = actual_shifts[e];
                        int shifted_z = (z + shift) % Z;
                        int bit_pos = col * Z + shifted_z;
                        syndrome ^= hard_bits[bit_pos];
                    }
                    if (syndrome != 0) {
                        syn_errors_this_row++;
                    }
                }
                if (syn_errors_this_row > 0) {
                    total_syn_errors += syn_errors_this_row;
                    rows_with_errors++;
                    if (has_col0 || has_col1) rows_col01_only_errors++;
                    if (rows_with_errors <= 5) {
                        printf("    Row %d: %d/%d syndrome errors (has_col0=%d has_col1=%d)\n",
                               row, syn_errors_this_row, Z, has_col0, has_col1);
                    }
                }
            }
            printf("    Total: %d syndrome errors in %d/%d rows\n",
                   total_syn_errors, rows_with_errors, M);
            printf("    Rows with errors that have col0/1: %d/%d\n",
                   rows_col01_only_errors, rows_with_errors);

            // Expected: all rows have errors because all rows involve col 0 or col 1
            // The number of errors per row should be ~Z/2 (random pattern from unknown punctured bits)
            // If errors >> Z/2 for rows without col 0/1, that indicates a real problem
        }

        // === Part C: E2E codeword syndrome check EXCLUDING cols 0-1 contributions ===
        // Compute partial syndrome using only known columns (2-67)
        // For a valid codeword, the partial syndrome should equal the (unknown) cols 0-1 contribution
        // Check if the pattern is consistent
        printf("  Part C: E2E partial syndrome (excluding col 0-1 contributions)\n");
        {
            float* cb_llrs = llrs.data();
            std::vector<uint8_t> hard_bits(N_full, 0);
            for (int i = 0; i < N_full; i++) {
                hard_bits[i] = (cb_llrs[i] < 0.0f) ? 1 : 0;
            }

            // For each row, compute syndrome using only cols >= 2
            // This partial syndrome should equal the contribution of cols 0-1
            // For consistency, rows that share col 0 should give consistent col-0 bit values
            // and rows that share col 1 should give consistent col-1 bit values

            // First, compute what col 0 and col 1 bits "should be" from each row
            std::vector<int> col0_inferred(Z, -1);  // -1 = unset
            std::vector<int> col1_inferred(Z, -1);
            int col0_conflicts = 0, col1_conflicts = 0;

            for (int row = 0; row < M; row++) {
                // Find col 0 and col 1 edges in this row
                int col0_shift = -1, col1_shift = -1;
                bool has_col0 = false, has_col1 = false;

                for (int e = bg1_row_ptr[row]; e < bg1_row_ptr[row + 1]; e++) {
                    if (bg1_col[e] == 0) { has_col0 = true; col0_shift = actual_shifts[e]; }
                    if (bg1_col[e] == 1) { has_col1 = true; col1_shift = actual_shifts[e]; }
                }

                for (int z = 0; z < Z; z++) {
                    // Compute partial syndrome (excluding cols 0 and 1)
                    int partial_syn = 0;
                    for (int e = bg1_row_ptr[row]; e < bg1_row_ptr[row + 1]; e++) {
                        int col = bg1_col[e];
                        if (col == 0 || col == 1) continue;  // Skip cols 0-1
                        int shift = actual_shifts[e];
                        int shifted_z = (z + shift) % Z;
                        partial_syn ^= hard_bits[col * Z + shifted_z];
                    }

                    // partial_syn should equal col0_contribution XOR col1_contribution
                    // where col_X_contribution = unknown_bit[X * Z + (z + shift_X) % Z]

                    if (has_col0 && !has_col1) {
                        // partial_syn = col0_bit[(z + col0_shift) % Z]
                        int inferred_z = (z + col0_shift) % Z;
                        if (col0_inferred[inferred_z] == -1) {
                            col0_inferred[inferred_z] = partial_syn;
                        } else if (col0_inferred[inferred_z] != partial_syn) {
                            col0_conflicts++;
                        }
                    } else if (!has_col0 && has_col1) {
                        // partial_syn = col1_bit[(z + col1_shift) % Z]
                        int inferred_z = (z + col1_shift) % Z;
                        if (col1_inferred[inferred_z] == -1) {
                            col1_inferred[inferred_z] = partial_syn;
                        } else if (col1_inferred[inferred_z] != partial_syn) {
                            col1_conflicts++;
                        }
                    }
                    // If both col0 and col1: partial_syn = col0_bit XOR col1_bit (can't separate)
                }
            }

            // Count how many col0/col1 bits were inferred
            int col0_set = 0, col1_set = 0;
            for (int z = 0; z < Z; z++) {
                if (col0_inferred[z] >= 0) col0_set++;
                if (col1_inferred[z] >= 0) col1_set++;
            }

            printf("    Col 0: %d/%d z-positions inferred, %d conflicts\n", col0_set, Z, col0_conflicts);
            printf("    Col 1: %d/%d z-positions inferred, %d conflicts\n", col1_set, Z, col1_conflicts);
            if (col0_conflicts == 0 && col1_conflicts == 0) {
                printf("    CONSISTENT: E2E codeword is valid for OCUDU PHY CUDA H matrix\n");
                printf("    (Decoder bug: valid codeword but decoder fails to converge)\n");
            } else {
                printf("    INCONSISTENT: E2E codeword violates OCUDU PHY CUDA H matrix!\n");
                printf("    (Encoder mismatch: srsRAN and OCUDU PHY CUDA produce different codewords)\n");
            }
        }

        // === Part D: Try INVERTED shift convention: (z - shift + Z) % Z ===
        printf("  Part D: E2E consistency check with INVERTED shift convention\n");
        {
            float* cb_llrs = llrs.data();
            std::vector<uint8_t> hard_bits(N_full, 0);
            for (int i = 0; i < N_full; i++) {
                hard_bits[i] = (cb_llrs[i] < 0.0f) ? 1 : 0;
            }

            std::vector<int> col0_inferred(Z, -1);
            std::vector<int> col1_inferred(Z, -1);
            int col0_conflicts = 0, col1_conflicts = 0;

            for (int row = 0; row < M; row++) {
                int col0_shift = -1, col1_shift = -1;
                bool has_col0 = false, has_col1 = false;
                for (int e = bg1_row_ptr[row]; e < bg1_row_ptr[row + 1]; e++) {
                    if (bg1_col[e] == 0) { has_col0 = true; col0_shift = actual_shifts[e]; }
                    if (bg1_col[e] == 1) { has_col1 = true; col1_shift = actual_shifts[e]; }
                }

                for (int z = 0; z < Z; z++) {
                    int partial_syn = 0;
                    for (int e = bg1_row_ptr[row]; e < bg1_row_ptr[row + 1]; e++) {
                        int col = bg1_col[e];
                        if (col == 0 || col == 1) continue;
                        int shift = actual_shifts[e];
                        // INVERTED: (z - shift + Z) % Z
                        int shifted_z = (z - shift + Z) % Z;
                        partial_syn ^= hard_bits[col * Z + shifted_z];
                    }

                    if (has_col0 && !has_col1) {
                        int inferred_z = (z - col0_shift + Z) % Z;
                        if (col0_inferred[inferred_z] == -1) {
                            col0_inferred[inferred_z] = partial_syn;
                        } else if (col0_inferred[inferred_z] != partial_syn) {
                            col0_conflicts++;
                        }
                    } else if (!has_col0 && has_col1) {
                        int inferred_z = (z - col1_shift + Z) % Z;
                        if (col1_inferred[inferred_z] == -1) {
                            col1_inferred[inferred_z] = partial_syn;
                        } else if (col1_inferred[inferred_z] != partial_syn) {
                            col1_conflicts++;
                        }
                    }
                }
            }

            int col0_set = 0, col1_set = 0;
            for (int z = 0; z < Z; z++) {
                if (col0_inferred[z] >= 0) col0_set++;
                if (col1_inferred[z] >= 0) col1_set++;
            }
            printf("    Col 0: %d/%d inferred, %d conflicts\n", col0_set, Z, col0_conflicts);
            printf("    Col 1: %d/%d inferred, %d conflicts\n", col1_set, Z, col1_conflicts);
            if (col0_conflicts == 0 && col1_conflicts == 0) {
                printf("    CONSISTENT with inverted shifts! ROOT CAUSE: shift convention mismatch\n");
            } else {
                printf("    Also inconsistent with inverted shifts\n");
            }
        }

        // === Part E: Verify OCUDU PHY CUDA-encoded codeword with INVERTED shifts (should FAIL) ===
        printf("  Part E: OCUDU PHY CUDA-encoded codeword with INVERTED shifts (control)\n");
        {
            ldpc_encoder_handle_t enc;
            ldpc_encoder_create(&enc);
            ldpc_encoder_configure(enc, &cfg);
            int enc_words = ldpc_encoder_get_output_words(enc);

            std::vector<uint32_t> h_input(K_words, 0);
            srand(42);
            for (int i = 0; i < K_words; i++) h_input[i] = rand();
            for (int bit = Kd; bit < K; bit++) {
                int word_idx = bit / 32;
                int bit_in_word = bit % 32;
                int byte_in_word = bit_in_word / 8;
                int bit_in_byte = 7 - (bit_in_word % 8);
                int bit_pos = byte_in_word * 8 + bit_in_byte;
                h_input[word_idx] &= ~(1u << bit_pos);
            }

            uint32_t *d_input, *d_encoded;
            cudaMalloc(&d_input, K_words * sizeof(uint32_t));
            cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
            cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
            ldpc_encoder_encode(enc, d_input, d_encoded, stream);
            cudaStreamSynchronize(stream);

            std::vector<uint32_t> h_encoded(enc_words);
            cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            int total_syn_errors = 0;
            for (int row = 0; row < M; row++) {
                for (int z = 0; z < Z; z++) {
                    int syndrome = 0;
                    for (int e = bg1_row_ptr[row]; e < bg1_row_ptr[row + 1]; e++) {
                        int col = bg1_col[e];
                        int shift = actual_shifts[e];
                        int shifted_z = (z - shift + Z) % Z;  // INVERTED
                        int bit_pos = col * Z + shifted_z;
                        syndrome ^= get_bit_msb(h_encoded.data(), bit_pos);
                    }
                    if (syndrome != 0) total_syn_errors++;
                }
            }
            printf("    Total: %d syndrome errors (should be >0 for inverted convention)\n", total_syn_errors);

            cudaFree(d_input);
            cudaFree(d_encoded);
            ldpc_encoder_destroy(enc);
        }

        // === Part F: Direct E2E syndrome check with INVERTED shifts ===
        printf("  Part F: E2E full syndrome with INVERTED shifts\n");
        {
            float* cb_llrs = llrs.data();
            std::vector<uint8_t> hard_bits(N_full, 0);
            for (int i = 0; i < N_full; i++) {
                hard_bits[i] = (cb_llrs[i] < 0.0f) ? 1 : 0;
            }

            int total_syn_errors = 0;
            int rows_with_errors = 0;
            for (int row = 0; row < M; row++) {
                int syn_errors_this_row = 0;
                for (int z = 0; z < Z; z++) {
                    int syndrome = 0;
                    for (int e = bg1_row_ptr[row]; e < bg1_row_ptr[row + 1]; e++) {
                        int col = bg1_col[e];
                        int shift = actual_shifts[e];
                        int shifted_z = (z - shift + Z) % Z;  // INVERTED
                        int bit_pos = col * Z + shifted_z;
                        syndrome ^= hard_bits[bit_pos];
                    }
                    if (syndrome != 0) syn_errors_this_row++;
                }
                if (syn_errors_this_row > 0) {
                    total_syn_errors += syn_errors_this_row;
                    rows_with_errors++;
                }
            }
            printf("    Total: %d syndrome errors in %d/%d rows\n",
                   total_syn_errors, rows_with_errors, M);
            // Compare with Part B: if fewer errors, inverted convention is closer
        }
    }

    cudaStreamDestroy(stream);
    ldpc_decoder_destroy(decoder);
    return 0;
}
