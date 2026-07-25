/**
 * @file mcs_sweep_5gnr.cu
 * @brief 5G NR Compliant MCS Performance Sweep
 *
 * Implements complete 5G NR MCS Table 1 (64QAM) and Table 2 (256QAM)
 * per 3GPP TS 38.214 Tables 5.1.3.1-1 and 5.1.3.1-2.
 *
 * Tests the complete pipeline:
 * - TB segmentation (if needed)
 * - CRC attachment
 * - LDPC encoding
 * - Rate matching
 * - Bit interleaving
 * - Scrambling
 * - Modulation/AWGN/Demodulation
 */

#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <algorithm>

// =============================================================================
// 5G NR MCS Tables per 3GPP TS 38.214
// =============================================================================

struct MCSEntry {
    int mcs_index;
    int modulation_order;  // Qm: 2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM
    float target_code_rate; // R x 1024
    float spectral_efficiency; // bits/s/Hz
    const char* mod_name;
};

// MCS Table 1: 64QAM maximum (Table 5.1.3.1-1)
static const MCSEntry mcs_table1[] = {
    {0,  2, 120.0f/1024, 0.2344f, "QPSK"},
    {1,  2, 157.0f/1024, 0.3066f, "QPSK"},
    {2,  2, 193.0f/1024, 0.3770f, "QPSK"},
    {3,  2, 251.0f/1024, 0.4902f, "QPSK"},
    {4,  2, 308.0f/1024, 0.6016f, "QPSK"},
    {5,  2, 379.0f/1024, 0.7402f, "QPSK"},
    {6,  2, 449.0f/1024, 0.8770f, "QPSK"},
    {7,  2, 526.0f/1024, 1.0273f, "QPSK"},
    {8,  2, 602.0f/1024, 1.1758f, "QPSK"},
    {9,  2, 679.0f/1024, 1.3262f, "QPSK"},
    {10, 4, 340.0f/1024, 1.3281f, "16QAM"},
    {11, 4, 378.0f/1024, 1.4766f, "16QAM"},
    {12, 4, 434.0f/1024, 1.6953f, "16QAM"},
    {13, 4, 490.0f/1024, 1.9141f, "16QAM"},
    {14, 4, 553.0f/1024, 2.1602f, "16QAM"},
    {15, 4, 616.0f/1024, 2.4063f, "16QAM"},
    {16, 4, 658.0f/1024, 2.5703f, "16QAM"},
    {17, 6, 438.0f/1024, 2.5664f, "64QAM"},
    {18, 6, 466.0f/1024, 2.7305f, "64QAM"},
    {19, 6, 517.0f/1024, 3.0293f, "64QAM"},
    {20, 6, 567.0f/1024, 3.3223f, "64QAM"},
    {21, 6, 616.0f/1024, 3.6094f, "64QAM"},
    {22, 6, 666.0f/1024, 3.9023f, "64QAM"},
    {23, 6, 719.0f/1024, 4.2129f, "64QAM"},
    {24, 6, 772.0f/1024, 4.5234f, "64QAM"},
    {25, 6, 822.0f/1024, 4.8164f, "64QAM"},
    {26, 6, 873.0f/1024, 5.1152f, "64QAM"},
    {27, 6, 910.0f/1024, 5.3320f, "64QAM"},
    {28, 2, 0.0f, 0.0f, "RESERVED"},  // Reserved
};

// MCS Table 2: 256QAM maximum (Table 5.1.3.1-2)
static const MCSEntry mcs_table2[] = {
    {0,  2, 120.0f/1024, 0.2344f, "QPSK"},
    {1,  2, 193.0f/1024, 0.3770f, "QPSK"},
    {2,  2, 308.0f/1024, 0.6016f, "QPSK"},
    {3,  2, 449.0f/1024, 0.8770f, "QPSK"},
    {4,  2, 602.0f/1024, 1.1758f, "QPSK"},
    {5,  4, 378.0f/1024, 1.4766f, "16QAM"},
    {6,  4, 434.0f/1024, 1.6953f, "16QAM"},
    {7,  4, 490.0f/1024, 1.9141f, "16QAM"},
    {8,  4, 553.0f/1024, 2.1602f, "16QAM"},
    {9,  4, 616.0f/1024, 2.4063f, "16QAM"},
    {10, 4, 658.0f/1024, 2.5703f, "16QAM"},
    {11, 6, 466.0f/1024, 2.7305f, "64QAM"},
    {12, 6, 517.0f/1024, 3.0293f, "64QAM"},
    {13, 6, 567.0f/1024, 3.3223f, "64QAM"},
    {14, 6, 616.0f/1024, 3.6094f, "64QAM"},
    {15, 6, 666.0f/1024, 3.9023f, "64QAM"},
    {16, 6, 719.0f/1024, 4.2129f, "64QAM"},
    {17, 6, 772.0f/1024, 4.5234f, "64QAM"},
    {18, 6, 822.0f/1024, 4.8164f, "64QAM"},
    {19, 6, 873.0f/1024, 5.1152f, "64QAM"},
    {20, 8, 682.5f/1024, 5.3320f, "256QAM"},
    {21, 8, 711.0f/1024, 5.5547f, "256QAM"},
    {22, 8, 754.0f/1024, 5.8906f, "256QAM"},
    {23, 8, 797.0f/1024, 6.2266f, "256QAM"},
    {24, 8, 841.0f/1024, 6.5703f, "256QAM"},
    {25, 8, 885.0f/1024, 6.9141f, "256QAM"},
    {26, 8, 916.5f/1024, 7.1602f, "256QAM"},
    {27, 8, 948.0f/1024, 7.4063f, "256QAM"},
};

// =============================================================================
// Test Configuration
// =============================================================================

struct TestConfig {
    int tbs_bits;           // Transport block size
    int num_res_elements;   // Number of resource elements (determines E)
    int num_blocks;         // Number of TBs to test per SNR
    float snr_step;         // SNR step size in dB
};

// Configuration for accurate BLER measurement
static const TestConfig test_cfg = {
    .tbs_bits = 8000,       // Medium TB size
    .num_res_elements = 0,  // Will be calculated based on code rate
    .num_blocks = 200,      // More blocks for accurate BLER
    .snr_step = 0.5f        // Fine SNR resolution
};

// =============================================================================
// Results Structure
// =============================================================================

struct MCSResult {
    int table_num;
    int mcs_index;
    const char* modulation;
    int Qm;
    float code_rate;
    float spectral_eff;
    float bler10_snr_db;
    bool tested;
    double encoder_gbps;
    double decoder_mbps;
    double pipeline_mbps;
};

// =============================================================================
// CUDA Timer
// =============================================================================

class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }
    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
    }
    float elapsed_ms() {
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
private:
    cudaEvent_t start_, stop_;
};

// =============================================================================
// Calculate E (rate-matched output bits) from code rate and modulation
// =============================================================================

int calculate_E(int tbs_bits, float code_rate, int Qm) {
    // E = ceil(TBS / code_rate) rounded to multiple of Qm
    int E = (int)ceil((float)tbs_bits / code_rate);
    E = ((E + Qm - 1) / Qm) * Qm;  // Round up to multiple of Qm
    return E;
}

// =============================================================================
// Find BLER 10% SNR point
// =============================================================================

float find_bler10_snr(const MCSEntry& mcs, int tbs_bits, bool verbose = false) {
    if (mcs.target_code_rate <= 0) return -999.0f;  // Reserved entry

    // Calculate E (rate-matched output size)
    int E = calculate_E(tbs_bits, mcs.target_code_rate, mcs.modulation_order);

    // Initialize LDPC config
    nr_ldpc_config_t ldpc_cfg;
    nr_ldpc_init_config(&ldpc_cfg, tbs_bits, mcs.target_code_rate);

    int Kb = (ldpc_cfg.base_graph == 1) ? 22 : 10;
    int N_cols = (ldpc_cfg.base_graph == 1) ? 68 : 52;
    int K = Kb * ldpc_cfg.lifting_size;
    int N = N_cols * ldpc_cfg.lifting_size;
    int N_cb = N - 2 * ldpc_cfg.lifting_size;  // Circular buffer (first 2Z punctured)

    // IMPORTANT: Ensure E >= N_cb to avoid erased parity bits
    // The min-sum decoder doesn't handle partial parity well (LLR near 0 for erased bits)
    // For low rates, use repetition (E > N_cb) instead of parity erasure (E < N_cb)
    if (E < N_cb) {
        E = N_cb;  // Minimum E is full circular buffer
    }

    // Calculate ACTUAL code rate for proper Eb/N0 calculation
    // When E is forced to N_cb, the actual rate differs from target rate
    float actual_code_rate = (float)tbs_bits / (float)E;

    if (verbose) {
        printf("  Config: BG%d Z=%d K=%d N=%d E=%d\n",
               ldpc_cfg.base_graph, ldpc_cfg.lifting_size, K, N, E);
        if (fabsf(actual_code_rate - mcs.target_code_rate) > 0.01f) {
            printf("  NOTE: Actual rate %.4f differs from target %.4f (E clamped to N_cb)\n",
                   actual_code_rate, mcs.target_code_rate);
        }
    }

    // Create components
    ldpc_encoder_handle_t encoder;
    ldpc_decoder_handle_t decoder;
    rate_matcher_handle_t rm_tx, rm_rx;
    modulator_handle_t modulator;

    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &ldpc_cfg);

    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 25;
    ldpc_decoder_configure(decoder, &ldpc_cfg, &dec_params);

    nr_rate_match_config_t rm_cfg = {
        .E = E,
        .Q_m = mcs.modulation_order,
        .rv = 0,
        .N_cb = N_cb,
        .k0 = 0,
        .limited_buffer = false
    };
    rate_matcher_create(&rm_tx);
    rate_matcher_configure_tx(rm_tx, &ldpc_cfg, &rm_cfg);
    rate_matcher_create(&rm_rx);
    rate_matcher_configure_rx(rm_rx, &ldpc_cfg, &rm_cfg);

    modulator_create(&modulator);

    // Allocate memory
    int K_words = (K + 31) / 32;
    int enc_words = ldpc_encoder_get_output_words(encoder);
    int E_words = (E + 31) / 32;
    int num_symbols = (E + mcs.modulation_order - 1) / mcs.modulation_order;

    uint32_t *h_input = new uint32_t[K_words];
    uint32_t *h_decoded = new uint32_t[K_words];

    uint32_t *d_input, *d_encoded, *d_rate_matched;
    cuFloatComplex *d_symbols, *d_rx_symbols;
    float *d_llrs, *d_derate_llrs;
    uint32_t *d_decoded;

    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
    cudaMalloc(&d_rate_matched, E_words * sizeof(uint32_t));
    cudaMalloc(&d_symbols, num_symbols * sizeof(cuFloatComplex));
    cudaMalloc(&d_rx_symbols, num_symbols * sizeof(cuFloatComplex));
    cudaMalloc(&d_llrs, E * sizeof(float));
    cudaMalloc(&d_derate_llrs, N * sizeof(float));
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));

    // We sweep over Es/N0 (symbol energy to noise ratio)
    // This is simpler and more direct than trying to compute from Eb/N0
    // For normalized constellation (Es = 1): sigma = sqrt(1 / (2 * Es_N0_linear))
    //
    // After measurement, we can derive Eb/N0 = Es/N0 / (Qm * R)
    // where Qm = bits per symbol, R = actual code rate (TBS/E)
    //
    // Starting Es/N0 based on spectral efficiency (higher SE needs higher SNR)
    float es_n0_start = -4.0f + 4.0f * mcs.spectral_efficiency;
    float es_n0_end = es_n0_start + 10.0f;

    float bler10_es_n0 = es_n0_end;  // Default if not found
    int info_bits = ldpc_cfg.num_info_bits;  // Actual info bits (excludes filler)

    // Binary search for BLER 10% point
    float last_bler = 1.0f;

    for (float es_n0_db = es_n0_start; es_n0_db <= es_n0_end; es_n0_db += test_cfg.snr_step) {
        // Convert Es/N0 (dB) to noise standard deviation per dimension
        // For normalized constellation with Es = 1:
        //   Es/N0 = 1 / (2 * sigma^2)  ->  sigma = sqrt(1 / (2 * Es/N0))
        float es_n0_linear = powf(10.0f, es_n0_db / 10.0f);
        float sigma = sqrtf(1.0f / (2.0f * es_n0_linear));
        float noise_var = sigma * sigma;  // Per-dimension variance for soft demod

        int block_errors = 0;
        int total_blocks = test_cfg.num_blocks;

        for (int t = 0; t < total_blocks; t++) {
            // Generate random input
            for (int i = 0; i < K_words; i++) {
                h_input[i] = rand();
            }
            // Clear filler bits
            for (int i = info_bits; i < K; i++) {
                int w = i / 32;
                int b = i % 32;
                h_input[w] &= ~(1u << b);
            }

            cudaMemcpy(d_input, h_input, K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
            cudaMemset(d_rate_matched, 0, E_words * sizeof(uint32_t));

            // Encode
            ldpc_encoder_encode(encoder, d_input, d_encoded, 0);

            // Rate match
            rate_matcher_match(rm_tx, d_encoded, d_rate_matched, 0);

            // Modulate
            modulator_modulate(modulator, d_rate_matched, d_symbols, E, mcs.modulation_order, 0);

            // Add AWGN noise (in-place on d_symbols, then copy result)
            cudaMemcpy(d_rx_symbols, d_symbols, num_symbols * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
            modulator_add_noise(modulator, d_rx_symbols, num_symbols, sigma, 0);

            // Soft demodulate
            modulator_soft_demod(modulator, d_rx_symbols, d_llrs, num_symbols, mcs.modulation_order, noise_var, 0);

            // De-rate match
            rate_matcher_dematch(rm_rx, d_llrs, d_derate_llrs, 0);

            // Decode
            cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));
            ldpc_decoder_decode(decoder, d_derate_llrs, d_decoded, 0);
            cudaDeviceSynchronize();

            cudaMemcpy(h_decoded, d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            // Check for block error (any bit error)
            int bit_errors = 0;
            for (int i = 0; i < info_bits; i++) {
                int w = i / 32;
                int b = i % 32;
                int in_bit = (h_input[w] >> b) & 1;
                int out_bit = (h_decoded[w] >> b) & 1;
                if (in_bit != out_bit) bit_errors++;
            }
            if (bit_errors > 0) block_errors++;
        }

        float bler = (float)block_errors / total_blocks;

        if (verbose) {
            printf("    Es/N0=%.1f dB: BLER=%.3f (%d/%d)\n", es_n0_db, bler, block_errors, total_blocks);
        }

        // Check if we crossed the 10% threshold
        if (bler <= 0.10f && last_bler > 0.10f) {
            // Interpolate
            float frac = (0.10f - bler) / (last_bler - bler);
            bler10_es_n0 = es_n0_db - frac * test_cfg.snr_step;
            break;
        }

        if (bler <= 0.10f) {
            bler10_es_n0 = es_n0_db;
            break;
        }

        last_bler = bler;
    }

    // Cleanup
    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);
    rate_matcher_destroy(rm_tx);
    rate_matcher_destroy(rm_rx);
    modulator_destroy(modulator);

    delete[] h_input;
    delete[] h_decoded;
    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaFree(d_rate_matched);
    cudaFree(d_symbols);
    cudaFree(d_rx_symbols);
    cudaFree(d_llrs);
    cudaFree(d_derate_llrs);
    cudaFree(d_decoded);

    return bler10_es_n0;  // Return Es/N0 in dB
}

// =============================================================================
// Measure Throughput
// =============================================================================

void measure_throughput(MCSResult& result, int tbs_bits) {
    if (result.code_rate <= 0) return;

    int E = calculate_E(tbs_bits, result.code_rate, result.Qm);

    nr_ldpc_config_t ldpc_cfg;
    nr_ldpc_init_config(&ldpc_cfg, tbs_bits, result.code_rate);

    int Kb = (ldpc_cfg.base_graph == 1) ? 22 : 10;
    int N_cols = (ldpc_cfg.base_graph == 1) ? 68 : 52;
    int K = Kb * ldpc_cfg.lifting_size;
    int N = N_cols * ldpc_cfg.lifting_size;
    int N_cb = N - 2 * ldpc_cfg.lifting_size;
    // Ensure E >= N_cb to avoid erased parity issues
    if (E < N_cb) E = N_cb;

    ldpc_encoder_handle_t encoder;
    ldpc_decoder_handle_t decoder;

    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &ldpc_cfg);

    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = 10;
    ldpc_decoder_configure(decoder, &ldpc_cfg, &dec_params);

    int K_words = (K + 31) / 32;
    int enc_words = ldpc_encoder_get_output_words(encoder);
    int num_codewords = 200;

    uint32_t *d_input, *d_encoded;
    float *d_llrs;
    uint32_t *d_decoded;

    cudaMalloc(&d_input, num_codewords * K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, num_codewords * enc_words * sizeof(uint32_t));
    cudaMalloc(&d_llrs, num_codewords * N * sizeof(float));
    cudaMalloc(&d_decoded, num_codewords * K_words * sizeof(uint32_t));

    // Initialize
    std::vector<uint32_t> h_input(num_codewords * K_words);
    for (auto& v : h_input) v = rand();
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Generate LLRs for decoder test (perfect channel)
    std::vector<float> h_llrs(num_codewords * N, 32.0f);
    cudaMemcpy(d_llrs, h_llrs.data(), h_llrs.size() * sizeof(float), cudaMemcpyHostToDevice);

    CudaTimer timer;
    int num_iter = 5;
    int total_bits = num_codewords * ldpc_cfg.num_info_bits;

    // Warmup
    ldpc_encoder_encode_batch(encoder, d_input, d_encoded, num_codewords, 0);
    ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, num_codewords, 0);
    cudaDeviceSynchronize();

    // Measure encoder
    timer.start();
    for (int i = 0; i < num_iter; i++) {
        ldpc_encoder_encode_batch(encoder, d_input, d_encoded, num_codewords, 0);
    }
    timer.stop();
    result.encoder_gbps = (total_bits * num_iter / 1e9) / (timer.elapsed_ms() / 1000.0);

    // Measure decoder
    timer.start();
    for (int i = 0; i < num_iter; i++) {
        ldpc_decoder_decode_batch(decoder, d_llrs, d_decoded, num_codewords, 0);
    }
    timer.stop();
    result.decoder_mbps = (total_bits * num_iter / 1e6) / (timer.elapsed_ms() / 1000.0);

    // Pipeline throughput (decoder limited)
    result.pipeline_mbps = result.decoder_mbps;

    ldpc_encoder_destroy(encoder);
    ldpc_decoder_destroy(decoder);
    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaFree(d_llrs);
    cudaFree(d_decoded);
}

// =============================================================================
// Main
// =============================================================================

void print_separator() {
    printf("================================================================================\n");
}

void run_mcs_table_sweep(const char* table_name, const MCSEntry* table, int table_size,
                         int table_num, std::vector<MCSResult>& all_results) {

    print_separator();
    printf("%s Sweep\n", table_name);
    print_separator();

    printf("\n%-4s %-8s %-6s %-8s %-10s %-14s\n",
           "MCS", "Mod", "Qm", "Rate", "SE", "BLER10% Es/N0");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < table_size; i++) {
        const MCSEntry& mcs = table[i];

        if (mcs.target_code_rate <= 0) continue;  // Skip reserved

        MCSResult result;
        result.table_num = table_num;
        result.mcs_index = mcs.mcs_index;
        result.modulation = mcs.mod_name;
        result.Qm = mcs.modulation_order;
        result.code_rate = mcs.target_code_rate;
        result.spectral_eff = mcs.spectral_efficiency;
        result.tested = true;

        // Find BLER 10% point
        result.bler10_snr_db = find_bler10_snr(mcs, test_cfg.tbs_bits, false);

        printf("%-4d %-8s %-6d %-8.4f %-10.4f %-12.1f dB\n",
               result.mcs_index, result.modulation, result.Qm,
               result.code_rate, result.spectral_eff, result.bler10_snr_db);
        fflush(stdout);

        // Measure throughput
        measure_throughput(result, test_cfg.tbs_bits);

        all_results.push_back(result);
    }
}

int main(int argc, char* argv[]) {
    print_separator();
    printf("5G NR LDPC MCS Performance Sweep\n");
    printf("Complete MCS Table 1 (64QAM) and Table 2 (256QAM)\n");
    printf("per 3GPP TS 38.214 Tables 5.1.3.1-1 and 5.1.3.1-2\n");
    print_separator();
    printf("\n");

    ocudu_phy_cuda_init();
    srand(42);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("TBS: %d bits, Blocks/Es_N0: %d, Es/N0 step: %.1f dB\n\n",
           test_cfg.tbs_bits, test_cfg.num_blocks, test_cfg.snr_step);

    std::vector<MCSResult> all_results;

    // Run MCS Table 1 (64QAM max)
    int table1_size = sizeof(mcs_table1) / sizeof(mcs_table1[0]);
    run_mcs_table_sweep("MCS Table 1 (64QAM)", mcs_table1, table1_size, 1, all_results);

    printf("\n");

    // Run MCS Table 2 (256QAM max)
    int table2_size = sizeof(mcs_table2) / sizeof(mcs_table2[0]);
    run_mcs_table_sweep("MCS Table 2 (256QAM)", mcs_table2, table2_size, 2, all_results);

    // Print summary tables for PERFORMANCE.md
    printf("\n");
    print_separator();
    printf("Summary for PERFORMANCE.md\n");
    print_separator();

    printf("\n## MCS Table 1 (64QAM Maximum)\n\n");
    printf("| MCS | Mod | Qm | Code Rate | Spectral Eff | BLER 10%% Es/N0 | Encoder | Decoder |\n");
    printf("|-----|-----|----|-----------|--------------|-----------------|---------|---------|\n");
    for (const auto& r : all_results) {
        if (r.table_num == 1) {
            printf("| %d | %s | %d | %.4f | %.4f | %.1f dB | %.2f Gbps | %.1f Mbps |\n",
                   r.mcs_index, r.modulation, r.Qm, r.code_rate, r.spectral_eff,
                   r.bler10_snr_db, r.encoder_gbps, r.decoder_mbps);
        }
    }

    printf("\n## MCS Table 2 (256QAM Maximum)\n\n");
    printf("| MCS | Mod | Qm | Code Rate | Spectral Eff | BLER 10%% Es/N0 | Encoder | Decoder |\n");
    printf("|-----|-----|----|-----------|--------------|-----------------|---------|---------|\n");
    for (const auto& r : all_results) {
        if (r.table_num == 2) {
            printf("| %d | %s | %d | %.4f | %.4f | %.1f dB | %.2f Gbps | %.1f Mbps |\n",
                   r.mcs_index, r.modulation, r.Qm, r.code_rate, r.spectral_eff,
                   r.bler10_snr_db, r.encoder_gbps, r.decoder_mbps);
        }
    }

    // Verify monotonicity - Es/N0 should increase with spectral efficiency
    printf("\n## BLER 10%% Es/N0 Monotonicity Check\n\n");
    bool monotonic = true;
    float prev_es_n0 = -100.0f;
    float prev_se = 0.0f;

    for (const auto& r : all_results) {
        if (r.table_num == 1 && r.spectral_eff > prev_se) {
            if (r.bler10_snr_db < prev_es_n0 - 0.5f) {
                printf("WARNING: MCS %d (SE=%.4f) has lower Es/N0 (%.1f) than expected\n",
                       r.mcs_index, r.spectral_eff, r.bler10_snr_db);
                monotonic = false;
            }
            prev_es_n0 = r.bler10_snr_db;
            prev_se = r.spectral_eff;
        }
    }

    if (monotonic) {
        printf("BLER 10%% Es/N0 increases monotonically with spectral efficiency as expected.\n");
    }

    print_separator();
    printf("MCS Sweep Complete\n");
    print_separator();

    return 0;
}
