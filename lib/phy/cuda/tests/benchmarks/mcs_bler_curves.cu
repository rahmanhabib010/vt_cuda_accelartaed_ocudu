/**
 * @file mcs_bler_curves.cu
 * @brief Generate BLER vs Es/N0 curves for all MCS values
 *
 * Outputs CSV-friendly data for plotting BLER curves across MCS tables.
 */

#include "ocudu_phy_cuda.h"
#include "transport_block.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <curand.h>
#include <chrono>

// AWGN kernel
__global__ void add_awgn_kernel(cuFloatComplex* symbols, const float* noise, int n, float noise_std) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        symbols[idx].x += noise[2*idx] * noise_std;
        symbols[idx].y += noise[2*idx+1] * noise_std;
    }
}

// MCS Table 1 (TS 38.214 Table 5.1.3.1-1) - for PDSCH
struct MCSEntry {
    int mcs;
    int mod_order;  // 2=QPSK, 4=16QAM, 6=64QAM
    int R_x1024;    // Rate * 1024
};

MCSEntry mcs_table1[] = {
    {0,  2, 120}, {1,  2, 157}, {2,  2, 193}, {3,  2, 251}, {4,  2, 308},
    {5,  2, 379}, {6,  2, 449}, {7,  2, 526}, {8,  2, 602}, {9,  2, 679},
    {10, 4, 340}, {11, 4, 378}, {12, 4, 434}, {13, 4, 490}, {14, 4, 553},
    {15, 4, 616}, {16, 4, 658},
    {17, 6, 438}, {18, 6, 466}, {19, 6, 517}, {20, 6, 567}, {21, 6, 616},
    {22, 6, 666}, {23, 6, 719}, {24, 6, 772}, {25, 6, 822}, {26, 6, 873},
    {27, 6, 910}, {28, 6, 948},
};
const int NUM_MCS_TABLE1 = sizeof(mcs_table1) / sizeof(MCSEntry);

// MCS Table 2 (TS 38.214 Table 5.1.3.1-2) - 256QAM
MCSEntry mcs_table2[] = {
    {0,  2, 120}, {1,  2, 193}, {2,  2, 308}, {3,  2, 449}, {4,  2, 602},
    {5,  4, 378}, {6,  4, 434}, {7,  4, 490}, {8,  4, 553}, {9,  4, 616},
    {10, 4, 658}, {11, 6, 466}, {12, 6, 517}, {13, 6, 567}, {14, 6, 616},
    {15, 6, 666}, {16, 6, 719}, {17, 6, 772}, {18, 6, 822}, {19, 6, 873},
    {20, 8, 682}, {21, 8, 711}, {22, 8, 754}, {23, 8, 797}, {24, 8, 841},
    {25, 8, 885}, {26, 8, 916}, {27, 8, 948},
};
const int NUM_MCS_TABLE2 = sizeof(mcs_table2) / sizeof(MCSEntry);

// MCS Table 3 (TS 38.214 Table 5.1.3.1-3) - Low SE (64QAM max)
MCSEntry mcs_table3[] = {
    {0,  2, 30},  {1,  2, 40},  {2,  2, 50},  {3,  2, 64},  {4,  2, 78},
    {5,  2, 99},  {6,  2, 120}, {7,  2, 157}, {8,  2, 193}, {9,  2, 251},
    {10, 2, 308}, {11, 2, 379}, {12, 2, 449}, {13, 2, 526}, {14, 2, 602},
    {15, 4, 340}, {16, 4, 378}, {17, 4, 434}, {18, 4, 490}, {19, 4, 553},
    {20, 4, 616}, {21, 6, 438}, {22, 6, 466}, {23, 6, 517}, {24, 6, 567},
    {25, 6, 616}, {26, 6, 666}, {27, 6, 719}, {28, 6, 772},
};
const int NUM_MCS_TABLE3 = sizeof(mcs_table3) / sizeof(MCSEntry);

struct TestResult {
    int mcs;
    int mod_order;
    float code_rate;
    float esn0_db;
    float bler;
    int num_trials;
    int num_errors;
};

float measure_bler(MCSEntry& m, int tbs, float EsN0_dB, int trials,
                   curandGenerator_t gen, modulator_handle_t mod,
                   int* out_errors = nullptr) {
    float code_rate = (float)m.R_x1024 / 1024.0f;
    int E = (int)ceilf((float)tbs / code_rate);
    E = ((E / m.mod_order) * m.mod_order);

    tb_encoder_handle_t encoder;
    tb_encoder_config_t enc_cfg = {};
    enc_cfg.tb_size_bits = tbs;
    enc_cfg.num_layers = 1;
    enc_cfg.modulation_order = m.mod_order;
    enc_cfg.num_allocated_res = E;
    enc_cfg.redundancy_version = 0;
    enc_cfg.code_rate = code_rate;
    enc_cfg.n_RNTI = 0x1234;
    enc_cfg.n_ID = 0;
    enc_cfg.enable_scrambling = true;
    tb_encoder_create(&encoder);
    if (tb_encoder_configure(encoder, &enc_cfg) != NR_LDPC_SUCCESS) {
        tb_encoder_destroy(encoder);
        return -1;
    }

    tb_decoder_handle_t decoder;
    tb_decoder_config_t dec_cfg = {};
    dec_cfg.tb_size_bits = tbs;
    dec_cfg.num_layers = 1;
    dec_cfg.modulation_order = m.mod_order;
    dec_cfg.num_received_bits = E;
    dec_cfg.redundancy_version = 0;
    dec_cfg.code_rate = code_rate;
    dec_cfg.max_iterations = 25;
    dec_cfg.n_RNTI = 0x1234;
    dec_cfg.n_ID = 0;
    dec_cfg.enable_scrambling = true;
    dec_cfg.llr_clamp = 32.0f;
    tb_decoder_create(&decoder);
    if (tb_decoder_configure(decoder, &dec_cfg) != NR_LDPC_SUCCESS) {
        tb_encoder_destroy(encoder);
        tb_decoder_destroy(decoder);
        return -1;
    }

    int tb_bytes = (tbs + 7) / 8;
    int enc_bytes = (E + 7) / 8;
    int num_symbols = E / m.mod_order;

    uint8_t *d_tx_tb, *d_rx_tb, *d_encoded;
    cuFloatComplex *d_symbols;
    float *d_llrs, *d_noise;
    cudaMalloc(&d_tx_tb, tb_bytes);
    cudaMalloc(&d_rx_tb, tb_bytes);
    cudaMalloc(&d_encoded, enc_bytes);
    cudaMalloc(&d_symbols, num_symbols * sizeof(cuFloatComplex));
    cudaMalloc(&d_llrs, E * sizeof(float));
    cudaMalloc(&d_noise, num_symbols * 2 * sizeof(float));

    std::vector<uint8_t> h_tx(tb_bytes);

    float EsN0_lin = powf(10.0f, EsN0_dB / 10.0f);
    float noise_var = 1.0f / EsN0_lin;
    float noise_std = sqrtf(noise_var / 2.0f);

    int errors = 0;
    for (int trial = 0; trial < trials; trial++) {
        srand(trial * 12345 + m.mcs);
        for (int i = 0; i < tb_bytes; i++) h_tx[i] = rand() & 0xFF;

        cudaMemcpy(d_tx_tb, h_tx.data(), tb_bytes, cudaMemcpyHostToDevice);
        tb_encoder_encode(encoder, d_tx_tb, d_encoded, 0);
        modulator_modulate(mod, (uint32_t*)d_encoded, d_symbols, E, m.mod_order, 0);

        // Add AWGN
        curandGenerateNormal(gen, d_noise, num_symbols * 2, 0.0f, 1.0f);
        int blocks = (num_symbols + 255) / 256;
        add_awgn_kernel<<<blocks, 256>>>(d_symbols, d_noise, num_symbols, noise_std);

        modulator_soft_demod(mod, d_symbols, d_llrs, num_symbols, m.mod_order, noise_var, 0);

        cudaMemset(d_rx_tb, 0, tb_bytes);
        tb_decode_result_t result;
        tb_decoder_decode(decoder, d_llrs, d_rx_tb, &result, 0);
        cudaDeviceSynchronize();

        if (!result.crc_pass) errors++;
    }

    cudaFree(d_tx_tb); cudaFree(d_rx_tb); cudaFree(d_encoded);
    cudaFree(d_symbols); cudaFree(d_llrs); cudaFree(d_noise);
    tb_encoder_destroy(encoder); tb_decoder_destroy(decoder);

    if (out_errors) *out_errors = errors;
    return (float)errors / trials;
}

void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --table <1|2|3>   MCS table (default: 1)\n");
    fprintf(stderr, "  --tbs <bits>      Transport block size (default: 2048)\n");
    fprintf(stderr, "  --trials <n>      Trials per SNR point (default: 100)\n");
    fprintf(stderr, "  --snr-min <dB>    Minimum Es/N0 (default: -8)\n");
    fprintf(stderr, "  --snr-max <dB>    Maximum Es/N0 (default: 25)\n");
    fprintf(stderr, "  --snr-step <dB>   Es/N0 step size (default: 1.0)\n");
    fprintf(stderr, "  --mcs <n>         Single MCS to test (default: all)\n");
    fprintf(stderr, "  --csv             Output CSV format\n");
    fprintf(stderr, "  --summary         Output summary table only (10%% BLER point)\n");
}

int main(int argc, char** argv) {
    int table = 1;
    int tbs = 2048;
    int trials = 100;
    float snr_min = -8.0f;
    float snr_max = 25.0f;
    float snr_step = 1.0f;
    int single_mcs = -1;
    bool csv_output = false;
    bool summary_only = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--table") == 0 && i+1 < argc) table = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tbs") == 0 && i+1 < argc) tbs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--trials") == 0 && i+1 < argc) trials = atoi(argv[++i]);
        else if (strcmp(argv[i], "--snr-min") == 0 && i+1 < argc) snr_min = atof(argv[++i]);
        else if (strcmp(argv[i], "--snr-max") == 0 && i+1 < argc) snr_max = atof(argv[++i]);
        else if (strcmp(argv[i], "--snr-step") == 0 && i+1 < argc) snr_step = atof(argv[++i]);
        else if (strcmp(argv[i], "--mcs") == 0 && i+1 < argc) single_mcs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0) csv_output = true;
        else if (strcmp(argv[i], "--summary") == 0) summary_only = true;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        }
    }

    MCSEntry* mcs_table;
    int num_mcs;
    if (table == 1) { mcs_table = mcs_table1; num_mcs = NUM_MCS_TABLE1; }
    else if (table == 2) { mcs_table = mcs_table2; num_mcs = NUM_MCS_TABLE2; }
    else if (table == 3) { mcs_table = mcs_table3; num_mcs = NUM_MCS_TABLE3; }
    else { fprintf(stderr, "Invalid table: %d\n", table); return 1; }

    // Auto-adjust SNR range for Table 3 if using defaults
    // Table 3 has very low code rates (0.029-0.076) that achieve 10% BLER
    // at much lower SNRs (-12 to -8 dB) than the default range
    if (table == 3 && snr_min == -8.0f) {
        snr_min = -20.0f;  // Extend range for low code rate MCS entries
    }

    ocudu_phy_cuda_init();

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);

    modulator_handle_t mod;
    modulator_create(&mod);

    auto start = std::chrono::high_resolution_clock::now();

    if (summary_only) {
        // Summary mode: find 10% BLER point for each MCS
        if (csv_output) {
            printf("MCS,ModOrder,Rate,EsN0_10pct_BLER_dB,Shannon_EsN0_dB,Gap_dB\n");
        } else {
            printf("MCS Table %d Summary (TBS=%d, %d trials):\n\n", table, tbs, trials);
            printf("MCS | Mod   | Rate   | Es/N0 @10%%BLER | Shannon | Gap\n");
            printf("----|-------|--------|----------------|---------|-----\n");
        }

        for (int m_idx = 0; m_idx < num_mcs; m_idx++) {
            MCSEntry& m = mcs_table[m_idx];
            if (single_mcs >= 0 && m.mcs != single_mcs) continue;

            float code_rate = (float)m.R_x1024 / 1024.0f;

            // Binary search for 10% BLER
            float low = snr_min, high = snr_max;
            float target = 0.1f;

            // Check endpoints
            float bler_high = measure_bler(m, tbs, high, trials/2, gen, mod);
            if (bler_high < 0) {
                if (csv_output) printf("%d,%d,%.4f,NaN,NaN,NaN\n", m.mcs, m.mod_order, code_rate);
                else printf("%3d | %-5s | %.4f | CONFIG FAIL    |         |\n",
                           m.mcs, (m.mod_order==2)?"QPSK":(m.mod_order==4)?"16QAM":(m.mod_order==6)?"64QAM":"256QAM", code_rate);
                continue;
            }
            if (bler_high > target) {
                if (csv_output) printf("%d,%d,%.4f,>%.0f,NaN,NaN\n", m.mcs, m.mod_order, code_rate, high);
                else printf("%3d | %-5s | %.4f | >+%.0f dB       |         |\n",
                           m.mcs, (m.mod_order==2)?"QPSK":(m.mod_order==4)?"16QAM":(m.mod_order==6)?"64QAM":"256QAM", code_rate, high);
                continue;
            }

            for (int iter = 0; iter < 8; iter++) {
                float mid = (low + high) / 2.0f;
                float bler = measure_bler(m, tbs, mid, trials, gen, mod);
                if (bler > target) low = mid;
                else high = mid;
            }

            float snr_10pct = (low + high) / 2.0f;

            // Shannon limit for AWGN channel
            // Spectral efficiency η = R × m (bits per symbol)
            // Shannon capacity: C = log2(1 + Es/N0) → Es/N0 = 2^C - 1
            // At capacity C = η: Es/N0_Shannon = 2^η - 1
            float eta = code_rate * m.mod_order;  // spectral efficiency (bits/symbol)
            float shannon_EsN0_lin = powf(2.0f, eta) - 1.0f;
            float shannon_EsN0_dB = 10.0f * log10f(shannon_EsN0_lin);
            float gap = snr_10pct - shannon_EsN0_dB;

            const char* mod_name = (m.mod_order == 2) ? "QPSK" :
                                   (m.mod_order == 4) ? "16QAM" :
                                   (m.mod_order == 6) ? "64QAM" : "256QAM";

            if (csv_output) {
                printf("%d,%d,%.4f,%.2f,%.2f,%.2f\n", m.mcs, m.mod_order, code_rate, snr_10pct, shannon_EsN0_dB, gap);
            } else {
                printf("%3d | %-5s | %.4f | %+6.1f dB      | %+5.1f   | %+.1f\n",
                       m.mcs, mod_name, code_rate, snr_10pct, shannon_EsN0_dB, gap);
            }
        }
    } else {
        // Full curve mode: BLER at each SNR point
        if (csv_output) {
            printf("MCS,ModOrder,Rate,EsN0_dB,BLER,Trials,Errors\n");
        } else {
            printf("BLER vs Es/N0 - MCS Table %d (TBS=%d, %d trials/point)\n\n", table, tbs, trials);
        }

        for (int m_idx = 0; m_idx < num_mcs; m_idx++) {
            MCSEntry& m = mcs_table[m_idx];
            if (single_mcs >= 0 && m.mcs != single_mcs) continue;

            float code_rate = (float)m.R_x1024 / 1024.0f;
            const char* mod_name = (m.mod_order == 2) ? "QPSK" :
                                   (m.mod_order == 4) ? "16QAM" :
                                   (m.mod_order == 6) ? "64QAM" : "256QAM";

            if (!csv_output) {
                printf("=== MCS %d (%s, R=%.4f) ===\n", m.mcs, mod_name, code_rate);
                printf("Es/N0 (dB) | BLER   | Errors\n");
                printf("-----------|--------|--------\n");
            }

            bool done = false;
            for (float snr = snr_min; snr <= snr_max && !done; snr += snr_step) {
                int errors;
                float bler = measure_bler(m, tbs, snr, trials, gen, mod, &errors);

                if (bler < 0) {
                    if (!csv_output) printf("CONFIG FAIL\n");
                    break;
                }

                if (csv_output) {
                    printf("%d,%d,%.4f,%.1f,%.4f,%d,%d\n",
                           m.mcs, m.mod_order, code_rate, snr, bler, trials, errors);
                } else {
                    printf("  %+6.1f   | %.4f | %d/%d\n", snr, bler, errors, trials);
                }

                // Stop if BLER is very low
                if (bler < 0.001f) done = true;
            }
            if (!csv_output) printf("\n");
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    if (!csv_output) {
        printf("Total time: %ld seconds\n", duration.count());
    }

    curandDestroyGenerator(gen);
    modulator_destroy(mod);

    return 0;
}
