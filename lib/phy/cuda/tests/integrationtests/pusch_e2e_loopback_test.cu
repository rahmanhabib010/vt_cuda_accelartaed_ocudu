/**
 * @file pusch_e2e_loopback_test.cu
 * @brief PUSCH End-to-End Loopback Test using PRODUCTION Kernel Paths
 *
 * Tests the complete PUSCH TX + RX chain using the exact same fused kernels
 * the gnb uses in production:
 *   TX: tb_encoder_encode_to_symbols_int8()  (CRC→LDPC→RM→Interleave→Scramble→Modulate→INT8)
 *   Grid: INT8→float→CBF16 resource grid + CPU-generated DMRS pilots
 *   RX: pusch_e2e_process_full_gpu_optimized() (ChEst→FD smooth→TimeInterp→MMSE EQ→SoftDemod→Descramble)
 *   Decode: tb_decoder_decode_half()  (Deinterleave→Rate Dematch→LDPC Decode→CRC)
 *
 * This validates the EXACT production kernel paths, catching bugs like
 * bit-ordering issues that standalone TB encode+modulate tests miss.
 */

#include "ocudu_phy_cuda.h"
#include "transport_block.h"
#include "pusch_e2e.h"
#include "dmrs_utils.h"
#include "modulation.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <complex>
#include <algorithm>

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Constants
// ============================================================================

static constexpr int CARRIER_PRB = 273;
static constexpr int SC_PER_PRB = 12;
static constexpr int CARRIER_SC = CARRIER_PRB * SC_PER_PRB;  // 3276
static constexpr int SLOT_SYMBOLS = 14;
static constexpr int NC_SKIP = 1600;  // Gold sequence Nc

// ============================================================================
// MCS Table per TS 38.214 Table 5.1.3.1-1 (64QAM table)
// ============================================================================

struct MCSEntry {
    int index;
    int mod_order;
    float code_rate;
    const char* name;
};

static const MCSEntry MCS_TABLE[] = {
    {0,  2, 120.0f/1024, "QPSK R=120/1024"},
    {1,  2, 157.0f/1024, "QPSK R=157/1024"},
    {2,  2, 193.0f/1024, "QPSK R=193/1024"},
    {3,  2, 251.0f/1024, "QPSK R=251/1024"},
    {4,  2, 308.0f/1024, "QPSK R=308/1024"},
    {5,  2, 379.0f/1024, "QPSK R=379/1024"},
    {6,  2, 449.0f/1024, "QPSK R=449/1024"},
    {7,  2, 526.0f/1024, "QPSK R=526/1024"},
    {8,  2, 602.0f/1024, "QPSK R=602/1024"},
    {9,  2, 679.0f/1024, "QPSK R=679/1024"},
    {10, 4, 340.0f/1024, "16QAM R=340/1024"},
    {11, 4, 378.0f/1024, "16QAM R=378/1024"},
    {12, 4, 434.0f/1024, "16QAM R=434/1024"},
    {13, 4, 490.0f/1024, "16QAM R=490/1024"},
    {14, 4, 553.0f/1024, "16QAM R=553/1024"},
    {15, 4, 616.0f/1024, "16QAM R=616/1024"},
    {16, 4, 658.0f/1024, "16QAM R=658/1024"},
    {17, 6, 438.0f/1024, "64QAM R=438/1024"},
    {18, 6, 466.0f/1024, "64QAM R=466/1024"},
    {19, 6, 517.0f/1024, "64QAM R=517/1024"},
    {20, 6, 567.0f/1024, "64QAM R=567/1024"},
    {21, 6, 616.0f/1024, "64QAM R=616/1024"},
    {22, 6, 666.0f/1024, "64QAM R=666/1024"},
    {23, 6, 719.0f/1024, "64QAM R=719/1024"},
    {24, 6, 772.0f/1024, "64QAM R=772/1024"},
    {25, 6, 822.0f/1024, "64QAM R=822/1024"},
    {26, 6, 873.0f/1024, "64QAM R=873/1024"},
    {27, 6, 910.0f/1024, "64QAM R=910/1024"},
    {28, 6, 948.0f/1024, "64QAM R=948/1024"},
};

// ============================================================================
// TBS Table per TS 38.214 Table 5.1.3.2-1 (partial)
// ============================================================================

static const int TBS_TABLE[] = {
    24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144,
    152, 160, 168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320,
    336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640,
    672, 704, 736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160,
    1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736,
    1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600,
    2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824
};
static constexpr int TBS_TABLE_SIZE = sizeof(TBS_TABLE) / sizeof(TBS_TABLE[0]);

// ============================================================================
// Helper Functions
// ============================================================================

static double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000.0 + ts.tv_nsec / 1000.0;
}

static int count_bits_set(int mask) {
    int count = 0;
    for (int i = 0; i < 14; i++)
        if (mask & (1 << i)) count++;
    return count;
}

static int find_closest_tbs(int n_info_prime) {
    for (int i = 0; i < TBS_TABLE_SIZE; i++)
        if (TBS_TABLE[i] >= n_info_prime) return TBS_TABLE[i];
    return TBS_TABLE[TBS_TABLE_SIZE - 1];
}

static int calculate_tbs(int nof_prb, int num_symbols, int num_dmrs_symbols,
                         int dmrs_re_per_prb_per_sym, int mod_order, float code_rate) {
    // Per TS 38.214 §5.1.3.2
    int re_per_prb = SC_PER_PRB * num_symbols - dmrs_re_per_prb_per_sym * num_dmrs_symbols;
    int n_re_prime = std::min(156, re_per_prb);
    int n_re = n_re_prime * nof_prb;
    float n_info = n_re * code_rate * mod_order;

    if (n_info <= 3824) {
        int n = std::max(3, (int)floor(log2(n_info)) - 6);
        int n_info_prime = std::max(24, (int)(pow(2, n) * floor(n_info / pow(2, n))));
        return find_closest_tbs(n_info_prime);
    } else {
        int n = (int)floor(log2(n_info - 24)) - 5;
        int n_info_prime = std::max(3840, (int)(pow(2, n) * round((n_info - 24) / pow(2, n))));
        int max_cb_size = 8424;
        int c = (n_info_prime > max_cb_size) ? (int)ceil((n_info_prime + 24.0) / max_cb_size) : 1;
        return 8 * c * (int)ceil((n_info_prime + 24.0) / (8.0 * c)) - 24;
    }
}

static float get_int8_norm_factor(int mod_order) {
    switch (mod_order) {
        case 2: return 1.0f / sqrtf(2.0f);
        case 4: return 1.0f / sqrtf(10.0f);
        case 6: return 1.0f / sqrtf(42.0f);
        case 8: return 1.0f / sqrtf(170.0f);
        default: return 1.0f;
    }
}

static void generate_random_data(uint8_t* data, int num_bytes) {
    for (int i = 0; i < num_bytes; i++)
        data[i] = rand() & 0xFF;
}

static int count_bit_errors(const uint8_t* a, const uint8_t* b, int num_bits) {
    int errors = 0;
    int num_bytes = (num_bits + 7) / 8;
    for (int i = 0; i < num_bytes; i++)
        errors += __builtin_popcount(a[i] ^ b[i]);
    int extra = num_bytes * 8 - num_bits;
    if (extra > 0) {
        uint8_t mask = (0xFF << extra) & 0xFF;
        errors -= __builtin_popcount(a[num_bytes-1] ^ b[num_bytes-1]);
        errors += __builtin_popcount((a[num_bytes-1] ^ b[num_bytes-1]) & mask);
    }
    return errors;
}

// ============================================================================
// CPU Gold Sequence Generator (3GPP TS 38.211)
// ============================================================================

static void cpu_lfsr_step(uint32_t& x1, uint32_t& x2) {
    uint32_t new_x1 = ((x1 >> 3) ^ x1) & 1;
    x1 = (x1 >> 1) | (new_x1 << 30);
    uint32_t new_x2 = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
    x2 = (x2 >> 1) | (new_x2 << 30);
}

/**
 * Generate DMRS pilot symbols (QPSK from Gold sequence).
 * Matches the E2E kernel's internal DMRS generation exactly.
 *
 * Per TS 38.211 §6.4.1.1:
 *   r(m) = (1/√2)(1 - 2·c(2m)) + j·(1/√2)(1 - 2·c(2m+1))
 * where c(n) is the Gold sequence with c_init per §6.4.1.1.1.1.
 */
static void generate_dmrs_pilots(std::vector<std::complex<float>>& pilots,
                                  uint32_t scrambling_id, int n_scid,
                                  int slot_idx, int symbol_idx, int nof_pilots) {
    uint32_t c_init = dmrs_compute_c_init(slot_idx, symbol_idx, scrambling_id, n_scid);

    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;

    // Advance by Nc = 1600
    for (int i = 0; i < NC_SKIP; i++)
        cpu_lfsr_step(x1, x2);

    pilots.resize(nof_pilots);
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);

    for (int m = 0; m < nof_pilots; m++) {
        uint32_t c_real = (x1 ^ x2) & 1;
        cpu_lfsr_step(x1, x2);
        uint32_t c_imag = (x1 ^ x2) & 1;
        cpu_lfsr_step(x1, x2);
        pilots[m] = std::complex<float>(
            (1.0f - 2.0f * c_real) * inv_sqrt2,
            (1.0f - 2.0f * c_imag) * inv_sqrt2);
    }
}

// ============================================================================
// CBF16 Packing (CPU)
// ============================================================================

static uint32_t float_to_cbf16(float re, float im) {
    uint32_t re_bits, im_bits;
    memcpy(&re_bits, &re, sizeof(float));
    memcpy(&im_bits, &im, sizeof(float));
    uint16_t re_bf16 = (uint16_t)(re_bits >> 16);
    uint16_t im_bf16 = (uint16_t)(im_bits >> 16);
    return (uint32_t)re_bf16 | ((uint32_t)im_bf16 << 16);
}

// ============================================================================
// Resource Grid Builder (CPU)
// ============================================================================

/**
 * Build a CBF16 resource grid with DMRS pilots and INT8 data symbols.
 *
 * Grid layout: [nof_symbols * grid_nof_subcarriers] as uint32_t (CBF16).
 * For identity channel (no noise): grid contains exact pilot+data values.
 *
 * @param h_grid        Output grid (host, preallocated)
 * @param h_int8_data   INT8 data symbols from production TX encoder
 * @param nof_prb       Number of allocated PRBs
 * @param start_prb     Starting PRB in carrier grid
 * @param nof_symbols   Number of OFDM symbols
 * @param dmrs_mask     Bitmask of DMRS symbols
 * @param scrambling_id DMRS scrambling ID
 * @param n_scid        DMRS n_scid
 * @param slot_idx      Slot index
 * @param mod_order     Modulation order (for INT8 normalization)
 * @param noise_std     Per-dimension noise std (0 = identity channel)
 * @param h_re_indices  Output: data RE indices for E2E kernel
 * @param nof_data_re   Output: number of data REs
 */
static void build_resource_grid(
    std::vector<uint32_t>& h_grid,
    const int8_t* h_int8_data,
    int nof_prb, int start_prb, int nof_symbols,
    int dmrs_mask, uint32_t scrambling_id, int n_scid, int slot_idx,
    int mod_order, float noise_std,
    std::vector<int>& h_re_indices, int& nof_data_re)
{
    int grid_sc = CARRIER_SC;
    int grid_size = nof_symbols * grid_sc;
    h_grid.assign(grid_size, 0);  // Zero-fill (unused REs = 0)

    float norm = get_int8_norm_factor(mod_order);
    int alloc_sc_start = start_prb * SC_PER_PRB;
    int alloc_sc_end = (start_prb + nof_prb) * SC_PER_PRB;

    // For nof_cdm_groups_without_data = 2, DMRS Type 1:
    // DMRS symbols: all 12 subcarriers per PRB are DMRS (no data)
    // Non-DMRS symbols: all 12 subcarriers per PRB are data

    int data_sym_idx = 0;
    h_re_indices.clear();

    // Simple Gaussian noise generator (Box-Muller)
    unsigned int noise_seed = 42 + slot_idx * 7;
    auto next_noise = [&]() -> float {
        if (noise_std <= 0.0f) return 0.0f;
        // Park-Miller LCG + Box-Muller
        noise_seed = noise_seed * 1103515245 + 12345;
        float u1 = (float)((noise_seed >> 1) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        if (u1 < 1e-10f) u1 = 1e-10f;
        noise_seed = noise_seed * 1103515245 + 12345;
        float u2 = (float)((noise_seed >> 1) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2) * noise_std;
    };

    for (int sym = 0; sym < nof_symbols; sym++) {
        bool is_dmrs = (dmrs_mask & (1 << sym)) != 0;

        if (is_dmrs) {
            // Generate DMRS pilots for this symbol
            // Must generate from position 0 up to (start_prb + nof_prb) * 6 because
            // the E2E kernel uses pilot_idx = (start_prb + prb_idx) * 6 + d
            int total_pilots = (start_prb + nof_prb) * 6;
            std::vector<std::complex<float>> pilots;
            generate_dmrs_pilots(pilots, scrambling_id, n_scid, slot_idx, sym, total_pilots);

            // Place pilots at DMRS positions (even subcarriers within allocation)
            // Also place pilots at odd subcarriers (CDM group 1 — use same pilots for simplicity,
            // the E2E kernel only reads CDM group 0 for single-layer)
            int pilot_idx = start_prb * 6;  // Account for start_prb offset in Gold sequence
            for (int prb = 0; prb < nof_prb; prb++) {
                int base_sc = alloc_sc_start + prb * SC_PER_PRB;
                for (int p = 0; p < 6; p++) {
                    // CDM group 0: even subcarriers
                    int sc_even = base_sc + p * 2;
                    float re = pilots[pilot_idx].real() + next_noise();
                    float im = pilots[pilot_idx].imag() + next_noise();
                    h_grid[sym * grid_sc + sc_even] = float_to_cbf16(re, im);

                    // CDM group 1: odd subcarriers (place zero or pilot — doesn't matter for single port)
                    int sc_odd = base_sc + p * 2 + 1;
                    float re2 = pilots[pilot_idx].real() + next_noise();
                    float im2 = pilots[pilot_idx].imag() + next_noise();
                    h_grid[sym * grid_sc + sc_odd] = float_to_cbf16(re2, im2);

                    pilot_idx++;
                }
            }
        } else {
            // Data symbol: place normalized data at all subcarriers within allocation
            for (int sc = alloc_sc_start; sc < alloc_sc_end; sc++) {
                float re = h_int8_data[2 * data_sym_idx] * norm + next_noise();
                float im = h_int8_data[2 * data_sym_idx + 1] * norm + next_noise();
                h_grid[sym * grid_sc + sc] = float_to_cbf16(re, im);

                // Record RE index for E2E kernel
                h_re_indices.push_back(sym * grid_sc + sc);
                data_sym_idx++;
            }
        }
    }

    nof_data_re = data_sym_idx;
}

// ============================================================================
// Test Configuration
// ============================================================================

struct PUSCHTestConfig {
    int nof_prb;
    int start_prb;
    int nof_symbols;
    int mcs_index;
    int dmrs_symbol_mask;
    uint16_t rnti;
    uint16_t n_id;
    uint32_t scrambling_id;
    int n_scid;
    int slot_idx;
    float snr_db;          // 0 = identity channel (no noise)
    const char* name;
};

// ============================================================================
// Run Single PUSCH E2E Loopback Test
// ============================================================================

static bool run_pusch_e2e_test(const PUSCHTestConfig& cfg, bool verbose) {
    int mod_order = MCS_TABLE[cfg.mcs_index].mod_order;
    float code_rate = MCS_TABLE[cfg.mcs_index].code_rate;
    int num_dmrs_symbols = count_bits_set(cfg.dmrs_symbol_mask);

    // For Type 1 with nof_cdm_groups_without_data=2: all 12 SCs per PRB are DMRS on DMRS symbols
    int dmrs_re_per_prb_per_sym = 12;  // Both CDM groups
    int num_data_symbols = cfg.nof_symbols - num_dmrs_symbols;
    int num_data_re = cfg.nof_prb * SC_PER_PRB * num_data_symbols;
    int encoded_bits = num_data_re * mod_order;

    // Calculate TBS
    int tbs = calculate_tbs(cfg.nof_prb, cfg.nof_symbols, num_dmrs_symbols,
                            dmrs_re_per_prb_per_sym, mod_order, code_rate);
    if (tbs < 24) tbs = 24;

    // Sanity check: encoded bits must exceed TBS
    if (encoded_bits <= tbs) {
        if (verbose) printf("  SKIP: encoded_bits(%d) <= tbs(%d)\n", encoded_bits, tbs);
        return true;  // Skip, not a failure
    }

    int tb_bytes = (tbs + 7) / 8;
    int num_symbols_tx = encoded_bits / mod_order;  // Must equal num_data_re

    if (verbose) {
        printf("  PRBs: %d@%d, MCS: %d (%s), TBS: %d bits, G: %d bits, DataRE: %d\n",
               cfg.nof_prb, cfg.start_prb, cfg.mcs_index, MCS_TABLE[cfg.mcs_index].name,
               tbs, encoded_bits, num_data_re);
        printf("  DMRS: mask=0x%04X (%d syms), RNTI=0x%04X, n_ID=%d, slot=%d\n",
               cfg.dmrs_symbol_mask, num_dmrs_symbols, cfg.rnti, cfg.n_id, cfg.slot_idx);
    }

    bool test_passed = true;
    nr_ldpc_status_t status;

    // Allocate host memory
    std::vector<uint8_t> h_tx_tb(tb_bytes);
    std::vector<uint8_t> h_rx_tb(tb_bytes);

    // Device memory
    uint8_t* d_tx_tb = nullptr;
    int8_t* d_symbols_int8 = nullptr;
    void* d_grid_cbf16 = nullptr;
    void* d_llrs_half = nullptr;
    int* d_re_indices = nullptr;
    uint8_t* d_rx_tb = nullptr;

    int grid_size = cfg.nof_symbols * CARRIER_SC;
    CHECK_CUDA(cudaMalloc(&d_tx_tb, tb_bytes));
    CHECK_CUDA(cudaMalloc(&d_symbols_int8, num_symbols_tx * 2));
    CHECK_CUDA(cudaMalloc(&d_grid_cbf16, grid_size * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_llrs_half, encoded_bits * sizeof(uint16_t)));  // FP16
    CHECK_CUDA(cudaMalloc(&d_re_indices, num_data_re * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rx_tb, tb_bytes));

    // Create handles
    tb_encoder_handle_t encoder = nullptr;
    tb_decoder_handle_t decoder = nullptr;
    pusch_e2e_handle_t e2e = nullptr;

    status = tb_encoder_create(&encoder);
    if (status != NR_LDPC_SUCCESS) { printf("ERROR: create encoder\n"); test_passed = false; goto cleanup; }
    status = tb_decoder_create(&decoder);
    if (status != NR_LDPC_SUCCESS) { printf("ERROR: create decoder\n"); test_passed = false; goto cleanup; }
    status = pusch_e2e_create(&e2e);
    if (status != NR_LDPC_SUCCESS) { printf("ERROR: create E2E\n"); test_passed = false; goto cleanup; }

    // Configure TB encoder — production config
    {
        tb_encoder_config_t enc_cfg = {};
        enc_cfg.tb_size_bits = tbs;
        enc_cfg.num_layers = 1;
        enc_cfg.modulation_order = mod_order;
        enc_cfg.num_allocated_res = encoded_bits;
        enc_cfg.redundancy_version = 0;
        enc_cfg.code_rate = (float)tbs / (float)encoded_bits;
        enc_cfg.n_RNTI = cfg.rnti;
        enc_cfg.n_ID = cfg.n_id;
        enc_cfg.q = 0;
        enc_cfg.enable_scrambling = true;
        status = tb_encoder_configure(encoder, &enc_cfg);
        if (status != NR_LDPC_SUCCESS) { printf("ERROR: configure encoder\n"); test_passed = false; goto cleanup; }
    }

    // Configure TB decoder — scrambling disabled (E2E kernel handles descrambling)
    {
        tb_decoder_config_t dec_cfg = {};
        dec_cfg.tb_size_bits = tbs;
        dec_cfg.num_layers = 1;
        dec_cfg.modulation_order = mod_order;
        dec_cfg.num_received_bits = encoded_bits;
        dec_cfg.redundancy_version = 0;
        dec_cfg.code_rate = (float)tbs / (float)encoded_bits;
        dec_cfg.max_iterations = 25;
        dec_cfg.llr_clamp = 32.0f;
        dec_cfg.n_RNTI = cfg.rnti;
        dec_cfg.n_ID = cfg.n_id;
        dec_cfg.q = 0;
        dec_cfg.enable_scrambling = false;  // E2E kernel already descrambled
        status = tb_decoder_configure(decoder, &dec_cfg);
        if (status != NR_LDPC_SUCCESS) { printf("ERROR: configure decoder\n"); test_passed = false; goto cleanup; }
    }

    // Configure PUSCH E2E — matches gnb pusch_demodulator_gpu_impl.cpp
    {
        pusch_e2e_config_t e2e_cfg = {};
        e2e_cfg.nof_prb = cfg.nof_prb;
        e2e_cfg.nof_symbols = cfg.nof_symbols;
        e2e_cfg.nof_rx_ports = 1;
        e2e_cfg.nof_tx_layers = 1;
        e2e_cfg.grid_nof_subcarriers = CARRIER_SC;
        e2e_cfg.grid_nof_symbols = cfg.nof_symbols;
        e2e_cfg.dmrs_type = DMRS_TYPE_1;
        e2e_cfg.dmrs_symbol_mask = cfg.dmrs_symbol_mask;
        e2e_cfg.nof_cdm_groups_without_data = 2;
        e2e_cfg.scrambling_id = cfg.scrambling_id;
        e2e_cfg.n_scid = cfg.n_scid;
        e2e_cfg.slot_idx = cfg.slot_idx;
        e2e_cfg.dmrs_scaling = 1.0f;  // Identity: no DMRS power boost
        e2e_cfg.mod_order = mod_order;
        e2e_cfg.rnti = cfg.rnti;
        e2e_cfg.n_id = cfg.n_id;
        e2e_cfg.start_prb = cfg.start_prb;
        e2e_cfg.start_symbol = 0;
        e2e_cfg.tx_scaling = 1.0f;
        e2e_cfg.equalizer_algorithm = EQUALIZER_MMSE;
        e2e_cfg.use_low_papr_dmrs = 0;
        e2e_cfg.n_rs_id = 0;

        status = pusch_e2e_configure(e2e, &e2e_cfg);
        if (status != NR_LDPC_SUCCESS) { printf("ERROR: configure E2E\n"); test_passed = false; goto cleanup; }
    }

    // Update slot config
    pusch_e2e_update_slot_config(e2e, cfg.nof_prb, cfg.start_prb, cfg.slot_idx, cfg.dmrs_symbol_mask);

    // Run test iterations
    {
        const int num_iterations = (cfg.snr_db <= 0.0f) ? 3 : 10;
        int total_bit_errors = 0;
        int total_block_errors = 0;

        for (int iter = 0; iter < num_iterations; iter++) {
            // Generate random TB data
            srand((unsigned)(time(NULL) ^ (iter * 31) ^ cfg.rnti));
            generate_random_data(h_tx_tb.data(), tb_bytes);
            CHECK_CUDA(cudaMemcpy(d_tx_tb, h_tx_tb.data(), tb_bytes, cudaMemcpyHostToDevice));

            // ========== TX: Production fused path ==========
            status = tb_encoder_encode_to_symbols_int8(encoder, d_tx_tb, d_symbols_int8, 0);
            if (status != NR_LDPC_SUCCESS) {
                printf("ERROR: production TX failed\n");
                test_passed = false;
                goto cleanup;
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            // Download INT8 symbols to CPU for grid building
            std::vector<int8_t> h_int8(num_symbols_tx * 2);
            CHECK_CUDA(cudaMemcpy(h_int8.data(), d_symbols_int8, num_symbols_tx * 2, cudaMemcpyDeviceToHost));

            // ========== Build Resource Grid (CPU) ==========
            std::vector<uint32_t> h_grid;
            std::vector<int> h_re_indices;
            int actual_data_re = 0;

            float noise_std = 0.0f;
            if (cfg.snr_db > 0.0f) {
                noise_std = snr_to_noise_std(cfg.snr_db, (float)tbs / (float)encoded_bits, mod_order);
            }

            build_resource_grid(h_grid, h_int8.data(),
                                cfg.nof_prb, cfg.start_prb, cfg.nof_symbols,
                                cfg.dmrs_symbol_mask, cfg.scrambling_id, cfg.n_scid, cfg.slot_idx,
                                mod_order, noise_std,
                                h_re_indices, actual_data_re);

            if (actual_data_re != num_data_re) {
                printf("ERROR: data RE mismatch: expected %d, got %d\n", num_data_re, actual_data_re);
                test_passed = false;
                goto cleanup;
            }

            // Upload grid and RE indices to GPU
            CHECK_CUDA(cudaMemcpy(d_grid_cbf16, h_grid.data(), grid_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_re_indices, h_re_indices.data(), num_data_re * sizeof(int), cudaMemcpyHostToDevice));

            // ========== RX: Production E2E path ==========
            status = pusch_e2e_process_full_gpu_optimized(
                e2e, d_grid_cbf16, d_llrs_half, d_re_indices, num_data_re, 0);
            if (status != NR_LDPC_SUCCESS) {
                printf("ERROR: E2E processing failed\n");
                test_passed = false;
                goto cleanup;
            }

            // ========== Decode: FP16 LLR path ==========
            tb_decode_result_t result;
            status = tb_decoder_decode_half(decoder, d_llrs_half, d_rx_tb, &result, 0);
            if (status != NR_LDPC_SUCCESS) {
                printf("ERROR: decode failed\n");
                test_passed = false;
                goto cleanup;
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            // Compare
            CHECK_CUDA(cudaMemcpy(h_rx_tb.data(), d_rx_tb, tb_bytes, cudaMemcpyDeviceToHost));
            int bit_errors = count_bit_errors(h_tx_tb.data(), h_rx_tb.data(), tbs);
            total_bit_errors += bit_errors;
            if (bit_errors > 0 || !result.crc_pass) {
                total_block_errors++;
            }

            if (verbose && iter < 3) {
                printf("    Iter %d: %s (CRC=%s, errors=%d, iters=%.1f)\n",
                       iter, bit_errors == 0 ? "PASS" : "FAIL",
                       result.crc_pass ? "OK" : "FAIL",
                       bit_errors, result.avg_iterations);
            }
        }

        float bler = (float)total_block_errors / num_iterations;
        float ber = (float)total_bit_errors / (num_iterations * tbs);

        if (cfg.snr_db <= 0.0f) {
            // Identity channel: must be 0% BLER
            if (total_block_errors > 0) {
                printf("  FAILED: %d block errors at identity channel (BLER=%.1f%%, BER=%.2e)\n",
                       total_block_errors, bler * 100, ber);
                test_passed = false;
            } else {
                if (verbose) printf("  PASSED (identity channel, 0%% BLER)\n");
            }
        } else {
            // Noisy channel: <10% BLER
            if (bler > 0.1f) {
                printf("  FAILED: BLER=%.1f%% at SNR=%.1f dB\n", bler * 100, cfg.snr_db);
                test_passed = false;
            } else {
                if (verbose) printf("  PASSED (SNR=%.1f dB, BLER=%.1f%%)\n", cfg.snr_db, bler * 100);
            }
        }
    }

cleanup:
    if (encoder) tb_encoder_destroy(encoder);
    if (decoder) tb_decoder_destroy(decoder);
    if (e2e) pusch_e2e_destroy(e2e);
    if (d_tx_tb) cudaFree(d_tx_tb);
    if (d_symbols_int8) cudaFree(d_symbols_int8);
    if (d_grid_cbf16) cudaFree(d_grid_cbf16);
    if (d_llrs_half) cudaFree(d_llrs_half);
    if (d_re_indices) cudaFree(d_re_indices);
    if (d_rx_tb) cudaFree(d_rx_tb);

    return test_passed;
}

// ============================================================================
// Predefined Test Matrix
// ============================================================================

static bool run_identity_channel_tests(bool verbose) {
    printf("\n=======================================================\n");
    printf("Identity Channel Tests (no noise, must be 0%% BLER)\n");
    printf("=======================================================\n");

    bool all_passed = true;

    // Test matrix: various PRB counts, modulations, DMRS configs
    static const PUSCHTestConfig tests[] = {
        // Small allocations — QPSK
        {3,   0, 14, 4,  0x0004, 0x1234, 100, 100, 0, 0, 0.0f, "3PRB QPSK single-DMRS"},
        {5,   0, 14, 4,  0x0004, 0x2345, 200, 200, 0, 1, 0.0f, "5PRB QPSK single-DMRS"},
        {10,  0, 14, 4,  0x0204, 0x3456, 300, 300, 0, 2, 0.0f, "10PRB QPSK double-DMRS"},

        // Medium allocations — QPSK, 16QAM
        {25,  0, 14, 7,  0x0004, 0x4567, 400, 400, 0, 3, 0.0f, "25PRB QPSK R=526/1024"},
        {25,  0, 14, 10, 0x0004, 0x5678, 500, 500, 0, 4, 0.0f, "25PRB 16QAM R=340/1024"},
        {52,  0, 14, 12, 0x0204, 0x6789, 600, 600, 0, 5, 0.0f, "52PRB 16QAM double-DMRS"},

        // Large allocations — 16QAM, 64QAM
        {106, 0, 14, 14, 0x0004, 0x789A, 700, 700, 0, 6, 0.0f, "106PRB 16QAM R=553/1024"},
        {106, 0, 14, 19, 0x0004, 0x89AB, 800, 800, 0, 7, 0.0f, "106PRB 64QAM R=517/1024"},
        {106, 0, 14, 19, 0x0204, 0x9ABC, 900, 900, 0, 8, 0.0f, "106PRB 64QAM double-DMRS"},

        // Full bandwidth — 273 PRBs
        {273, 0, 14, 12, 0x0004, 0xABCD, 1, 1, 0, 9, 0.0f, "273PRB 16QAM full-BW"},
        {273, 0, 14, 19, 0x0204, 0xBCDE, 2, 2, 0, 10, 0.0f, "273PRB 64QAM full-BW double-DMRS"},

        // Non-zero start_prb (offset allocations)
        {24,  8, 14, 4,  0x0204, 0xCDEF, 3, 3, 0, 11, 0.0f, "24PRB@8 QPSK offset"},
        {50, 50, 14, 12, 0x0004, 0xDEF0, 4, 4, 0, 12, 0.0f, "50PRB@50 16QAM offset"},
    };
    int num_tests = sizeof(tests) / sizeof(tests[0]);

    for (int t = 0; t < num_tests; t++) {
        printf("\nTest %d/%d: %s\n", t + 1, num_tests, tests[t].name);
        bool passed = run_pusch_e2e_test(tests[t], verbose);
        if (!passed) all_passed = false;
    }

    return all_passed;
}

static bool run_noisy_channel_tests(bool verbose) {
    printf("\n=======================================================\n");
    printf("Noisy Channel Tests (AWGN, target <10%% BLER)\n");
    printf("=======================================================\n");

    bool all_passed = true;

    // Representative configs with per-MCS SNR targets (generous margins)
    static const PUSCHTestConfig tests[] = {
        // QPSK with plenty of margin
        {25,  0, 14, 4,  0x0004, 0x1111, 100, 100, 0, 0, 18.0f, "25PRB QPSK SNR=18dB"},
        {52,  0, 14, 7,  0x0204, 0x2222, 200, 200, 0, 1, 18.0f, "52PRB QPSK SNR=18dB"},

        // 16QAM
        {52,  0, 14, 12, 0x0004, 0x3333, 300, 300, 0, 2, 24.0f, "52PRB 16QAM SNR=24dB"},
        {106, 0, 14, 14, 0x0204, 0x4444, 400, 400, 0, 3, 26.0f, "106PRB 16QAM SNR=26dB"},

        // 64QAM
        {106, 0, 14, 19, 0x0004, 0x5555, 500, 500, 0, 4, 28.0f, "106PRB 64QAM SNR=28dB"},
        {273, 0, 14, 19, 0x0204, 0x6666, 600, 600, 0, 5, 28.0f, "273PRB 64QAM SNR=28dB"},
    };
    int num_tests = sizeof(tests) / sizeof(tests[0]);

    for (int t = 0; t < num_tests; t++) {
        printf("\nTest %d/%d: %s\n", t + 1, num_tests, tests[t].name);
        bool passed = run_pusch_e2e_test(tests[t], verbose);
        if (!passed) all_passed = false;
    }

    return all_passed;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    bool verbose = false;
    bool run_noisy = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) verbose = true;
        else if (strcmp(argv[i], "--identity-only") == 0) run_noisy = false;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  -v, --verbose       Verbose output\n");
            printf("  --identity-only     Skip noisy channel tests\n");
            return 0;
        }
    }

    printf("================================================\n");
    printf("PUSCH E2E Loopback Test (Production Kernel Paths)\n");
    printf("================================================\n");
    printf("TX: tb_encoder_encode_to_symbols_int8()\n");
    printf("RX: pusch_e2e_process_full_gpu_optimized() + tb_decoder_decode_half()\n");
    printf("================================================\n");

    nr_ldpc_status_t status = ocudu_phy_cuda_init();
    if (status != NR_LDPC_SUCCESS) {
        printf("ERROR: Failed to initialize OCUDU PHY CUDA: %s\n", nr_ldpc_get_error_string(status));
        return 1;
    }
    ocudu_phy_cuda_print_info();

    bool all_passed = true;

    // Identity channel tests (must be 0% BLER)
    all_passed &= run_identity_channel_tests(verbose);

    // Noisy channel tests (<10% BLER at target SNR)
    if (run_noisy) {
        all_passed &= run_noisy_channel_tests(verbose);
    }

    printf("\n================================================\n");
    printf("Result: %s\n", all_passed ? "ALL PASSED" : "SOME FAILED");
    printf("================================================\n");

    ocudu_phy_cuda_cleanup();
    return all_passed ? 0 : 1;
}
