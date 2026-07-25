/**
 * @file nr_constants.cu
 * @brief 5G NR LDPC Constants and Utility Functions
 */

#include "nr_ldpc_defs.h"
#include "ocudu_phy_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

// ============================================================================
// Base Graph Shift Matrices (stored in device constant memory)
// ============================================================================

// Base Graph 1 CSR format row pointers
__device__ __constant__ int16_t BG1_CSR_ROW_PTR[NR_LDPC_BG1_ROWS + 1] = {
    0, 19, 38, 57, 76, 79, 87, 96, 103, 113, 122, 129, 137, 144, 150, 157, 164,
    170, 176, 182, 188, 194, 200, 205, 210, 216, 221, 226, 230, 235, 240, 245,
    250, 255, 260, 265, 270, 275, 279, 284, 289, 293, 298, 302, 307, 312, 316
};

// Base Graph 1 column indices
__device__ __constant__ int8_t BG1_CSR_COL[NR_LDPC_BG1_NNZ] = {
    0, 1, 2, 3, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23,
    0, 2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 21, 22, 23, 24,
    0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18, 19, 20, 24, 25,
    0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 25,
    0, 1, 26,
    0, 1, 3, 12, 16, 21, 22, 27,
    0, 6, 10, 11, 13, 17, 18, 20, 28,
    0, 1, 4, 7, 8, 14, 29,
    0, 1, 3, 12, 16, 19, 21, 22, 24, 30,
    0, 1, 10, 11, 13, 17, 18, 20, 31,
    1, 2, 4, 7, 8, 14, 32,
    0, 1, 12, 16, 21, 22, 23, 33,
    0, 1, 10, 11, 13, 18, 34,
    0, 3, 7, 20, 23, 35,
    0, 12, 15, 16, 17, 21, 36,
    0, 1, 10, 13, 18, 25, 37,
    1, 3, 11, 20, 22, 38,
    0, 14, 16, 17, 21, 39,
    1, 12, 13, 18, 19, 40,
    0, 1, 7, 8, 10, 41,
    0, 3, 9, 11, 22, 42,
    1, 5, 16, 20, 21, 43,
    0, 12, 13, 17, 44,
    1, 2, 10, 18, 45,
    0, 3, 4, 11, 22, 46,
    1, 6, 7, 14, 47,
    0, 2, 4, 15, 48,
    1, 6, 8, 49,
    0, 4, 19, 21, 50,
    1, 14, 18, 25, 51,
    0, 10, 13, 24, 52,
    1, 7, 22, 25, 53,
    0, 12, 14, 24, 54,
    1, 2, 11, 21, 55,
    0, 7, 15, 17, 56,
    1, 6, 12, 22, 57,
    0, 14, 15, 18, 58,
    1, 13, 23, 59,
    0, 9, 10, 12, 60,
    1, 3, 7, 19, 61,
    0, 8, 17, 62,
    1, 3, 9, 18, 63,
    0, 4, 24, 64,
    1, 16, 18, 25, 65,
    0, 7, 9, 22, 66,
    1, 6, 10, 67
};

// Base Graph 2 CSR format row pointers - 3GPP TS 38.212 compliant (42 rows)
__device__ __constant__ int16_t BG2_CSR_ROW_PTR[NR_LDPC_BG2_ROWS + 1] = {
    0, 8, 18, 26, 36, 40, 46, 52, 58, 62, 67, 72,
    77, 81, 86, 91, 95, 100, 105, 109, 113, 117, 121, 124,
    128, 132, 135, 140, 143, 147, 150, 155, 158, 162, 166, 170,
    174, 178, 181, 185, 189, 193, 197
};

// Base Graph 2 column indices - 3GPP TS 38.212 compliant
__device__ __constant__ int8_t BG2_CSR_COL[NR_LDPC_BG2_NNZ] = {
     0,  1,  2,  3,  6,  9, 10, 11,  // row 0
     0,  3,  4,  5,  6,  7,  8,  9, 11, 12,  // row 1
     0,  1,  3,  4,  8, 10, 12, 13,  // row 2
     1,  2,  4,  5,  6,  7,  8,  9, 10, 13,  // row 3
     0,  1, 11, 14,  // row 4
     0,  1,  5,  7, 11, 15,  // row 5
     0,  5,  7,  9, 11, 16,  // row 6
     1,  5,  7, 11, 13, 17,  // row 7
     0,  1, 12, 18,  // row 8
     1,  8, 10, 11, 19,  // row 9
     0,  1,  6,  7, 20,  // row 10
     0,  7,  9, 13, 21,  // row 11
     1,  3, 11, 22,  // row 12
     0,  1,  8, 13, 23,  // row 13
     1,  6, 11, 13, 24,  // row 14
     0, 10, 11, 25,  // row 15
     1,  9, 11, 12, 26,  // row 16
     1,  5, 11, 12, 27,  // row 17
     0,  6,  7, 28,  // row 18
     0,  1, 10, 29,  // row 19
     1,  4, 11, 30,  // row 20
     0,  8, 13, 31,  // row 21
     1,  2, 32,  // row 22
     0,  3,  5, 33,  // row 23
     1,  2,  9, 34,  // row 24
     0,  5, 35,  // row 25
     2,  7, 12, 13, 36,  // row 26
     0,  6, 37,  // row 27
     1,  2,  5, 38,  // row 28
     0,  4, 39,  // row 29
     2,  5,  7,  9, 40,  // row 30
     1, 13, 41,  // row 31
     0,  5, 12, 42,  // row 32
     2,  7, 10, 43,  // row 33
     0, 12, 13, 44,  // row 34
     1,  5, 11, 45,  // row 35
     0,  2,  7, 46,  // row 36
    10, 13, 47,  // row 37
     1,  5, 11, 48,  // row 38
     0,  7, 12, 49,  // row 39
     2, 10, 13, 50,  // row 40
     1,  5, 11, 51   // row 41
};

// ============================================================================
// Lifting Size Tables
// ============================================================================

// Mapping from Z to set index (iLS)
__host__ __device__ int get_lifting_set_index(int Z) {
    // Set 0: Z = 2, 4, 8, 16, 32, 64, 128, 256
    // Set 1: Z = 3, 6, 12, 24, 48, 96, 192, 384
    // Set 2: Z = 5, 10, 20, 40, 80, 160, 320
    // Set 3: Z = 7, 14, 28, 56, 112, 224
    // Set 4: Z = 9, 18, 36, 72, 144, 288
    // Set 5: Z = 11, 22, 44, 88, 176, 352
    // Set 6: Z = 13, 26, 52, 104, 208
    // Set 7: Z = 15, 30, 60, 120, 240

    static const int set_bases[8] = {2, 3, 5, 7, 9, 11, 13, 15};

    for (int i = 0; i < 8; i++) {
        int base = set_bases[i];
        if (Z % base == 0) {
            int ratio = Z / base;
            // Check if ratio is power of 2
            if (ratio > 0 && (ratio & (ratio - 1)) == 0) {
                return i;
            }
        }
    }
    return -1; // Invalid Z
}

// ============================================================================
// Library Utility Functions
// ============================================================================

extern "C" {

const char* ocudu_phy_cuda_version(void) {
    static char version[32];
    snprintf(version, sizeof(version), "%d.%d.%d",
             OCUDU_PHY_CUDA_VERSION_MAJOR,
             OCUDU_PHY_CUDA_VERSION_MINOR,
             OCUDU_PHY_CUDA_VERSION_PATCH);
    return version;
}

nr_ldpc_status_t ocudu_phy_cuda_init(void) {
    // Initialize CUDA
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }
    return NR_LDPC_SUCCESS;
}

void ocudu_phy_cuda_cleanup(void) {
    cudaDeviceReset();
}

const char* nr_ldpc_get_error_string(nr_ldpc_status_t status) {
    switch (status) {
        case NR_LDPC_SUCCESS: return "Success";
        case NR_LDPC_ERROR_INVALID_BG: return "Invalid base graph";
        case NR_LDPC_ERROR_INVALID_Z: return "Invalid lifting size";
        case NR_LDPC_ERROR_INVALID_CONFIG: return "Invalid configuration";
        case NR_LDPC_ERROR_CUDA_FAILED: return "CUDA operation failed";
        case NR_LDPC_ERROR_ALLOC_FAILED: return "Memory allocation failed";
        case NR_LDPC_ERROR_SIZE_MISMATCH: return "Size mismatch";
        default: return "Unknown error";
    }
}

void ocudu_phy_cuda_print_info(void) {
    printf("=== OCUDU PHY CUDA - CUDA acceleration for OCUDU ===\n");
    printf("Version: %s\n", ocudu_phy_cuda_version());
    printf("Max lifting size: %d\n", NR_LDPC_MAX_LIFTING_SIZE);
    printf("BG1: %d info nodes, %d parity nodes\n", NR_LDPC_BG1_INFO_NODES, NR_LDPC_BG1_PARITY_NODES);
    printf("BG2: %d info nodes, %d parity nodes\n", NR_LDPC_BG2_INFO_NODES, NR_LDPC_BG2_PARITY_NODES);

    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("CUDA Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("=====================================\n");
}

int nr_ldpc_get_lifting_set_index(int Z) {
    return get_lifting_set_index(Z);
}

int nr_ldpc_get_Kb(int tbs, int bg) {
    if (bg == 1) {
        return NR_LDPC_BG1_INFO_NODES; // 22
    } else {
        // BG2: Per 3GPP TS 38.212, K_b is always 10 for BG2
        // The old implementation had incorrect TBS-dependent logic that caused
        // wrong lifting sizes for TBS values like 280 (would select Z=40 instead of Z=30)
        return NR_LDPC_BG2_INFO_NODES; // 10
    }
}

int nr_ldpc_select_base_graph(int tbs, float rate) {
    // Per TS 38.212 Section 7.2.2
    // BG2 is selected if any of these conditions are met:
    //   - A ≤ 292
    //   - A ≤ 3824 AND R ≤ 0.67
    //   - R ≤ 0.25
    // Note: Large TBS can still use BG2 with multiple code blocks!
    // Note: Use small epsilon for rate comparison to handle float precision
    //       (TBS is derived from target rate, so TBS/G may slightly exceed target)
    const float rate_epsilon = 0.001f;
    bool cond1 = (tbs <= 292);
    bool cond2 = (tbs <= 3824 && rate <= (0.67f + rate_epsilon));
    bool cond3 = (rate <= (0.25f + rate_epsilon));
    int bg = (cond1 || cond2 || cond3) ? 2 : 1;
    return bg;
}

int nr_ldpc_compute_lifting_size(int K, int Kb) {
    // Find minimum Z such that K <= Kb * Z
    // Z must be from the valid lifting size set
    int min_Z = (K + Kb - 1) / Kb;

    for (int i = 0; i < NR_LDPC_NUM_LIFTING_SIZES; i++) {
        if (NR_LDPC_LIFTING_SIZES[i] >= min_Z) {
            return NR_LDPC_LIFTING_SIZES[i];
        }
    }
    return NR_LDPC_MAX_LIFTING_SIZE;
}

nr_ldpc_status_t nr_ldpc_init_config(nr_ldpc_config_t* cfg, int cb_info_bits, float rate) {
    if (!cfg) return NR_LDPC_ERROR_INVALID_CONFIG;

    // cb_info_bits is the code block size including any CRC already attached
    // (TB CRC for single CB, or TB portion + CB CRC for multi-CB)
    // We do NOT add additional CRC here - that's handled at the TB layer

    // Select base graph based on the CB size
    cfg->base_graph = nr_ldpc_select_base_graph(cb_info_bits, rate);

    // Get Kb for lifting size calculation (variable: 6, 8, 9, or 10 for BG2)
    int Kb_for_Z = nr_ldpc_get_Kb(cb_info_bits, cfg->base_graph);

    // K is the code block size (already includes CRC from TB layer)
    int K = cb_info_bits;

    // Compute lifting size using the variable Kb
    cfg->lifting_size = nr_ldpc_compute_lifting_size(K, Kb_for_Z);
    cfg->lifting_set_index = get_lifting_set_index(cfg->lifting_size);

    if (cfg->lifting_set_index < 0) {
        return NR_LDPC_ERROR_INVALID_Z;
    }

    // Per 3GPP TS 38.212, the LDPC encoder ALWAYS uses the full Kb columns:
    // - BG1: Kb = 22 (K_b = 22 systematic columns)
    // - BG2: Kb = 10 (K_b = 10 systematic columns)
    // The filler bits fill the gap between actual info bits and Kb * Z.
    int Kb_full = (cfg->base_graph == 1) ? NR_LDPC_BG1_INFO_NODES : NR_LDPC_BG2_INFO_NODES;

    // Information bits K' = Kb_full * Z (this is the message length for LDPC encoding)
    int K_prime = Kb_full * cfg->lifting_size;

    // Filler bits = K' - actual info bits
    cfg->num_filler_bits = K_prime - K;
    cfg->num_info_bits = K;

    // Number of parity bits (N - K)
    int N = (cfg->base_graph == 1) ?
            NR_LDPC_BG1_UNPUNCTURED_VARS * cfg->lifting_size :
            NR_LDPC_BG2_UNPUNCTURED_VARS * cfg->lifting_size;

    cfg->num_codeword_bits = N;
    cfg->num_parity_bits = N - K_prime;

    // Default settings
    cfg->puncture = true;  // Always puncture first 2*Z bits
    cfg->max_parity_nodes = (cfg->base_graph == 1) ?
                            NR_LDPC_BG1_PARITY_NODES : NR_LDPC_BG2_PARITY_NODES;
    cfg->redundancy_version = 0;

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t nr_tb_init_config(nr_tb_config_t* tb_cfg, int tbs, float rate) {
    if (!tb_cfg) return NR_LDPC_ERROR_INVALID_CONFIG;

    tb_cfg->tb_size_bits = tbs;

    // TB CRC size (16 or 24 bits)
    tb_cfg->tb_crc_bits = (tbs > 3824) ? NR_TB_CRC24A_BITS : NR_TB_CRC16_BITS;

    // B = A + L (TB + CRC)
    int B = tbs + tb_cfg->tb_crc_bits;

    // Maximum code block size depends on base graph
    int bg = nr_ldpc_select_base_graph(tbs, rate);
    int K_cb_max = (bg == 1) ? NR_LDPC_MAX_CB_INFO_BITS : NR_LDPC_MAX_CB_INFO_BITS_BG2;

    // Code block segmentation
    if (B <= K_cb_max) {
        // Single code block
        tb_cfg->num_code_blocks = 1;
        tb_cfg->cb_crc_bits = 0;  // No CB CRC for single block
        tb_cfg->cb_size_bits = B;
    } else {
        // Multiple code blocks with CB CRC
        tb_cfg->cb_crc_bits = NR_CB_CRC24B_BITS;
        int B_prime = B + 24;  // Add segment-level CRC

        // Number of code blocks
        tb_cfg->num_code_blocks = (B_prime + K_cb_max - 1 - 24) / (K_cb_max - 24);

        // Per-CB size
        int total_cb_bits = B_prime + (tb_cfg->num_code_blocks - 1) * 24;
        tb_cfg->cb_size_bits = (total_cb_bits + tb_cfg->num_code_blocks - 1) / tb_cfg->num_code_blocks;
    }

    // Initialize LDPC config for code blocks
    // NOTE: LDPC encoder receives the full CB including CB CRC (per 3GPP TS 38.212).
    // The CB CRC is attached BEFORE LDPC encoding, so K = cb_size_bits (NOT cb_size_bits - cb_crc_bits).
    return nr_ldpc_init_config(&tb_cfg->ldpc_cfg, tb_cfg->cb_size_bits, rate);
}

} // extern "C"
