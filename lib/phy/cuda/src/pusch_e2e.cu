/*
 * OCUDU PHY CUDA - CUDA-accelerated 5G NR PHY processing
 *
 * Fused PUSCH End-to-End Processing with Integrated Channel Estimation
 */

#include "pusch_e2e.h"
#include "scrambling.h"
#include "ocudu_phy_cuda_mimo_math.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <new>
#include <mutex>
#include <cstring>
#include <cstdlib>

#ifdef OCUDU_PHY_CUDA_ENABLE_VKFFT
#include "vkFFT.h"
#endif

/* ============================================================================
 * LFSR Jump Tables for on-the-fly Gold sequence generation
 * NOTE: These are defined here (not extern) because CUDA __constant__
 * memory is module-local and cannot be linked across compilation units.
 * ============================================================================ */
#define NC_SKIP 1600
#define LFSR_BITS 31
#define MAX_JUMP_POWER 24
#define PUSCH_DEPRECODE_MAX_DFT_SIZE 3240
#define PUSCH_DEPRECODE_MAX_SYMBOLS 14

static uint64_t pack_deprecoding_fft_factors(int n, int* nof_factors);

static nr_ldpc_status_t sync_pusch_precompute_stream(cudaStream_t stream, const char* phase)
{
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "[OCUDU PHY CUDA] PUSCH E2E configure %s failed: %s\n",
                     phase ? phase : "precompute",
                     cudaGetErrorString(err));
    }
    return (err == cudaSuccess) ? NR_LDPC_SUCCESS : NR_LDPC_ERROR_CUDA_FAILED;
}

static bool pusch_e2e_env_flag_disabled(const char* name)
{
    const char* value = std::getenv(name);
    return value && (std::strcmp(value, "0") == 0 ||
                     std::strcmp(value, "false") == 0 ||
                     std::strcmp(value, "off") == 0 ||
                     std::strcmp(value, "no") == 0);
}

static unsigned pusch_e2e_env_unsigned(const char* name, unsigned default_value)
{
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value) {
        return default_value;
    }
    return static_cast<unsigned>(parsed);
}

enum pusch_transform_deprecoder_backend {
    PUSCH_DEPRECODER_CUSTOM = 0,
    PUSCH_DEPRECODER_VKFFT = 1,
    PUSCH_DEPRECODER_AUTO = 2
};

#ifdef OCUDU_PHY_CUDA_ENABLE_VKFFT
struct pusch_vkfft_deprecoder_plan {
    VkFFTApplication app;
    pfUINT buffer_size;
    int dft_size;
    bool initialized;
};
#endif

/* Define the jump tables in this module (not extern!) */
__constant__ uint32_t d_pusch_e2e_x1_jump[MAX_JUMP_POWER + 1][LFSR_BITS];
__constant__ uint32_t d_pusch_e2e_x2_jump[MAX_JUMP_POWER + 1][LFSR_BITS];

/* CRITICAL FIX: Thread-safe initialization for jump tables
 * Multiple PUSCH processor instances created in parallel were racing to initialize
 * the same constant memory symbols, causing "invalid resource handle" errors that
 * contaminated all subsequent CUDA operations.
 */
static std::once_flag g_pusch_e2e_jump_tables_init_flag;

/* ============================================================================
 * Raised Cosine FIR Filter for Frequency-Domain Smoothing
 *
 * Matches CPU port_channel_estimator_helpers.cpp filter_type constructor:
 *   RC_FILTER with roll-off=0.2, span=3 symbols, resampled with stride=2
 *   (DMRS Type 1: pilots on every other subcarrier), nof_rbs=min(nof_prb, 3).
 *
 * For nof_rbs >= 3: 15 taps operating on pilot positions (6 per PRB).
 * Filter center is at index 7, so it looks 7 pilot positions each side.
 *
 * Tail correction factors compensate for truncation at band edges.
 * Applied symmetrically: left edge uses reversed tail_correction,
 * right edge uses tail_correction directly.
 * ============================================================================ */
#define FD_SMOOTH_FILTER_LEN 15

__constant__ float d_rc_filter[FD_SMOOTH_FILTER_LEN] = {
    -0.0391585909f, -0.0287990728f,  0.0000000000f,  0.0445245383f,
     0.0970724536f,  0.1467045930f,  0.1821531663f,  0.1950058248f,
     0.1821531663f,  0.1467045930f,  0.0970724536f,  0.0445245383f,
     0.0000000000f, -0.0287990728f, -0.0391585909f
};

/* ============================================================================
 * 64QAM Piecewise-Linear Demodulation Lookup Tables
 * Per TS 38.211: Optimal LLR approximation using interval functions
 * M = 1/sqrt(42) = 0.154303349962092
 * Pre-computed slope and intercept values for constant memory
 * ============================================================================ */

/* Bits 0,1: 8 intervals, width = 2M */
__constant__ float SLOPE_01[8] = {
    2.46885359939347f,  1.85164019954510f,  1.23442679969673f,  0.61721339984837f,
    0.61721339984837f,  1.23442679969673f,  1.85164019954510f,  2.46885359939347f
};
__constant__ float INTERCEPT_01[8] = {
    1.14285714285714f,  0.57142857142857f,  0.19047619047619f,  0.0f,
    0.0f,              -0.19047619047619f, -0.57142857142857f, -1.14285714285714f
};

/* Bits 2,3: 8 intervals, width = 2M */
__constant__ float SLOPE_23[8] = {
    1.23442679969673f,   0.61721339984837f,  0.61721339984837f,  1.23442679969673f,
   -1.23442679969673f,  -0.61721339984837f, -0.61721339984837f, -1.23442679969673f
};
__constant__ float INTERCEPT_23[8] = {
    0.95238095238095f,  0.38095238095238f,  0.38095238095238f,  0.57142857142857f,
    0.57142857142857f,  0.38095238095238f,  0.38095238095238f,  0.95238095238095f
};

/* Bits 4,5: 4 intervals, width = 4M */
__constant__ float SLOPE_45[4] = {
    0.61721339984837f, -0.61721339984837f,
    0.61721339984837f, -0.61721339984837f
};
__constant__ float INTERCEPT_45[4] = {
    0.57142857142857f, -0.19047619047619f,
   -0.19047619047619f,  0.57142857142857f
};

/* ============================================================================
 * 256QAM Piecewise-Linear Demodulation Lookup Tables
 * Keep these in constant memory. Declaring them inside soft_demod_256qam_piecewise
 * creates a 448-byte per-thread stack frame in every inlined 256QAM MIMO kernel.
 * ============================================================================ */

__constant__ float QAM256_SLOPE_01[16] = {
    32.0f * 0.07669649888473704f, 28.0f * 0.07669649888473704f,
    24.0f * 0.07669649888473704f, 20.0f * 0.07669649888473704f,
    16.0f * 0.07669649888473704f, 12.0f * 0.07669649888473704f,
     8.0f * 0.07669649888473704f,  4.0f * 0.07669649888473704f,
     4.0f * 0.07669649888473704f,  8.0f * 0.07669649888473704f,
    12.0f * 0.07669649888473704f, 16.0f * 0.07669649888473704f,
    20.0f * 0.07669649888473704f, 24.0f * 0.07669649888473704f,
    28.0f * 0.07669649888473704f, 32.0f * 0.07669649888473704f
};
__constant__ float QAM256_INTERCEPT_01[16] = {
    112.0f/85, 84.0f/85, 60.0f/85, 40.0f/85, 24.0f/85, 12.0f/85, 4.0f/85, 0.0f,
      0.0f,   -4.0f/85, -12.0f/85, -24.0f/85, -40.0f/85, -60.0f/85, -84.0f/85, -112.0f/85
};

__constant__ float QAM256_SLOPE_23[16] = {
    16.0f * 0.07669649888473704f,  12.0f * 0.07669649888473704f,
     8.0f * 0.07669649888473704f,   4.0f * 0.07669649888473704f,
     4.0f * 0.07669649888473704f,   8.0f * 0.07669649888473704f,
    12.0f * 0.07669649888473704f,  16.0f * 0.07669649888473704f,
   -16.0f * 0.07669649888473704f, -12.0f * 0.07669649888473704f,
    -8.0f * 0.07669649888473704f,  -4.0f * 0.07669649888473704f,
    -4.0f * 0.07669649888473704f,  -8.0f * 0.07669649888473704f,
   -12.0f * 0.07669649888473704f, -16.0f * 0.07669649888473704f
};
__constant__ float QAM256_INTERCEPT_23[16] = {
    88.0f/85, 60.0f/85, 36.0f/85, 16.0f/85, 16.0f/85, 28.0f/85, 36.0f/85, 40.0f/85,
    40.0f/85, 36.0f/85, 28.0f/85, 16.0f/85, 16.0f/85, 36.0f/85, 60.0f/85, 88.0f/85
};

__constant__ float QAM256_SLOPE_45[16] = {
     8.0f * 0.07669649888473704f,   4.0f * 0.07669649888473704f,
     4.0f * 0.07669649888473704f,   8.0f * 0.07669649888473704f,
    -8.0f * 0.07669649888473704f,  -4.0f * 0.07669649888473704f,
    -4.0f * 0.07669649888473704f,  -8.0f * 0.07669649888473704f,
     8.0f * 0.07669649888473704f,   4.0f * 0.07669649888473704f,
     4.0f * 0.07669649888473704f,   8.0f * 0.07669649888473704f,
    -8.0f * 0.07669649888473704f,  -4.0f * 0.07669649888473704f,
    -4.0f * 0.07669649888473704f,  -8.0f * 0.07669649888473704f
};
__constant__ float QAM256_INTERCEPT_45[16] = {
    52.0f/85, 24.0f/85, 24.0f/85, 44.0f/85, -20.0f/85, -8.0f/85, -8.0f/85, -12.0f/85,
   -12.0f/85, -8.0f/85, -8.0f/85, -20.0f/85, 44.0f/85, 24.0f/85, 24.0f/85, 52.0f/85
};

__constant__ float QAM256_SLOPE_67[8] = {
     4.0f * 0.07669649888473704f, -4.0f * 0.07669649888473704f,
     4.0f * 0.07669649888473704f, -4.0f * 0.07669649888473704f,
     4.0f * 0.07669649888473704f, -4.0f * 0.07669649888473704f,
     4.0f * 0.07669649888473704f, -4.0f * 0.07669649888473704f
};
__constant__ float QAM256_INTERCEPT_67[8] = {
    28.0f/85, -20.0f/85, 12.0f/85, -4.0f/85, -4.0f/85, 12.0f/85, -20.0f/85, 28.0f/85
};

/* Host-side matrix operations for computing jump tables */
static void matrix_multiply_gf2_local(const uint32_t A[LFSR_BITS],
                                       const uint32_t B[LFSR_BITS],
                                       uint32_t C[LFSR_BITS]) {
    uint32_t B_T[LFSR_BITS] = {0};
    for (int i = 0; i < LFSR_BITS; i++) {
        for (int j = 0; j < LFSR_BITS; j++) {
            if (B[j] & (1u << i)) {
                B_T[i] |= (1u << j);
            }
        }
    }
    for (int i = 0; i < LFSR_BITS; i++) {
        C[i] = 0;
        for (int j = 0; j < LFSR_BITS; j++) {
            uint32_t dot = A[i] & B_T[j];
            if (__builtin_popcount(dot) & 1) {
                C[i] |= (1u << j);
            }
        }
    }
}

static void init_x1_base_matrix_local(uint32_t M[LFSR_BITS]) {
    for (int i = 0; i < LFSR_BITS; i++) M[i] = 0;
    for (int i = 0; i < LFSR_BITS - 1; i++) M[i] = 1u << (i + 1);
    M[LFSR_BITS - 1] = (1u << 0) | (1u << 3);
}

static void init_x2_base_matrix_local(uint32_t M[LFSR_BITS]) {
    for (int i = 0; i < LFSR_BITS; i++) M[i] = 0;
    for (int i = 0; i < LFSR_BITS - 1; i++) M[i] = 1u << (i + 1);
    M[LFSR_BITS - 1] = (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3);
}

// Helper to do the actual initialization (called once via std::call_once)
static cudaError_t do_initialize_jump_tables() {
    uint32_t h_x1_jump[MAX_JUMP_POWER + 1][LFSR_BITS];
    uint32_t h_x2_jump[MAX_JUMP_POWER + 1][LFSR_BITS];

    init_x1_base_matrix_local(h_x1_jump[0]);
    init_x2_base_matrix_local(h_x2_jump[0]);

    for (int p = 1; p <= MAX_JUMP_POWER; p++) {
        matrix_multiply_gf2_local(h_x1_jump[p-1], h_x1_jump[p-1], h_x1_jump[p]);
        matrix_multiply_gf2_local(h_x2_jump[p-1], h_x2_jump[p-1], h_x2_jump[p]);
    }

    cudaError_t err;
    err = cudaMemcpyToSymbol(d_pusch_e2e_x1_jump, h_x1_jump, sizeof(h_x1_jump));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyToSymbol(d_pusch_e2e_x2_jump, h_x2_jump, sizeof(h_x2_jump));
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

static cudaError_t initialize_pusch_e2e_jump_tables() {
    // CRITICAL FIX: Use std::call_once to ensure thread-safe initialization
    // This prevents multiple threads from racing to write to the same constant memory
    static cudaError_t init_result = cudaSuccess;
    std::call_once(g_pusch_e2e_jump_tables_init_flag, [&]() {
        init_result = do_initialize_jump_tables();
    });
    return init_result;
}

/* Apply a pre-computed jump matrix to LFSR state */
__device__ __forceinline__ uint32_t apply_jump_matrix_local(uint32_t state,
                                                             const uint32_t jump[LFSR_BITS]) {
    uint32_t result = 0;
    #pragma unroll
    for (int i = 0; i < LFSR_BITS; i++) {
        uint32_t masked = state & jump[i];
        if (__popc(masked) & 1) {
            result |= (1u << i);
        }
    }
    return result;
}

/* Advance x1 LFSR by N steps using matrix exponentiation - O(log N) */
__device__ __forceinline__ uint32_t advance_x1_local(uint32_t x1, int n) {
    while (n) {
        int p = __ffs(n) - 1;
        x1 = apply_jump_matrix_local(x1, d_pusch_e2e_x1_jump[p]);
        n &= (n - 1);
    }
    return x1;
}

/* Advance x2 LFSR by N steps using matrix exponentiation - O(log N) */
__device__ __forceinline__ uint32_t advance_x2_local(uint32_t x2, int n) {
    while (n) {
        int p = __ffs(n) - 1;
        x2 = apply_jump_matrix_local(x2, d_pusch_e2e_x2_jump[p]);
        n &= (n - 1);
    }
    return x2;
}

/* Single-step LFSR advancement for sequential bit generation */
__device__ __forceinline__ uint32_t step_x1_local(uint32_t x1) {
    uint32_t new_bit = ((x1 >> 3) ^ x1) & 1;
    return (x1 >> 1) | (new_bit << 30);
}

__device__ __forceinline__ uint32_t step_x2_local(uint32_t x2) {
    uint32_t new_bit = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
    return (x2 >> 1) | (new_bit << 30);
}

/* Maximum supported configuration */
#define MAX_PRB 273
#define MAX_SYMBOLS 14
#define MAX_PORTS 8
#define MAX_DMRS_SYMBOLS 4
#define MAX_SLOTS_PER_FRAME 20  /* 30kHz SCS: 20 slots per 10ms frame */

#define MIMO_CFO_DATA_PHASOR_COUNT (MAX_SYMBOLS * MAX_DMRS_SYMBOLS)
#define MIMO_CFO_RESIDUAL_PHASOR_COUNT (MAX_DMRS_SYMBOLS * MAX_DMRS_SYMBOLS)
#define MIMO_CFO_PHASOR_COUNT (MIMO_CFO_DATA_PHASOR_COUNT + MIMO_CFO_RESIDUAL_PHASOR_COUNT)
#define MIMO_CFO_RESIDUAL_PHASOR_OFFSET MIMO_CFO_DATA_PHASOR_COUNT

__device__ __host__ __forceinline__ int pusch_dmrs_pilots_per_cdm(int dmrs_type)
{
    return (dmrs_type == DMRS_TYPE_2) ? 4 : 6;
}

__device__ __host__ __forceinline__ int pusch_dmrs_sc_offset(int dmrs_type, int cdm_group, int pilot_re)
{
    if (dmrs_type == DMRS_TYPE_2) {
        return cdm_group * 2 + (pilot_re & 1) + 6 * (pilot_re >> 1);
    }
    return cdm_group + 2 * pilot_re;
}

__device__ __forceinline__ __half2 interpolate_dmrs_pilots_half2(
    const __half2* __restrict__ pilots,
    int nof_pilots,
    int dmrs_type,
    int cdm_group,
    int sc)
{
    int left = 0;
    int right = 0;
    int left_sc = pusch_dmrs_sc_offset(dmrs_type, cdm_group, 0);
    int right_sc = left_sc;

    #pragma unroll
    for (int p = 0; p < 6; p++) {
        if (p >= nof_pilots) break;
        int pilot_sc = pusch_dmrs_sc_offset(dmrs_type, cdm_group, p);
        if (pilot_sc == sc) {
            return pilots[p];
        }
        if (pilot_sc < sc) {
            left = p;
            left_sc = pilot_sc;
        } else {
            right = p;
            right_sc = pilot_sc;
            break;
        }
    }

    if (sc <= left_sc) {
        return pilots[left];
    }
    if (right_sc <= left_sc) {
        return pilots[left];
    }

    float alpha = (float)(sc - left_sc) / (float)(right_sc - left_sc);
    __half2 left_h = pilots[left];
    __half2 right_h = pilots[right];
    return __hadd2(__hmul2(left_h, __float2half2_rn(1.0f - alpha)),
                   __hmul2(right_h, __float2half2_rn(alpha)));
}

static int pusch_e2e_get_dmrs_symbol_indices_checked(int dmrs_symbol_mask, int* dmrs_symbol_indices)
{
    if ((dmrs_symbol_mask & ~((1 << MAX_SYMBOLS) - 1)) != 0) {
        return -1;
    }

    int nof_dmrs_symbols = 0;
    for (int s = 0; s < MAX_SYMBOLS; s++) {
        if (dmrs_symbol_mask & (1 << s)) {
            if (nof_dmrs_symbols == MAX_DMRS_SYMBOLS) {
                return -1;
            }
            if (dmrs_symbol_indices) {
                dmrs_symbol_indices[nof_dmrs_symbols] = s;
            }
            nof_dmrs_symbols++;
        }
    }

    return nof_dmrs_symbols;
}

/* PUSCH E2E context */
struct pusch_e2e_ctx {
    pusch_e2e_config_t config;
    bool configured;

    /* Scrambler for data descrambling */
    scrambler_handle_t data_scrambler;

    /* Scrambler for DMRS pilot generation */
    scrambler_handle_t dmrs_scrambler;

    /* Device memory for DMRS pilots */
    cuFloatComplex* d_dmrs_pilots;
    size_t dmrs_pilots_capacity;

    /* Device memory for channel estimates (all REs, all ports) */
    cuFloatComplex* d_ch_estimates;
    size_t ch_estimates_capacity;
    int nof_estimates;

    /* Device memory for DMRS symbol estimates (before time interpolation) */
    cuFloatComplex* d_dmrs_estimates;
    size_t dmrs_estimates_capacity;

    /* Device memory for GPU-computed noise variances (per port) */
    float* d_noise_vars;
    size_t noise_vars_capacity;

    /* Device memory for noise variance reduction (per port, per block) */
    float* d_noise_var_partial;
    size_t noise_var_partial_capacity;

    /* Device memory for accumulated post-equalization noise variance (for SINR) */
    float* d_eq_noise_var_sum;
    unsigned int* d_eq_noise_var_count;

    /* Device memory for CPU-style hard-decision EVM accumulation */
    float* d_evm_error_sum;
    unsigned int* d_evm_symbol_count;

    /* Pinned host memory for async SINR/EVM readback */
    void* h_sinr_eq_block;            /* Single pinned alloc: [sinr sum/count, evm sum/count] */
    float* h_sinr_noise_var_sum;      /* Alias: h_sinr_eq_block + 0 */
    unsigned int* h_sinr_count;       /* Alias: h_sinr_eq_block + sizeof(float) */
    float* h_evm_error_sum;           /* Alias: h_sinr_eq_block + 8 */
    unsigned int* h_evm_symbol_count; /* Alias: h_sinr_eq_block + 12 */

    /* DMRS-based RSRP accumulation (per port, for SINR matching CPU ch.est) */
    float* d_rsrp_accum;        /* [nof_ports] — sum of |H|² at DMRS positions */

    /* DMRS-based EPRE accumulation (per port) */
    float* d_epre_accum;        /* [nof_ports] — sum of |y|² at DMRS positions */

    /* Residual SINR accumulators (per port, for fused CV+SINR kernel) */
    float* d_sinr_noise_accum;  /* [8] — per-port residual noise accumulator */
    float* d_sinr_rsrp_accum;   /* [8] — per-port signal power accumulator */

    /* DMRS c_init values for inline gold sequence generation (per DMRS symbol) */
    uint32_t* d_dmrs_c_inits;  /* [MAX_DMRS_SYMBOLS] */

    /* Results: DMRS-based SINR and EPRE computed on GPU */
    float* d_sinr_epre_result;  /* [5] — {sinr_db, epre_db, rsrp_db, ta_seconds, cfo_hz} */
    float* h_sinr_epre_result;  /* [5] — pinned host for async D2H */

    /* Pre-allocated buffers to avoid per-slot malloc (HUGE efficiency win!) */
    int* d_dmrs_indices;
    size_t dmrs_indices_capacity;
    cuFloatComplex* d_freq_interp_estimates;
    size_t freq_interp_capacity;
    cuFloatComplex* d_lse_averaged;
    size_t lse_averaged_capacity;
    __half* d_llrs_half_temp;  /* Temp buffer for INT8 conversion path */
    size_t llrs_half_temp_capacity;

    /* Optimized FP16 channel estimate buffers */
    __half2* d_estimates_fp16;
    size_t estimates_fp16_capacity;

    /* Buffer for DMRS received symbols (avoids grid re-read in noise var kernel) */
    cuFloatComplex* d_dmrs_received;
    size_t dmrs_received_capacity;

    /* Buffer for fused LSE+noise variance kernel (atomic counter) */
    unsigned int* d_noise_count;

    /* Consolidated per-iteration accumulator block — single memset zeros all.
     * Layout: [noise_count(4) | rsrp(32) | epre(32) | sinr_noise(32) | sinr_rsrp(32) |
     *          eq_sum(4) | eq_count(4) | evm_sum(4) | evm_count(4) |
     *          ta_cfo_accum(24) | cv_noise_accum(32) | cv_done_counter(4)]
     * Total: 208 bytes (max 8 ports). Pointer aliases point into this block. */
    void*  d_accum_block;
    size_t accum_block_size;

    /* TA/CFO/RSRP/EPRE accumulator (alias into d_accum_block, auto-zeroed).
     * Layout: [ta_r, ta_i, cfo_r, cfo_i, rsrp_sum, epre_sum] — 6 floats. */
    float* d_ta_cfo_accum;

    /* CV noise accumulator + done counter (aliases into d_accum_block, auto-zeroed).
     * Used by fused CV+finalize kernel. Separate from d_noise_vars so no mid-processing
     * re-zero is needed — the initial d_accum_block memset covers it. */
    float* d_cv_noise_accum;            /* [8] — per-port CV noise accumulator */
    unsigned int* d_cv_done_counter;    /* [1] — last-block atomic counter */

    /* Pre-computed scrambling sequence buffer (packed bits, 32 bits per word) */
    uint32_t* d_scrambling_seq;
    size_t scrambling_seq_capacity;
    uint32_t cached_data_scrambling_c_init;
    int cached_data_scrambling_bits;

    /* Pre-computed DMRS pilots (packed bits: 2 bits per pilot = real_sign, imag_sign)
     * Format: bit 2n = real sign, bit 2n+1 = imag sign for pilot n
     * Size: ceil(nof_prb * 6 * 2 * 2 / 32) words per DMRS symbol */
    uint32_t* d_dmrs_pilot_bits;
    size_t dmrs_pilot_bits_capacity;

    /* DMRS symbol mask for mega-fused kernel (14 ints, 1 per symbol slot) */
    int* d_dmrs_mask;
    size_t dmrs_mask_capacity;

    /* Pre-computed DMRS pilots for ALL 14 OFDM symbols × ALL slots
     * Layout: [slot_idx * 14 * words_per_sym + ofdm_symbol * words_per_sym]
     * This eliminates runtime kernel_generate_dmrs_pilots regardless of DMRS mask.
     * Precomputed at configure() time since scrambling_id/n_scid are fixed. */
    uint32_t* d_precomputed_dmrs_pilots;
    size_t precomputed_dmrs_pilots_capacity;
    int precomputed_nof_slots;           /* Number of slots precomputed */
    int precomputed_pilots_per_symbol;   /* Pilots per DMRS symbol */
    int precomputed_words_per_slot;      /* uint32_t words per slot (14 * words_per_sym) */
    int precomputed_words_per_sym;       /* uint32_t words per single OFDM symbol */
    bool dmrs_pilots_precomputed;        /* Flag indicating pilots are ready */

    /* Low-PAPR DMRS support for transform precoding */
    cuFloatComplex* d_low_papr_pilots;   /* Device buffer for low-PAPR pilot symbols */
    size_t low_papr_pilots_capacity;
    int cached_low_papr_nof_prb;
    int cached_low_papr_n_rs_id;

    /* Bounded transform-deprecoding plans for all valid 5G NR DFT sizes.
     * Index by dft_size; invalid entries have nof_factors == 0. */
    uint64_t deprecoding_fft_factors[PUSCH_DEPRECODE_MAX_DFT_SIZE + 1];
    int      deprecoding_fft_nof_factors[PUSCH_DEPRECODE_MAX_DFT_SIZE + 1];
    int      transform_deprecoder_backend;
    int      vkfft_auto_min_dft_size;
    cuFloatComplex* d_deprecode_symbols;
    float*   d_deprecode_noise_vars;
    size_t   deprecode_symbols_capacity;
    size_t   deprecode_noise_capacity;
#ifdef OCUDU_PHY_CUDA_ENABLE_VKFFT
    pusch_vkfft_deprecoder_plan* vkfft_plans[PUSCH_DEPRECODE_MAX_DFT_SIZE + 1];
    void* vkfft_buffer_ptr;
    CUdevice vkfft_device;
    cudaStream_t vkfft_stream;
    bool vkfft_preplanned;
#endif

    /* Precomputed OFDM symbol start times (seconds from slot start) */
    float* d_symbol_start_times;         /* Device memory [14] */
    float  h_symbol_start_times[14];     /* Host cache */
    int    symbol_times_scs_khz;         /* SCS used for last precomputation (0 = not yet computed) */
    float2* d_mimo_cfo_phasors;          /* [data symbol x DMRS] + [DMRS x DMRS] CFO rotation cache */

    /* Cached DMRS indices to avoid per-slot sync cudaMemcpy.
     * d_dmrs_indices is already uploaded when cached_dmrs_symbol_mask matches. */
    int    cached_dmrs_symbol_mask;      /* Mask used for last d_dmrs_indices upload */
    int    cached_nof_dmrs_symbols;      /* Number of DMRS symbols in cached indices */

    /* Cached DMRS pilot precompute parameters — skip 280-kernel precompute
     * when only RNTI/n_id changes (DMRS c_init depends only on scrambling_id + n_scid). */
    uint32_t cached_dmrs_scrambling_id;
    int      cached_dmrs_n_scid;
    int      cached_dmrs_nof_prb;
    int      cached_dmrs_type;

    /* External eq-output buffers (set via pusch_e2e_set_eq_output, default NULL) */
    float2* d_eq_symbols_out;
    float*  d_eq_noise_var_out;
};

/* ============================================================================
 * Low-PAPR DMRS Sequence Tables (TS 38.211 Table 5.2.2.2-1 to 5.2.2.2-4)
 * Used for transform precoding (MSG3 / DFT-s-OFDM)
 * ============================================================================ */

/* Phi table for M_zc = 6 (30 sequence groups x 6 values) */
__constant__ int d_phi_M_sc_6[30][6] = {
    {-3, -1, 3, 3, -1, -3},  {-3, 3, -1, -1, 3, -3},  {-3, -3, -3, 3, 1, -3},  {1, 1, 1, 3, -1, -3},
    {1, 1, 1, -3, -1, 3},    {-3, 1, -1, -3, -3, -3}, {-3, 1, 3, -3, -3, -3},  {-3, -1, 1, -3, 1, -1},
    {-3, -1, -3, 1, -3, -3}, {-3, -3, 1, -3, 3, -3},  {-3, 1, 3, 1, -3, -3},   {-3, -1, -3, 1, 1, -3},
    {1, 1, 3, -1, -3, 3},    {1, 1, 3, 3, -1, 3},     {1, 1, 1, -3, 3, -1},    {1, 1, 1, -1, 3, -3},
    {-3, -1, -1, -1, 3, -1}, {-3, -3, -1, 1, -1, -3}, {-3, -3, -3, 1, -3, -1}, {-3, 1, 1, -3, -1, -3},
    {-3, 3, -3, 1, 1, -3},   {-3, 1, -3, -3, -3, -1}, {1, 1, -3, 3, 1, 3},     {1, 1, -3, -3, 1, -3},
    {1, 1, 3, -1, 3, 3},     {1, 1, -3, 1, 3, 3},     {1, 1, -1, -1, 3, -1},   {1, 1, -1, 3, -1, -1},
    {1, 1, -1, 3, -3, -1},   {1, 1, -3, 1, -1, -1}
};

/* Phi table for M_zc = 12 (30 sequence groups x 12 values) */
__constant__ int d_phi_M_sc_12[30][12] = {
    {-3, 1, -3, -3, -3, 3, -3, -1, 1, 1, 1, -3},  {-3, 3, 1, -3, 1, 3, -1, -1, 1, 3, 3, 3},
    {-3, 3, 3, 1, -3, 3, -1, 1, 3, -3, 3, -3},    {-3, -3, -1, 3, 3, 3, -3, 3, -3, 1, -1, -3},
    {-3, -1, -1, 1, 3, 1, 1, -1, 1, -1, -3, 1},   {-3, -3, 3, 1, -3, -3, -3, -1, 3, -1, 1, 3},
    {1, -1, 3, -1, -1, -1, -3, -1, 1, 1, 1, -3},  {-1, -3, 3, -1, -3, -3, -3, -1, 1, -1, 1, -3},
    {-3, -1, 3, 1, -3, -1, -3, 3, 1, 3, 3, 1},    {-3, -1, -1, -3, -3, -1, -3, 3, 1, 3, -1, -3},
    {-3, 3, -3, 3, 3, -3, -1, -1, 3, 3, 1, -3},   {-3, -1, -3, -1, -1, -3, 3, 3, -1, -1, 1, -3},
    {-3, -1, 3, -3, -3, -1, -3, 1, -1, -3, 3, 3}, {-3, 1, -1, -1, 3, 3, -3, -1, -1, -3, -1, -3},
    {1, 3, -3, 1, 3, 3, 3, 1, -1, 1, -1, 3},      {-3, 1, 3, -1, -1, -3, -3, -1, -1, 3, 1, -3},
    {-1, -1, -1, -1, 1, -3, -1, 3, 3, -1, -3, 1}, {-1, 1, 1, -1, 1, 3, 3, -1, -1, -3, 1, -3},
    {-3, 1, 3, 3, -1, -1, -3, 3, 3, -3, 3, -3},   {-3, -3, 3, -3, -1, 3, 3, 3, -1, -3, 1, -3},
    {3, 1, 3, 1, 3, -3, -1, 1, 3, 1, -1, -3},     {-3, 3, 1, 3, -3, 1, 1, 1, 1, 3, -3, 3},
    {-3, 3, 3, 3, -1, -3, -3, -1, -3, 1, 3, -3},  {3, -1, -3, 3, -3, -1, 3, 3, 3, -3, -1, -3},
    {-3, -1, 1, -3, 1, 3, 3, 3, -1, -3, 3, 3},    {-3, 3, 1, -1, 3, 3, -3, 1, -1, 1, -1, 1},
    {-1, 1, 3, -3, 1, -1, 1, -1, -1, -3, 1, -1},  {-3, -3, 3, 3, 3, -3, -1, 1, -3, 3, 1, -3},
    {1, -1, 3, 1, 1, -1, -1, -1, 1, 3, -3, 1},    {-3, 3, -3, 3, -3, -3, 3, -1, -1, 1, 3, -3}
};

/* Phi table for M_zc = 18 (30 sequence groups x 18 values) */
__constant__ int d_phi_M_sc_18[30][18] = {
    {-1, 3, -1, -3, 3, 1, -3, -1, 3, -3, -1, -1, 1, 1, 1, -1, -1, -1},
    {3, -3, 3, -1, 1, 3, -3, -1, -3, -3, -1, -3, 3, 1, -1, 3, -3, 3},
    {-3, 3, 1, -1, -1, 3, -3, -1, 1, 1, 1, 1, 1, -1, 3, -1, -3, -1},
    {-3, -3, 3, 3, 3, 1, -3, 1, 3, 3, 1, -3, -3, 3, -1, -3, -1, 1},
    {1, 1, -1, -1, -3, -1, 1, -3, -3, -3, 1, -3, -1, -1, 1, -1, 3, 1},
    {3, -3, 1, 1, 3, -1, 1, -1, -1, -3, 1, 1, -1, 3, 3, -3, 3, -1},
    {-3, 3, -1, 1, 3, 1, -3, -1, 1, 1, -3, 1, 3, 3, -1, -3, -3, -3},
    {1, 1, -3, 3, 3, 1, 3, -3, 3, -1, 1, 1, -1, 1, -3, -3, -1, 3},
    {-3, 1, -3, -3, 1, -3, -3, 3, 1, -3, -1, -3, -3, -3, -1, 1, 1, 3},
    {3, -1, 3, 1, -3, -3, -1, 1, -3, -3, 3, 3, 3, 1, 3, -3, 3, -3},
    {-3, -3, -3, 1, -3, 3, 1, 1, 3, -3, -3, 1, 3, -1, 3, -3, -3, 3},
    {-3, -3, 3, 3, 3, -1, -1, -3, -1, -1, -1, 3, 1, -3, -3, -1, 3, -1},
    {-3, -1, -3, -3, 1, 1, -1, -3, -1, -3, -1, -1, 3, 3, -1, 3, 1, 3},
    {1, 1, -3, -3, -3, -3, 1, 3, -3, 3, 3, 1, -3, -1, 3, -1, -3, 1},
    {-3, 3, -1, -3, -1, -3, 1, 1, -3, -3, -1, -1, 3, -3, 1, 3, 1, 1},
    {3, 1, -3, 1, -3, 3, 3, -1, -3, -3, -1, -3, -3, 3, -3, -1, 1, 3},
    {-3, -1, -3, -1, -3, 1, 3, -3, -1, 3, 3, 3, 1, -1, -3, 3, -1, -3},
    {-3, -1, 3, 3, -1, 3, -1, -3, -1, 1, -1, -3, -1, -1, -1, 3, 3, 1},
    {-3, 1, -3, -1, -1, 3, 1, -3, -3, -3, -1, -3, -3, 1, 1, 1, -1, -1},
    {3, 3, 3, -3, -1, -3, -1, 3, -1, 1, -1, -3, 1, -3, -3, -1, 3, 3},
    {-3, 1, 1, -3, 1, 1, 3, -3, -1, -3, -1, 3, -3, 3, -1, -1, -1, -3},
    {1, -3, -1, -3, 3, 3, -1, -3, 1, -3, -3, -1, -3, -1, 1, 3, 3, 3},
    {-3, -3, 1, -1, -1, 1, 1, -3, -1, 3, 3, 3, 3, -1, 3, 1, 3, 1},
    {3, -1, -3, 1, -3, -3, -3, 3, 3, -1, 1, -3, -1, 3, 1, 1, 3, 3},
    {3, -1, -1, 1, -3, -1, -3, -1, -3, -3, -1, -3, 1, 1, 1, -3, -3, 3},
    {-3, -3, 1, -3, 3, 3, 3, -1, 3, 1, 1, -3, -3, -3, 3, -3, -1, -1},
    {-3, -1, -1, -3, 1, -3, 3, -1, -1, -3, 3, 3, -3, -1, 3, -1, -1, -1},
    {-3, -3, 3, 3, -3, 1, 3, -1, -3, 1, -1, -3, 3, -3, -1, -1, -1, 3},
    {-1, -3, 1, -3, -3, -3, 1, 1, 3, 3, -3, 3, 3, -3, -1, 3, -3, 1},
    {-3, 3, 1, -1, -1, -1, -1, 1, -1, 3, 3, -3, -1, 1, 3, -1, 3, -1}
};

/* Phi table for M_zc = 24 (30 sequence groups x 24 values) */
__constant__ int d_phi_M_sc_24[30][24] = {
    {-1, -3, 3, -1, 3, 1, 3, -1, 1, -3, -1, -3, -1, 1, 3, -3, -1, -3, 3, 3, 3, -3, -3, -3},
    {-1, -3, 3, 1, 1, -3, 1, -3, -3, 1, -3, -1, -1, 3, -3, 3, 3, 3, -3, 1, 3, 3, -3, -3},
    {-1, -3, -3, 1, -1, -1, -3, 1, 3, -1, -3, -1, -1, -3, 1, 1, 3, 1, -3, -1, -1, 3, -3, -3},
    {1, -3, 3, -1, -3, -1, 3, 3, 1, -1, 1, 1, 3, -3, -1, -3, -3, -3, -1, 3, -3, -1, -3, -3},
    {-1, 3, -3, -3, -1, 3, -1, -1, 1, 3, 1, 3, -1, -1, -3, 1, 3, 1, -1, -3, 1, -1, -3, -3},
    {-3, -1, 1, -3, -3, 1, 1, -3, 3, -1, -1, -3, 1, 3, 1, -1, -3, -1, -3, 1, -3, -3, -3, -3},
    {-3, 3, 1, 3, -1, 1, -3, 1, -3, 1, -1, -3, -1, -3, -3, -3, -3, -1, -1, -1, 1, 1, -3, -3},
    {-3, 1, 3, -1, 1, -1, 3, -3, 3, -1, -3, -1, -3, 3, -1, -1, -1, -3, -1, -1, -3, 3, 3, -3},
    {-3, 1, -3, 3, -1, -1, -1, -3, 3, 1, -1, -3, -1, 1, 3, -1, 1, -1, 1, -3, -3, -3, -3, -3},
    {1, 1, -1, -3, -1, 1, 1, -3, 1, -1, 1, -3, 3, -3, -3, 3, -1, -3, 1, 3, -3, 1, -3, -3},
    {-3, -3, -3, -1, 3, -3, 3, 1, 3, 1, -3, -1, -1, -3, 1, 1, 3, 1, -1, -3, 3, 1, 3, -3},
    {-3, 3, -1, 3, 1, -1, -1, -1, 3, 3, 1, 1, 1, 3, 3, 1, -3, -3, -1, 1, -3, 1, 3, -3},
    {3, -3, 3, -1, -3, 1, 3, 1, -1, -1, -3, -1, 3, -3, 3, -1, -1, 3, 3, -3, -3, 3, -3, -3},
    {-3, 3, -1, 3, -1, 3, 3, 1, 1, -3, 1, 3, -3, 3, -3, -3, -1, 1, 3, -3, -1, -1, -3, -3},
    {-3, 1, -3, -1, -1, 3, 1, 3, -3, 1, -1, 3, 3, -1, -3, 3, -3, -1, -1, -3, -3, -3, 3, -3},
    {-3, -1, -1, -3, 1, -3, -3, -1, -1, 3, -1, 1, -1, 3, 1, -3, -1, 3, 1, 1, -1, -1, -3, -3},
    {-3, -3, 1, -1, 3, 3, -3, -1, 1, -1, -1, 1, 1, -1, -1, 3, -3, 1, -3, 1, -1, -1, -1, -3},
    {3, -1, 3, -1, 1, -3, 1, 1, -3, -3, 3, -3, -1, -1, -1, -1, -1, -3, -3, -1, 1, 1, -3, -3},
    {-3, 1, -3, 1, -3, -3, 1, -3, 1, -3, -3, -3, -3, -3, 1, -3, -3, 1, 1, -3, 1, 1, -3, -3},
    {-3, -3, 3, 3, 1, -1, -1, -1, 1, -3, -1, 1, -1, 3, -3, -1, -3, -1, -1, 1, -3, 3, -1, -3},
    {-3, -3, -1, -1, -1, -3, 1, -1, -3, -1, 3, -3, 1, -3, 3, -3, 3, 3, 1, -1, -1, 1, -3, -3},
    {3, -1, 1, -1, 3, -3, 1, 1, 3, -1, -3, 3, 1, -3, 3, -1, -1, -1, -1, 1, -3, -3, -3, -3},
    {-3, 1, -3, 3, -3, 1, -3, 3, 1, -1, -3, -1, -3, -3, -3, -3, 1, 3, -1, 1, 3, 3, 3, -3},
    {-3, -1, 1, -3, -1, -1, 1, 1, 1, 3, 3, -1, 1, -1, 1, -1, -1, -3, -3, -3, 3, 1, -1, -3},
    {-3, 3, -1, -3, -1, -1, -1, 3, -1, -1, 3, -3, -1, 3, -3, 3, -3, -1, 3, 1, 1, -1, -3, -3},
    {-3, 1, -1, -3, -3, -1, 1, -3, -1, -3, 1, 1, -1, 1, 1, 3, 3, 3, -1, 1, -1, 1, -1, -3},
    {-1, 3, -1, -1, 3, 3, -1, -1, -1, 3, -1, -3, 1, 3, 1, 1, -3, -3, -3, -1, -3, -1, -3, -3},
    {3, -3, -3, -1, 3, 3, -3, -1, 3, 1, 1, 1, 3, -1, 3, -3, -1, 3, -1, 3, 1, -1, -3, -3},
    {-3, 1, -3, 1, -3, 1, 1, 3, 1, -3, -3, -1, 1, 3, -1, -3, 3, 1, -1, -3, -3, -3, -3, -3},
    {3, -3, -1, 1, 3, -1, -1, -3, -1, 3, -1, -3, -1, -3, 3, -1, 3, 1, 1, -3, 3, -3, -3, -3}
};

/* Number of sequence groups for low-PAPR */
#define NOF_LOW_PAPR_GROUPS 30

/**
 * Get largest prime less than n (used for Zadoff-Chu sequence generation).
 * Covers all valid low-PAPR sequence lengths used by transform-precoded PUSCH.
 */
__device__ __forceinline__ int get_prime_less_than_device(int n) {
    if (n <= 6) return 5;
    if (n <= 12) return 11;
    if (n <= 18) return 17;
    if (n <= 24) return 23;
    if (n <= 30) return 29;
    if (n <= 36) return 31;
    if (n <= 48) return 47;
    if (n <= 54) return 53;
    if (n <= 60) return 59;
    if (n <= 72) return 71;
    if (n <= 84) return 83;
    if (n <= 90) return 89;
    if (n <= 96) return 89;
    if (n <= 108) return 107;
    if (n <= 120) return 113;
    if (n <= 132) return 131;
    if (n <= 144) return 139;
    if (n <= 150) return 149;
    if (n <= 156) return 151;
    if (n <= 162) return 157;
    if (n <= 168) return 167;
    if (n <= 180) return 179;
    if (n <= 192) return 191;
    if (n <= 204) return 199;
    if (n <= 216) return 211;
    if (n <= 228) return 227;
    if (n <= 240) return 239;
    if (n <= 252) return 251;
    if (n <= 264) return 263;
    if (n <= 270) return 269;
    if (n <= 276) return 271;
    if (n <= 288) return 283;
    if (n <= 300) return 293;
    if (n <= 312) return 311;
    if (n <= 324) return 317;
    if (n <= 336) return 331;
    if (n <= 360) return 359;
    if (n <= 384) return 383;
    if (n <= 396) return 389;
    if (n <= 408) return 401;
    if (n <= 432) return 431;
    if (n <= 450) return 449;
    if (n <= 456) return 449;
    if (n <= 480) return 479;
    if (n <= 486) return 479;
    if (n <= 504) return 503;
    if (n <= 528) return 523;
    if (n <= 540) return 523;
    if (n <= 552) return 547;
    if (n <= 576) return 571;
    if (n <= 600) return 599;
    if (n <= 624) return 619;
    if (n <= 648) return 647;
    if (n <= 672) return 661;
    if (n <= 720) return 719;
    if (n <= 750) return 743;
    if (n <= 768) return 761;
    if (n <= 792) return 787;
    if (n <= 810) return 809;
    if (n <= 816) return 811;
    if (n <= 864) return 863;
    if (n <= 900) return 887;
    if (n <= 912) return 911;
    if (n <= 960) return 953;
    if (n <= 972) return 971;
    if (n <= 1008) return 997;
    if (n <= 1056) return 1051;
    if (n <= 1080) return 1069;
    if (n <= 1104) return 1103;
    if (n <= 1152) return 1151;
    if (n <= 1200) return 1193;
    if (n <= 1248) return 1237;
    if (n <= 1296) return 1291;
    if (n <= 1344) return 1327;
    if (n <= 1350) return 1327;
    if (n <= 1440) return 1439;
    if (n <= 1458) return 1453;
    if (n <= 1500) return 1499;
    if (n <= 1536) return 1531;
    if (n <= 1584) return 1583;
    if (n <= 1620) return 1619;
    return 1627;
}

/**
 * Compute Zadoff-Chu sequence parameter q.
 * TS 38.211 Section 5.2.2.1
 */
__device__ __forceinline__ int zc_sequence_q_device(int u, int v, int N_zc) {
    float n_sz = (float)N_zc;
    float q_hat = n_sz * (u + 1) / 31.0f;
    float q;
    if (((int)(2.0f * q_hat)) % 2 == 0) {
        q = q_hat + 0.5f + v;
    } else {
        q = q_hat + 0.5f - v;
    }
    return (int)q;
}

/**
 * Get N_zc for low-PAPR sequence based on M_zc.
 * For M_zc < 36, N_zc is determined by lookup table divisor.
 * For M_zc >= 36, N_zc is largest prime < M_zc.
 * For M_zc = 30, N_zc = 31 (special case from TS 38.211).
 */
__device__ __forceinline__ int get_N_zc_device(int M_zc) {
    if (M_zc >= 36) {
        return get_prime_less_than_device(M_zc);
    }
    if (M_zc == 30) {
        return 31;
    }
    /* For M_zc in {6, 12, 18, 24}, the tables are normalized to 4 */
    return 4;
}

/**
 * Generate low-PAPR DMRS pilot sequence for transform precoding.
 * Implements TS 38.211 Section 6.4.1.1.2 (PUSCH DMRS with transform precoding).
 *
 * @param d_pilots Output: complex pilot symbols [M_zc]
 * @param u Sequence group = n_rs_id % 30
 * @param M_zc Sequence length = nof_prb * 6 (DMRS RE per symbol)
 */
__global__ void kernel_generate_low_papr_pilots(
    cuFloatComplex* __restrict__ d_pilots,
    int u,
    int M_zc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M_zc) return;

    int arg;
    int N_zc;

    if (M_zc == 6) {
        arg = d_phi_M_sc_6[u][idx];
        N_zc = 4;
    } else if (M_zc == 12) {
        arg = d_phi_M_sc_12[u][idx];
        N_zc = 4;
    } else if (M_zc == 18) {
        arg = d_phi_M_sc_18[u][idx];
        N_zc = 4;
    } else if (M_zc == 24) {
        arg = d_phi_M_sc_24[u][idx];
        N_zc = 4;
    } else if (M_zc == 30) {
        /* TS 38.211 formula for M_zc = 30 */
        N_zc = 31;
        /* arg[n] = -((u+1)*(n+1)*(n+2)) mod (2*N_zc) */
        int64_t prod = (int64_t)(u + 1) * (int64_t)(idx + 1) * (int64_t)(idx + 2);
        arg = -((int)(prod % (2 * N_zc)));
    } else {
        /* Zadoff-Chu for M_zc >= 36 */
        N_zc = get_prime_less_than_device(M_zc);
        int q = zc_sequence_q_device(u, 0, N_zc);  /* v = 0 for low-PAPR DMRS */
        int m = idx % N_zc;
        /* arg[n] = -(q*m*(m+1)) mod (2*N_zc) */
        int64_t prod = (int64_t)q * (int64_t)m * (int64_t)(m + 1);
        arg = -((int)(prod % (2 * N_zc)));
    }

    /* Compute exp(j * π * arg / N_zc) */
    float phase = M_PI * (float)arg / (float)N_zc;
    d_pilots[idx] = make_cuFloatComplex(cosf(phase), sinf(phase));
}

/* ============================================================================
 * Inline device functions
 * ============================================================================ */

__device__ __forceinline__ cuFloatComplex cbf16_to_fp32(unsigned int packed) {
    unsigned int real_bits = packed & 0xFFFF;
    unsigned int imag_bits = (packed >> 16) & 0xFFFF;
    unsigned int real_fp32 = real_bits << 16;
    unsigned int imag_fp32 = imag_bits << 16;
    float real_f = *reinterpret_cast<float*>(&real_fp32);
    float imag_f = *reinterpret_cast<float*>(&imag_fp32);
    return make_cuFloatComplex(real_f, imag_f);
}

__device__ __forceinline__ cuFloatComplex cuCmulf_fast(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ cuFloatComplex cuConjf_fast(cuFloatComplex a) {
    return make_cuFloatComplex(a.x, -a.y);
}

/* Deep-fade sentinel: eq_noise_var set to this when channel is dead.
 * Makes inv_noise ≈ 0 → LLR ≈ 0 (erasure), matching CPU behavior.
 * SINR accumulation excludes this exact value as the GPU infinity-equivalent. */
static constexpr float EQ_NOISE_VAR_DEEP_FADE = 1e6f;

/* The CPU post-equalization SINR path drops infinities, but keeps large
 * finite equalizer noise variances. The GPU uses EQ_NOISE_VAR_DEEP_FADE as
 * its infinity-equivalent erasure value, so only that exact sentinel is
 * excluded from the average. */
__device__ __forceinline__ bool sinr_noise_var_is_valid(float eq_noise_var)
{
    return isfinite(eq_noise_var) && (eq_noise_var != EQ_NOISE_VAR_DEEP_FADE);
}

/* Accumulate eq_noise_var for SINR reporting, excluding deep-fade REs. */
__device__ __forceinline__ void sinr_accumulate(
    float* d_sum, unsigned int* d_count, float eq_noise_var)
{
    if (sinr_noise_var_is_valid(eq_noise_var)) {
        atomicAdd(d_sum, eq_noise_var);
        atomicAdd(d_count, 1u);
    }
}

__device__ __forceinline__ void warp_accumulate_sum_count(
    float* d_sum, unsigned int* d_count, float value, unsigned int count)
{
    unsigned mask = __activemask();
    int lane = threadIdx.x & 31;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
        count += __shfl_down_sync(mask, count, offset);
    }

    int leader = __ffs(mask) - 1;
    if ((lane == leader) && (count > 0)) {
        atomicAdd(d_sum, value);
        atomicAdd(d_count, count);
    }
}

__device__ __forceinline__ void sinr_accumulate_warp(
    float* d_sum, unsigned int* d_count, float eq_noise_var)
{
    unsigned int valid_count = sinr_noise_var_is_valid(eq_noise_var) ? 1u : 0u;
    warp_accumulate_sum_count(d_sum, d_count, valid_count ? eq_noise_var : 0.0f, valid_count);
}

__device__ __forceinline__ float nearest_qam_axis(float value, int max_abs_level, float scale)
{
    float clipped = fminf((float)max_abs_level, fmaxf((float)-max_abs_level, value / scale));
    int idx = (int)floorf((clipped + (float)max_abs_level) * 0.5f + 0.5f);
    idx = max(0, min(max_abs_level, idx));
    return ((float)(-max_abs_level + 2 * idx)) * scale;
}

__device__ __forceinline__ float evm_error_power(cuFloatComplex eq, int mod_order)
{
    float ref_re;
    float ref_im;
    if (mod_order == 2) {
        constexpr float scale = 0.7071067811865475f;
        ref_re = (eq.x >= 0.0f) ? scale : -scale;
        ref_im = (eq.y >= 0.0f) ? scale : -scale;
    } else if (mod_order == 4) {
        constexpr float scale = 0.31622776601683794f;
        ref_re = nearest_qam_axis(eq.x, 3, scale);
        ref_im = nearest_qam_axis(eq.y, 3, scale);
    } else if (mod_order == 6) {
        constexpr float scale = 0.1543033499620919f;
        ref_re = nearest_qam_axis(eq.x, 7, scale);
        ref_im = nearest_qam_axis(eq.y, 7, scale);
    } else {
        constexpr float scale = 0.07669649888473704f;
        ref_re = nearest_qam_axis(eq.x, 15, scale);
        ref_im = nearest_qam_axis(eq.y, 15, scale);
    }

    float err_re = eq.x - ref_re;
    float err_im = eq.y - ref_im;
    return err_re * err_re + err_im * err_im;
}

__device__ __forceinline__ void evm_accumulate(
    float* d_sum, unsigned int* d_count, cuFloatComplex eq, int mod_order)
{
    if (d_sum && d_count && isfinite(eq.x) && isfinite(eq.y)) {
        atomicAdd(d_sum, evm_error_power(eq, mod_order));
        atomicAdd(d_count, 1u);
    }
}

__device__ __forceinline__ void evm_accumulate_warp(
    float* d_sum, unsigned int* d_count, float evm_sum, unsigned int symbol_count)
{
    if (d_sum && d_count) {
        warp_accumulate_sum_count(d_sum, d_count, evm_sum, symbol_count);
    }
}

/**
 * Equalize a single symbol using the specified algorithm.
 *
 * @tparam NOF_PORTS Number of receive ports
 * @tparam ALGORITHM Equalization algorithm (EQUALIZER_ZF, EQUALIZER_MMSE, EQUALIZER_MMSE_IRC)
 * @param y Received symbols per port
 * @param h Channel estimates per port
 * @param noise_vars Per-port noise variances
 * @param tx_scaling TX amplitude scaling
 * @param eq_symbol Output equalized symbol
 * @param eq_noise_var Output post-equalization noise variance
 */
template <int NOF_PORTS, int ALGORITHM>
__device__ __forceinline__ void equalize_symbol(
    const cuFloatComplex* y,
    const cuFloatComplex* h,
    const float* noise_vars,
    float tx_scaling,
    cuFloatComplex& eq_symbol,
    float& eq_noise_var)
{
    float inv_scaling = 1.0f / tx_scaling;

    if (ALGORITHM == EQUALIZER_ZF) {
        /* Zero Forcing: w = h^H / |h|^2 */
        float H_sq = 0.0f;
        cuFloatComplex hH_y = make_cuFloatComplex(0.0f, 0.0f);

        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            H_sq += h[p].x * h[p].x + h[p].y * h[p].y;
            /* h^H * y = conj(h) * y */
            hH_y.x += h[p].x * y[p].x + h[p].y * y[p].y;
            hH_y.y += h[p].x * y[p].y - h[p].y * y[p].x;
        }

        float inv_H_sq = (H_sq > 1e-10f) ? (1.0f / H_sq) : 0.0f;
        eq_symbol.x = hH_y.x * inv_H_sq * inv_scaling;
        eq_symbol.y = hH_y.y * inv_H_sq * inv_scaling;

        /* ZF noise enhancement: sigma^2 / |h|^2 */
        float avg_noise = 0.0f;
        for (int p = 0; p < NOF_PORTS; p++) {
            avg_noise += noise_vars[p];
        }
        avg_noise /= NOF_PORTS;
        eq_noise_var = (H_sq > 1e-10f)
            ? fmaxf(avg_noise * inv_H_sq / (tx_scaling * tx_scaling), 1e-10f)
            : EQ_NOISE_VAR_DEEP_FADE;

    } else if (ALGORITHM == EQUALIZER_MMSE) {
        /* MMSE: w = h^H / (|h|^2 + sigma^2) with gain normalization
         *
         * CPU formula (equalize_mmse_mxn_simd.h, single-layer):
         *   correction = H_sq / (H_sq + σ²)    where σ² = avg noise across ports
         *   gain_norm  = 1 / correction
         *   eq_symbol  = (h^H y) / (H_sq + σ²) * gain_norm / tx_scaling
         *             = (h^H y) / H_sq / tx_scaling
         *   noise_var  = gain_norm - 1 = σ² / H_sq   (scaled by 1/tx_scaling²)
         */
        float H_sq = 0.0f;
        float avg_noise = 0.0f;
        cuFloatComplex hH_y = make_cuFloatComplex(0.0f, 0.0f);

        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            float h_norm_sq = h[p].x * h[p].x + h[p].y * h[p].y;
            H_sq += h_norm_sq;
            avg_noise += noise_vars[p];
            hH_y.x += h[p].x * y[p].x + h[p].y * y[p].y;
            hH_y.y += h[p].x * y[p].y - h[p].y * y[p].x;
        }
        avg_noise /= NOF_PORTS;

        float denom = H_sq + avg_noise;
        float inv_denom = (denom > 1e-10f) ? (1.0f / denom) : 0.0f;

        /* CPU-matching gain normalization: correction = H_sq / (H_sq + σ²) */
        float correction = (H_sq > 1e-10f) ? (H_sq * inv_denom) : 0.0f;
        float gain_norm = (correction > 1e-10f) ? (1.0f / correction) : 1.0f;

        eq_symbol.x = hH_y.x * inv_denom * inv_scaling * gain_norm;
        eq_symbol.y = hH_y.y * inv_denom * inv_scaling * gain_norm;

        /* CPU-matching noise: gain_norm - 1 = σ²/H_sq, scaled by 1/tx_scaling² */
        eq_noise_var = (correction > 1e-10f)
            ? fmaxf((gain_norm - 1.0f) / (tx_scaling * tx_scaling), 1e-10f)
            : EQ_NOISE_VAR_DEEP_FADE;

    } else {
        /* MMSE-IRC: w = (H^H R_n^{-1} H + I)^{-1} H^H R_n^{-1} y
         *
         * Uses per-port noise weighting but CPU-matching gain normalization:
         *   correction = Σ Re(w_p * h_p)     (effective combining gain)
         *   gain_norm  = 1 / correction
         *   noise_var  = gain_norm - 1
         */
        float HRnH = 1.0f;
        cuFloatComplex Rn_inv_H[8];

        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            float inv_sigma = (noise_vars[p] > 1e-10f) ? (1.0f / noise_vars[p]) : 0.0f;
            Rn_inv_H[p] = make_cuFloatComplex(inv_sigma * h[p].x, inv_sigma * h[p].y);
            HRnH += h[p].x * Rn_inv_H[p].x + h[p].y * Rn_inv_H[p].y;
        }

        float inv_HRnH = 1.0f / HRnH;  /* Always > 0 due to +1 regularization */

        cuFloatComplex eq = make_cuFloatComplex(0.0f, 0.0f);
        float correction = 0.0f;
        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            cuFloatComplex w = make_cuFloatComplex(Rn_inv_H[p].x * inv_HRnH, Rn_inv_H[p].y * inv_HRnH);
            eq.x += w.x * y[p].x + w.y * y[p].y;
            eq.y += w.x * y[p].y - w.y * y[p].x;
            /* correction = Σ Re(w_p * h_p) = Σ Re(h_p^H R_n^{-1} h_p) / (H^H R_n^{-1} H + 1) */
            correction += w.x * h[p].x + w.y * h[p].y;
        }

        float gain_norm = (correction > 1e-10f) ? (1.0f / correction) : 1.0f;

        eq_symbol.x = eq.x * inv_scaling * gain_norm;
        eq_symbol.y = eq.y * inv_scaling * gain_norm;
        /* CPU-matching noise: gain_norm - 1, scaled by 1/tx_scaling² */
        eq_noise_var = (correction > 1e-10f)
            ? fmaxf((gain_norm - 1.0f) / (tx_scaling * tx_scaling), 1e-10f)
            : EQ_NOISE_VAR_DEEP_FADE;
    }
}

/* ============================================================================
 * MIMO EQUALIZATION HELPERS (2-layer and 4-layer support)
 * ============================================================================ */

/**
 * Equalize a single RE for 2-layer MIMO using ZF or MMSE.
 *
 * For 2-layer MIMO, we compute:
 *   G = H^H * H (2x2 Gram matrix)
 *   eq = G^{-1} * H^H * y
 *
 * @tparam NOF_PORTS Number of receive ports (must be >= 2)
 * @tparam ALGORITHM Equalization algorithm (ZF or MMSE)
 * @param y Received symbols per port [NOF_PORTS]
 * @param H Channel matrix H[port][layer] with shape [NOF_PORTS][2]
 * @param noise_vars Per-port noise variances [NOF_PORTS]
 * @param tx_scaling TX amplitude scaling
 * @param eq_symbols Output equalized symbols [2]
 * @param eq_noise_vars Output per-layer noise variances [2]
 */
template <int NOF_PORTS, int ALGORITHM>
__device__ __forceinline__ void equalize_symbol_2layer(
    const cuFloatComplex* y,
    const cuFloatComplex H[NOF_PORTS][2],
    const float* noise_vars,
    float tx_scaling,
    cuFloatComplex eq_symbols[2],
    float eq_noise_vars[2])
{
    // CPU multi-layer equalizer uses the most pessimistic per-port estimate.
    float noise_var_est = 0.0f;
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        noise_var_est = fmaxf(noise_var_est, noise_vars[p]);
    }

    // Compute Gram matrix G = H^H * H [2x2 Hermitian]
    cuFloatComplex G00 = make_cuFloatComplex(0.0f, 0.0f);
    cuFloatComplex G01 = make_cuFloatComplex(0.0f, 0.0f);
    cuFloatComplex G11 = make_cuFloatComplex(0.0f, 0.0f);

    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        cuFloatComplex h0 = H[p][0];
        cuFloatComplex h1 = H[p][1];

        // G00 = sum |h0|² (real)
        G00.x += h0.x * h0.x + h0.y * h0.y;

        // G01 = sum conj(h0) * h1
        G01 = cuCaddf(G01, cuCmulf(cuConjf(h0), h1));

        // G11 = sum |h1|² (real)
        G11.x += h1.x * h1.x + h1.y * h1.y;
    }

    float tx_scaling_sq = tx_scaling * tx_scaling;
    float noise_reg = noise_var_est / tx_scaling_sq;

    // CPU scales H by tx_scaling before MMSE regularization. Since this GPU
    // path keeps H unscaled, use the equivalent noise / tx_scaling^2 term.
    if constexpr (ALGORITHM == EQUALIZER_MMSE || ALGORITHM == EQUALIZER_MMSE_IRC) {
        G00.x += noise_reg;
        G11.x += noise_reg;
    }

    // Invert Gram matrix
    cuFloatComplex G_inv[4];
    mimo_invert_2x2_hermitian(G00, G01, G11, G_inv, 1e-6f);

    // Compute matched filter: mf = H^H * y [2x1]
    cuFloatComplex mf[2] = {make_cuFloatComplex(0.0f, 0.0f), make_cuFloatComplex(0.0f, 0.0f)};

    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        // mf[layer] = sum conj(H[p][layer]) * y[p]
        mf[0] = cuCaddf(mf[0], cuCmulf(cuConjf(H[p][0]), y[p]));
        mf[1] = cuCaddf(mf[1], cuCmulf(cuConjf(H[p][1]), y[p]));
    }

    // Equalization: eq = G^{-1} * mf
    float inv_scaling = 1.0f / tx_scaling;

    cuFloatComplex eq0 = cuCaddf(cuCmulf(G_inv[0], mf[0]), cuCmulf(G_inv[1], mf[1]));
    cuFloatComplex eq1 = cuCaddf(cuCmulf(G_inv[2], mf[0]), cuCmulf(G_inv[3], mf[1]));

    if constexpr (ALGORITHM == EQUALIZER_MMSE || ALGORITHM == EQUALIZER_MMSE_IRC) {
        float c0 = 1.0f - noise_reg * G_inv[0].x;
        float c1 = 1.0f - noise_reg * G_inv[3].x;
        float corr0 = (c0 > 1e-10f) ? (1.0f / c0) : 0.0f;
        float corr1 = (c1 > 1e-10f) ? (1.0f / c1) : 0.0f;
        eq_symbols[0] = make_cuFloatComplex(eq0.x * inv_scaling * corr0, eq0.y * inv_scaling * corr0);
        eq_symbols[1] = make_cuFloatComplex(eq1.x * inv_scaling * corr1, eq1.y * inv_scaling * corr1);
        eq_noise_vars[0] = (corr0 > 1.0f) ? (corr0 - 1.0f) : EQ_NOISE_VAR_DEEP_FADE;
        eq_noise_vars[1] = (corr1 > 1.0f) ? (corr1 - 1.0f) : EQ_NOISE_VAR_DEEP_FADE;
    } else {
        eq_symbols[0] = make_cuFloatComplex(eq0.x * inv_scaling, eq0.y * inv_scaling);
        eq_symbols[1] = make_cuFloatComplex(eq1.x * inv_scaling, eq1.y * inv_scaling);

        float noise_scale = noise_var_est / tx_scaling_sq;
        eq_noise_vars[0] = (noise_scale * G_inv[0].x > 1e-10f)
            ? noise_scale * G_inv[0].x : EQ_NOISE_VAR_DEEP_FADE;
        eq_noise_vars[1] = (noise_scale * G_inv[3].x > 1e-10f)
            ? noise_scale * G_inv[3].x : EQ_NOISE_VAR_DEEP_FADE;
    }
}

__device__ __forceinline__ cuFloatComplex cuCnegf_local(cuFloatComplex a)
{
    return make_cuFloatComplex(-a.x, -a.y);
}

__device__ __forceinline__ cuFloatComplex cuCdivf_regularized(cuFloatComplex a, cuFloatComplex b, float reg)
{
    float denom = b.x * b.x + b.y * b.y + reg;
    return make_cuFloatComplex((a.x * b.x + a.y * b.y) / denom,
                               (a.y * b.x - a.x * b.y) / denom);
}

__device__ __forceinline__ bool mimo_invert_4x4_gauss_jordan_inplace(
    cuFloatComplex* in,
    cuFloatComplex* G_inv)
{
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        G_inv[i] = make_cuFloatComplex(0.0f, 0.0f);
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        G_inv[i * 4 + i] = make_cuFloatComplex(1.0f, 0.0f);
    }

    bool valid = true;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        cuFloatComplex pivot = in[i * 4 + i];
        float pivot_norm = pivot.x * pivot.x + pivot.y * pivot.y;
        valid = valid && (pivot_norm > 0.0f) && isfinite(pivot_norm);

        float inv_pivot_norm = (pivot_norm > 0.0f) ? (1.0f / pivot_norm) : 0.0f;
        cuFloatComplex pivot_inv = make_cuFloatComplex(pivot.x * inv_pivot_norm, -pivot.y * inv_pivot_norm);

        #pragma unroll
        for (int j = i + 1; j < 4; j++) {
            in[i * 4 + j] = cuCmulf(in[i * 4 + j], pivot_inv);
        }

        #pragma unroll
        for (int j = 0; j <= i; j++) {
            G_inv[i * 4 + j] = cuCmulf(G_inv[i * 4 + j], pivot_inv);
        }

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            if (k != i) {
                cuFloatComplex factor = in[k * 4 + i];

                #pragma unroll
                for (int j = i; j < 4; j++) {
                    in[k * 4 + j] = cuCsubf(in[k * 4 + j], cuCmulf(factor, in[i * 4 + j]));
                }

                #pragma unroll
                for (int j = 0; j < i; j++) {
                    G_inv[k * 4 + j] = cuCsubf(G_inv[k * 4 + j], cuCmulf(factor, G_inv[i * 4 + j]));
                }

                G_inv[k * 4 + i] = cuCnegf_local(cuCmulf(factor, G_inv[i * 4 + i]));
            }
        }
    }

    return valid;
}

__device__ __forceinline__ void mimo_invert_3x3(
    const cuFloatComplex* G,
    cuFloatComplex* G_inv,
    float reg = 1e-6f)
{
    cuFloatComplex a = G[0], b = G[1], c = G[2];
    cuFloatComplex d = G[3], e = G[4], f = G[5];
    cuFloatComplex g = G[6], h = G[7], i = G[8];

    cuFloatComplex ei_fh = cuCsubf(cuCmulf(e, i), cuCmulf(f, h));
    cuFloatComplex di_fg = cuCsubf(cuCmulf(d, i), cuCmulf(f, g));
    cuFloatComplex dh_eg = cuCsubf(cuCmulf(d, h), cuCmulf(e, g));
    cuFloatComplex det = cuCaddf(cuCsubf(cuCmulf(a, ei_fh), cuCmulf(b, di_fg)), cuCmulf(c, dh_eg));

    cuFloatComplex cof0 = ei_fh;
    cuFloatComplex cof1 = cuCnegf_local(di_fg);
    cuFloatComplex cof2 = dh_eg;
    cuFloatComplex cof3 = cuCnegf_local(cuCsubf(cuCmulf(b, i), cuCmulf(c, h)));
    cuFloatComplex cof4 = cuCsubf(cuCmulf(a, i), cuCmulf(c, g));
    cuFloatComplex cof5 = cuCnegf_local(cuCsubf(cuCmulf(a, h), cuCmulf(b, g)));
    cuFloatComplex cof6 = cuCsubf(cuCmulf(b, f), cuCmulf(c, e));
    cuFloatComplex cof7 = cuCnegf_local(cuCsubf(cuCmulf(a, f), cuCmulf(c, d)));
    cuFloatComplex cof8 = cuCsubf(cuCmulf(a, e), cuCmulf(b, d));

    G_inv[0] = cuCdivf_regularized(cof0, det, reg);
    G_inv[1] = cuCdivf_regularized(cof3, det, reg);
    G_inv[2] = cuCdivf_regularized(cof6, det, reg);
    G_inv[3] = cuCdivf_regularized(cof1, det, reg);
    G_inv[4] = cuCdivf_regularized(cof4, det, reg);
    G_inv[5] = cuCdivf_regularized(cof7, det, reg);
    G_inv[6] = cuCdivf_regularized(cof2, det, reg);
    G_inv[7] = cuCdivf_regularized(cof5, det, reg);
    G_inv[8] = cuCdivf_regularized(cof8, det, reg);
}

template <int NOF_PORTS, int ALGORITHM>
__device__ __forceinline__ void equalize_symbol_3layer(
    const cuFloatComplex* y,
    const cuFloatComplex H[NOF_PORTS][3],
    const float* noise_vars,
    float tx_scaling,
    cuFloatComplex eq_symbols[3],
    float eq_noise_vars[3])
{
    float noise_var_est = 0.0f;
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        noise_var_est = fmaxf(noise_var_est, noise_vars[p]);
    }

    float tx_scaling_sq = tx_scaling * tx_scaling;
    float noise_reg = noise_var_est / tx_scaling_sq;

    cuFloatComplex G[9];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int j = i; j < 3; j++) {
            cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
            #pragma unroll
            for (int p = 0; p < NOF_PORTS; p++) {
                sum = cuCaddf(sum, cuCmulf(cuConjf(H[p][i]), H[p][j]));
            }
            if constexpr (ALGORITHM == EQUALIZER_MMSE || ALGORITHM == EQUALIZER_MMSE_IRC) {
                if (i == j) {
                    sum.x += noise_reg;
                }
            }
            G[i * 3 + j] = sum;
            if (i != j) {
                G[j * 3 + i] = cuConjf(sum);
            }
        }
    }

    cuFloatComplex G_inv[9];
    mimo_invert_3x3(G, G_inv, 1e-6f);

    cuFloatComplex mf[3];
    #pragma unroll
    for (int l = 0; l < 3; l++) {
        mf[l] = make_cuFloatComplex(0.0f, 0.0f);
    }

    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        #pragma unroll
        for (int l = 0; l < 3; l++) {
            mf[l] = cuCaddf(mf[l], cuCmulf(cuConjf(H[p][l]), y[p]));
        }
    }

    float inv_scaling = 1.0f / tx_scaling;

    #pragma unroll
    for (int l = 0; l < 3; l++) {
        cuFloatComplex eq = make_cuFloatComplex(0.0f, 0.0f);
        #pragma unroll
        for (int k = 0; k < 3; k++) {
            eq = cuCaddf(eq, cuCmulf(G_inv[l * 3 + k], mf[k]));
        }
        if constexpr (ALGORITHM == EQUALIZER_MMSE || ALGORITHM == EQUALIZER_MMSE_IRC) {
            float c = 1.0f - noise_reg * G_inv[l * 3 + l].x;
            float corr = (c > 1e-10f) ? (1.0f / c) : 0.0f;
            eq_symbols[l] = make_cuFloatComplex(eq.x * inv_scaling * corr, eq.y * inv_scaling * corr);
            eq_noise_vars[l] = (corr > 1.0f) ? (corr - 1.0f) : EQ_NOISE_VAR_DEEP_FADE;
        } else {
            float noise_scale = noise_var_est / tx_scaling_sq;
            eq_symbols[l] = make_cuFloatComplex(eq.x * inv_scaling, eq.y * inv_scaling);
            eq_noise_vars[l] = (noise_scale * G_inv[l * 3 + l].x > 1e-10f)
                ? noise_scale * G_inv[l * 3 + l].x : EQ_NOISE_VAR_DEEP_FADE;
        }
    }
}

/**
 * Equalize a single RE for 4-layer MIMO using ZF or MMSE.
 *
 * @tparam NOF_PORTS Number of receive ports (must be >= 4)
 * @tparam ALGORITHM Equalization algorithm (ZF or MMSE)
 * @param y Received symbols per port [NOF_PORTS]
 * @param H Channel matrix H[port][layer] with shape [NOF_PORTS][4]
 * @param noise_vars Per-port noise variances [NOF_PORTS]
 * @param tx_scaling TX amplitude scaling
 * @param eq_symbols Output equalized symbols [4]
 * @param eq_noise_vars Output per-layer noise variances [4]
 */
template <int NOF_PORTS, int ALGORITHM>
__device__ __forceinline__ void equalize_symbol_4layer(
    const cuFloatComplex* y,
    const cuFloatComplex H[NOF_PORTS][4],
    const float* noise_vars,
    float tx_scaling,
    cuFloatComplex eq_symbols[4],
    float eq_noise_vars[4])
{
    // CPU multi-layer equalizer uses the most pessimistic per-port estimate.
    float noise_var_est = 0.0f;
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        noise_var_est = fmaxf(noise_var_est, noise_vars[p]);
    }

    float tx_scaling_sq = tx_scaling * tx_scaling;
    float noise_reg = noise_var_est / tx_scaling_sq;

    // Compute Gram matrix G = H^H * H [4x4 Hermitian]
    cuFloatComplex G[16];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = i; j < 4; j++) {
            cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
            #pragma unroll
            for (int p = 0; p < NOF_PORTS; p++) {
                sum = cuCaddf(sum, cuCmulf(cuConjf(H[p][i]), H[p][j]));
            }
            // For MMSE, add noise to diagonal
            if constexpr (ALGORITHM == EQUALIZER_MMSE || ALGORITHM == EQUALIZER_MMSE_IRC) {
                if (i == j) {
                    sum.x += noise_reg;
                }
            }
            G[i * 4 + j] = sum;
            if (i != j) {
                G[j * 4 + i] = cuConjf(sum);  // Lower triangle
            }
        }
    }

    // Invert Gram matrix with Gauss-Jordan elimination. This avoids the
    // determinant-damped block inverse losing gain near ill-conditioned REs.
    cuFloatComplex G_inv[16];
    bool valid_inverse = mimo_invert_4x4_gauss_jordan_inplace(G, G_inv);

    // Compute matched filter: mf = H^H * y [4x1]
    cuFloatComplex mf[4];
    #pragma unroll
    for (int l = 0; l < 4; l++) {
        mf[l] = make_cuFloatComplex(0.0f, 0.0f);
    }

    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            mf[l] = cuCaddf(mf[l], cuCmulf(cuConjf(H[p][l]), y[p]));
        }
    }

    // Equalization: eq = G^{-1} * mf
    float inv_scaling = 1.0f / tx_scaling;

    #pragma unroll
    for (int l = 0; l < 4; l++) {
        cuFloatComplex eq = make_cuFloatComplex(0.0f, 0.0f);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            eq = cuCaddf(eq, cuCmulf(G_inv[l * 4 + k], mf[k]));
        }
        if (!valid_inverse) {
            eq_symbols[l] = make_cuFloatComplex(0.0f, 0.0f);
            eq_noise_vars[l] = EQ_NOISE_VAR_DEEP_FADE;
            continue;
        }
        if constexpr (ALGORITHM == EQUALIZER_MMSE || ALGORITHM == EQUALIZER_MMSE_IRC) {
            float c = 1.0f - noise_reg * G_inv[l * 4 + l].x;
            float corr = (c > 1e-10f) ? (1.0f / c) : 0.0f;
            eq_symbols[l] = make_cuFloatComplex(eq.x * inv_scaling * corr, eq.y * inv_scaling * corr);
            eq_noise_vars[l] = (corr > 1.0f) ? (corr - 1.0f) : EQ_NOISE_VAR_DEEP_FADE;
        } else {
            eq_symbols[l] = make_cuFloatComplex(eq.x * inv_scaling, eq.y * inv_scaling);

            float noise_scale = noise_var_est / tx_scaling_sq;
            eq_noise_vars[l] = (noise_scale * G_inv[l * 4 + l].x > 1e-10f)
                ? noise_scale * G_inv[l * 4 + l].x : EQ_NOISE_VAR_DEEP_FADE;
        }
    }
}

/* ============================================================================
 * OPTIMIZED FUSED KERNELS (FP16 output, parallel DMRS processing)
 * ============================================================================ */

/**
 * Fused LSE + Frequency Interpolation kernel with FP16 output.
 *
 * This kernel combines DMRS LSE estimation and frequency interpolation into a
 * single kernel, outputting FP16 channel estimates for 50% memory savings.
 *
 * Processing: One PRB per thread block, all 12 REs output per PRB.
 * - DMRS REs (0,2,4,6,8,10): Direct LSE from received signal
 * - Data REs (1,3,5,7,9,11): Linear interpolation between adjacent DMRS
 *
 * Input: RX grid in cbf16 format
 * Output: Channel estimates in FP16 complex (__half2) for all 12 REs/PRB
 */
__global__ void kernel_fused_lse_freq_interp_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const unsigned int* __restrict__ d_dmrs_seq,  /* Packed Gold sequence bits */
    __half2* __restrict__ d_estimates_fp16,       /* FP16 complex output [prb, port, 12] */
    int nof_prb,
    int nof_ports,
    int grid_stride,           /* nof_symbols * nof_subcarriers */
    int symbol_stride,         /* nof_subcarriers */
    int start_prb,
    int dmrs_symbol_idx,       /* Which symbol in the slot (absolute) */
    float dmrs_scaling)
{
    int prb_idx = blockIdx.x;
    int port = threadIdx.x;

    if (prb_idx >= nof_prb || port >= nof_ports) return;

    /* Type 1 DMRS: REs at subcarriers 0, 2, 4, 6, 8, 10 within PRB */
    int prb_start_sc = (start_prb + prb_idx) * 12;

    /* DMRS RE indices within PRB for Type 1 */
    const int dmrs_sc_offset[6] = {0, 2, 4, 6, 8, 10};

    float inv_sqrt2 = 0.70710678118f;
    float h_normalizer = 1.0f / dmrs_scaling;

    /* Compute LSE at all 6 DMRS positions first */
    __half2 h_dmrs[6];

    #pragma unroll
    for (int d = 0; d < 6; d++) {
        int sc = prb_start_sc + dmrs_sc_offset[d];

        /* Load received symbol */
        int grid_idx = port * grid_stride + dmrs_symbol_idx * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        /* Generate DMRS pilot from Gold sequence */
        int pilot_idx = (start_prb + prb_idx) * 6 + d;
        int bit_idx_real = 2 * pilot_idx;
        int bit_idx_imag = 2 * pilot_idx + 1;

        int word_idx_real = bit_idx_real / 32;
        int bit_pos_real = 31 - (bit_idx_real % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_real = (d_dmrs_seq[word_idx_real] >> bit_pos_real) & 1;

        int word_idx_imag = bit_idx_imag / 32;
        int bit_pos_imag = 31 - (bit_idx_imag % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_imag = (d_dmrs_seq[word_idx_imag] >> bit_pos_imag) & 1;

        /* QPSK mapping */
        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        /* LSE: h = y × conj(pilot) / dmrs_scaling */
        float h_real = (y.x * p_real + y.y * p_imag) * h_normalizer;
        float h_imag = (y.y * p_real - y.x * p_imag) * h_normalizer;

        /* Store as FP16 complex */
        h_dmrs[d] = __halves2half2(__float2half(h_real), __float2half(h_imag));
    }

    /* Output base index: [prb, port, sc_in_prb] */
    int out_base = prb_idx * (nof_ports * 12) + port * 12;

    /* Now output all 12 REs with interpolation for odd subcarriers */
    #pragma unroll
    for (int sc = 0; sc < 12; sc++) {
        __half2 h_out;
        if (sc % 2 == 0) {
            /* DMRS position - direct copy */
            h_out = h_dmrs[sc / 2];
        } else {
            /* Interpolate between adjacent DMRS */
            int left_idx = sc / 2;
            int right_idx = (left_idx + 1 < 6) ? left_idx + 1 : 5;

            /* Linear interpolation in FP16 */
            __half2 h_left = h_dmrs[left_idx];
            __half2 h_right = h_dmrs[right_idx];

            /* Average: (left + right) / 2 */
            __half2 h_sum = __hadd2(h_left, h_right);
            h_out = __hmul2(h_sum, __float2half2_rn(0.5f));
        }
        d_estimates_fp16[out_base + sc] = h_out;
    }
}

/**
 * Parallel DMRS symbol processing: Process all DMRS symbols in one kernel launch.
 *
 * This kernel processes all DMRS symbols in parallel using 2D grid indexing:
 * - blockIdx.x = PRB index
 * - blockIdx.y = DMRS symbol index
 * - threadIdx.x = port index
 *
 * Output: FP16 estimates for all DMRS symbols [dmrs_sym, prb, port, 12]
 */
__global__ void kernel_parallel_dmrs_lse_freq_interp_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const unsigned int* __restrict__ d_dmrs_seq,    /* Single pre-generated sequence */
    const int* __restrict__ dmrs_symbol_indices,    /* Array of DMRS symbol indices */
    __half2* __restrict__ d_estimates_fp16,         /* FP16 output [dmrs_sym, prb, port, 12] */
    cuFloatComplex* __restrict__ d_dmrs_received,   /* Store received DMRS symbols [dmrs_sym, prb, port, 6] */
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling)
{
    int prb_idx = blockIdx.x;
    int dmrs_sym_idx = blockIdx.y;
    int port = threadIdx.x;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols || port >= nof_ports) return;

    int dmrs_symbol = dmrs_symbol_indices[dmrs_sym_idx];

    int prb_start_sc = (start_prb + prb_idx) * 12;
    const int dmrs_sc_offset[6] = {0, 2, 4, 6, 8, 10};

    float inv_sqrt2 = 0.70710678118f;
    float h_normalizer = 1.0f / dmrs_scaling;

    /* Compute LSE at all 6 DMRS positions */
    __half2 h_dmrs[6];
    cuFloatComplex y_dmrs[6];  /* Store received symbols for noise var computation */

    #pragma unroll
    for (int d = 0; d < 6; d++) {
        int sc = prb_start_sc + dmrs_sc_offset[d];
        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);
        y_dmrs[d] = y;  /* Store for noise variance */

        int pilot_idx = (start_prb + prb_idx) * 6 + d;
        int bit_idx_real = 2 * pilot_idx;
        int bit_idx_imag = 2 * pilot_idx + 1;

        int word_idx_real = bit_idx_real / 32;
        int bit_pos_real = 31 - (bit_idx_real % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_real = (d_dmrs_seq[word_idx_real] >> bit_pos_real) & 1;

        int word_idx_imag = bit_idx_imag / 32;
        int bit_pos_imag = 31 - (bit_idx_imag % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_imag = (d_dmrs_seq[word_idx_imag] >> bit_pos_imag) & 1;

        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        float h_real = (y.x * p_real + y.y * p_imag) * h_normalizer;
        float h_imag = (y.y * p_real - y.x * p_imag) * h_normalizer;

        h_dmrs[d] = __halves2half2(__float2half(h_real), __float2half(h_imag));
    }

    /* Output estimates: [dmrs_sym, prb, port, 12] */
    int re_per_dmrs_sym = nof_prb * nof_ports * 12;
    int out_base = dmrs_sym_idx * re_per_dmrs_sym + prb_idx * (nof_ports * 12) + port * 12;

    #pragma unroll
    for (int sc = 0; sc < 12; sc++) {
        __half2 h_out;
        if (sc % 2 == 0) {
            h_out = h_dmrs[sc / 2];
        } else {
            int left_idx = sc / 2;
            int right_idx = (left_idx + 1 < 6) ? left_idx + 1 : 5;
            __half2 h_left = h_dmrs[left_idx];
            __half2 h_right = h_dmrs[right_idx];
            __half2 h_sum = __hadd2(h_left, h_right);
            h_out = __hmul2(h_sum, __float2half2_rn(0.5f));
        }
        d_estimates_fp16[out_base + sc] = h_out;
    }

    /* Output received DMRS symbols: [dmrs_sym, prb, port, 6] */
    if (d_dmrs_received) {
        int recv_per_dmrs_sym = nof_prb * nof_ports * 6;
        int recv_base = dmrs_sym_idx * recv_per_dmrs_sym + prb_idx * (nof_ports * 6) + port * 6;
        #pragma unroll
        for (int d = 0; d < 6; d++) {
            d_dmrs_received[recv_base + d] = y_dmrs[d];
        }
    }
}

/* ============================================================================
 * MIMO CHANNEL ESTIMATION WITH OCC DESPREADING
 *
 * For multi-layer MIMO, DMRS layers are separated using OCC (Orthogonal Cover
 * Codes) within CDM groups. For Type 1 DMRS:
 *
 * 2-layer MIMO (CDM Group 0 only):
 *   - Layer 0: OCC [+1, +1] on consecutive DMRS RE pairs
 *   - Layer 1: OCC [+1, -1] on consecutive DMRS RE pairs
 *   - h0 = (lse[2k] + lse[2k+1]) / 2
 *   - h1 = (lse[2k] - lse[2k+1]) / 2
 *
 * 4-layer MIMO (CDM Groups 0 and 1):
 *   - CDM Group 0 (subcarriers 0,2,4,6,8,10): Layers 0,1
 *   - CDM Group 1 (subcarriers 1,3,5,7,9,11): Layers 2,3
 *   - Same OCC patterns within each group
 * ============================================================================ */

/**
 * 2-layer MIMO LSE kernel with OCC despreading.
 *
 * This kernel processes all DMRS symbols in parallel and outputs per-layer
 * channel estimates for 2-layer MIMO configurations.
 *
 * Grid: (nof_prb, nof_dmrs_symbols)
 * Block: (nof_ports)
 *
 * Output layout: [dmrs_sym, prb, port, layer, 6_dmrs_re] -> interpolated to 12 RE
 * For efficiency, we output interpolated [dmrs_sym, prb, port, layer, 12_re]
 */
__global__ void kernel_mimo_lse_2layer_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const uint32_t* __restrict__ d_dmrs_seq,       /* Packed Gold sequence bits */
    const int* __restrict__ dmrs_symbol_indices,   /* Array of DMRS symbol indices */
    __half2* __restrict__ d_estimates_fp16,        /* FP16 output [dmrs_sym, prb, port, layer, 12] */
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling,
    int words_per_dmrs_sym_stride,
    int dmrs_type)
{
    int prb_idx = blockIdx.x;
    int dmrs_sym_idx = blockIdx.y;
    int port = threadIdx.x;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols || port >= nof_ports) return;

    int dmrs_symbol = dmrs_symbol_indices[dmrs_sym_idx];
    int prb_start_sc = (start_prb + prb_idx) * 12;
    const uint32_t* d_dmrs_seq_symbol = d_dmrs_seq + dmrs_sym_idx * words_per_dmrs_sym_stride;

    int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);

    float inv_sqrt2 = 0.70710678118f;
    float h_normalizer = 1.0f / dmrs_scaling;

    /* Compute raw LSE at all 6 DMRS positions first */
    cuFloatComplex lse[6];

    #pragma unroll
    for (int d = 0; d < nof_pilots; d++) {
        int sc = prb_start_sc + pusch_dmrs_sc_offset(dmrs_type, 0, d);
        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        int pilot_idx = (start_prb + prb_idx) * nof_pilots + d;
        int word_idx = pilot_idx / 16;
        int bit_offset = (pilot_idx % 16) * 2;
        uint32_t pilot_word = d_dmrs_seq_symbol[word_idx];
        int c_real = (pilot_word >> bit_offset) & 1;
        int c_imag = (pilot_word >> (bit_offset + 1)) & 1;

        /* QPSK pilot generation */
        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        /* LSE: h = y × conj(pilot) / dmrs_scaling */
        lse[d] = make_cuFloatComplex(
            (y.x * p_real + y.y * p_imag) * h_normalizer,
            (y.y * p_real - y.x * p_imag) * h_normalizer);
    }

    /* OCC despreading for 2-layer MIMO.
     * DMRS Type 1, 2-layer uses OCC in frequency domain:
     *   - Layer 0: OCC [+1, +1] → h0[k] = (lse[2k] + lse[2k+1]) / 2
     *   - Layer 1: OCC [+1, -1] → h1[k] = (lse[2k] - lse[2k+1]) / 2
     * where k = 0,1,2 (3 OCC pairs from 6 DMRS REs) */
    __half2 h_layer0[6], h_layer1[6];  /* Per-layer estimates at DMRS positions */

    #pragma unroll
    for (int k = 0; k < 3; k++) {
        if (2 * k + 1 >= nof_pilots) break;
        cuFloatComplex lse_even = lse[2 * k];
        cuFloatComplex lse_odd = lse[2 * k + 1];

        /* Layer 0: (even + odd) / 2 */
        cuFloatComplex h0 = make_cuFloatComplex(
            (lse_even.x + lse_odd.x) * 0.5f,
            (lse_even.y + lse_odd.y) * 0.5f);

        /* Layer 1: (even - odd) / 2 */
        cuFloatComplex h1 = make_cuFloatComplex(
            (lse_even.x - lse_odd.x) * 0.5f,
            (lse_even.y - lse_odd.y) * 0.5f);

        /* Store at both DMRS positions (OCC gives same estimate for paired REs) */
        __half2 h0_fp16 = __halves2half2(__float2half(h0.x), __float2half(h0.y));
        __half2 h1_fp16 = __halves2half2(__float2half(h1.x), __float2half(h1.y));

        h_layer0[2 * k] = h0_fp16;
        h_layer0[2 * k + 1] = h0_fp16;
        h_layer1[2 * k] = h1_fp16;
        h_layer1[2 * k + 1] = h1_fp16;
    }

    /* Output layout: [dmrs_sym, prb, port, layer, 12]
     * Total per dmrs_sym: nof_prb * nof_ports * 2_layers * 12 */
    int re_per_layer = 12;
    int re_per_port = 2 * re_per_layer;  /* 2 layers */
    int re_per_prb = nof_ports * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    int out_base = dmrs_sym_idx * re_per_dmrs_sym +
                   prb_idx * re_per_prb +
                   port * re_per_port;

    /* Output with frequency interpolation for odd subcarriers */
    #pragma unroll
    for (int sc = 0; sc < 12; sc++) {
        __half2 h0_out, h1_out;

        h0_out = interpolate_dmrs_pilots_half2(h_layer0, nof_pilots, dmrs_type, 0, sc);
        h1_out = interpolate_dmrs_pilots_half2(h_layer1, nof_pilots, dmrs_type, 0, sc);

        d_estimates_fp16[out_base + sc] = h0_out;              /* Layer 0 */
        d_estimates_fp16[out_base + re_per_layer + sc] = h1_out;  /* Layer 1 */
    }
}

__device__ __forceinline__ __half2 interpolate_mimo_dmrs_pilot_half2(
    const __half2* __restrict__ pilots,
    int sc,
    int cdm_offset)
{
    if (cdm_offset == 0) {
        if ((sc & 1) == 0) {
            return pilots[sc / 2];
        }
        int left = sc / 2;
        int right = (left + 1 < 6) ? left + 1 : 5;
        return __hmul2(__hadd2(pilots[left], pilots[right]), __float2half2_rn(0.5f));
    }

    if (sc <= 1) {
        return pilots[0];
    }
    if ((sc & 1) == 1) {
        return pilots[sc / 2];
    }
    int left = sc / 2 - 1;
    int right = sc / 2;
    return __hmul2(__hadd2(pilots[left], pilots[right]), __float2half2_rn(0.5f));
}

/**
 * 3-layer MIMO LSE kernel with OCC despreading.
 *
 * Layers 0 and 1 use CDM group 0. Layer 2 uses CDM group 1 with the +,+ OCC
 * branch; layer 3 from the 4-layer pattern is absent.
 */
__global__ void kernel_mimo_lse_3layer_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const uint32_t* __restrict__ d_dmrs_seq,
    const int* __restrict__ dmrs_symbol_indices,
    __half2* __restrict__ d_estimates_fp16,
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling,
    int words_per_dmrs_sym_stride,
    int dmrs_type)
{
    int prb_idx = blockIdx.x;
    int dmrs_sym_idx = blockIdx.y;
    int port = threadIdx.x;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols || port >= nof_ports) return;

    int dmrs_symbol = dmrs_symbol_indices[dmrs_sym_idx];
    int prb_start_sc = (start_prb + prb_idx) * 12;
    const uint32_t* d_dmrs_seq_symbol = d_dmrs_seq + dmrs_sym_idx * words_per_dmrs_sym_stride;

    float inv_sqrt2 = 0.70710678118f;
    float h_normalizer = 1.0f / dmrs_scaling;

    int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);
    cuFloatComplex lse_cdm0[6], lse_cdm1[6];

    #pragma unroll
    for (int d = 0; d < 6; d++) {
        if (d >= nof_pilots) break;
        int pilot_idx = (start_prb + prb_idx) * nof_pilots + d;
        int word_idx = pilot_idx / 16;
        int bit_offset = (pilot_idx % 16) * 2;
        uint32_t pilot_word = d_dmrs_seq_symbol[word_idx];
        int c_real = (pilot_word >> bit_offset) & 1;
        int c_imag = (pilot_word >> (bit_offset + 1)) & 1;

        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        int sc0 = prb_start_sc + pusch_dmrs_sc_offset(dmrs_type, 0, d);
        int grid_idx0 = port * grid_stride + dmrs_symbol * symbol_stride + sc0;
        cuFloatComplex y0 = cbf16_to_fp32(d_grid_cbf16[grid_idx0]);
        lse_cdm0[d] = make_cuFloatComplex(
            (y0.x * p_real + y0.y * p_imag) * h_normalizer,
            (y0.y * p_real - y0.x * p_imag) * h_normalizer);

        int sc1 = prb_start_sc + pusch_dmrs_sc_offset(dmrs_type, 1, d);
        int grid_idx1 = port * grid_stride + dmrs_symbol * symbol_stride + sc1;
        cuFloatComplex y1 = cbf16_to_fp32(d_grid_cbf16[grid_idx1]);
        lse_cdm1[d] = make_cuFloatComplex(
            (y1.x * p_real + y1.y * p_imag) * h_normalizer,
            (y1.y * p_real - y1.x * p_imag) * h_normalizer);
    }

    __half2 h_layer[3][6];
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        if (2 * k + 1 >= nof_pilots) break;
        cuFloatComplex lse0_even = lse_cdm0[2 * k];
        cuFloatComplex lse0_odd = lse_cdm0[2 * k + 1];
        cuFloatComplex h0 = make_cuFloatComplex((lse0_even.x + lse0_odd.x) * 0.5f,
                                                (lse0_even.y + lse0_odd.y) * 0.5f);
        cuFloatComplex h1 = make_cuFloatComplex((lse0_even.x - lse0_odd.x) * 0.5f,
                                                (lse0_even.y - lse0_odd.y) * 0.5f);

        cuFloatComplex lse1_even = lse_cdm1[2 * k];
        cuFloatComplex lse1_odd = lse_cdm1[2 * k + 1];
        cuFloatComplex h2 = make_cuFloatComplex((lse1_even.x + lse1_odd.x) * 0.5f,
                                                (lse1_even.y + lse1_odd.y) * 0.5f);

        __half2 h0_fp16 = __halves2half2(__float2half(h0.x), __float2half(h0.y));
        __half2 h1_fp16 = __halves2half2(__float2half(h1.x), __float2half(h1.y));
        __half2 h2_fp16 = __halves2half2(__float2half(h2.x), __float2half(h2.y));

        h_layer[0][2 * k] = h0_fp16;
        h_layer[0][2 * k + 1] = h0_fp16;
        h_layer[1][2 * k] = h1_fp16;
        h_layer[1][2 * k + 1] = h1_fp16;
        h_layer[2][2 * k] = h2_fp16;
        h_layer[2][2 * k + 1] = h2_fp16;
    }

    int re_per_layer = 12;
    int re_per_port = 3 * re_per_layer;
    int re_per_prb = nof_ports * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;
    int out_base = dmrs_sym_idx * re_per_dmrs_sym + prb_idx * re_per_prb + port * re_per_port;

    #pragma unroll
    for (int layer = 0; layer < 3; layer++) {
        int layer_base = out_base + layer * re_per_layer;
        int cdm_offset = layer / 2;
        #pragma unroll
        for (int sc = 0; sc < 12; sc++) {
            d_estimates_fp16[layer_base + sc] =
                interpolate_dmrs_pilots_half2(h_layer[layer], nof_pilots, dmrs_type, cdm_offset, sc);
        }
    }
}

/**
 * 4-layer MIMO LSE kernel with OCC despreading.
 *
 * For 4-layer MIMO, we use both CDM groups:
 *   - CDM Group 0 (subcarriers 0,2,4,6,8,10): Layers 0,1
 *   - CDM Group 1 (subcarriers 1,3,5,7,9,11): Layers 2,3
 *
 * Each CDM group uses the same OCC pattern as 2-layer.
 *
 * Output layout: [dmrs_sym, prb, port, layer, 12_re]
 */
__global__ void kernel_mimo_lse_4layer_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const uint32_t* __restrict__ d_dmrs_seq,       /* Packed Gold sequence bits */
    const int* __restrict__ dmrs_symbol_indices,   /* Array of DMRS symbol indices */
    __half2* __restrict__ d_estimates_fp16,        /* FP16 output [dmrs_sym, prb, port, layer, 12] */
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling,
    int words_per_dmrs_sym_stride,
    int dmrs_type)
{
    int prb_idx = blockIdx.x;
    int dmrs_sym_idx = blockIdx.y;
    int port = threadIdx.x;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols || port >= nof_ports) return;

    int dmrs_symbol = dmrs_symbol_indices[dmrs_sym_idx];
    int prb_start_sc = (start_prb + prb_idx) * 12;
    const uint32_t* d_dmrs_seq_symbol = d_dmrs_seq + dmrs_sym_idx * words_per_dmrs_sym_stride;

    float inv_sqrt2 = 0.70710678118f;
    float h_normalizer = 1.0f / dmrs_scaling;

    int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);

    /* Compute raw LSE at all 12 DMRS positions */
    cuFloatComplex lse_cdm0[6], lse_cdm1[6];

    /* CDM Group 0 (layers 0,1) */
    #pragma unroll
    for (int d = 0; d < 6; d++) {
        if (d >= nof_pilots) break;
        int sc = prb_start_sc + pusch_dmrs_sc_offset(dmrs_type, 0, d);
        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        /* CDM Group 0 pilots: indices 0,1,2,3,4,5 */
        int pilot_idx = (start_prb + prb_idx) * nof_pilots + d;
        int word_idx = pilot_idx / 16;
        int bit_offset = (pilot_idx % 16) * 2;
        uint32_t pilot_word = d_dmrs_seq_symbol[word_idx];
        int c_real = (pilot_word >> bit_offset) & 1;
        int c_imag = (pilot_word >> (bit_offset + 1)) & 1;

        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        lse_cdm0[d] = make_cuFloatComplex(
            (y.x * p_real + y.y * p_imag) * h_normalizer,
            (y.y * p_real - y.x * p_imag) * h_normalizer);
    }

    /* CDM Group 1 (layers 2,3) - uses different pilot sequence offset.
     * For Type 1 DMRS, CDM group 1 pilots are at indices 6-11 (offset by 6 from CDM0).
     * NOTE: This is a simplification; full 3GPP spec may require different scrambling. */
    #pragma unroll
    for (int d = 0; d < 6; d++) {
        if (d >= nof_pilots) break;
        int sc = prb_start_sc + pusch_dmrs_sc_offset(dmrs_type, 1, d);
        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        int pilot_idx = (start_prb + prb_idx) * nof_pilots + d;
        int word_idx = pilot_idx / 16;
        int bit_offset = (pilot_idx % 16) * 2;
        uint32_t pilot_word = d_dmrs_seq_symbol[word_idx];
        int c_real = (pilot_word >> bit_offset) & 1;
        int c_imag = (pilot_word >> (bit_offset + 1)) & 1;

        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        lse_cdm1[d] = make_cuFloatComplex(
            (y.x * p_real + y.y * p_imag) * h_normalizer,
            (y.y * p_real - y.x * p_imag) * h_normalizer);
    }

    /* OCC despreading for all 4 layers */
    __half2 h_layer[4][6];  /* [layer][dmrs_re] */

    /* CDM Group 0 → Layers 0,1 */
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        if (2 * k + 1 >= nof_pilots) break;
        cuFloatComplex lse_even = lse_cdm0[2 * k];
        cuFloatComplex lse_odd = lse_cdm0[2 * k + 1];

        cuFloatComplex h0 = make_cuFloatComplex(
            (lse_even.x + lse_odd.x) * 0.5f,
            (lse_even.y + lse_odd.y) * 0.5f);
        cuFloatComplex h1 = make_cuFloatComplex(
            (lse_even.x - lse_odd.x) * 0.5f,
            (lse_even.y - lse_odd.y) * 0.5f);

        __half2 h0_fp16 = __halves2half2(__float2half(h0.x), __float2half(h0.y));
        __half2 h1_fp16 = __halves2half2(__float2half(h1.x), __float2half(h1.y));

        h_layer[0][2 * k] = h0_fp16;
        h_layer[0][2 * k + 1] = h0_fp16;
        h_layer[1][2 * k] = h1_fp16;
        h_layer[1][2 * k + 1] = h1_fp16;
    }

    /* CDM Group 1 → Layers 2,3 */
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        if (2 * k + 1 >= nof_pilots) break;
        cuFloatComplex lse_even = lse_cdm1[2 * k];
        cuFloatComplex lse_odd = lse_cdm1[2 * k + 1];

        cuFloatComplex h2 = make_cuFloatComplex(
            (lse_even.x + lse_odd.x) * 0.5f,
            (lse_even.y + lse_odd.y) * 0.5f);
        cuFloatComplex h3 = make_cuFloatComplex(
            (lse_even.x - lse_odd.x) * 0.5f,
            (lse_even.y - lse_odd.y) * 0.5f);

        __half2 h2_fp16 = __halves2half2(__float2half(h2.x), __float2half(h2.y));
        __half2 h3_fp16 = __halves2half2(__float2half(h3.x), __float2half(h3.y));

        h_layer[2][2 * k] = h2_fp16;
        h_layer[2][2 * k + 1] = h2_fp16;
        h_layer[3][2 * k] = h3_fp16;
        h_layer[3][2 * k + 1] = h3_fp16;
    }

    /* Output layout: [dmrs_sym, prb, port, layer, 12] */
    int re_per_layer = 12;
    int re_per_port = 4 * re_per_layer;  /* 4 layers */
    int re_per_prb = nof_ports * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    int out_base = dmrs_sym_idx * re_per_dmrs_sym +
                   prb_idx * re_per_prb +
                   port * re_per_port;

    /* Output with frequency interpolation for odd subcarriers */
    #pragma unroll
    for (int layer = 0; layer < 4; layer++) {
        int layer_base = out_base + layer * re_per_layer;
        int cdm_offset = layer / 2;

        #pragma unroll
        for (int sc = 0; sc < 12; sc++) {
            d_estimates_fp16[layer_base + sc] =
                interpolate_dmrs_pilots_half2(h_layer[layer], nof_pilots, dmrs_type, cdm_offset, sc);
        }
    }
}

/* ============================================================================
 * ULTRA-OPTIMIZED KERNELS: Fused LSE+NoiseVar and On-the-fly Scrambling
 *
 * These kernels eliminate additional kernel launches and memory traffic:
 * 1. Fused LSE: Computes channel estimates AND noise variance in one pass
 * 2. On-the-fly scrambling: Generates Gold sequence bits directly instead
 *    of reading from pre-generated buffer
 * ============================================================================ */

/**
 * Compute DMRS c_init on device (matches dmrs_compute_c_init).
 * TS 38.211 6.4.1.1.1.1:
 *   c_init = (2^17 × (14 × n_s + l + 1) × (2×N_ID + 1) + 2×N_ID + n_SCID) mod 2^31
 */
__device__ __forceinline__ uint32_t compute_dmrs_c_init_device(
    int slot_idx, int symbol_idx, uint32_t scrambling_id, int n_scid)
{
    uint64_t term1 = (1ULL << 17) * (14ULL * slot_idx + symbol_idx + 1) * (2ULL * scrambling_id + 1);
    uint64_t term2 = 2ULL * scrambling_id + n_scid;
    return static_cast<uint32_t>((term1 + term2) & 0x7FFFFFFF);
}

/**
 * Pre-generate DMRS pilot bits in parallel.
 *
 * Each thread generates 16 pilot pairs (32 pilots = 64 bits = 2 words).
 * Uses LFSR jump to starting position, then sequential generation.
 *
 * Output format: packed bits where bit 2n = real_sign, bit 2n+1 = imag_sign
 * Pilot value = (1 - 2*real_sign) + j*(1 - 2*imag_sign)) / sqrt(2)
 *
 * Grid: (nof_words/2 + 255) / 256, Threads: 256
 */
__global__ void kernel_generate_dmrs_pilots(
    uint32_t* __restrict__ d_pilot_bits,
    uint32_t c_init,
    int nof_pilots)  /* Total pilots = nof_prb * 6 per DMRS symbol */
{
    int word_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;  /* Each thread does 2 words (32 pilots) */
    int pilot_start = word_idx * 16;  /* 16 pilots per word (2 bits each) */

    if (pilot_start >= nof_pilots) return;

    /* Initialize LFSRs and jump to starting position */
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;
    int bit_start = pilot_start * 2;  /* 2 bits per pilot */
    int total_advance = NC_SKIP + bit_start;
    x1 = advance_x1_local(x1, total_advance);
    x2 = advance_x2_local(x2, total_advance);

    /* Generate up to 32 pilots (64 bits = 2 words) */
    uint32_t word0 = 0, word1 = 0;
    int pilots_remaining = nof_pilots - pilot_start;
    int pilots_to_gen = min(32, pilots_remaining);

    for (int p = 0; p < pilots_to_gen; p++) {
        /* Generate 2 bits: real_sign, imag_sign */
        uint32_t c_real = (x1 ^ x2) & 1;
        uint32_t new_x1 = ((x1 >> 3) ^ x1) & 1;
        x1 = (x1 >> 1) | (new_x1 << 30);
        uint32_t new_x2 = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
        x2 = (x2 >> 1) | (new_x2 << 30);

        uint32_t c_imag = (x1 ^ x2) & 1;
        new_x1 = ((x1 >> 3) ^ x1) & 1;
        x1 = (x1 >> 1) | (new_x1 << 30);
        new_x2 = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
        x2 = (x2 >> 1) | (new_x2 << 30);

        /* Pack into words: 2 bits per pilot, 16 pilots per word */
        int bit_pos = (p % 16) * 2;
        if (p < 16) {
            word0 |= (c_real << bit_pos) | (c_imag << (bit_pos + 1));
        } else {
            word1 |= (c_real << bit_pos) | (c_imag << (bit_pos + 1));
        }
    }

    d_pilot_bits[word_idx] = word0;
    if (word_idx + 1 < (nof_pilots * 2 + 31) / 32) {
        d_pilot_bits[word_idx + 1] = word1;
    }
}

/**
 * Batched version of kernel_generate_dmrs_pilots: generates pilots for ALL
 * (slot, symbol) pairs in a single kernel launch. blockIdx.y indexes the
 * flattened (slot * 14 + symbol) pair; blockIdx.x + threadIdx.x index
 * within a single symbol's pilots, identical to the original kernel.
 *
 * Eliminates 279 kernel launch overheads per configure call (280 → 1).
 */
__global__ void kernel_generate_dmrs_pilots_batched(
    uint32_t* __restrict__ d_pilot_bits,
    int nof_pilots,
    int words_per_sym,
    int words_per_slot,
    int nof_slots,
    uint32_t scrambling_id,
    int n_scid)
{
    int pair_idx = blockIdx.y;
    int slot = pair_idx / 14;
    int sym  = pair_idx % 14;
    if (slot >= nof_slots) return;

    /* Compute c_init per TS 38.211 §6.4.1.1.1.1 */
    uint64_t term1 = (1ULL << 17) * (14ULL * slot + sym + 1) * (2ULL * scrambling_id + 1);
    uint64_t term2 = 2ULL * scrambling_id + n_scid;
    uint32_t c_init = static_cast<uint32_t>((term1 + term2) & 0x7FFFFFFF);

    int word_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int pilot_start = word_idx * 16;
    if (pilot_start >= nof_pilots) return;

    /* Initialize LFSRs and jump to starting position */
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;
    int bit_start = pilot_start * 2;
    int total_advance = NC_SKIP + bit_start;
    x1 = advance_x1_local(x1, total_advance);
    x2 = advance_x2_local(x2, total_advance);

    /* Generate up to 32 pilots (64 bits = 2 words) */
    uint32_t word0 = 0, word1 = 0;
    int pilots_remaining = nof_pilots - pilot_start;
    int pilots_to_gen = min(32, pilots_remaining);

    for (int p = 0; p < pilots_to_gen; p++) {
        uint32_t c_real = (x1 ^ x2) & 1;
        uint32_t new_x1 = ((x1 >> 3) ^ x1) & 1;
        x1 = (x1 >> 1) | (new_x1 << 30);
        uint32_t new_x2 = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
        x2 = (x2 >> 1) | (new_x2 << 30);

        uint32_t c_imag = (x1 ^ x2) & 1;
        new_x1 = ((x1 >> 3) ^ x1) & 1;
        x1 = (x1 >> 1) | (new_x1 << 30);
        new_x2 = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
        x2 = (x2 >> 1) | (new_x2 << 30);

        int bit_pos = (p % 16) * 2;
        if (p < 16) {
            word0 |= (c_real << bit_pos) | (c_imag << (bit_pos + 1));
        } else {
            word1 |= (c_real << bit_pos) | (c_imag << (bit_pos + 1));
        }
    }

    uint32_t* dst = d_pilot_bits + slot * words_per_slot + sym * words_per_sym;
    dst[word_idx] = word0;
    if (word_idx + 1 < (nof_pilots * 2 + 31) / 32) {
        dst[word_idx + 1] = word1;
    }
}

/**
 * Gather DMRS pilot bits from the all-14-symbol precomputed layout into a
 * contiguous per-DMRS-symbol buffer.  Replaces 2-4 cudaMemcpyAsync D2D calls
 * per slot with a single kernel launch.
 *
 * Grid:  nof_dmrs_symbols blocks x 1
 * Block: words_per_sym threads (rounded up to warp multiple, max 128)
 */
__global__ void kernel_gather_dmrs_pilot_bits(
    uint32_t* __restrict__       d_dst,
    const uint32_t* __restrict__ d_src,
    const int* __restrict__      d_symbol_indices,
    int                          words_per_sym)
{
    int word = threadIdx.x;
    if (word >= words_per_sym) return;
    int sym = d_symbol_indices[blockIdx.x];
    d_dst[blockIdx.x * words_per_sym + word] = d_src[sym * words_per_sym + word];
}

/**
 * ULTRA-FAST LSE kernel with PRE-COMPUTED DMRS pilots.
 *
 * This version reads pilot bits from a pre-generated buffer instead of
 * computing them on-the-fly. Eliminates O(log N) LFSR advancement per PRB.
 *
 * Expected speedup: ~100x (from 340µs to ~3µs for 100 MHz)
 */
__global__ void kernel_lse_precomputed_pilots_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const uint32_t* __restrict__ d_pilot_bits,  /* Pre-computed pilot bits */
    const int* __restrict__ dmrs_symbol_indices,
    __half2* __restrict__ d_estimates_fp16,
    float* __restrict__ d_noise_var_accum,
    unsigned int* __restrict__ d_noise_count,
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling,
    int precomputed_words_per_dmrs_sym)  /* uint32_t words per DMRS symbol (word-aligned) */
{
    /* Thread layout: 1 thread per PRB per port */
    int prb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dmrs_sym_idx = blockIdx.y;
    int port = blockIdx.z;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols || port >= nof_ports) return;

    int dmrs_symbol = dmrs_symbol_indices[dmrs_sym_idx];
    int prb_start_sc = (start_prb + prb_idx) * 12;
    const int dmrs_sc_offset[6] = {0, 2, 4, 6, 8, 10};

    float inv_sqrt2 = 0.70710678118f;
    float h_normalizer = 1.0f / dmrs_scaling;

    __half2 h_dmrs[6];
    float local_noise_sum = 0.0f;

    /* Process all 6 DMRS pilots for this PRB */
    #pragma unroll
    for (int d = 0; d < 6; d++) {
        int sc = prb_start_sc + dmrs_sc_offset[d];
        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        /* Read pilot bits from pre-computed buffer.
         * CRITICAL: Pilots are stored with word-aligned boundaries per DMRS symbol.
         * Use precomputed_words_per_dmrs_sym (in uint32_t words) for the DMRS symbol stride,
         * not pilots_per_symbol. The buffer layout is:
         *   [DMRS sym 0: words 0..W-1] [DMRS sym 1: words W..2W-1] ...
         * where W = words_per_dmrs_sym = ceil(pilots_per_symbol * 2 / 32). */
        int pilot_idx = (start_prb + prb_idx) * 6 + d;  /* Pilot index within this DMRS symbol */
        int dmrs_word_offset = dmrs_sym_idx * precomputed_words_per_dmrs_sym;
        int word_idx_within_dmrs = pilot_idx / 16;  /* 16 pilots per word */
        int word_idx = dmrs_word_offset + word_idx_within_dmrs;
        int bit_offset = (pilot_idx % 16) * 2;
        uint32_t pilot_word = d_pilot_bits[word_idx];
        int c_real = (pilot_word >> bit_offset) & 1;
        int c_imag = (pilot_word >> (bit_offset + 1)) & 1;

        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        /* LSE: h = y × conj(pilot) / dmrs_scaling */
        float h_real = (y.x * p_real + y.y * p_imag) * h_normalizer;
        float h_imag = (y.y * p_real - y.x * p_imag) * h_normalizer;

        h_dmrs[d] = __halves2half2(__float2half(h_real), __float2half(h_imag));

        /* Compute noise residual */
        float h_scaled_real = h_real * dmrs_scaling;
        float h_scaled_imag = h_imag * dmrs_scaling;
        float y_ideal_real = h_scaled_real * p_real - h_scaled_imag * p_imag;
        float y_ideal_imag = h_scaled_real * p_imag + h_scaled_imag * p_real;
        float res_real = y.x - y_ideal_real;
        float res_imag = y.y - y_ideal_imag;
        local_noise_sum += res_real * res_real + res_imag * res_imag;
    }

    /* Frequency interpolation: output 12 estimates from 6 pilots */
    int re_per_dmrs_sym = nof_prb * nof_ports * 12;
    int out_base = dmrs_sym_idx * re_per_dmrs_sym + prb_idx * (nof_ports * 12) + port * 12;

    #pragma unroll
    for (int sc = 0; sc < 12; sc++) {
        __half2 h_out;
        if (sc % 2 == 0) {
            h_out = h_dmrs[sc / 2];
        } else {
            int left_idx = sc / 2;
            int right_idx = (left_idx + 1 < 6) ? left_idx + 1 : 5;
            __half2 h_left = h_dmrs[left_idx];
            __half2 h_right = h_dmrs[right_idx];
            __half2 h_sum = __hadd2(h_left, h_right);
            h_out = __hmul2(h_sum, __float2half2_rn(0.5f));
        }
        d_estimates_fp16[out_base + sc] = h_out;
    }

    /* Atomic accumulate noise variance */
    atomicAdd(&d_noise_var_accum[port], local_noise_sum);

    /* Count total DMRS REs (only one thread does this) */
    if (prb_idx == 0 && dmrs_sym_idx == 0 && port == 0) {
        atomicAdd(d_noise_count, nof_prb * nof_dmrs_symbols * 6);
    }
}

/**
 * Low-PAPR DMRS LSE kernel for transform precoding (MSG3/DFT-s-OFDM).
 *
 * This kernel uses complex-valued low-PAPR pilots instead of QPSK pilots.
 * The pilots are pre-generated by kernel_generate_low_papr_pilots.
 *
 * @param d_grid_cbf16 Input grid in cbf16 format
 * @param d_low_papr_pilots Pre-generated low-PAPR pilot symbols [nof_pilots_per_sym]
 * @param dmrs_symbol_indices DMRS symbol indices
 * @param d_estimates_fp16 Output channel estimates in FP16 format
 * @param d_noise_var_accum Per-port noise variance accumulator
 * @param d_noise_count Total DMRS RE count
 * @param nof_prb Number of PRBs
 * @param nof_ports Number of receive ports
 * @param nof_dmrs_symbols Number of DMRS symbols
 * @param grid_stride Grid stride (symbols * subcarriers)
 * @param symbol_stride Symbol stride (subcarriers)
 * @param start_prb Starting PRB index
 * @param dmrs_scaling DMRS amplitude scaling factor
 */
__global__ void kernel_lse_low_papr_pilots_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const cuFloatComplex* __restrict__ d_low_papr_pilots,
    const int* __restrict__ dmrs_symbol_indices,
    __half2* __restrict__ d_estimates_fp16,
    float* __restrict__ d_noise_var_accum,
    unsigned int* __restrict__ d_noise_count,
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling)
{
    /* Thread layout: 1 thread per PRB per port (same as standard LSE kernel) */
    int prb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dmrs_sym_idx = blockIdx.y;
    int port = blockIdx.z;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols || port >= nof_ports) return;

    int dmrs_symbol = dmrs_symbol_indices[dmrs_sym_idx];
    int prb_start_sc = (start_prb + prb_idx) * 12;

    /* For low-PAPR (transform precoding), DMRS uses all 12 subcarriers per PRB,
     * not alternating like Type 1 DMRS. The pilots are at every subcarrier. */
    float h_normalizer = 1.0f / dmrs_scaling;

    __half2 h_dmrs[12];
    float local_noise_sum = 0.0f;

    /* Process all 12 DMRS pilots for this PRB (low-PAPR uses all subcarriers) */
    #pragma unroll
    for (int d = 0; d < 12; d++) {
        int sc = prb_start_sc + d;
        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        /* Get the complex pilot from pre-generated low-PAPR sequence.
         * Pilot index: prb_idx * 12 + d (12 pilots per PRB, contiguous) */
        int pilot_idx = prb_idx * 12 + d;
        cuFloatComplex p = d_low_papr_pilots[pilot_idx];

        /* LSE: h = y × conj(pilot) / dmrs_scaling */
        float h_real = (y.x * p.x + y.y * p.y) * h_normalizer;
        float h_imag = (y.y * p.x - y.x * p.y) * h_normalizer;

        h_dmrs[d] = __halves2half2(__float2half(h_real), __float2half(h_imag));

        /* Compute noise residual */
        float h_scaled_real = h_real * dmrs_scaling;
        float h_scaled_imag = h_imag * dmrs_scaling;
        float y_ideal_real = h_scaled_real * p.x - h_scaled_imag * p.y;
        float y_ideal_imag = h_scaled_real * p.y + h_scaled_imag * p.x;
        float res_real = y.x - y_ideal_real;
        float res_imag = y.y - y_ideal_imag;
        local_noise_sum += res_real * res_real + res_imag * res_imag;
    }

    /* Output channel estimates (no frequency interpolation needed for low-PAPR,
     * as we have estimates at all 12 subcarriers) */
    int re_per_dmrs_sym = nof_prb * nof_ports * 12;
    int out_base = dmrs_sym_idx * re_per_dmrs_sym + prb_idx * (nof_ports * 12) + port * 12;

    #pragma unroll
    for (int sc = 0; sc < 12; sc++) {
        d_estimates_fp16[out_base + sc] = h_dmrs[sc];
    }

    /* Atomic accumulate noise variance */
    atomicAdd(&d_noise_var_accum[port], local_noise_sum);

    /* Count total DMRS REs (only one thread does this) */
    if (prb_idx == 0 && dmrs_sym_idx == 0 && port == 0) {
        atomicAdd(d_noise_count, nof_prb * nof_dmrs_symbols * 12);  /* 12 pilots/PRB for low-PAPR */
    }
}

/**
 * ULTRA-OPTIMIZED: Fused LSE + Freq Interp + Noise with ON-THE-FLY DMRS generation.
 * WARP-OPTIMIZED VERSION: Multiple PRBs per block for better GPU utilization.
 *
 * Optimization strategy:
 * - 32 threads per block (full warp), each thread handles 1 PRB completely
 * - No shared memory needed, no __syncthreads()
 * - Each thread: LFSR advance → 6 pilots → 12 outputs (all sequential)
 * - Block-level reduction for noise variance
 *
 * Grid: (ceil(nof_prb/32), nof_dmrs_symbols), Threads: min(32, nof_prb) * nof_ports
 */
__global__ void kernel_ultra_fused_lse_onthefly_dmrs_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const int* __restrict__ dmrs_symbol_indices,
    __half2* __restrict__ d_estimates_fp16,
    float* __restrict__ d_noise_var_accum,
    unsigned int* __restrict__ d_noise_count,
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling,
    int slot_idx,
    uint32_t scrambling_id,
    int n_scid)
{
    // Thread layout: Each thread handles one complete PRB
    int tid = threadIdx.x;
    int port = tid / 32;
    int lane = tid % 32;  // Lane within the port's warp
    int prb_idx = blockIdx.x * 32 + lane;
    int dmrs_sym_idx = blockIdx.y;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols || port >= nof_ports) return;

    int dmrs_symbol = dmrs_symbol_indices[dmrs_sym_idx];
    int prb_start_sc = (start_prb + prb_idx) * 12;
    const int dmrs_sc_offset[6] = {0, 2, 4, 6, 8, 10};

    // Compute c_init for this DMRS symbol
    uint32_t c_init = compute_dmrs_c_init_device(slot_idx, dmrs_symbol, scrambling_id, n_scid);

    // Initialize LFSRs and advance to starting pilot position
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;
    int first_pilot_bit = (start_prb + prb_idx) * 6 * 2;
    int total_advance = NC_SKIP + first_pilot_bit;
    x1 = advance_x1_local(x1, total_advance);
    x2 = advance_x2_local(x2, total_advance);

    float inv_sqrt2 = 0.70710678118f;
    float h_normalizer = 1.0f / dmrs_scaling;

    // Store pilot estimates in registers (no shared memory needed)
    __half2 h_dmrs[6];
    float local_noise_sum = 0.0f;

    // Compute all 6 pilots sequentially
    #pragma unroll
    for (int d = 0; d < 6; d++) {
        int sc = prb_start_sc + dmrs_sc_offset[d];
        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        // Generate DMRS pilot bits on-the-fly
        int c_real = (x1 ^ x2) & 1;
        x1 = step_x1_local(x1);
        x2 = step_x2_local(x2);
        int c_imag = (x1 ^ x2) & 1;
        x1 = step_x1_local(x1);
        x2 = step_x2_local(x2);

        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        // LSE: h = y × conj(pilot) / dmrs_scaling
        float h_real = (y.x * p_real + y.y * p_imag) * h_normalizer;
        float h_imag = (y.y * p_real - y.x * p_imag) * h_normalizer;

        h_dmrs[d] = __halves2half2(__float2half(h_real), __float2half(h_imag));

        // Compute noise residual
        float h_scaled_real = h_real * dmrs_scaling;
        float h_scaled_imag = h_imag * dmrs_scaling;
        float y_ideal_real = h_scaled_real * p_real - h_scaled_imag * p_imag;
        float y_ideal_imag = h_scaled_real * p_imag + h_scaled_imag * p_real;
        float res_real = y.x - y_ideal_real;
        float res_imag = y.y - y_ideal_imag;
        local_noise_sum += res_real * res_real + res_imag * res_imag;
    }

    // Frequency interpolation and output (12 outputs from 6 pilots)
    int re_per_dmrs_sym = nof_prb * nof_ports * 12;
    int out_base = dmrs_sym_idx * re_per_dmrs_sym + prb_idx * (nof_ports * 12) + port * 12;

    #pragma unroll
    for (int sc = 0; sc < 12; sc++) {
        __half2 h_out;
        if (sc % 2 == 0) {
            h_out = h_dmrs[sc / 2];
        } else {
            int left_idx = sc / 2;
            int right_idx = (left_idx + 1 < 6) ? left_idx + 1 : 5;
            __half2 h_left = h_dmrs[left_idx];
            __half2 h_right = h_dmrs[right_idx];
            __half2 h_sum = __hadd2(h_left, h_right);
            h_out = __hmul2(h_sum, __float2half2_rn(0.5f));
        }
        d_estimates_fp16[out_base + sc] = h_out;
    }

    // Atomic accumulate noise variance
    atomicAdd(&d_noise_var_accum[port], local_noise_sum);

    if (lane == 0 && blockIdx.x == 0 && dmrs_sym_idx == 0) {
        atomicAdd(d_noise_count, nof_prb * nof_dmrs_symbols * 6);
    }
}

/**
 * Fused LSE + Freq Interpolation + Noise Variance Accumulation
 *
 * This kernel performs channel estimation AND accumulates noise variance
 * residuals in a single pass, eliminating the separate noise variance kernel.
 *
 * Grid: (nof_prb, nof_dmrs_symbols), Threads: nof_ports
 * Output: FP16 estimates + atomically accumulated noise variance
 */
__global__ void kernel_fused_lse_freq_interp_noise_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const uint32_t* __restrict__ d_dmrs_c_inits,  /* Per-DMRS-symbol c_init array */
    const int* __restrict__ dmrs_symbol_indices,
    __half2* __restrict__ d_estimates_fp16,
    float* __restrict__ d_noise_var_accum,     /* Per-port atomic accumulator */
    unsigned int* __restrict__ d_noise_count,  /* Total DMRS REs processed (single atomic) */
    float* __restrict__ d_rsrp_accum,          /* Per-port RSRP accumulator (|H|²) */
    float* __restrict__ d_epre_accum,          /* Per-port EPRE accumulator (|y|²) */
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling,
    int dmrs_type)
{
    int prb_idx = blockIdx.x;
    int dmrs_sym_idx = blockIdx.y;
    int port = threadIdx.x;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols || port >= nof_ports) return;

    int dmrs_symbol = dmrs_symbol_indices[dmrs_sym_idx];
    int prb_start_sc = (start_prb + prb_idx) * 12;
    int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);

    /* DMRS pilot amplitude: Use the actual TX amplitude including power boost.
     * TX sends pilots with amplitude = dmrs_scaling * 1/sqrt(2).
     * For LSE: h = y × conj(p) / |p|^2 where |p|^2 = (dmrs_scaling * 0.707)^2 * 2 = dmrs_scaling^2.
     * Note: This matches the original code behavior when dmrs_scaling=1. */
    float pilot_amp = dmrs_scaling * 0.70710678118f;
    float pilot_pwr = dmrs_scaling * dmrs_scaling;  // |p|^2
    float h_normalizer = 1.0f / pilot_pwr;

    /* Compute LSE at all 6 DMRS positions, also accumulate noise, RSRP, EPRE */
    __half2 h_dmrs[6];
    float local_noise_sum = 0.0f;
    float local_rsrp_sum = 0.0f;
    float local_epre_sum = 0.0f;

    /* Inline DMRS gold sequence generation: advance LFSR to this PRB's first pilot bit */
    uint32_t dmrs_c_init = d_dmrs_c_inits[dmrs_sym_idx];
    int bit_start = (start_prb + prb_idx) * nof_pilots * 2;
    uint32_t x1 = advance_x1_local(1, NC_SKIP + bit_start);
    uint32_t x2 = advance_x2_local(dmrs_c_init, NC_SKIP + bit_start);

    #pragma unroll
    for (int d = 0; d < nof_pilots; d++) {
        int sc = prb_start_sc + pusch_dmrs_sc_offset(dmrs_type, 0, d);
        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        /* Generate DMRS pilot bits inline from LFSR */
        int c_real = (x1 ^ x2) & 1; x1 = step_x1_local(x1); x2 = step_x2_local(x2);
        int c_imag = (x1 ^ x2) & 1; x1 = step_x1_local(x1); x2 = step_x2_local(x2);

        float p_real = (1 - 2 * c_real) * pilot_amp;
        float p_imag = (1 - 2 * c_imag) * pilot_amp;

        /* LSE: h = y × conj(pilot) / |pilot|^2 */
        float h_real = (y.x * p_real + y.y * p_imag) * h_normalizer;
        float h_imag = (y.y * p_real - y.x * p_imag) * h_normalizer;

        h_dmrs[d] = __halves2half2(__float2half(h_real), __float2half(h_imag));

        /* Accumulate RSRP: |H|² per DMRS pilot */
        local_rsrp_sum += h_real * h_real + h_imag * h_imag;

        /* Accumulate EPRE: |y|² per DMRS pilot */
        local_epre_sum += y.x * y.x + y.y * y.y;

        /* Compute noise residual: |y - h * pilot|^2
         * Since h = y × conj(p) / |p|^2, we have h × p = y (ideally).
         * For noise estimation with multiple DMRS, use cross-symbol averaging.
         * For now, compute the residual which will be near-zero for pure LSE. */
        float y_ideal_real = h_real * p_real - h_imag * p_imag;
        float y_ideal_imag = h_real * p_imag + h_imag * p_real;
        float res_real = y.x - y_ideal_real;
        float res_imag = y.y - y_ideal_imag;
        local_noise_sum += res_real * res_real + res_imag * res_imag;
    }

    /* Output estimates: [dmrs_sym, prb, port, 12] */
    int re_per_dmrs_sym = nof_prb * nof_ports * 12;
    int out_base = dmrs_sym_idx * re_per_dmrs_sym + prb_idx * (nof_ports * 12) + port * 12;

    #pragma unroll
    for (int sc = 0; sc < 12; sc++) {
        __half2 h_out;
        h_out = interpolate_dmrs_pilots_half2(h_dmrs, nof_pilots, dmrs_type, 0, sc);
        d_estimates_fp16[out_base + sc] = h_out;
    }

    /* Atomic accumulate noise variance (per-port) */
    atomicAdd(&d_noise_var_accum[port], local_noise_sum);

    /* Atomic accumulate RSRP and EPRE (per-port) */
    atomicAdd(&d_rsrp_accum[port], local_rsrp_sum);
    atomicAdd(&d_epre_accum[port], local_epre_sum);

    /* Count total DMRS REs (only one thread per port needs to do this) */
    if (prb_idx == 0 && dmrs_sym_idx == 0) {
        atomicAdd(d_noise_count, nof_prb * nof_dmrs_symbols * nof_pilots);
    }
}

/**
 * Finalize noise variance after fused LSE kernel.
 * Simple kernel to divide accumulated sum by count and apply scaling.
 */
__global__ void kernel_finalize_noise_variance(
    float* __restrict__ d_noise_var_accum,
    const unsigned int* __restrict__ d_noise_count,
    int nof_ports,
    float dmrs_scaling,
    int nof_dmrs_symbols)
{
    int port = threadIdx.x;
    if (port >= nof_ports) return;

    unsigned int count = *d_noise_count;
    float sum = d_noise_var_accum[port];

    /* Normalize */
    float divisor = (count > 1) ? (float)(count - 1) : 1.0f;
    float noise_var = sum / divisor;

    /* Apply same scaling as original kernel */
    float N = (float)nof_dmrs_symbols;
    float beta_sq = dmrs_scaling * dmrs_scaling;
    float base_scale = beta_sq / sqrtf(N);
    float alloc_factor = (count < 50) ? 1.2f : 1.0f;

    /* Noise variance floor for fallback path (when CPU noise not provided).
     * The GPU DMRS-based noise estimation is unreliable because h is estimated from
     * the same y samples (so |y - h*pilot| ≈ 0). Use a conservative floor. */
    d_noise_var_accum[port] = fmaxf(noise_var * base_scale * alloc_factor, 0.001f);
}

/**
 * CROSS-VALIDATION NOISE ESTIMATION KERNEL
 *
 * Computes noise variance by comparing channel estimates across DMRS symbols.
 * This is the CORRECT way to estimate noise - the previous method computed
 * |y - h*pilot| where h = y/pilot, which gives zero!
 *
 * For N DMRS symbols with estimates h[0], h[1], ..., h[N-1]:
 *   h[i] = h_true + n[i]   where n[i] ~ CN(0, σ²)
 *   h_mean = mean(h[i])
 *   variance = mean(|h[i] - h_mean|²) = σ² * (N-1)/N
 *   Therefore: σ² = variance * N / (N-1)
 *
 * For the common case of N=2:
 *   |h[0] - h[1]|² = |n[0] - n[1]|² has E[·] = 2σ²
 *   So σ² = 0.5 * |h[0] - h[1]|²
 *
 * Grid: (nof_prb, 1), Threads: 12 * nof_ports
 * Each thread handles one (prb, port, sc) triplet, sums over DMRS symbols.
 */
__global__ void kernel_cross_validation_noise_variance(
    const __half2* __restrict__ d_estimates_fp16,
    float* __restrict__ d_noise_var_accum,
    const unsigned int* __restrict__ d_grid_cbf16,
    const uint32_t* __restrict__ d_dmrs_pilot_bits,
    const int* __restrict__ d_dmrs_symbol_indices,
    float* __restrict__ d_sinr_noise_accum,
    float* __restrict__ d_sinr_rsrp_accum,
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling,
    int words_per_dmrs_sym,
    const float* __restrict__ d_cfo_hz_ptr,
    const float* __restrict__ d_symbol_start_times,
    const float2* __restrict__ d_cfo_phasors,
    /* Fused finalize parameters (eliminates separate kernel launch) */
    float* __restrict__ d_noise_vars_out,
    unsigned int* __restrict__ d_cv_done_counter,
    const float* __restrict__ d_epre_accum,
    float* __restrict__ d_sinr_epre_result,
    int noise_mode,  /* 0=cross-validation (default), 1=pilot-residual */
    int dmrs_type)
{
    int prb_idx = blockIdx.x;
    int local_idx = threadIdx.x;  /* 0..12*nof_ports-1 */
    int port = local_idx / 12;
    int sc = local_idx % 12;

    if (prb_idx >= nof_prb || port >= nof_ports) return;

    /* CFO de-rotation phasors: exp(-j·2π·cfo·t_d) per DMRS symbol.
     * Thread 0 computes sincosf into shared memory; all threads use it. */
    __shared__ float s_derot_cos[4];  /* max 4 DMRS symbols */
    __shared__ float s_derot_sin[4];

    if (local_idx == 0) {
        float cfo_hz = (d_cfo_hz_ptr != nullptr) ? *d_cfo_hz_ptr : 0.0f;
        for (int d = 0; d < nof_dmrs_symbols && d < 4; d++) {
            if (d_cfo_hz_ptr != nullptr && d_cfo_phasors != nullptr) {
                float2 ph = d_cfo_phasors[MIMO_CFO_RESIDUAL_PHASOR_OFFSET + d];
                s_derot_cos[d] = ph.x;
                s_derot_sin[d] = ph.y;
            } else if (d_cfo_hz_ptr != nullptr && d_symbol_start_times != nullptr && d_dmrs_symbol_indices != nullptr) {
                float t_d = d_symbol_start_times[d_dmrs_symbol_indices[d]];
                sincosf(-6.2831853071795864f * cfo_hz * t_d, &s_derot_sin[d], &s_derot_cos[d]);
            } else {
                s_derot_cos[d] = 1.0f;
                s_derot_sin[d] = 0.0f;
            }
        }
    }
    __syncthreads();

    /* Estimate layout: [dmrs_sym, prb, port, 12] */
    int re_per_dmrs_sym = nof_prb * nof_ports * 12;
    int base_idx = prb_idx * (nof_ports * 12) + port * 12 + sc;

    /* FAST PATH: 2 DMRS symbols (most common case) */
    if (nof_dmrs_symbols == 2) {
        __half2 h0 = d_estimates_fp16[0 * re_per_dmrs_sym + base_idx];
        __half2 h1 = d_estimates_fp16[1 * re_per_dmrs_sym + base_idx];

        /* Convert to FP32 */
        float h0_real = __half2float(__low2half(h0));
        float h0_imag = __half2float(__high2half(h0));
        float h1_real = __half2float(__low2half(h1));
        float h1_imag = __half2float(__high2half(h1));

        /* De-rotate each estimate by exp(-j·2π·cfo·t_d) to remove CFO phase */
        float h0r_d = h0_real * s_derot_cos[0] - h0_imag * s_derot_sin[0];
        float h0i_d = h0_real * s_derot_sin[0] + h0_imag * s_derot_cos[0];
        float h1r_d = h1_real * s_derot_cos[1] - h1_imag * s_derot_sin[1];
        float h1i_d = h1_real * s_derot_sin[1] + h1_imag * s_derot_cos[1];

        /* Compute |h0_derot - h1_derot|² — CFO component removed */
        float diff_real = h0r_d - h1r_d;
        float diff_imag = h0i_d - h1i_d;
        float diff_sq = diff_real * diff_real + diff_imag * diff_imag;

        /* σ² = 0.5 * E[|h0 - h1|²] for two independent estimates */
        float noise_contribution = 0.5f * diff_sq;

        atomicAdd(&d_noise_var_accum[port], noise_contribution);

        /* Fused SINR: at pilot positions (sc % 2 == 0), compute residual noise */
        int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);
        int d_pilot = -1;
        for (int p = 0; p < nof_pilots; ++p) {
            if (pusch_dmrs_sc_offset(dmrs_type, 0, p) == sc) {
                d_pilot = p;
                break;
            }
        }

        if ((d_pilot >= 0) && d_sinr_noise_accum != nullptr) {
            /* Coherent average of de-rotated estimates */
            float h_avg_re = (h0r_d + h1r_d) * 0.5f;
            float h_avg_im = (h0i_d + h1i_d) * 0.5f;

            float local_rsrp = h_avg_re * h_avg_re + h_avg_im * h_avg_im;
            float local_noise = 0.0f;
            float pilot_amp = dmrs_scaling * 0.70710678118f;

            int pilot_idx = (start_prb + prb_idx) * nof_pilots + d_pilot;
            int pilot_word_idx = pilot_idx / 16;
            int pilot_bit_offset = (pilot_idx % 16) * 2;

            for (int d = 0; d < 2; d++) {
                int dmrs_symbol = d_dmrs_symbol_indices[d];
                int sc_grid = (start_prb + prb_idx) * 12 + sc;

                int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc_grid;
                cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

                const uint32_t* dmrs_seq = d_dmrs_pilot_bits + d * words_per_dmrs_sym;
                uint32_t pilot_word = dmrs_seq[pilot_word_idx];
                int c_real = (pilot_word >> pilot_bit_offset) & 1;
                int c_imag = (pilot_word >> (pilot_bit_offset + 1)) & 1;

                float p_real = (1 - 2 * c_real) * pilot_amp;
                float p_imag = (1 - 2 * c_imag) * pilot_amp;

                /* pred = h_avg_derot × pilot */
                float pred_real = h_avg_re * p_real - h_avg_im * p_imag;
                float pred_imag = h_avg_re * p_imag + h_avg_im * p_real;

                /* Re-rotate prediction to match received signal phase:
                 * pred *= exp(+j·2π·cfo·t_d) = conjugate of de-rotation.
                 * conj(cos_d + j·sin_d) = cos_d - j·sin_d where sin_d = sin(-2π·cfo·t_d)
                 * So: (a+jb)(cos_d - j·sin_d) = (a·cos_d + b·sin_d) + j(b·cos_d - a·sin_d) */
                float pr = pred_real * s_derot_cos[d] + pred_imag * s_derot_sin[d];
                float pi = pred_imag * s_derot_cos[d] - pred_real * s_derot_sin[d];

                float res_real = y.x - pr;
                float res_imag = y.y - pi;

                local_noise += res_real * res_real + res_imag * res_imag;
            }

            atomicAdd(&d_sinr_noise_accum[port], local_noise);
            atomicAdd(&d_sinr_rsrp_accum[port], local_rsrp);
        }
    }
    /* GENERAL PATH: N DMRS symbols */
    else if (nof_dmrs_symbols > 2) {
        /* First pass: compute de-rotated mean */
        float sum_real = 0.0f, sum_imag = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            __half2 h = d_estimates_fp16[d * re_per_dmrs_sym + base_idx];
            float hr = __half2float(__low2half(h));
            float hi = __half2float(__high2half(h));
            /* De-rotate by exp(-j·2π·cfo·t_d) */
            sum_real += hr * s_derot_cos[d] - hi * s_derot_sin[d];
            sum_imag += hr * s_derot_sin[d] + hi * s_derot_cos[d];
        }
        float mean_real = sum_real / nof_dmrs_symbols;
        float mean_imag = sum_imag / nof_dmrs_symbols;

        /* Second pass: compute variance from de-rotated estimates */
        float var_sum = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            __half2 h = d_estimates_fp16[d * re_per_dmrs_sym + base_idx];
            float hr = __half2float(__low2half(h));
            float hi = __half2float(__high2half(h));
            float dr = (hr * s_derot_cos[d] - hi * s_derot_sin[d]) - mean_real;
            float di = (hr * s_derot_sin[d] + hi * s_derot_cos[d]) - mean_imag;
            var_sum += dr * dr + di * di;
        }

        /* variance = sum / N, then σ² = variance * N / (N-1) */
        float variance = var_sum / nof_dmrs_symbols;
        float noise_est = variance * nof_dmrs_symbols / (nof_dmrs_symbols - 1);

        atomicAdd(&d_noise_var_accum[port], noise_est);

        int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);
        int d_pilot = -1;
        for (int p = 0; p < nof_pilots; ++p) {
            if (pusch_dmrs_sc_offset(dmrs_type, 0, p) == sc) {
                d_pilot = p;
                break;
            }
        }

        if ((d_pilot >= 0) && d_sinr_noise_accum != nullptr) {
            float local_rsrp = mean_real * mean_real + mean_imag * mean_imag;
            float local_noise = 0.0f;
            float pilot_amp = dmrs_scaling * 0.70710678118f;

            int pilot_idx = (start_prb + prb_idx) * nof_pilots + d_pilot;
            int pilot_word_idx = pilot_idx / 16;
            int pilot_bit_offset = (pilot_idx % 16) * 2;

            for (int d = 0; d < nof_dmrs_symbols; d++) {
                int dmrs_symbol = d_dmrs_symbol_indices[d];
                int sc_grid = (start_prb + prb_idx) * 12 + sc;

                int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc_grid;
                cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

                const uint32_t* dmrs_seq = d_dmrs_pilot_bits + d * words_per_dmrs_sym;
                uint32_t pilot_word = dmrs_seq[pilot_word_idx];
                int c_real = (pilot_word >> pilot_bit_offset) & 1;
                int c_imag = (pilot_word >> (pilot_bit_offset + 1)) & 1;

                float p_real = (1 - 2 * c_real) * pilot_amp;
                float p_imag = (1 - 2 * c_imag) * pilot_amp;

                /* pred = h_avg_derot × pilot */
                float pred_real = mean_real * p_real - mean_imag * p_imag;
                float pred_imag = mean_real * p_imag + mean_imag * p_real;

                /* Re-rotate: pred *= exp(+j·2π·cfo·t_d) = conjugate of de-rotation */
                float pr = pred_real * s_derot_cos[d] + pred_imag * s_derot_sin[d];
                float pi = pred_imag * s_derot_cos[d] - pred_real * s_derot_sin[d];

                float res_real = y.x - pr;
                float res_imag = y.y - pi;

                local_noise += res_real * res_real + res_imag * res_imag;
            }

            atomicAdd(&d_sinr_noise_accum[port], local_noise);
            atomicAdd(&d_sinr_rsrp_accum[port], local_rsrp);
        }
    }
    /* Edge case: only 1 DMRS symbol - cannot estimate noise, use floor */
    /* This will be handled by the standalone finalize kernel */

    /* ---- Fused finalize: last-block pattern (same as TA/CFO kernel) ----
     * After all blocks complete their atomicAdds into d_noise_var_accum,
     * the last block normalizes the accumulated noise and writes finalized
     * values to d_noise_vars_out. This eliminates a separate kernel launch. */
    __syncthreads();  /* Ensure all threads in this block are done with atomicAdds */

    if (threadIdx.x == 0 && d_cv_done_counter != nullptr) {
        __threadfence();  /* Make this block's atomicAdds visible to all SMs */
        unsigned int done = atomicAdd(d_cv_done_counter, 1u) + 1;

        if (done == (unsigned int)nof_prb) {
            /* Last block: all CV accumulations are globally visible.
             * Normalize per-port noise variance and write to output. */
            int nof_sc = nof_prb * 12;
            int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);
            int nof_pilot_positions = nof_prb * nof_pilots;
            int nof_dmrs_re_per_port = nof_prb * nof_dmrs_symbols * nof_pilots;

            if (noise_mode == 1 && d_sinr_noise_accum != nullptr && nof_dmrs_re_per_port > 1) {
                /* Pilot-residual noise: σ² = Σ|y - H_avg*pilot|² / (N-1)
                 * More robust under fading where CV overestimates noise. */
                for (int p = 0; p < nof_ports; p++) {
                    float noise_var = d_sinr_noise_accum[p] / (float)(nof_dmrs_re_per_port - 1);
                    d_noise_vars_out[p] = fmaxf(noise_var, 1e-6f);
                }
            } else {
                /* Cross-validation noise (default) */
                for (int p = 0; p < nof_ports; p++) {
                    float noise_var = (nof_sc > 0)
                        ? (d_noise_var_accum[p] / (float)nof_sc) : 0.001f;
                    /* Calibration: CV variance is ~40% of CPU noise */
                    noise_var *= 2.5f;
                    d_noise_vars_out[p] = fmaxf(noise_var, 1e-6f);
                }
            }

            /* Compute SINR/EPRE/RSRP in dB from fused CV accumulators */
            if (d_sinr_epre_result != nullptr && d_sinr_noise_accum != nullptr) {
                float total_signal = 0.0f, total_noise = 0.0f, total_epre = 0.0f;
                for (int p = 0; p < nof_ports; p++) {
                    float signal_power = d_sinr_rsrp_accum[p] / (float)nof_pilot_positions;
                    float sinr_noise_var = d_sinr_noise_accum[p] / (float)(nof_dmrs_re_per_port - 1);
                    sinr_noise_var = fmaxf(sinr_noise_var, signal_power * 1e-10f);
                    float mean_epre_port = d_epre_accum[p] / (float)nof_dmrs_re_per_port;
                    total_signal += signal_power;
                    total_noise += sinr_noise_var;
                    total_epre += mean_epre_port;
                }

                float mean_noise = total_noise / nof_ports;
                d_sinr_epre_result[0] = (mean_noise > 1e-10f)
                    ? 10.0f * log10f(total_signal / mean_noise) : 60.0f;
                float mean_epre = total_epre / nof_ports;
                d_sinr_epre_result[1] = (mean_epre > 1e-10f)
                    ? 10.0f * log10f(mean_epre) : -INFINITY;
                float mean_rsrp = (total_signal / nof_ports) * dmrs_scaling * dmrs_scaling;
                d_sinr_epre_result[2] = (mean_rsrp > 1e-10f)
                    ? 10.0f * log10f(mean_rsrp) : -INFINITY;
            }
        }
    }
}

/**
 * Cross-validation noise estimator for MIMO channel estimates.
 *
 * MIMO LSE kernels store estimates as [dmrs_sym, prb, port, layer, 12].  The
 * single-layer CV kernel above uses [dmrs_sym, prb, port, 12]; using it on MIMO
 * buffers under-strides the DMRS dimension and can compare slices from the same
 * DMRS symbol, making the demodulator overconfident.
 *
 * Grid: (nof_prb, 1), Threads: 12 * nof_ports * nof_layers
 */
__global__ void kernel_cross_validation_noise_variance_mimo(
    const __half2* __restrict__ d_estimates_fp16,
    float* __restrict__ d_noise_var_accum,
    int nof_prb,
    int nof_ports,
    int nof_layers,
    int nof_dmrs_symbols,
    float* __restrict__ d_noise_vars_out,
    unsigned int* __restrict__ d_cv_done_counter)
{
    int prb_idx = blockIdx.x;
    int local_idx = threadIdx.x;
    int sc = local_idx % 12;
    int layer = (local_idx / 12) % nof_layers;
    int port = local_idx / (12 * nof_layers);

    if (prb_idx >= nof_prb || port >= nof_ports || layer >= nof_layers) return;

    int re_per_layer = 12;
    int re_per_port = nof_layers * re_per_layer;
    int re_per_prb = nof_ports * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;
    int base_idx = prb_idx * re_per_prb + port * re_per_port + layer * re_per_layer + sc;

    if (nof_dmrs_symbols == 2) {
        __half2 h0 = d_estimates_fp16[base_idx];
        __half2 h1 = d_estimates_fp16[re_per_dmrs_sym + base_idx];

        float h0_real = __half2float(__low2half(h0));
        float h0_imag = __half2float(__high2half(h0));
        float h1_real = __half2float(__low2half(h1));
        float h1_imag = __half2float(__high2half(h1));

        float diff_real = h0_real - h1_real;
        float diff_imag = h0_imag - h1_imag;
        float diff_sq = diff_real * diff_real + diff_imag * diff_imag;

        atomicAdd(&d_noise_var_accum[port], 0.5f * diff_sq);
    } else if (nof_dmrs_symbols > 2) {
        float sum_real = 0.0f;
        float sum_imag = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            __half2 h = d_estimates_fp16[d * re_per_dmrs_sym + base_idx];
            sum_real += __half2float(__low2half(h));
            sum_imag += __half2float(__high2half(h));
        }
        float mean_real = sum_real / nof_dmrs_symbols;
        float mean_imag = sum_imag / nof_dmrs_symbols;

        float var_sum = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            __half2 h = d_estimates_fp16[d * re_per_dmrs_sym + base_idx];
            float dr = __half2float(__low2half(h)) - mean_real;
            float di = __half2float(__high2half(h)) - mean_imag;
            var_sum += dr * dr + di * di;
        }

        float variance = var_sum / nof_dmrs_symbols;
        float noise_est = variance * nof_dmrs_symbols / (nof_dmrs_symbols - 1);
        atomicAdd(&d_noise_var_accum[port], noise_est);
    }

    __syncthreads();

    if (threadIdx.x == 0 && d_cv_done_counter != nullptr) {
        __threadfence();
        unsigned int done = atomicAdd(d_cv_done_counter, 1u) + 1;

        if (done == (unsigned int)nof_prb) {
            int nof_sc_layer = nof_prb * nof_layers * 12;
            for (int p = 0; p < nof_ports; p++) {
                float noise_var = (nof_sc_layer > 0)
                    ? (d_noise_var_accum[p] / (float)nof_sc_layer) : 0.001f;
                noise_var *= 2.5f;
                d_noise_vars_out[p] = fmaxf(noise_var, 1e-6f);
            }
        }
    }
}

__global__ void kernel_mimo_dmrs_residual_noise_variance(
    const unsigned int* __restrict__ d_grid_cbf16,
    const uint32_t* __restrict__ d_dmrs_pilot_bits,
    const int* __restrict__ d_dmrs_symbol_indices,
    const __half2* __restrict__ d_estimates_fp16,
    float* __restrict__ d_noise_var_accum,
    int nof_prb,
    int nof_ports,
    int nof_layers,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling,
    int words_per_dmrs_sym,
    float* __restrict__ d_noise_vars_out,
    unsigned int* __restrict__ d_done_counter,
    const float* __restrict__ d_cfo_hz_ptr,
    const float* __restrict__ d_symbol_start_times,
    const float2* __restrict__ d_cfo_phasors,
    int dmrs_type)
{
    int prb_idx = blockIdx.x;
    int port = blockIdx.y;
    int local_idx = threadIdx.x;
    int nof_cdm = (nof_layers + 1) / 2;
    int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);
    int cdm = local_idx / nof_pilots;
    int pilot_re = local_idx % nof_pilots;

    if (prb_idx >= nof_prb || port >= nof_ports || cdm >= nof_cdm) return;

    int re_per_layer = 12;
    int re_per_port = nof_layers * re_per_layer;
    int re_per_prb = nof_ports * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;
    int prb_base = prb_idx * re_per_prb + port * re_per_port;

    int sc_in_prb = pusch_dmrs_sc_offset(dmrs_type, cdm, pilot_re);
    float occ = ((pilot_re & 1) == 0) ? 1.0f : -1.0f;

    float residual_sum = 0.0f;
    float inv_sqrt2 = 0.70710678118f;
    float cfo_hz = (d_cfo_hz_ptr != nullptr) ? *d_cfo_hz_ptr : 0.0f;

    for (int d = 0; d < nof_dmrs_symbols; d++) {
        int dmrs_symbol = d_dmrs_symbol_indices[d];
        int grid_sc = (start_prb + prb_idx) * 12 + sc_in_prb;
        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + grid_sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        int pilot_idx = (start_prb + prb_idx) * nof_pilots + pilot_re;
        int pilot_word_idx = pilot_idx / 16;
        int pilot_bit_offset = (pilot_idx % 16) * 2;
        const uint32_t* dmrs_seq = d_dmrs_pilot_bits + d * words_per_dmrs_sym;
        uint32_t pilot_word = dmrs_seq[pilot_word_idx];
        int c_real = (pilot_word >> pilot_bit_offset) & 1;
        int c_imag = (pilot_word >> (pilot_bit_offset + 1)) & 1;

        float p_real = (1 - 2 * c_real) * inv_sqrt2 * dmrs_scaling;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2 * dmrs_scaling;

        cuFloatComplex h_eff = make_cuFloatComplex(0.0f, 0.0f);
        int first_layer = cdm * 2;
        int second_layer = first_layer + 1;

        for (int l = first_layer; l < nof_layers && l <= second_layer; l++) {
            float h_re = 0.0f;
            float h_im = 0.0f;
            for (int avg_d = 0; avg_d < nof_dmrs_symbols; avg_d++) {
                int est_idx = avg_d * re_per_dmrs_sym + prb_base + l * re_per_layer + sc_in_prb;
                __half2 h = d_estimates_fp16[est_idx];
                float hr = __half2float(__low2half(h));
                float hi = __half2float(__high2half(h));
                if (cfo_hz != 0.0f) {
                    float cos_d = 1.0f;
                    float sin_d = 0.0f;
                    if (d_cfo_phasors != nullptr) {
                        float2 ph = d_cfo_phasors[MIMO_CFO_RESIDUAL_PHASOR_OFFSET +
                                                   d * MAX_DMRS_SYMBOLS + avg_d];
                        cos_d = ph.x;
                        sin_d = ph.y;
                    } else if (d_symbol_start_times != nullptr) {
                        float dt = d_symbol_start_times[dmrs_symbol] -
                                   d_symbol_start_times[d_dmrs_symbol_indices[avg_d]];
                        sincosf(6.2831853071795864f * cfo_hz * dt, &sin_d, &cos_d);
                    }
                    float r2 = hr * cos_d - hi * sin_d;
                    float i2 = hr * sin_d + hi * cos_d;
                    hr = r2;
                    hi = i2;
                }
                h_re += hr;
                h_im += hi;
            }
            h_re /= nof_dmrs_symbols;
            h_im /= nof_dmrs_symbols;

            float layer_occ = (l == second_layer) ? occ : 1.0f;
            h_eff.x += layer_occ * h_re;
            h_eff.y += layer_occ * h_im;
        }

        float pred_re = h_eff.x * p_real - h_eff.y * p_imag;
        float pred_im = h_eff.x * p_imag + h_eff.y * p_real;
        float err_re = y.x - pred_re;
        float err_im = y.y - pred_im;
        residual_sum += err_re * err_re + err_im * err_im;
    }

    atomicAdd(&d_noise_var_accum[port], residual_sum);
    __syncthreads();

    if (threadIdx.x == 0 && d_done_counter != nullptr) {
        __threadfence();
        unsigned int done = atomicAdd(d_done_counter, 1u) + 1;
        if (done == (unsigned int)(nof_prb * nof_ports)) {
            int nof_dmrs_pilots = nof_prb * nof_dmrs_symbols * nof_pilots;
            int divisor = nof_dmrs_pilots * nof_cdm - 1;
            for (int p = 0; p < nof_ports; p++) {
                float noise_var = (divisor > 0) ? (d_noise_var_accum[p] / (float)divisor) : 0.001f;
                d_noise_vars_out[p] = fmaxf(noise_var, 1e-6f);
            }
        }
    }
}

/**
 * Finalize cross-validation noise variance + fused SINR/EPRE.
 * Divides accumulated sum by the number of REs per port and applies floor.
 * If SINR accumulators are provided (non-null), thread 0 also computes
 * SINR and EPRE in dB from the fused accumulations.
 */
__global__ void kernel_finalize_cross_validation_noise(
    float* __restrict__ d_noise_vars,
    const float* __restrict__ d_sinr_noise_accum,
    const float* __restrict__ d_sinr_rsrp_accum,
    const float* __restrict__ d_epre_accum,
    float* __restrict__ d_sinr_epre_result,
    int nof_ports,
    int nof_prb,
    int nof_dmrs_symbols,
    float dmrs_scaling)
{
    int port = threadIdx.x;
    if (port >= nof_ports) return;

    /* Each port has nof_prb * 12 subcarriers contributing to the sum */
    int nof_sc = nof_prb * 12;
    float noise_sum = d_noise_vars[port];

    /* Normalize by number of subcarriers */
    float noise_var = (nof_sc > 0) ? (noise_sum / nof_sc) : 0.001f;

    /* CALIBRATION: Cross-validation variance is ~40% of CPU-computed noise variance.
     * Empirically, GPU cross-validation gives ratio=0.33-0.44 vs CPU noise.
     * Scale by 2.5 to match CPU behavior (1/0.4 = 2.5).
     *
     * Possible reasons for the difference:
     * 1. CPU uses different time-averaging window
     * 2. CPU may include additional noise sources
     * 3. Frequency interpolation correlation effects */
    noise_var *= 2.5f;

    /* The estimate is on the channel estimate (after dividing by dmrs_scaling).
     * The actual noise on received symbols is scaled by dmrs_scaling².
     * For equalization, we want noise variance relative to the estimated channel,
     * which is what we computed. No additional scaling needed.
     *
     * Apply floor to avoid division by zero in equalization.
     * Floor of 1e-6 corresponds to ~60 dB SNR (very high). */
    d_noise_vars[port] = fmaxf(noise_var, 1e-6f);

    /* Thread 0 computes final SINR and EPRE in dB (fused from CV kernel) */
    if (port == 0 && d_sinr_epre_result != nullptr) {
        int nof_pilot_positions = nof_prb * 6;
        int nof_dmrs_re_per_port = nof_prb * nof_dmrs_symbols * 6;

        float total_signal = 0.0f, total_noise = 0.0f, total_epre = 0.0f;
        for (int p = 0; p < nof_ports; p++) {
            float signal_power = d_sinr_rsrp_accum[p] / (float)nof_pilot_positions;
            float sinr_noise_var = d_sinr_noise_accum[p] / (float)(nof_dmrs_re_per_port - 1);
            sinr_noise_var = fmaxf(sinr_noise_var, signal_power * 1e-10f);
            float mean_epre_port = d_epre_accum[p] / (float)nof_dmrs_re_per_port;
            total_signal += signal_power;
            total_noise += sinr_noise_var;
            total_epre += mean_epre_port;
        }

        float mean_noise = total_noise / nof_ports;
        d_sinr_epre_result[0] = (mean_noise > 1e-10f)
            ? 10.0f * log10f(total_signal / mean_noise) : 60.0f;
        float mean_epre = total_epre / nof_ports;
        d_sinr_epre_result[1] = (mean_epre > 1e-10f)
            ? 10.0f * log10f(mean_epre) : -INFINITY;
        /* RSRP: averaged per-port signal power in dB.
         * d_sinr_rsrp_accum stores |H|² where H = y/(pilot*dmrs_scaling),
         * so multiply by dmrs_scaling² to recover actual received signal power. */
        float mean_rsrp = (total_signal / nof_ports) * dmrs_scaling * dmrs_scaling;
        d_sinr_epre_result[2] = (mean_rsrp > 1e-10f)
            ? 10.0f * log10f(mean_rsrp) : -INFINITY;
    }
}

/**
 * Optimized residual-based SINR/EPRE kernel (v2).
 *
 * Same algorithm as v1: noise = |y - h_avg × pilot|² where h_avg is the
 * time-averaged smoothed channel estimate across DMRS symbols.
 *
 * Key optimizations over v1:
 * - Single block launch <<<1, 256>>> — full warps, zero inter-block sync
 * - Reads precomputed pilot bits instead of LFSR generation per thread
 * - Shared memory reduction instead of global atomics + threadfence
 * - No external accumulators needed (all in shared memory)
 *
 * Each thread handles one or more (prb, port) pairs via strided loop.
 */
__global__ void kernel_compute_residual_sinr_epre_v2(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_estimates_fp16,  /* smoothed [dmrs_sym, prb, port, 12] */
    const uint32_t* __restrict__ d_dmrs_pilot_bits, /* precomputed packed bits per DMRS sym */
    const int* __restrict__ d_dmrs_symbol_indices,
    const float* __restrict__ d_epre_accum,        /* [nof_ports] — from LSE kernel */
    float* __restrict__ d_sinr_epre_result,        /* [4] output: {sinr_db, epre_db, rsrp_db, ta_s} */
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling,
    int words_per_dmrs_sym)
{
    __shared__ float s_noise[8];  /* max 8 ports */
    __shared__ float s_rsrp[8];

    if (threadIdx.x < 8) {
        s_noise[threadIdx.x] = 0.0f;
        s_rsrp[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int total_work = nof_prb * nof_ports;
    int re_per_dmrs_sym = nof_prb * nof_ports * 12;
    float pilot_amp = dmrs_scaling * 0.70710678118f;

    for (int work_idx = threadIdx.x; work_idx < total_work; work_idx += blockDim.x) {
        int prb_idx = work_idx / nof_ports;
        int port = work_idx % nof_ports;

        float local_noise = 0.0f;
        float local_rsrp = 0.0f;

        for (int d_pilot = 0; d_pilot < 6; d_pilot++) {
            int sc_offset = d_pilot * 2;

            /* Time-average the smoothed channel estimates across DMRS symbols */
            float h_avg_re = 0.0f, h_avg_im = 0.0f;
            for (int d = 0; d < nof_dmrs_symbols; d++) {
                int est_idx = d * re_per_dmrs_sym + prb_idx * (nof_ports * 12) + port * 12 + sc_offset;
                __half2 h = d_estimates_fp16[est_idx];
                h_avg_re += __half2float(__low2half(h));
                h_avg_im += __half2float(__high2half(h));
            }
            float inv_n = 1.0f / (float)nof_dmrs_symbols;
            h_avg_re *= inv_n;
            h_avg_im *= inv_n;

            local_rsrp += h_avg_re * h_avg_re + h_avg_im * h_avg_im;

            /* Compute residual noise for each DMRS symbol */
            int pilot_idx = (start_prb + prb_idx) * 6 + d_pilot;
            int pilot_word_idx = pilot_idx / 16;
            int pilot_bit_offset = (pilot_idx % 16) * 2;

            for (int d = 0; d < nof_dmrs_symbols; d++) {
                int dmrs_symbol = d_dmrs_symbol_indices[d];
                int sc = (start_prb + prb_idx) * 12 + sc_offset;

                /* Read received signal y from grid */
                int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
                cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

                /* Read pilot bits from precomputed buffer (LSB-first packing) */
                const uint32_t* dmrs_seq = d_dmrs_pilot_bits + d * words_per_dmrs_sym;
                uint32_t pilot_word = dmrs_seq[pilot_word_idx];
                int c_real = (pilot_word >> pilot_bit_offset) & 1;
                int c_imag = (pilot_word >> (pilot_bit_offset + 1)) & 1;

                float p_real = (1 - 2 * c_real) * pilot_amp;
                float p_imag = (1 - 2 * c_imag) * pilot_amp;

                /* predicted = h_avg × pilot */
                float pred_real = h_avg_re * p_real - h_avg_im * p_imag;
                float pred_imag = h_avg_re * p_imag + h_avg_im * p_real;

                /* residual = y - predicted */
                float res_real = y.x - pred_real;
                float res_imag = y.y - pred_imag;

                local_noise += res_real * res_real + res_imag * res_imag;
            }
        }

        atomicAdd(&s_noise[port], local_noise);
        atomicAdd(&s_rsrp[port], local_rsrp);
    }

    __syncthreads();

    /* Thread 0 computes final SINR and EPRE in dB */
    if (threadIdx.x == 0) {
        int nof_pilot_positions = nof_prb * 6;
        int nof_dmrs_re_per_port = nof_prb * nof_dmrs_symbols * 6;

        float total_signal = 0.0f, total_noise = 0.0f, total_epre = 0.0f;
        for (int p = 0; p < nof_ports; p++) {
            float signal_power = s_rsrp[p] / (float)nof_pilot_positions;
            float noise_var = s_noise[p] / (float)(nof_dmrs_re_per_port - 1);
            noise_var = fmaxf(noise_var, signal_power * 1e-10f);
            float mean_epre_port = d_epre_accum[p] / (float)nof_dmrs_re_per_port;
            total_signal += signal_power;
            total_noise += noise_var;
            total_epre += mean_epre_port;
        }

        float mean_noise = total_noise / nof_ports;
        d_sinr_epre_result[0] = (mean_noise > 1e-10f)
            ? 10.0f * log10f(total_signal / mean_noise) : 60.0f;
        float mean_epre = total_epre / nof_ports;
        d_sinr_epre_result[1] = (mean_epre > 1e-10f)
            ? 10.0f * log10f(mean_epre) : -INFINITY;
        /* RSRP: averaged per-port signal power in dB.
         * s_rsrp stores |H|² where H = y/(pilot*dmrs_scaling),
         * so multiply by dmrs_scaling² to recover actual received signal power. */
        float mean_rsrp = (total_signal / nof_ports) * dmrs_scaling * dmrs_scaling;
        d_sinr_epre_result[2] = (mean_rsrp > 1e-10f)
            ? 10.0f * log10f(mean_rsrp) : -INFINITY;
    }
}

/**
 * Multi-block residual-based SINR/EPRE kernel (v3).
 *
 * Same algorithm as v2 but parallelized across PRBs:
 * - Launch <<<nof_prb, nof_ports * 6>>> — one block per PRB
 * - Each thread handles one (port, pilot) pair
 * - Global atomics to per-port accumulators (max 8 ports, fine granularity)
 *
 * Requires separate kernel_finalize_sinr_epre to convert accumulators to dB.
 */
__global__ void kernel_compute_residual_sinr_epre_v3(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_estimates_fp16,
    const uint32_t* __restrict__ d_dmrs_pilot_bits,
    const int* __restrict__ d_dmrs_symbol_indices,
    float* __restrict__ d_sinr_noise_accum,
    float* __restrict__ d_sinr_rsrp_accum,
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling,
    int words_per_dmrs_sym,
    int dmrs_type)
{
    int prb_idx = blockIdx.x;
    int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);
    int port = threadIdx.x / nof_pilots;
    int d_pilot = threadIdx.x % nof_pilots;

    if (prb_idx >= nof_prb || port >= nof_ports) return;

    int re_per_dmrs_sym = nof_prb * nof_ports * 12;
    int sc_offset = pusch_dmrs_sc_offset(dmrs_type, 0, d_pilot);
    float pilot_amp = dmrs_scaling * 0.70710678118f;

    /* Time-average the smoothed channel estimates across DMRS symbols */
    float h_avg_re = 0.0f, h_avg_im = 0.0f;
    for (int d = 0; d < nof_dmrs_symbols; d++) {
        int est_idx = d * re_per_dmrs_sym + prb_idx * (nof_ports * 12) + port * 12 + sc_offset;
        __half2 h = d_estimates_fp16[est_idx];
        h_avg_re += __half2float(__low2half(h));
        h_avg_im += __half2float(__high2half(h));
    }
    float inv_n = 1.0f / (float)nof_dmrs_symbols;
    h_avg_re *= inv_n;
    h_avg_im *= inv_n;

    float local_rsrp = h_avg_re * h_avg_re + h_avg_im * h_avg_im;
    float local_noise = 0.0f;

    /* Compute residual noise for each DMRS symbol */
    int pilot_idx = (start_prb + prb_idx) * nof_pilots + d_pilot;
    int pilot_word_idx = pilot_idx / 16;
    int pilot_bit_offset = (pilot_idx % 16) * 2;

    for (int d = 0; d < nof_dmrs_symbols; d++) {
        int dmrs_symbol = d_dmrs_symbol_indices[d];
        int sc = (start_prb + prb_idx) * 12 + sc_offset;

        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        const uint32_t* dmrs_seq = d_dmrs_pilot_bits + d * words_per_dmrs_sym;
        uint32_t pilot_word = dmrs_seq[pilot_word_idx];
        int c_real = (pilot_word >> pilot_bit_offset) & 1;
        int c_imag = (pilot_word >> (pilot_bit_offset + 1)) & 1;

        float p_real = (1 - 2 * c_real) * pilot_amp;
        float p_imag = (1 - 2 * c_imag) * pilot_amp;

        float pred_real = h_avg_re * p_real - h_avg_im * p_imag;
        float pred_imag = h_avg_re * p_imag + h_avg_im * p_real;

        float res_real = y.x - pred_real;
        float res_imag = y.y - pred_imag;

        local_noise += res_real * res_real + res_imag * res_imag;
    }

    atomicAdd(&d_sinr_noise_accum[port], local_noise);
    atomicAdd(&d_sinr_rsrp_accum[port], local_rsrp);
}

/**
 * Finalize SINR/EPRE/RSRP from accumulated residual noise and signal power.
 * Launched <<<1, 1>>> — single thread doing dB conversion.
 */
__global__ void kernel_finalize_sinr_epre(
    const float* __restrict__ d_sinr_noise_accum,
    const float* __restrict__ d_sinr_rsrp_accum,
    const float* __restrict__ d_epre_accum,
    float* __restrict__ d_noise_vars_out,
    float* __restrict__ d_sinr_epre_result,
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    float dmrs_scaling,
    int dmrs_type)
{
    int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);
    int nof_pilot_positions = nof_prb * nof_pilots;
    int nof_dmrs_re_per_port = nof_prb * nof_dmrs_symbols * nof_pilots;

    float total_signal = 0.0f, total_noise = 0.0f, total_epre = 0.0f;
    for (int p = 0; p < nof_ports; p++) {
        float signal_power = d_sinr_rsrp_accum[p] / (float)nof_pilot_positions;
        float noise_var = d_sinr_noise_accum[p] / (float)(nof_dmrs_re_per_port - 1);
        noise_var = fmaxf(noise_var, signal_power * 1e-10f);
        if (d_noise_vars_out != nullptr) {
            d_noise_vars_out[p] = noise_var;
        }
        float mean_epre_port = d_epre_accum[p] / (float)nof_dmrs_re_per_port;
        total_signal += signal_power;
        total_noise += noise_var;
        total_epre += mean_epre_port;
    }

    if (d_sinr_epre_result == nullptr) {
        return;
    }

    float mean_noise = total_noise / nof_ports;
    d_sinr_epre_result[0] = (mean_noise > 1e-10f)
        ? 10.0f * log10f(total_signal / mean_noise) : 60.0f;
    float mean_epre = total_epre / nof_ports;
    d_sinr_epre_result[1] = (mean_epre > 1e-10f)
        ? 10.0f * log10f(mean_epre) : -INFINITY;
    /* RSRP: averaged per-port signal power in dB.
     * d_sinr_rsrp_accum stores |H|² where H = y/(pilot*dmrs_scaling),
     * so multiply by dmrs_scaling² to recover actual received signal power. */
    float mean_rsrp = (total_signal / nof_ports) * dmrs_scaling * dmrs_scaling;
    d_sinr_epre_result[2] = (mean_rsrp > 1e-10f)
        ? 10.0f * log10f(mean_rsrp) : -INFINITY;
}

__global__ void kernel_reset_sinr_epre_result(float* __restrict__ d_sinr_epre_result)
{
    if (d_sinr_epre_result == nullptr) {
        return;
    }
    d_sinr_epre_result[0] = -INFINITY; /* SINR */
    d_sinr_epre_result[1] = -INFINITY; /* EPRE */
    d_sinr_epre_result[2] = -INFINITY; /* RSRP */
    d_sinr_epre_result[3] = NAN;       /* TA */
    d_sinr_epre_result[4] = NAN;       /* CFO */
}

__device__ __forceinline__ float2 pusch_phasor_mul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 pusch_phasor_conj(float2 a)
{
    return make_float2(a.x, -a.y);
}

__device__ void pusch_fill_cfo_phasor_table(float cfo_hz,
                                             const float* __restrict__ d_symbol_start_times,
                                             const int* __restrict__ d_dmrs_symbol_indices,
                                             float2* __restrict__ d_cfo_phasors,
                                             int nof_dmrs_symbols,
                                             int start_symbol)
{
    if (d_cfo_phasors == nullptr) {
        return;
    }

    float2 sym_phasors[MAX_SYMBOLS];
    if (cfo_hz == 0.0f || d_symbol_start_times == nullptr ||
        d_dmrs_symbol_indices == nullptr || nof_dmrs_symbols <= 0) {
        #pragma unroll
        for (int s = 0; s < MAX_SYMBOLS; ++s) {
            sym_phasors[s] = make_float2(1.0f, 0.0f);
        }
    } else {
        float omega = 6.2831853071795864f * cfo_hz;
        float sn, cs;
        sincosf(omega * d_symbol_start_times[0], &sn, &cs);
        sym_phasors[0] = make_float2(cs, sn);

        float cached_dt[3];
        float2 cached_step[3];
        int cached_count = 0;

        #pragma unroll
        for (int sidx = 1; sidx < MAX_SYMBOLS; ++sidx) {
            float dt = d_symbol_start_times[sidx] - d_symbol_start_times[sidx - 1];
            int cache_idx = -1;
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                if (c < cached_count && fabsf(dt - cached_dt[c]) < 1.0e-12f) {
                    cache_idx = c;
                }
            }
            if (cache_idx < 0) {
                float step_s, step_c;
                sincosf(omega * dt, &step_s, &step_c);
                cache_idx = cached_count;
                if (cached_count < 3) {
                    cached_dt[cached_count] = dt;
                    cached_step[cached_count] = make_float2(step_c, step_s);
                    cached_count++;
                } else {
                    cached_step[cache_idx - 1] = make_float2(step_c, step_s);
                    cache_idx--;
                }
            }
            sym_phasors[sidx] = pusch_phasor_mul(sym_phasors[sidx - 1], cached_step[cache_idx]);
        }
    }

    #pragma unroll
    for (int data_symbol = 0; data_symbol < MAX_SYMBOLS; ++data_symbol) {
        #pragma unroll
        for (int d = 0; d < MAX_DMRS_SYMBOLS; ++d) {
            float2 ph = make_float2(1.0f, 0.0f);
            if (d < nof_dmrs_symbols) {
                int dmrs_symbol = d_dmrs_symbol_indices[d] + start_symbol;
                if ((unsigned)dmrs_symbol < MAX_SYMBOLS) {
                    ph = pusch_phasor_mul(sym_phasors[data_symbol],
                                          pusch_phasor_conj(sym_phasors[dmrs_symbol]));
                }
            }
            d_cfo_phasors[data_symbol * MAX_DMRS_SYMBOLS + d] = ph;
        }
    }

    #pragma unroll
    for (int target_dmrs_idx = 0; target_dmrs_idx < MAX_DMRS_SYMBOLS; ++target_dmrs_idx) {
        #pragma unroll
        for (int source_dmrs_idx = 0; source_dmrs_idx < MAX_DMRS_SYMBOLS; ++source_dmrs_idx) {
            float2 ph = make_float2(1.0f, 0.0f);
            if (target_dmrs_idx < nof_dmrs_symbols && source_dmrs_idx < nof_dmrs_symbols) {
                int target_symbol = d_dmrs_symbol_indices[target_dmrs_idx];
                int source_symbol = d_dmrs_symbol_indices[source_dmrs_idx];
                if ((unsigned)target_symbol < MAX_SYMBOLS && (unsigned)source_symbol < MAX_SYMBOLS) {
                    ph = pusch_phasor_mul(sym_phasors[target_symbol],
                                          pusch_phasor_conj(sym_phasors[source_symbol]));
                }
            }
            d_cfo_phasors[MIMO_CFO_RESIDUAL_PHASOR_OFFSET +
                          target_dmrs_idx * MAX_DMRS_SYMBOLS + source_dmrs_idx] = ph;
        }
    }
}

/**
 * Fused Time Alignment (TA) + Carrier Frequency Offset (CFO) estimation
 * from DMRS channel estimates.
 *
 * TA algorithm (phase slope across frequency):
 * 1. Time-average pilot estimates into shared memory
 * 2. Cross-correlate adjacent pilots: sum_k H_avg[k+1] * conj(H_avg[k])
 * 3. ta_seconds = -atan2f(imag, real) / (2 * PI * pilot_spacing_hz)
 *
 * CFO algorithm (phase rotation across time, requires >= 2 DMRS symbols):
 * 1. Store first and last DMRS symbol estimates in shared memory
 * 2. Cross-correlate all pilots: sum_k H_last[k] * conj(H_first[k])
 * 3. cfo_hz = atan2f(imag, real) / (2 * PI * delta_t_s)
 *
 * Shared memory: 3 arrays × nof_pilots × 2 floats (avg, sym0, sym_last).
 * Launched <<<nof_ports, 256, smem>>> — one block per port for parallel execution.
 * Uses atomicAdd to accumulate cross-port results, last block computes atan2.
 */
__global__ void kernel_compute_ta_and_cfo(
    const __half2* __restrict__ d_estimates_fp16,
    float* __restrict__ d_ta_result,
    float* __restrict__ d_cfo_result,
    float* __restrict__ d_accum,   /* [ta_r, ta_i, cfo_r, cfo_i, done_counter] — pre-zeroed */
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    float pilot_spacing_hz,
    float delta_t_s,
    const float* __restrict__ d_symbol_start_times,
    const int* __restrict__ d_dmrs_symbol_indices,
    float2* __restrict__ d_cfo_phasors,
    int start_symbol,
    int dmrs_type)
{
    extern __shared__ float s_data[];

    int pilots_per_prb = pusch_dmrs_pilots_per_cdm(dmrs_type);
    int nof_pilots = nof_prb * pilots_per_prb;
    float* s_avg      = s_data;
    float* s_sym0     = s_data + nof_pilots * 2;
    float* s_sym_last = s_data + nof_pilots * 4;
    /* Cross-warp reduction scratch: 4 accumulators × up to 8 warps, after pilot arrays. */
    float* s_warp_reduce = s_data + nof_pilots * 6;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;
    int nwarps  = nthreads / 32;
    int port    = blockIdx.x;  /* Each block handles one port */

    /* Phase 1: Load all DMRS symbols for this port, store sym0, sym_last, and average.
     * Layout: [dmrs_sym][prb][port][12 subcarriers] as __half2. */
    for (int pilot = tid; pilot < nof_pilots; pilot += nthreads) {
        int prb = pilot / pilots_per_prb;
        int sc_in_prb = pusch_dmrs_sc_offset(dmrs_type, 0, pilot % pilots_per_prb);
        float sum_r = 0.0f, sum_i = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int idx = d * nof_prb * nof_ports * 12
                    + prb * nof_ports * 12 + port * 12 + sc_in_prb;
            __half2 h = d_estimates_fp16[idx];
            float r = __half2float(h.x), i = __half2float(h.y);
            sum_r += r; sum_i += i;
            if (d == 0) {
                s_sym0[pilot * 2] = r; s_sym0[pilot * 2 + 1] = i;
            }
            if (d == nof_dmrs_symbols - 1) {
                s_sym_last[pilot * 2] = r; s_sym_last[pilot * 2 + 1] = i;
            }
        }
        s_avg[pilot * 2]     = sum_r / nof_dmrs_symbols;
        s_avg[pilot * 2 + 1] = sum_i / nof_dmrs_symbols;
    }
    __syncthreads();

    /* Phase 2a: TA — adjacent-frequency cross-correlation on averaged estimates. */
    float ta_lr = 0.0f, ta_li = 0.0f;
    for (int k = 1 + tid; k < nof_pilots; k += nthreads) {
        float cr = s_avg[k * 2],       ci = s_avg[k * 2 + 1];
        float pr = s_avg[(k - 1) * 2], pi = s_avg[(k - 1) * 2 + 1];
        ta_lr += cr * pr + ci * pi;   /* Re{H[k] · conj(H[k-1])} */
        ta_li += ci * pr - cr * pi;   /* Im{H[k] · conj(H[k-1])} */
    }

    /* Phase 2b: CFO — all-pilot time cross-correlation (sym_last vs sym0). */
    float cfo_lr = 0.0f, cfo_li = 0.0f;
    if (d_cfo_result) {
        for (int k = tid; k < nof_pilots; k += nthreads) {
            float h0r = s_sym0[k * 2],     h0i = s_sym0[k * 2 + 1];
            float h1r = s_sym_last[k * 2], h1i = s_sym_last[k * 2 + 1];
            cfo_lr += h1r * h0r + h1i * h0i;   /* Re{H_last[k] · conj(H_first[k])} */
            cfo_li += h1i * h0r - h1r * h0i;   /* Im{H_last[k] · conj(H_first[k])} */
        }
    }

    /* Phase 3: Intra-warp shuffle reduction (4 accumulators). */
    for (int off = 16; off > 0; off >>= 1) {
        ta_lr  += __shfl_down_sync(0xFFFFFFFF, ta_lr,  off);
        ta_li  += __shfl_down_sync(0xFFFFFFFF, ta_li,  off);
        cfo_lr += __shfl_down_sync(0xFFFFFFFF, cfo_lr, off);
        cfo_li += __shfl_down_sync(0xFFFFFFFF, cfo_li, off);
    }

    /* Phase 4: Cross-warp reduction via shared memory. */
    if (lane == 0) {
        s_warp_reduce[warp_id * 4 + 0] = ta_lr;
        s_warp_reduce[warp_id * 4 + 1] = ta_li;
        s_warp_reduce[warp_id * 4 + 2] = cfo_lr;
        s_warp_reduce[warp_id * 4 + 3] = cfo_li;
    }
    __syncthreads();

    /* Phase 5: Thread 0 reduces across warps, then atomicAdd to global accumulators. */
    if (tid == 0) {
        float ta_r = 0.0f, ta_i = 0.0f, cfo_r = 0.0f, cfo_i = 0.0f;
        for (int w = 0; w < nwarps; w++) {
            ta_r  += s_warp_reduce[w * 4 + 0];
            ta_i  += s_warp_reduce[w * 4 + 1];
            cfo_r += s_warp_reduce[w * 4 + 2];
            cfo_i += s_warp_reduce[w * 4 + 3];
        }
        atomicAdd(&d_accum[0], ta_r);
        atomicAdd(&d_accum[1], ta_i);
        atomicAdd(&d_accum[2], cfo_r);
        atomicAdd(&d_accum[3], cfo_i);

        /* Last block to finish computes atan2 and writes final results. */
        __threadfence();
        int done = atomicAdd((int*)&d_accum[4], 1) + 1;
        if (done == nof_ports) {
            float ta_phase = atan2f(d_accum[1], d_accum[0]);
            d_ta_result[0] = -ta_phase / (2.0f * 3.14159265358979f * pilot_spacing_hz);

            if (d_cfo_result && delta_t_s > 0.0f) {
                float cfo_phase = atan2f(d_accum[3], d_accum[2]);
                float cfo_hz = cfo_phase / (2.0f * 3.14159265358979f * delta_t_s);
                d_cfo_result[0] = cfo_hz;
                pusch_fill_cfo_phasor_table(cfo_hz,
                                             d_symbol_start_times,
                                             d_dmrs_symbol_indices,
                                             d_cfo_phasors,
                                             nof_dmrs_symbols,
                                             start_symbol);
            }
        }
    }
}

/**
 * MIMO CSI helper.
 *
 * Uses the layer-0 channel estimates to estimate TA/CFO. All layers use the
 * same propagation timing and oscillator offset, so layer 0 is enough and keeps
 * the kernel small. RSRP is averaged across all estimated layers and ports for
 * scheduler reporting.
 */
__global__ void kernel_compute_mimo_csi_from_estimates(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_estimates_fp16,
    const int* __restrict__ d_dmrs_symbol_indices,
    float* __restrict__ d_sinr_epre_result,
    float* __restrict__ d_accum,   /* [ta_r, ta_i, cfo_r, cfo_i, rsrp_sum, epre_sum] */
    int nof_prb,
    int nof_ports,
    int nof_layers,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float pilot_spacing_hz,
    float delta_t_s,
    const float* __restrict__ d_symbol_start_times,
    float2* __restrict__ d_cfo_phasors,
    int start_symbol,
    float dmrs_scaling,
    int dmrs_type)
{
    int tid = threadIdx.x;
    int pilots_per_prb = pusch_dmrs_pilots_per_cdm(dmrs_type);
    int total_pilots = nof_prb * pilots_per_prb;
    int re_per_port = nof_layers * 12;
    int re_per_prb = nof_ports * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    float ta_r = 0.0f;
    float ta_i = 0.0f;
    float cfo_r = 0.0f;
    float cfo_i = 0.0f;
    float rsrp_sum = 0.0f;
    float epre_sum = 0.0f;

    for (int pilot = tid; pilot < total_pilots; pilot += blockDim.x) {
        int prb = pilot / pilots_per_prb;
        int sc_in_prb = pusch_dmrs_sc_offset(dmrs_type, 0, pilot % pilots_per_prb);

        /* TA and CFO from layer 0, accumulated across ports. */
        for (int port = 0; port < nof_ports; ++port) {
            int base = prb * re_per_prb + port * re_per_port + sc_in_prb;

            float avg_re = 0.0f;
            float avg_im = 0.0f;
            float first_re = 0.0f;
            float first_im = 0.0f;
            float last_re = 0.0f;
            float last_im = 0.0f;

            for (int d = 0; d < nof_dmrs_symbols; ++d) {
                int idx = d * re_per_dmrs_sym + base;
                __half2 h = d_estimates_fp16[idx];
                float re = __half2float(__low2half(h));
                float im = __half2float(__high2half(h));
                avg_re += re;
                avg_im += im;
                if (d == 0) {
                    first_re = re;
                    first_im = im;
                }
                if (d == nof_dmrs_symbols - 1) {
                    last_re = re;
                    last_im = im;
                }
            }

            avg_re /= (float)nof_dmrs_symbols;
            avg_im /= (float)nof_dmrs_symbols;

            if (pilot > 0) {
                int prev_pilot = pilot - 1;
                int prev_prb = prev_pilot / pilots_per_prb;
                int prev_sc = pusch_dmrs_sc_offset(dmrs_type, 0, prev_pilot % pilots_per_prb);
                int prev_base = prev_prb * re_per_prb + port * re_per_port + prev_sc;
                float prev_re = 0.0f;
                float prev_im = 0.0f;
                for (int d = 0; d < nof_dmrs_symbols; ++d) {
                    int prev_idx = d * re_per_dmrs_sym + prev_base;
                    __half2 prev_h = d_estimates_fp16[prev_idx];
                    prev_re += __half2float(__low2half(prev_h));
                    prev_im += __half2float(__high2half(prev_h));
                }
                prev_re /= (float)nof_dmrs_symbols;
                prev_im /= (float)nof_dmrs_symbols;

                ta_r += avg_re * prev_re + avg_im * prev_im;
                ta_i += avg_im * prev_re - avg_re * prev_im;
            }

            if (nof_dmrs_symbols >= 2) {
                cfo_r += last_re * first_re + last_im * first_im;
                cfo_i += last_im * first_re - last_re * first_im;
            }
        }

        /* RSRP from all layers and ports. */
        for (int d = 0; d < nof_dmrs_symbols; ++d) {
            int dmrs_symbol = d_dmrs_symbol_indices[d];
            for (int port = 0; port < nof_ports; ++port) {
                int grid_sc = (start_prb + prb) * 12 + sc_in_prb;
                int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + grid_sc;
                cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);
                epre_sum += y.x * y.x + y.y * y.y;

                for (int layer = 0; layer < nof_layers; ++layer) {
                    int idx = d * re_per_dmrs_sym + prb * re_per_prb + port * re_per_port + layer * 12 + sc_in_prb;
                    __half2 h = d_estimates_fp16[idx];
                    float re = __half2float(__low2half(h));
                    float im = __half2float(__high2half(h));
                    rsrp_sum += re * re + im * im;
                }
            }
        }
    }

    atomicAdd(&d_accum[0], ta_r);
    atomicAdd(&d_accum[1], ta_i);
    atomicAdd(&d_accum[2], cfo_r);
    atomicAdd(&d_accum[3], cfo_i);
    atomicAdd(&d_accum[4], rsrp_sum);
    atomicAdd(&d_accum[5], epre_sum);
    __syncthreads();

    if (tid == 0) {
        float ta_phase = atan2f(d_accum[1], d_accum[0]);
        d_sinr_epre_result[3] = -ta_phase / (2.0f * 3.14159265358979f * pilot_spacing_hz);

        if (nof_dmrs_symbols >= 2 && delta_t_s > 0.0f) {
            float cfo_phase = atan2f(d_accum[3], d_accum[2]);
            float cfo_hz = cfo_phase / (2.0f * 3.14159265358979f * delta_t_s);
            d_sinr_epre_result[4] = cfo_hz;
            pusch_fill_cfo_phasor_table(cfo_hz,
                                         d_symbol_start_times,
                                         d_dmrs_symbol_indices,
                                         d_cfo_phasors,
                                         nof_dmrs_symbols,
                                         start_symbol);
        } else {
            d_sinr_epre_result[4] = NAN;
        }

        int rsrp_count = total_pilots * nof_dmrs_symbols * nof_ports * nof_layers;
        float mean_rsrp = (rsrp_count > 0) ? (d_accum[4] / (float)rsrp_count) * dmrs_scaling * dmrs_scaling : 0.0f;
        d_sinr_epre_result[2] = (mean_rsrp > 1e-10f) ? 10.0f * log10f(mean_rsrp) : -INFINITY;

        int epre_count = total_pilots * nof_dmrs_symbols * nof_ports;
        float layer_norm = (float)nof_layers * 0.25f;
        float mean_epre =
            (epre_count > 0) ? (d_accum[5] / (float)epre_count) * dmrs_scaling * dmrs_scaling * layer_norm : 0.0f;
        d_sinr_epre_result[1] = (mean_epre > 1e-10f) ? 10.0f * log10f(mean_epre) : -INFINITY;
    }
}

/**
 * Pre-generate scrambling sequence kernel.
 *
 * Each thread generates 32 consecutive bits (one uint32_t word).
 * Thread i generates bits [i*32, (i+1)*32).
 *
 * This eliminates per-RE LFSR advancement in the E2E kernel, replacing
 * O(log N) matrix operations per RE with a single memory load.
 *
 * Memory: ceil(nof_bits / 32) * 4 bytes (e.g., 30 KB for 100 MHz 64-QAM)
 */
__global__ void kernel_generate_scrambling_sequence(
    uint32_t* __restrict__ d_scrambling_seq,
    uint32_t c_init,
    int nof_bits)
{
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bit_start = word_idx * 32;

    if (bit_start >= nof_bits) return;

    /* Initialize LFSRs */
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;

    /* Jump to starting position: Nc + word_idx*32 */
    int total_advance = NC_SKIP + bit_start;
    x1 = advance_x1_local(x1, total_advance);
    x2 = advance_x2_local(x2, total_advance);

    /* Generate 32 bits */
    uint32_t packed = 0;
    int bits_to_generate = min(32, nof_bits - bit_start);

    #pragma unroll
    for (int b = 0; b < 32; b++) {
        if (b < bits_to_generate) {
            uint32_t scr_bit = (x1 ^ x2) & 1;
            packed |= (scr_bit << b);

            /* Step LFSRs */
            uint32_t new_x1 = ((x1 >> 3) ^ x1) & 1;
            x1 = (x1 >> 1) | (new_x1 << 30);

            uint32_t new_x2 = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
            x2 = (x2 >> 1) | (new_x2 << 30);
        }
    }

    d_scrambling_seq[word_idx] = packed;
}

/**
 * Soft demodulation for 64QAM using piecewise-linear approximation.
 * Matches the CPU implementation for accurate LLR computation at all SINR levels.
 */
__device__ __forceinline__ void soft_demod_64qam_piecewise(
    cuFloatComplex eq, float inv_noise, float llr_vals[6])
{
    /* M = 1/sqrt(42) = 0.154303349962092 */
    const float INTERVAL_WIDTH_01 = 0.308606699924184f;  // Bits 0,1: 2M intervals
    const float INTERVAL_WIDTH_23 = 0.308606699924184f;  // Bits 2,3: 2M intervals
    const float INTERVAL_WIDTH_45 = 0.617213399848368f;  // Bits 4,5: 4M intervals

    /* Bits 0,1: Sign bits (8 intervals) - CRITICAL: offset must be OUTSIDE floor() */
    int idx_x_01 = max(0, min(7, (int)floorf(eq.x / INTERVAL_WIDTH_01) + 4));
    int idx_y_01 = max(0, min(7, (int)floorf(eq.y / INTERVAL_WIDTH_01) + 4));
    llr_vals[0] = (SLOPE_01[idx_x_01] * eq.x + INTERCEPT_01[idx_x_01]) * inv_noise;
    llr_vals[1] = (SLOPE_01[idx_y_01] * eq.y + INTERCEPT_01[idx_y_01]) * inv_noise;

    /* Bits 2,3: Inner/outer group (8 intervals) */
    int idx_x_23 = max(0, min(7, (int)floorf(eq.x / INTERVAL_WIDTH_23) + 4));
    int idx_y_23 = max(0, min(7, (int)floorf(eq.y / INTERVAL_WIDTH_23) + 4));
    llr_vals[2] = (SLOPE_23[idx_x_23] * eq.x + INTERCEPT_23[idx_x_23]) * inv_noise;
    llr_vals[3] = (SLOPE_23[idx_y_23] * eq.y + INTERCEPT_23[idx_y_23]) * inv_noise;

    /* Bits 4,5: Nested magnitude (4 intervals) */
    int idx_x_45 = max(0, min(3, (int)floorf(eq.x / INTERVAL_WIDTH_45) + 2));
    int idx_y_45 = max(0, min(3, (int)floorf(eq.y / INTERVAL_WIDTH_45) + 2));
    llr_vals[4] = (SLOPE_45[idx_x_45] * eq.x + INTERCEPT_45[idx_x_45]) * inv_noise;
    llr_vals[5] = (SLOPE_45[idx_y_45] * eq.y + INTERCEPT_45[idx_y_45]) * inv_noise;
}

/**
 * Soft demodulation for 256QAM using piecewise-linear approximation.
 * Matches the CPU implementation for accurate LLR computation at all SINR levels.
 */
__device__ __forceinline__ void soft_demod_256qam_piecewise(
    cuFloatComplex eq, float inv_noise, float llr_vals[8])
{
    /* M = 1/sqrt(170) = 0.07669649888473704 */
    const float M = 0.07669649888473704f;
    const float INTERVAL_WIDTH_01 = 2.0f * M;  // Bits 0,1: 2M intervals (16 intervals)
    const float INTERVAL_WIDTH_23 = 2.0f * M;  // Bits 2,3: 2M intervals (16 intervals)
    const float INTERVAL_WIDTH_45 = 2.0f * M;  // Bits 4,5: 2M intervals (16 intervals)
    const float INTERVAL_WIDTH_67 = 4.0f * M;  // Bits 6,7: 4M intervals (8 intervals)

    /* Compute LLRs - CRITICAL: offset must be OUTSIDE floor() */

    /* Bits 0,1: Sign bits (16 intervals, offset by 8) */
    int idx_x_01 = max(0, min(15, (int)floorf(eq.x / INTERVAL_WIDTH_01) + 8));
    int idx_y_01 = max(0, min(15, (int)floorf(eq.y / INTERVAL_WIDTH_01) + 8));
    llr_vals[0] = (QAM256_SLOPE_01[idx_x_01] * eq.x + QAM256_INTERCEPT_01[idx_x_01]) * inv_noise;
    llr_vals[1] = (QAM256_SLOPE_01[idx_y_01] * eq.y + QAM256_INTERCEPT_01[idx_y_01]) * inv_noise;

    /* Bits 2,3: Inner/outer group (16 intervals, offset by 8) */
    int idx_x_23 = max(0, min(15, (int)floorf(eq.x / INTERVAL_WIDTH_23) + 8));
    int idx_y_23 = max(0, min(15, (int)floorf(eq.y / INTERVAL_WIDTH_23) + 8));
    llr_vals[2] = (QAM256_SLOPE_23[idx_x_23] * eq.x + QAM256_INTERCEPT_23[idx_x_23]) * inv_noise;
    llr_vals[3] = (QAM256_SLOPE_23[idx_y_23] * eq.y + QAM256_INTERCEPT_23[idx_y_23]) * inv_noise;

    /* Bits 4,5: Nested magnitude (16 intervals, offset by 8) */
    int idx_x_45 = max(0, min(15, (int)floorf(eq.x / INTERVAL_WIDTH_45) + 8));
    int idx_y_45 = max(0, min(15, (int)floorf(eq.y / INTERVAL_WIDTH_45) + 8));
    llr_vals[4] = (QAM256_SLOPE_45[idx_x_45] * eq.x + QAM256_INTERCEPT_45[idx_x_45]) * inv_noise;
    llr_vals[5] = (QAM256_SLOPE_45[idx_y_45] * eq.y + QAM256_INTERCEPT_45[idx_y_45]) * inv_noise;

    /* Bits 6,7: Finest level (8 intervals, offset by 4) */
    int idx_x_67 = max(0, min(7, (int)floorf(eq.x / INTERVAL_WIDTH_67) + 4));
    int idx_y_67 = max(0, min(7, (int)floorf(eq.y / INTERVAL_WIDTH_67) + 4));
    llr_vals[6] = (QAM256_SLOPE_67[idx_x_67] * eq.x + QAM256_INTERCEPT_67[idx_x_67]) * inv_noise;
    llr_vals[7] = (QAM256_SLOPE_67[idx_y_67] * eq.y + QAM256_INTERCEPT_67[idx_y_67]) * inv_noise;
}

/**
 * Ultra-optimized E2E kernel with ON-THE-FLY scrambling sequence generation.
 *
 * Instead of reading scrambling bits from a pre-generated buffer, this kernel
 * generates them directly using LFSR advancement. This eliminates:
 * - The scrambling sequence kernel launch
 * - Memory traffic for reading the sequence buffer
 *
 * Each thread advances its LFSR state to its starting bit position using
 * O(log N) matrix exponentiation, then generates mod_order bits sequentially.
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void kernel_fused_e2e_onthefly_scramble_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    __half* __restrict__ llrs_half,
    uint32_t scramble_c_init,                  /* c_init for scrambling (replaces buffer!) */
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int mod_order,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (re_idx >= nof_re) return;

    /* Load noise variances to shared memory */
    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    /* Get source RE index */
    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;

    /* Compute relative indices for estimate lookup */
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

    /* On-the-fly time interpolation: average FP16 estimates */
    cuFloatComplex h[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum = 0.0f;
        float h_imag_sum = 0.0f;

        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
            __half2 h_fp16 = d_dmrs_estimates_fp16[est_idx];
            h_real_sum += __half2float(__low2half(h_fp16));
            h_imag_sum += __half2float(__high2half(h_fp16));
        }

        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        h[p] = make_cuFloatComplex(h_real_sum * inv_n, h_imag_sum * inv_n);
    }

    /* Load received symbols from grid */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        int grid_idx = p * grid_stride + src_re;
        y[p] = cbf16_to_fp32(d_grid_cbf16[grid_idx]);
    }

    /* Equalization */
    cuFloatComplex eq;
    float eq_noise_var;
    equalize_symbol<NOF_PORTS, ALGORITHM>(y, h, s_noise_vars, tx_scaling, eq, eq_noise_var);

    /* Accumulate noise variance for SINR */
    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, eq_noise_var);

    /* Soft demodulation */
    float llr_vals[8];
    float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);

    if (mod_order == 2) {
        float scale = inv_noise * 2.8284271f  /* 2*sqrt(2) - matches CPU QPSK demod */;
        llr_vals[0] = eq.x * scale;
        llr_vals[1] = eq.y * scale;
    } else if (mod_order == 4) {
        float scale = inv_noise * 0.6324555f;
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.6324555f * 2.0f - fabsf(eq.x)) * scale * 2.0f;
        llr_vals[3] = (0.6324555f * 2.0f - fabsf(eq.y)) * scale * 2.0f;
    } else if (mod_order == 6) {
        // 64QAM: piecewise-linear approximation matching CPU implementation
        soft_demod_64qam_piecewise(eq, inv_noise, llr_vals);
    } else if (mod_order == 8) {
        // 256QAM: piecewise-linear approximation matching CPU implementation
        soft_demod_256qam_piecewise(eq, inv_noise, llr_vals);
    }

    /* ON-THE-FLY DESCRAMBLING using LFSR advancement */
    int llr_base = re_idx * mod_order;

    /* Initialize LFSRs and advance to our starting bit position */
    uint32_t x1 = 1;
    uint32_t x2 = scramble_c_init & 0x7FFFFFFF;

    /* Advance by Nc + llr_base steps using O(log N) matrix exponentiation */
    int total_advance = NC_SKIP + llr_base;
    x1 = advance_x1_local(x1, total_advance);
    x2 = advance_x2_local(x2, total_advance);

    /* Generate mod_order scrambling bits and apply descrambling */
    for (int b = 0; b < mod_order; b++) {
        uint32_t scr_bit = (x1 ^ x2) & 1;
        float llr = scr_bit ? -llr_vals[b] : llr_vals[b];
        llr = fmaxf(-65504.0f, fminf(65504.0f, llr));
        llrs_half[llr_base + b] = __float2half(llr);

        /* Step LFSRs for next bit */
        x1 = step_x1_local(x1);
        x2 = step_x2_local(x2);
    }
}

/* ============================================================================
 * ULTRA-OPTIMIZED E2E KERNEL v2: Cached scrambling + reductions
 *
 * Optimizations:
 * 1. Cached scrambling sequence: use configure-time Gold sequence precompute
 *    when available; keep warp-cooperative generation as a fallback.
 * 2. Block-level noise variance reduction via warp shuffle
 * 3. Template on NOF_DMRS for compile-time unrolling
 * 4. Vectorized half2 LLR writes
 * ============================================================================ */

/**
 * Warp-cooperative soft demod helper for 256QAM
 * Computes 8 LLRs per RE with proper max-log approximation
 */
__device__ __forceinline__ void soft_demod_256qam_optimized(
    cuFloatComplex eq, float inv_noise, float llr_vals[8])
{
    const float s = 0.07669650f;  // 1/sqrt(170)
    float abs_re = fabsf(eq.x);
    float abs_im = fabsf(eq.y);
    float scale = inv_noise * s * 2.0f;

    llr_vals[0] = eq.x * scale;
    llr_vals[1] = eq.y * scale;
    llr_vals[2] = (s * 8.0f - abs_re) * scale;
    llr_vals[3] = (s * 8.0f - abs_im) * scale;

    float level2 = s * 4.0f;
    llr_vals[4] = (level2 - fabsf(abs_re - level2 * 2.0f)) * scale;
    llr_vals[5] = (level2 - fabsf(abs_im - level2 * 2.0f)) * scale;

    float level3 = s * 2.0f;
    float d_re = fabsf(abs_re - level2 * 2.0f);
    float d_im = fabsf(abs_im - level2 * 2.0f);
    llr_vals[6] = (level3 - fabsf(d_re - level3 * 2.0f)) * scale;
    llr_vals[7] = (level3 - fabsf(d_im - level3 * 2.0f)) * scale;
}

/**
 * Piecewise-linear interval function for 64QAM soft demodulation
 * Implements optimal LLR approximation per TS 38.211
 *
 * @param value        Input symbol component (real or imag)
 * @param inv_noise    1/noise_variance scaling factor
 * @param interval_width  Width of each interval (2*M or 4*M)
 * @param num_intervals   Number of intervals (8 for bits 0-3, 4 for bits 4-5)
 * @param slopes       Slope lookup table
 * @param intercepts   Intercept lookup table
 * @return Scaled LLR value
 */
__device__ __forceinline__ float interval_function_64qam(
    float value,
    float inv_noise,
    float interval_width,
    int num_intervals,
    const float* slopes,
    const float* intercepts)
{
    /* Determine interval index: map value to [-num_intervals/2, num_intervals/2) range */
    float normalized = value / interval_width + (float)num_intervals * 0.5f;
    int idx = (int)floorf(normalized);

    /* Clamp to valid interval range [0, num_intervals-1] */
    idx = max(0, min(num_intervals - 1, idx));

    /* Apply piecewise-linear formula: LLR = (slope * value + intercept) / noise_var */
    return (slopes[idx] * value + intercepts[idx]) * inv_noise;
}

/**
 * Warp-level float reduction for noise variance accumulation
 * (Named with _f32 suffix to avoid conflict with later definition)
 */
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * OPTIMIZED E2E kernel with cached scrambling and warp-cooperative fallback.
 * Template parameters: NOF_PORTS, ALGORITHM, NOF_DMRS (for unrolling)
 */
template <int NOF_PORTS, int ALGORITHM, int NOF_DMRS, bool TIME_AVG = false, int MOD_ORDER = 2>
__global__ void __launch_bounds__(256, 4) kernel_fused_e2e_warp_cooperative_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    __half* __restrict__ llrs_half,
    const uint32_t* __restrict__ scramble_seq,
    uint32_t scramble_c_init,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    float* __restrict__ d_evm_error_sum,
    unsigned int* __restrict__ d_evm_symbol_count,
    int nof_re,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    const int* __restrict__ d_dmrs_indices,
    const float* __restrict__ d_cfo_hz_ptr,
    const float* __restrict__ d_symbol_start_times,
    const float2* __restrict__ d_cfo_phasors,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    const int WARP_SIZE = 32;
    const int WARPS_PER_BLOCK = 8;  // 256 threads / 32
    constexpr int mod_order = MOD_ORDER;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid & (WARP_SIZE - 1);
    int global_thread_id = blockIdx.x * blockDim.x + tid;

    /* ================================================================
     * Shared memory layout:
     * - Noise variances (8 floats)
     * - DMRS symbol indices (up to 4 ints)
     * - Fallback scrambling bits per warp (32 threads x 8 bits max = 256 bytes per warp)
     * - Partial sums for noise variance reduction
     * ================================================================ */
    __shared__ float s_noise_vars[8];
    __shared__ int s_dmrs_indices[4];
    __shared__ uint8_t s_scramble_bits[WARPS_PER_BLOCK][WARP_SIZE * 8];
    __shared__ float s_noise_var_sums[WARPS_PER_BLOCK];
    __shared__ int s_noise_var_counts[WARPS_PER_BLOCK];
    __shared__ float s_evm_error_sums[WARPS_PER_BLOCK];
    __shared__ int s_evm_symbol_counts[WARPS_PER_BLOCK];
    __shared__ float s_cfo_hz;
    __shared__ float s_sym_times[14];
    __shared__ float2 s_cfo_phasors[MIMO_CFO_DATA_PHASOR_COUNT];
    bool use_cfo_fallback =
        (d_cfo_hz_ptr != nullptr && d_cfo_phasors == nullptr && d_symbol_start_times != nullptr);

    /* Load noise variances, DMRS indices, CFO, and symbol times to shared memory */
    if (tid < NOF_PORTS) {
        s_noise_vars[tid] = noise_vars[tid];
    }
    if (tid < NOF_DMRS) {
        s_dmrs_indices[tid] = d_dmrs_indices[tid];
    }
    if (tid == 0) {
        s_cfo_hz = (d_cfo_hz_ptr != nullptr) ? *d_cfo_hz_ptr : 0.0f;
    }
    if (use_cfo_fallback && tid < 14) {
        s_sym_times[tid] = d_symbol_start_times[tid];
    }
    if (d_cfo_phasors != nullptr) {
        for (int i = tid; i < MIMO_CFO_DATA_PHASOR_COUNT; i += blockDim.x) {
            s_cfo_phasors[i] = d_cfo_phasors[i];
        }
    }
    __syncthreads();

    /* ================================================================
     * OPTIMIZATION #1: Cached scrambling sequence.
     * The normal path consumes the configure-time precomputed Gold sequence.
     * The fallback keeps the previous warp-cooperative generation for tests
     * and for configurations where the cached sequence is unavailable.
     * ================================================================ */
    int warp_re_base = (blockIdx.x * blockDim.x + warp_id * WARP_SIZE);

    if (scramble_seq == nullptr && lane == 0 && warp_re_base < nof_re) {
        /* Compute starting bit position for this warp */
        int bit_base = warp_re_base * mod_order + NC_SKIP;

        /* Advance LFSRs to starting position (O(log N)) - done once per warp! */
        uint32_t x1 = advance_x1_local(1, bit_base);
        uint32_t x2 = advance_x2_local(scramble_c_init, bit_base);

        /* Compute bits needed for this warp */
        int warp_end_re = min(warp_re_base + WARP_SIZE, nof_re);
        int bits_needed = (warp_end_re - warp_re_base) * mod_order;

        /* Generate scrambling bits for entire warp (max 256 bits) */
        for (int i = 0; i < bits_needed; i++) {
            s_scramble_bits[warp_id][i] = (x1 ^ x2) & 1;
            x1 = step_x1_local(x1);
            x2 = step_x2_local(x2);
        }
    }
    if (scramble_seq == nullptr) {
        __syncwarp();
    }

    /* ================================================================
     * Main processing - each thread handles one RE
     * ================================================================ */
    int re_idx = global_thread_id;
    float my_eq_noise_var = 0.0f;
    float my_evm_error = 0.0f;
    int my_evm_valid = 0;
    int my_valid = 0;

    if (re_idx < nof_re) {
        my_valid = 1;

        /* Get source RE index */
        int src_re = re_indices[re_idx];
        int subcarrier = src_re % symbol_stride;

        /* Compute relative indices for estimate lookup */
        int re_in_alloc = subcarrier - start_subcarrier;
        int prb_idx = re_in_alloc / 12;
        int sc_in_prb = re_in_alloc % 12;
        int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

        /* ================================================================
         * Time interpolation of channel estimates
         *
         * TIME_AVG=true:  Average all DMRS symbol estimates (matches CPU
         *                 "average" strategy). Gives σ²/N noise per estimate.
         * TIME_AVG=false: Linear interpolation between bracketing DMRS
         *                 symbols + edge extrapolation. Higher noise at edges.
         * ================================================================ */
        int ofdm_symbol = src_re / symbol_stride;

        cuFloatComplex h[8];
        if constexpr (TIME_AVG) {
            /* Time-average all DMRS estimates (matches CPU "average" strategy) */
            float inv_n = 1.0f / (float)NOF_DMRS;
            #pragma unroll
            for (int p = 0; p < NOF_PORTS; p++) {
                float sum_re = 0.0f, sum_im = 0.0f;
                #pragma unroll
                for (int d = 0; d < NOF_DMRS; d++) {
                    int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
                    __half2 hd = d_dmrs_estimates_fp16[est_idx];
                    float hr = __half2float(__low2half(hd));
                    float hi = __half2float(__high2half(hd));
                    /* CFO compensation: rotate each DMRS estimate to data symbol time */
                    if (s_cfo_hz != 0.0f) {
                        float cos_d = 1.0f;
                        float sin_d = 0.0f;
                        if (d_cfo_phasors != nullptr && (unsigned)ofdm_symbol < MAX_SYMBOLS) {
                            float2 ph = s_cfo_phasors[ofdm_symbol * MAX_DMRS_SYMBOLS + d];
                            cos_d = ph.x;
                            sin_d = ph.y;
                        } else if (use_cfo_fallback) {
                            int dmrs_sym = s_dmrs_indices[d] + start_symbol;
                            float dt = s_sym_times[ofdm_symbol] - s_sym_times[dmrs_sym];
                            sincosf(6.2831853071795864f * s_cfo_hz * dt, &sin_d, &cos_d);
                        }
                        float r2 = hr * cos_d - hi * sin_d;
                        float i2 = hr * sin_d + hi * cos_d;
                        hr = r2; hi = i2;
                    }
                    sum_re += hr;
                    sum_im += hi;
                }
                h[p] = make_cuFloatComplex(sum_re * inv_n, sum_im * inv_n);
            }
        } else {
            /* Linear time interpolation between bracketing DMRS symbols */

            /* Find bracketing DMRS symbols */
            int d_before = -1, d_after = -1;
            #pragma unroll
            for (int d = 0; d < NOF_DMRS; d++) {
                int dmrs_sym = s_dmrs_indices[d] + start_symbol;
                if (dmrs_sym <= ofdm_symbol) { d_before = d; }
                if (dmrs_sym >= ofdm_symbol && d_after < 0) { d_after = d; }
            }

            /* Edge cases: extrapolate from nearest two DMRS */
            if (d_before < 0) {
                d_before = d_after;
                d_after = (d_before + 1 < NOF_DMRS) ? d_before + 1 : d_before;
            }
            if (d_after < 0 || d_after == d_before) {
                d_after = d_before;
                d_before = (d_after > 0) ? d_after - 1 : d_after;
            }

            int dmrs_before = s_dmrs_indices[d_before] + start_symbol;
            int dmrs_after = s_dmrs_indices[d_after] + start_symbol;

            /* Compute interpolation weight */
            float weight = 0.0f;
            if (dmrs_after != dmrs_before) {
                weight = (float)(ofdm_symbol - dmrs_before) / (float)(dmrs_after - dmrs_before);
            }

            /* CFO compensation phasors: rotate each DMRS estimate to the data symbol time */
            float cos_b = 1.0f, sin_b = 0.0f;
            float cos_a = 1.0f, sin_a = 0.0f;
            if (s_cfo_hz != 0.0f) {
                if (d_cfo_phasors != nullptr && (unsigned)ofdm_symbol < MAX_SYMBOLS) {
                    float2 ph_b = s_cfo_phasors[ofdm_symbol * MAX_DMRS_SYMBOLS + d_before];
                    float2 ph_a = s_cfo_phasors[ofdm_symbol * MAX_DMRS_SYMBOLS + d_after];
                    cos_b = ph_b.x;
                    sin_b = ph_b.y;
                    cos_a = ph_a.x;
                    sin_a = ph_a.y;
                } else if (use_cfo_fallback) {
                    float dt_b = s_sym_times[ofdm_symbol] - s_sym_times[dmrs_before];
                    float dt_a = s_sym_times[ofdm_symbol] - s_sym_times[dmrs_after];
                    sincosf(6.2831853071795864f * s_cfo_hz * dt_b, &sin_b, &cos_b);
                    sincosf(6.2831853071795864f * s_cfo_hz * dt_a, &sin_a, &cos_a);
                }
            }

            #pragma unroll
            for (int p = 0; p < NOF_PORTS; p++) {
                int est_idx = prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
                __half2 h_b = d_dmrs_estimates_fp16[d_before * re_per_dmrs_sym + est_idx];
                __half2 h_a = d_dmrs_estimates_fp16[d_after * re_per_dmrs_sym + est_idx];
                float rb = __half2float(__low2half(h_b)), ib = __half2float(__high2half(h_b));
                float ra = __half2float(__low2half(h_a)), ia = __half2float(__high2half(h_a));

                float rb2 = rb * cos_b - ib * sin_b;
                float ib2 = rb * sin_b + ib * cos_b;
                float ra2 = ra * cos_a - ia * sin_a;
                float ia2 = ra * sin_a + ia * cos_a;

                h[p] = make_cuFloatComplex(rb2 + weight * (ra2 - rb2),
                                            ib2 + weight * (ia2 - ib2));
            }
        }

        /* Load received symbols from grid */
        cuFloatComplex y[8];
        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            int grid_idx = p * grid_stride + src_re;
            y[p] = cbf16_to_fp32(d_grid_cbf16[grid_idx]);
        }

        /* Equalization */
        cuFloatComplex eq;
        float eq_noise_var;
        equalize_symbol<NOF_PORTS, ALGORITHM>(y, h, s_noise_vars, tx_scaling, eq, eq_noise_var);
        my_eq_noise_var = sinr_noise_var_is_valid(eq_noise_var) ? eq_noise_var : 0.0f;
        if (d_evm_error_sum && d_evm_symbol_count && isfinite(eq.x) && isfinite(eq.y)) {
            my_evm_error = evm_error_power(eq, mod_order);
            my_evm_valid = 1;
        }

        /* Write equalized symbols to external buffer if requested */
        if (d_eq_symbols_out) {
            d_eq_symbols_out[re_idx] = make_float2(eq.x, eq.y);
            d_eq_noise_var_out[re_idx] = eq_noise_var;
        }

        /* Soft demodulation */
        float llr_vals[8];
        float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);

        if (mod_order == 2) {
            /* QPSK: L = 2*sqrt(2) * symbol / noise_var */
            float scale = 2.0f * 1.41421356f * inv_noise;
            llr_vals[0] = eq.x * scale;
            llr_vals[1] = eq.y * scale;
        } else if (mod_order == 4) {
            /* 16QAM: piecewise-linear soft demodulation per TS 38.211.
             * M = 1/sqrt(10), GAIN = 4*M, THRESHOLD = 2*M. */
            const float M_SQRT1_10 = 0.31622776601683794f;
            const float GAIN_FIRST = 4.0f * M_SQRT1_10;
            const float THRESHOLD = 2.0f * M_SQRT1_10;
            const float CONST_0_8 = 0.8f;

            float abs_re = fabsf(eq.x);
            float abs_im = fabsf(eq.y);

            float l_first_re = GAIN_FIRST * eq.x;
            float l_first_im = GAIN_FIRST * eq.y;

            float l_second_re = 2.0f * l_first_re - copysignf(CONST_0_8, eq.x);
            float l_second_im = 2.0f * l_first_im - copysignf(CONST_0_8, eq.y);

            float l_01_re = (abs_re > THRESHOLD) ? l_second_re : l_first_re;
            float l_01_im = (abs_im > THRESHOLD) ? l_second_im : l_first_im;

            float l_23_re = CONST_0_8 - fabsf(l_first_re);
            float l_23_im = CONST_0_8 - fabsf(l_first_im);

            llr_vals[0] = l_01_re * inv_noise;
            llr_vals[1] = l_01_im * inv_noise;
            llr_vals[2] = l_23_re * inv_noise;
            llr_vals[3] = l_23_im * inv_noise;
        } else if (mod_order == 6) {
            /* 64QAM: piecewise-linear approximation per TS 38.211.
             * Uses optimal interval functions matching CPU implementation for
             * accurate LLR computation at all SINR levels. */
            soft_demod_64qam_piecewise(eq, inv_noise, llr_vals);
        } else if (mod_order == 8) {
            /* 256QAM: piecewise-linear approximation matching CPU implementation */
            soft_demod_256qam_piecewise(eq, inv_noise, llr_vals);
        }

        /* ================================================================
         * OPTIMIZATION #1 cont'd: Apply descrambling from cached sequence
         * or from fallback shared-memory bits.
         * ================================================================ */
        int my_bit_offset = lane * mod_order;
        int llr_base = re_idx * mod_order;
        uint64_t scr_window = 0;
        if (scramble_seq != nullptr) {
            int word_idx = llr_base >> 5;
            int bit_offset = llr_base & 31;
            uint32_t scr_word = scramble_seq[word_idx];
            uint32_t scr_word_next = ((bit_offset + mod_order) > 32) ? scramble_seq[word_idx + 1] : 0;
            scr_window = (static_cast<uint64_t>(scr_word_next) << 32) | scr_word;
            scr_window >>= bit_offset;
        }

        /* ================================================================
         * OPTIMIZATION #4: Vectorized LLR writes using half2
         * ================================================================ */
        if (mod_order == 8) {
            /* 256QAM: 8 LLRs = 4 × half2 writes */
            half2 llr_pack[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float llr0 = llr_vals[i*2];
                float llr1 = llr_vals[i*2 + 1];

                /* Apply descrambling */
                uint32_t scr0 = (scramble_seq != nullptr) ? ((scr_window >> (i*2)) & 1u) :
                                                           s_scramble_bits[warp_id][my_bit_offset + i*2];
                uint32_t scr1 = (scramble_seq != nullptr) ? ((scr_window >> (i*2 + 1)) & 1u) :
                                                           s_scramble_bits[warp_id][my_bit_offset + i*2 + 1];
                llr0 = scr0 ? -llr0 : llr0;
                llr1 = scr1 ? -llr1 : llr1;

                /* Clamp to FP16 range */
                llr0 = fmaxf(-65504.0f, fminf(65504.0f, llr0));
                llr1 = fmaxf(-65504.0f, fminf(65504.0f, llr1));

                llr_pack[i] = __floats2half2_rn(llr0, llr1);
            }
            /* Vectorized store - 4 × 4-byte writes */
            *((half2*)(llrs_half + llr_base + 0)) = llr_pack[0];
            *((half2*)(llrs_half + llr_base + 2)) = llr_pack[1];
            *((half2*)(llrs_half + llr_base + 4)) = llr_pack[2];
            *((half2*)(llrs_half + llr_base + 6)) = llr_pack[3];
        } else if (mod_order == 4) {
            /* 16QAM: 4 LLRs = 2 × half2 writes */
            half2 llr_pack[2];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                float llr0 = llr_vals[i*2];
                float llr1 = llr_vals[i*2 + 1];
                uint32_t scr0 = (scramble_seq != nullptr) ? ((scr_window >> (i*2)) & 1u) :
                                                           s_scramble_bits[warp_id][my_bit_offset + i*2];
                uint32_t scr1 = (scramble_seq != nullptr) ? ((scr_window >> (i*2 + 1)) & 1u) :
                                                           s_scramble_bits[warp_id][my_bit_offset + i*2 + 1];
                llr0 = scr0 ? -llr0 : llr0;
                llr1 = scr1 ? -llr1 : llr1;
                llr0 = fmaxf(-65504.0f, fminf(65504.0f, llr0));
                llr1 = fmaxf(-65504.0f, fminf(65504.0f, llr1));
                llr_pack[i] = __floats2half2_rn(llr0, llr1);
            }
            *((half2*)(llrs_half + llr_base + 0)) = llr_pack[0];
            *((half2*)(llrs_half + llr_base + 2)) = llr_pack[1];
        } else if (mod_order == 6) {
            /* 64QAM: 6 LLRs = 3 × half2 writes */
            half2 llr_pack[3];
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                float llr0 = llr_vals[i*2];
                float llr1 = llr_vals[i*2 + 1];
                uint32_t scr0 = (scramble_seq != nullptr) ? ((scr_window >> (i*2)) & 1u) :
                                                           s_scramble_bits[warp_id][my_bit_offset + i*2];
                uint32_t scr1 = (scramble_seq != nullptr) ? ((scr_window >> (i*2 + 1)) & 1u) :
                                                           s_scramble_bits[warp_id][my_bit_offset + i*2 + 1];
                llr0 = scr0 ? -llr0 : llr0;
                llr1 = scr1 ? -llr1 : llr1;
                llr0 = fmaxf(-65504.0f, fminf(65504.0f, llr0));
                llr1 = fmaxf(-65504.0f, fminf(65504.0f, llr1));
                llr_pack[i] = __floats2half2_rn(llr0, llr1);
            }
            *((half2*)(llrs_half + llr_base + 0)) = llr_pack[0];
            *((half2*)(llrs_half + llr_base + 2)) = llr_pack[1];
            *((half2*)(llrs_half + llr_base + 4)) = llr_pack[2];
        } else {
            /* QPSK: 2 LLRs = 1 × half2 write */
            float llr0 = llr_vals[0];
            float llr1 = llr_vals[1];
            uint32_t scr0 = (scramble_seq != nullptr) ? (scr_window & 1u) :
                                                       s_scramble_bits[warp_id][my_bit_offset];
            uint32_t scr1 = (scramble_seq != nullptr) ? ((scr_window >> 1) & 1u) :
                                                       s_scramble_bits[warp_id][my_bit_offset + 1];
            llr0 = scr0 ? -llr0 : llr0;
            llr1 = scr1 ? -llr1 : llr1;
            llr0 = fmaxf(-65504.0f, fminf(65504.0f, llr0));
            llr1 = fmaxf(-65504.0f, fminf(65504.0f, llr1));
            *((half2*)(llrs_half + llr_base)) = __floats2half2_rn(llr0, llr1);
        }
    }

    /* ================================================================
     * OPTIMIZATION #2: Block-level noise variance reduction
     * Warp shuffle → shared memory → single atomic per block
     * ================================================================ */

    /* Warp-level reduction using shuffle (deep-fade REs contribute 0) */
    float warp_sum = warp_reduce_sum_f32(my_eq_noise_var);
    int sinr_valid = my_valid && (my_eq_noise_var > 0.0f);
    int warp_count = __popc(__ballot_sync(0xffffffff, sinr_valid));
    float warp_evm_sum = warp_reduce_sum_f32(my_evm_error);
    int warp_evm_count = __popc(__ballot_sync(0xffffffff, my_evm_valid != 0));

    /* Lane 0 stores warp result to shared memory */
    if (lane == 0) {
        s_noise_var_sums[warp_id] = warp_sum;
        s_noise_var_counts[warp_id] = warp_count;
        s_evm_error_sums[warp_id] = warp_evm_sum;
        s_evm_symbol_counts[warp_id] = warp_evm_count;
    }
    __syncthreads();

    /* Thread 0 does final block reduction and single global atomic */
    if (tid == 0) {
        float block_sum = 0.0f;
        int block_count = 0;
        float block_evm_sum = 0.0f;
        int block_evm_count = 0;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            block_sum += s_noise_var_sums[w];
            block_count += s_noise_var_counts[w];
            block_evm_sum += s_evm_error_sums[w];
            block_evm_count += s_evm_symbol_counts[w];
        }
        if (block_count > 0) {
            atomicAdd(d_eq_noise_var_sum, block_sum);
            atomicAdd(d_eq_noise_var_count, (unsigned int)block_count);
        }
        if (block_evm_count > 0) {
            atomicAdd(d_evm_error_sum, block_evm_sum);
            atomicAdd(d_evm_symbol_count, (unsigned int)block_evm_count);
        }
    }
}

/**
 * Ultra-optimized E2E kernel with on-the-fly scrambling, INT8 output.
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void kernel_fused_e2e_onthefly_scramble_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    uint32_t scramble_c_init,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int mod_order,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (re_idx >= nof_re) return;

    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

    /* Time interpolation */
    cuFloatComplex h[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum = 0.0f, h_imag_sum = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
            __half2 h_fp16 = d_dmrs_estimates_fp16[est_idx];
            h_real_sum += __half2float(__low2half(h_fp16));
            h_imag_sum += __half2float(__high2half(h_fp16));
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        h[p] = make_cuFloatComplex(h_real_sum * inv_n, h_imag_sum * inv_n);
    }

    /* Load received symbols */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    /* Equalization */
    cuFloatComplex eq;
    float eq_noise_var;
    equalize_symbol<NOF_PORTS, ALGORITHM>(y, h, s_noise_vars, tx_scaling, eq, eq_noise_var);

    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, eq_noise_var);

    /* Soft demodulation */
    float llr_vals[8];
    float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);

    if (mod_order == 2) {
        float scale = inv_noise * 2.8284271f  /* 2*sqrt(2) - matches CPU QPSK demod */;
        llr_vals[0] = eq.x * scale;
        llr_vals[1] = eq.y * scale;
    } else if (mod_order == 4) {
        float scale = inv_noise * 0.6324555f;
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.6324555f * 2.0f - fabsf(eq.x)) * scale * 2.0f;
        llr_vals[3] = (0.6324555f * 2.0f - fabsf(eq.y)) * scale * 2.0f;
    } else if (mod_order == 6) {
        // Proper max-log LLR computation for 64QAM
        const float s = 0.15430335f;  // 1/sqrt(42)
        const float s1 = s, s3 = 3.0f * s, s5 = 5.0f * s, s7 = 7.0f * s;
        float yi = eq.x, yq = eq.y;
        #define SQ(x) ((x)*(x))
        float d0_b0 = fminf(fminf(SQ(yi-s1),SQ(yi-s3)), fminf(SQ(yi-s5),SQ(yi-s7)));
        float d1_b0 = fminf(fminf(SQ(yi+s1),SQ(yi+s3)), fminf(SQ(yi+s5),SQ(yi+s7)));
        llr_vals[0] = (d1_b0 - d0_b0) * inv_noise;
        float d0_b1 = fminf(fminf(SQ(yq-s1),SQ(yq-s3)), fminf(SQ(yq-s5),SQ(yq-s7)));
        float d1_b1 = fminf(fminf(SQ(yq+s1),SQ(yq+s3)), fminf(SQ(yq+s5),SQ(yq+s7)));
        llr_vals[1] = (d1_b1 - d0_b1) * inv_noise;
        float d0_b2 = fminf(fminf(SQ(yi-s1),SQ(yi+s1)), fminf(SQ(yi-s3),SQ(yi+s3)));
        float d1_b2 = fminf(fminf(SQ(yi-s5),SQ(yi+s5)), fminf(SQ(yi-s7),SQ(yi+s7)));
        llr_vals[2] = (d1_b2 - d0_b2) * inv_noise;
        float d0_b3 = fminf(fminf(SQ(yq-s1),SQ(yq+s1)), fminf(SQ(yq-s3),SQ(yq+s3)));
        float d1_b3 = fminf(fminf(SQ(yq-s5),SQ(yq+s5)), fminf(SQ(yq-s7),SQ(yq+s7)));
        llr_vals[3] = (d1_b3 - d0_b3) * inv_noise;
        float d0_b4 = fminf(fminf(SQ(yi-s1),SQ(yi+s1)), fminf(SQ(yi-s5),SQ(yi+s5)));
        float d1_b4 = fminf(fminf(SQ(yi-s3),SQ(yi+s3)), fminf(SQ(yi-s7),SQ(yi+s7)));
        llr_vals[4] = (d1_b4 - d0_b4) * inv_noise;
        float d0_b5 = fminf(fminf(SQ(yq-s1),SQ(yq+s1)), fminf(SQ(yq-s5),SQ(yq+s5)));
        float d1_b5 = fminf(fminf(SQ(yq-s3),SQ(yq+s3)), fminf(SQ(yq-s7),SQ(yq+s7)));
        llr_vals[5] = (d1_b5 - d0_b5) * inv_noise;
        #undef SQ
    } else if (mod_order == 8) {
        float scale = inv_noise * 0.07669650f;
        float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.07669650f * 8.0f - abs_re) * scale * 2.0f;
        llr_vals[3] = (0.07669650f * 8.0f - abs_im) * scale * 2.0f;
        float level2 = 0.07669650f * 4.0f;
        llr_vals[4] = (level2 - fabsf(abs_re - level2 * 2.0f)) * scale * 2.0f;
        llr_vals[5] = (level2 - fabsf(abs_im - level2 * 2.0f)) * scale * 2.0f;
        float level3 = 0.07669650f * 2.0f;
        llr_vals[6] = (level3 - fabsf(fabsf(abs_re - level2 * 2.0f) - level3 * 2.0f)) * scale * 2.0f;
        llr_vals[7] = (level3 - fabsf(fabsf(abs_im - level2 * 2.0f) - level3 * 2.0f)) * scale * 2.0f;
    }

    /* On-the-fly descrambling with INT8 quantization */
    int llr_base = re_idx * mod_order;
    uint32_t x1 = 1;
    uint32_t x2 = scramble_c_init & 0x7FFFFFFF;
    int total_advance = NC_SKIP + llr_base;
    x1 = advance_x1_local(x1, total_advance);
    x2 = advance_x2_local(x2, total_advance);

    const float INT8_SCALE = 4.0f;  /* Quantization scale - must match LDPC decoder expectation */
    for (int b = 0; b < mod_order; b++) {
        uint32_t scr_bit = (x1 ^ x2) & 1;
        float llr = scr_bit ? -llr_vals[b] : llr_vals[b];
        llr = fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE));
        llrs_int8[llr_base + b] = (int8_t)rintf(llr);

        x1 = step_x1_local(x1);
        x2 = step_x2_local(x2);
    }
}

/**
 * Ultra-optimized E2E kernel with WARP-COOPERATIVE scrambling.
 *
 * Key optimization: Lane 0 of each warp generates scrambling bits for all 32 threads,
 * then broadcasts via warp shuffle. This reduces LFSR jumps from 32 to 1 per warp.
 *
 * For 64QAM (mod_order=6), each warp needs 192 scrambling bits.
 * Lane 0 does one O(log N) jump then 192 sequential LFSR steps.
 * Other 31 lanes save their O(log N) jumps entirely.
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void kernel_fused_e2e_warp_scramble_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    uint32_t scramble_c_init,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int mod_order,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    unsigned int mask = __activemask();

    /* Load noise variances to shared memory */
    __shared__ float s_noise_vars[8];
    /* Shared memory for warp scrambling bits: max 8 words per warp (256 bits) */
    __shared__ uint32_t s_scr_bits[8 * 8];  /* 8 warps max × 8 words */

    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }

    /* Lane 0 of each warp generates scrambling bits for the entire warp */
    int warp_base_re = (blockIdx.x * blockDim.x + warp_id * 32);
    int warp_nof_re = min(32, nof_re - warp_base_re);

    if (lane == 0 && warp_nof_re > 0) {
        /* Jump LFSR to warp's starting bit position */
        int warp_base_bit = warp_base_re * mod_order;
        uint32_t x1 = 1;
        uint32_t x2 = scramble_c_init & 0x7FFFFFFF;
        int total_advance = NC_SKIP + warp_base_bit;
        x1 = advance_x1_local(x1, total_advance);
        x2 = advance_x2_local(x2, total_advance);

        /* Generate all bits needed for this warp (32 REs × mod_order bits) */
        int total_bits = warp_nof_re * mod_order;
        int nof_words = (total_bits + 31) / 32;

        for (int w = 0; w < nof_words && w < 8; w++) {
            uint32_t packed = 0;
            int bits_in_word = min(32, total_bits - w * 32);
            for (int b = 0; b < bits_in_word; b++) {
                uint32_t scr_bit = (x1 ^ x2) & 1;
                packed |= (scr_bit << b);
                /* Step LFSRs */
                uint32_t new_x1 = ((x1 >> 3) ^ x1) & 1;
                x1 = (x1 >> 1) | (new_x1 << 30);
                uint32_t new_x2 = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
                x2 = (x2 >> 1) | (new_x2 << 30);
            }
            s_scr_bits[warp_id * 8 + w] = packed;
        }
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    /* Get source RE index */
    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

    /* Time interpolation */
    cuFloatComplex h[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum = 0.0f, h_imag_sum = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
            __half2 h_fp16 = d_dmrs_estimates_fp16[est_idx];
            h_real_sum += __half2float(__low2half(h_fp16));
            h_imag_sum += __half2float(__high2half(h_fp16));
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        h[p] = make_cuFloatComplex(h_real_sum * inv_n, h_imag_sum * inv_n);
    }

    /* Load received symbols */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    /* Equalization */
    cuFloatComplex eq;
    float eq_noise_var;
    equalize_symbol<NOF_PORTS, ALGORITHM>(y, h, s_noise_vars, tx_scaling, eq, eq_noise_var);

    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, eq_noise_var);

    /* Soft demodulation */
    float llr_vals[8];
    float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);

    if (mod_order == 2) {
        float scale = inv_noise * 2.8284271f  /* 2*sqrt(2) - matches CPU QPSK demod */;
        llr_vals[0] = eq.x * scale;
        llr_vals[1] = eq.y * scale;
    } else if (mod_order == 4) {
        float scale = inv_noise * 0.6324555f;
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.6324555f * 2.0f - fabsf(eq.x)) * scale * 2.0f;
        llr_vals[3] = (0.6324555f * 2.0f - fabsf(eq.y)) * scale * 2.0f;
    } else if (mod_order == 6) {
        // Proper max-log LLR computation for 64QAM
        const float s = 0.15430335f;  // 1/sqrt(42)
        const float s1 = s, s3 = 3.0f * s, s5 = 5.0f * s, s7 = 7.0f * s;
        float yi = eq.x, yq = eq.y;
        #define SQ(x) ((x)*(x))
        float d0_b0 = fminf(fminf(SQ(yi-s1),SQ(yi-s3)), fminf(SQ(yi-s5),SQ(yi-s7)));
        float d1_b0 = fminf(fminf(SQ(yi+s1),SQ(yi+s3)), fminf(SQ(yi+s5),SQ(yi+s7)));
        llr_vals[0] = (d1_b0 - d0_b0) * inv_noise;
        float d0_b1 = fminf(fminf(SQ(yq-s1),SQ(yq-s3)), fminf(SQ(yq-s5),SQ(yq-s7)));
        float d1_b1 = fminf(fminf(SQ(yq+s1),SQ(yq+s3)), fminf(SQ(yq+s5),SQ(yq+s7)));
        llr_vals[1] = (d1_b1 - d0_b1) * inv_noise;
        float d0_b2 = fminf(fminf(SQ(yi-s1),SQ(yi+s1)), fminf(SQ(yi-s3),SQ(yi+s3)));
        float d1_b2 = fminf(fminf(SQ(yi-s5),SQ(yi+s5)), fminf(SQ(yi-s7),SQ(yi+s7)));
        llr_vals[2] = (d1_b2 - d0_b2) * inv_noise;
        float d0_b3 = fminf(fminf(SQ(yq-s1),SQ(yq+s1)), fminf(SQ(yq-s3),SQ(yq+s3)));
        float d1_b3 = fminf(fminf(SQ(yq-s5),SQ(yq+s5)), fminf(SQ(yq-s7),SQ(yq+s7)));
        llr_vals[3] = (d1_b3 - d0_b3) * inv_noise;
        float d0_b4 = fminf(fminf(SQ(yi-s1),SQ(yi+s1)), fminf(SQ(yi-s5),SQ(yi+s5)));
        float d1_b4 = fminf(fminf(SQ(yi-s3),SQ(yi+s3)), fminf(SQ(yi-s7),SQ(yi+s7)));
        llr_vals[4] = (d1_b4 - d0_b4) * inv_noise;
        float d0_b5 = fminf(fminf(SQ(yq-s1),SQ(yq+s1)), fminf(SQ(yq-s5),SQ(yq+s5)));
        float d1_b5 = fminf(fminf(SQ(yq-s3),SQ(yq+s3)), fminf(SQ(yq-s7),SQ(yq+s7)));
        llr_vals[5] = (d1_b5 - d0_b5) * inv_noise;
        #undef SQ
    } else if (mod_order == 8) {
        float scale = inv_noise * 0.07669650f;
        float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.07669650f * 8.0f - abs_re) * scale * 2.0f;
        llr_vals[3] = (0.07669650f * 8.0f - abs_im) * scale * 2.0f;
        float level2 = 0.07669650f * 4.0f;
        llr_vals[4] = (level2 - fabsf(abs_re - level2 * 2.0f)) * scale * 2.0f;
        llr_vals[5] = (level2 - fabsf(abs_im - level2 * 2.0f)) * scale * 2.0f;
        float level3 = 0.07669650f * 2.0f;
        llr_vals[6] = (level3 - fabsf(fabsf(abs_re - level2 * 2.0f) - level3 * 2.0f)) * scale * 2.0f;
        llr_vals[7] = (level3 - fabsf(fabsf(abs_im - level2 * 2.0f) - level3 * 2.0f)) * scale * 2.0f;
    }

    /* Read scrambling bits from shared memory (generated by lane 0) */
    int lane_bit_start = lane * mod_order;
    int llr_base = re_idx * mod_order;
    const float INT8_SCALE = 4.0f;

    for (int b = 0; b < mod_order; b++) {
        int bit_idx = lane_bit_start + b;
        int word_idx = bit_idx / 32;
        int bit_offset = bit_idx % 32;
        uint32_t scr_word = s_scr_bits[warp_id * 8 + word_idx];
        uint32_t scr_bit = (scr_word >> bit_offset) & 1;

        float llr = scr_bit ? -llr_vals[b] : llr_vals[b];
        llr = fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE));
        llrs_int8[llr_base + b] = (int8_t)rintf(llr);
    }
}

/* ============================================================================
 * ULTRA-OPTIMIZED E2E KERNELS with compile-time modulation order dispatch
 *
 * These kernels eliminate runtime conditionals by using template specialization
 * for modulation order. Combined with launch bounds for optimal GPU occupancy.
 *
 * Key optimizations:
 * - __launch_bounds__(256, 4) for optimal register allocation
 * - Compile-time soft demod constants (no runtime conditionals)
 * - Pre-computed QAM scaling factors
 * - Template dispatch by port count AND modulation order
 * - Vectorized LLR output where possible
 * ============================================================================ */

/* Pre-computed QAM scaling constants (compile-time) */
constexpr float QPSK_SCALE = 2.8284271247461903f;   /* 2 * sqrt(2) - matches srsRAN CPU demod */
[[maybe_unused]] constexpr float QAM16_SCALE = 0.6324555320336759f;  /* 1/sqrt(10) */
[[maybe_unused]] constexpr float QAM64_SCALE = 0.4082482904638631f;  /* 1/sqrt(42) */
[[maybe_unused]] constexpr float QAM256_SCALE = 0.2581988897471611f; /* 1/sqrt(170) */

__device__ __forceinline__ void store_descrambled_half_llr(
    __half* __restrict__ llrs_half,
    const uint32_t* __restrict__ scramble_seq,
    int bit_start,
    const float* llr_vals,
    int mod_order)
{
    int word_idx = bit_start / 32;
    int bit_off = bit_start % 32;
    uint32_t scr_word = scramble_seq[word_idx];
    uint32_t scr_word_next = scramble_seq[word_idx + 1];
    uint64_t scr_window = ((uint64_t)scr_word_next << 32) | scr_word;

    #pragma unroll
    for (int b = 0; b < 8; b++) {
        if (b < mod_order) {
            uint32_t scr = (scr_window >> (bit_off + b)) & 1;
            float llr = scr ? -llr_vals[b] : llr_vals[b];
            llr = fmaxf(-65504.0f, fminf(65504.0f, llr));
            llrs_half[bit_start + b] = __float2half(llr);
        }
    }
}

__device__ __forceinline__ void soft_demod_symbol_runtime(cuFloatComplex eq, float eq_noise_var, int mod_order, float* llr_vals)
{
    float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);

    if (mod_order == 2) {
        float scale = QPSK_SCALE * inv_noise;
        llr_vals[0] = eq.x * scale;
        llr_vals[1] = eq.y * scale;
    } else if (mod_order == 4) {
        const float M_SQRT1_10 = 0.31622776601683794f;
        const float GAIN_FIRST = 4.0f * M_SQRT1_10;
        const float THRESHOLD = 2.0f * M_SQRT1_10;
        const float CONST_0_8 = 0.8f;

        float abs_re = fabsf(eq.x);
        float abs_im = fabsf(eq.y);

        float l_first_re = GAIN_FIRST * eq.x;
        float l_first_im = GAIN_FIRST * eq.y;
        float l_second_re = 2.0f * l_first_re - copysignf(CONST_0_8, eq.x);
        float l_second_im = 2.0f * l_first_im - copysignf(CONST_0_8, eq.y);

        llr_vals[0] = ((abs_re > THRESHOLD) ? l_second_re : l_first_re) * inv_noise;
        llr_vals[1] = ((abs_im > THRESHOLD) ? l_second_im : l_first_im) * inv_noise;
        llr_vals[2] = (CONST_0_8 - fabsf(l_first_re)) * inv_noise;
        llr_vals[3] = (CONST_0_8 - fabsf(l_first_im)) * inv_noise;
    } else if (mod_order == 6) {
        soft_demod_64qam_piecewise(eq, inv_noise, llr_vals);
    } else if (mod_order == 8) {
        soft_demod_256qam_piecewise(eq, inv_noise, llr_vals);
    }
}

/* ============================================================================
 * WARP REDUCTION HELPERS for efficient SINR accumulation
 *
 * These functions reduce per-thread values across a warp using shuffle
 * instructions, then use a single atomicAdd per warp instead of per-thread.
 * This reduces atomic contention by 32x.
 * ============================================================================ */

__device__ __forceinline__ float warp_reduce_sum(float val) {
    /* Butterfly reduction across 32-thread warp */
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  /* Only lane 0 has the final sum */
}

__device__ __forceinline__ unsigned int warp_reduce_sum_uint(unsigned int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/* Block-level reduction using shared memory for inter-warp communication */
template<int BLOCK_SIZE>
__device__ __forceinline__ void block_reduce_atomic(
    float val, unsigned int count,
    float* __restrict__ d_sum, unsigned int* __restrict__ d_count)
{
    __shared__ float s_sum[32];    /* One slot per warp */
    __shared__ unsigned int s_count[32];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = BLOCK_SIZE >> 5;

    /* Warp-level reduction */
    float warp_sum = warp_reduce_sum(val);
    unsigned int warp_count = warp_reduce_sum_uint(count);

    /* Lane 0 of each warp writes to shared memory */
    if (lane == 0) {
        s_sum[warp_id] = warp_sum;
        s_count[warp_id] = warp_count;
    }
    __syncthreads();

    /* First warp reduces across all warps */
    if (warp_id == 0) {
        float final_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        unsigned int final_count = (lane < num_warps) ? s_count[lane] : 0;

        final_sum = warp_reduce_sum(final_sum);
        final_count = warp_reduce_sum_uint(final_count);

        /* Only thread 0 does the atomic */
        if (lane == 0) {
            atomicAdd(d_sum, final_sum);
            atomicAdd(d_count, final_count);
        }
    }
}

/**
 * @brief Ultra-optimized QPSK E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_ultra_e2e_qpsk_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    const int* __restrict__ d_dmrs_indices)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    __shared__ int s_dmrs_indices[4];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    if (threadIdx.x < 4 && threadIdx.x < nof_dmrs_symbols) {
        s_dmrs_indices[threadIdx.x] = d_dmrs_indices[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    /* Get source RE index */
    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;
    int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

    /* Time-average all DMRS estimates (matches CPU "average" strategy) */
    cuFloatComplex h[8];
    float inv_n = 1.0f / (float)nof_dmrs_symbols;
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float sum_re = 0.0f, sum_im = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
            __half2 hd = d_dmrs_estimates_fp16[est_idx];
            sum_re += __half2float(__low2half(hd));
            sum_im += __half2float(__high2half(hd));
        }
        h[p] = make_cuFloatComplex(sum_re * inv_n, sum_im * inv_n);
    }

    /* Load received symbols */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    /* Equalization */
    cuFloatComplex eq;
    float eq_noise_var;

    /* OPTIMIZED MMSE with gain normalization (matches CPU equalize_mmse_1xn.h)
     *
     * Key insight: MMSE weights don't sum to unity, causing signal attenuation.
     * We normalize by the effective combining gain: G = Σ[|H_p|² / (|H_p|² + σ_p²)]
     * This preserves constellation scaling for soft demodulation.
     */
    if constexpr (NOF_PORTS == 1) {
        float h_sq = h[0].x * h[0].x + h[0].y * h[0].y;
        float noise = s_noise_vars[0];

        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            /* ZF: w = h* / |h|² */
            float inv_h_sq = 1.0f / fmaxf(h_sq, 1e-10f);
            eq.x = (y[0].x * h[0].x + y[0].y * h[0].y) * inv_h_sq / tx_scaling;
            eq.y = (y[0].y * h[0].x - y[0].x * h[0].y) * inv_h_sq / tx_scaling;
            eq_noise_var = noise * inv_h_sq / (tx_scaling * tx_scaling);
        } else {
            /* MMSE with gain normalization */
            float denom = h_sq + noise;
            float inv_denom = 1.0f / fmaxf(denom, 1e-10f);
            float mmse_gain = h_sq * inv_denom;  /* |H|² / (|H|² + σ²) */
            float gain_norm = (mmse_gain > 1e-10f) ? (1.0f / mmse_gain) : 1.0f;

            eq.x = (y[0].x * h[0].x + y[0].y * h[0].y) * inv_denom * gain_norm / tx_scaling;
            eq.y = (y[0].y * h[0].x - y[0].x * h[0].y) * inv_denom * gain_norm / tx_scaling;

            /* Post-EQ noise variance: nvar = |H|² * σ² / (|H|² + σ²)² * gain_norm² / tx_scaling² */
            float nvar_acc = (h_sq * noise) * inv_denom * inv_denom;
            eq_noise_var = nvar_acc * gain_norm * gain_norm / (tx_scaling * tx_scaling);
        }
        eq_noise_var = (eq_noise_var > 1e-10f) ? eq_noise_var : EQ_NOISE_VAR_DEEP_FADE;
    } else {
        /* Multi-port MMSE with per-port gain normalization */
        float H_sq = 0.0f;
        float mmse_gain = 0.0f;
        float nvar_acc = 0.0f;
        cuFloatComplex hhy = make_cuFloatComplex(0.0f, 0.0f);

        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            float h_norm_sq = h[p].x * h[p].x + h[p].y * h[p].y;
            float noise_p = s_noise_vars[p];
            H_sq += h_norm_sq;
            hhy.x += h[p].x * y[p].x + h[p].y * y[p].y;
            hhy.y += h[p].x * y[p].y - h[p].y * y[p].x;

            if constexpr (ALGORITHM != EQUALIZER_ZF) {
                float denom_p = h_norm_sq + noise_p;
                if (denom_p > 1e-10f) {
                    mmse_gain += h_norm_sq / denom_p;
                    nvar_acc += (h_norm_sq * noise_p) / (denom_p * denom_p);
                }
            }
        }

        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            float inv_H_sq = 1.0f / fmaxf(H_sq, 1e-10f);
            eq.x = hhy.x * inv_H_sq / tx_scaling;
            eq.y = hhy.y * inv_H_sq / tx_scaling;
            float avg_noise = 0.0f;
            for (int p = 0; p < NOF_PORTS; p++) avg_noise += s_noise_vars[p];
            eq_noise_var = avg_noise / NOF_PORTS * inv_H_sq / (tx_scaling * tx_scaling);
        } else {
            float avg_noise = 0.0f;
            for (int p = 0; p < NOF_PORTS; p++) avg_noise += s_noise_vars[p];
            avg_noise /= NOF_PORTS;

            float denom = H_sq + avg_noise;
            float inv_denom = 1.0f / fmaxf(denom, 1e-10f);
            float gain_norm = (mmse_gain > 1e-10f) ? (1.0f / mmse_gain) : 1.0f;

            eq.x = hhy.x * inv_denom * gain_norm / tx_scaling;
            eq.y = hhy.y * inv_denom * gain_norm / tx_scaling;
            eq_noise_var = nvar_acc * gain_norm * gain_norm / (tx_scaling * tx_scaling);
        }
        eq_noise_var = (eq_noise_var > 1e-10f) ? eq_noise_var : EQ_NOISE_VAR_DEEP_FADE;
    }

    /* Accumulate equalized-noise statistics for slot-level SINR reporting. */
    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, eq_noise_var);

    /* QPSK soft demodulation - compile-time constants */
    float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);
    float llr_scale = inv_noise * QPSK_SCALE;
    float llr0 = eq.x * llr_scale;
    float llr1 = eq.y * llr_scale;

    /* Descrambling + INT8 quantization */
    int bit_start = re_idx * 2;
    int word_idx0 = bit_start / 32;
    int bit_off0 = bit_start % 32;
    uint32_t scr_word = scramble_seq[word_idx0];
    uint32_t scr0 = (scr_word >> bit_off0) & 1;
    uint32_t scr1 = (scr_word >> (bit_off0 + 1)) & 1;

    constexpr float INT8_SCALE = 6.0f;  /* Matches srsRAN: value/20.0*120 = value*6 */
    llr0 = scr0 ? -llr0 : llr0;
    llr1 = scr1 ? -llr1 : llr1;

    int llr_base = re_idx * 2;
    llrs_int8[llr_base + 0] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr0 * INT8_SCALE)));
    llrs_int8[llr_base + 1] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr1 * INT8_SCALE)));
}

/**
 * @brief Ultra-optimized 16QAM E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_ultra_e2e_16qam_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    const int* __restrict__ d_dmrs_indices)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    __shared__ int s_dmrs_indices[4];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    if (threadIdx.x < 4 && threadIdx.x < nof_dmrs_symbols) {
        s_dmrs_indices[threadIdx.x] = d_dmrs_indices[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    /* Get source RE index */
    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;
    int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

    /* Time-average all DMRS estimates (matches CPU "average" strategy) */
    cuFloatComplex h[8];
    float inv_n = 1.0f / (float)nof_dmrs_symbols;
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float sum_re = 0.0f, sum_im = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
            __half2 hd = d_dmrs_estimates_fp16[est_idx];
            sum_re += __half2float(__low2half(hd));
            sum_im += __half2float(__high2half(hd));
        }
        h[p] = make_cuFloatComplex(sum_re * inv_n, sum_im * inv_n);
    }

    /* Load received symbols */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    /* OPTIMIZED MMSE with gain normalization (matches CPU equalize_mmse_1xn.h) */
    cuFloatComplex eq;
    float eq_noise_var;

    if constexpr (NOF_PORTS == 1) {
        float h_sq = h[0].x * h[0].x + h[0].y * h[0].y;
        float noise = s_noise_vars[0];

        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            float inv_h_sq = 1.0f / fmaxf(h_sq, 1e-10f);
            eq.x = (y[0].x * h[0].x + y[0].y * h[0].y) * inv_h_sq / tx_scaling;
            eq.y = (y[0].y * h[0].x - y[0].x * h[0].y) * inv_h_sq / tx_scaling;
            eq_noise_var = noise * inv_h_sq / (tx_scaling * tx_scaling);
        } else {
            float denom = h_sq + noise;
            float inv_denom = 1.0f / fmaxf(denom, 1e-10f);
            float mmse_gain = h_sq * inv_denom;
            float gain_norm = (mmse_gain > 1e-10f) ? (1.0f / mmse_gain) : 1.0f;

            eq.x = (y[0].x * h[0].x + y[0].y * h[0].y) * inv_denom * gain_norm / tx_scaling;
            eq.y = (y[0].y * h[0].x - y[0].x * h[0].y) * inv_denom * gain_norm / tx_scaling;

            float nvar_acc = (h_sq * noise) * inv_denom * inv_denom;
            eq_noise_var = nvar_acc * gain_norm * gain_norm / (tx_scaling * tx_scaling);
        }
        eq_noise_var = (eq_noise_var > 1e-10f) ? eq_noise_var : EQ_NOISE_VAR_DEEP_FADE;
    } else {
        float H_sq = 0.0f;
        float mmse_gain = 0.0f;
        float nvar_acc = 0.0f;
        cuFloatComplex hhy = make_cuFloatComplex(0.0f, 0.0f);

        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            float h_norm_sq = h[p].x * h[p].x + h[p].y * h[p].y;
            float noise_p = s_noise_vars[p];
            H_sq += h_norm_sq;
            hhy.x += h[p].x * y[p].x + h[p].y * y[p].y;
            hhy.y += h[p].x * y[p].y - h[p].y * y[p].x;

            if constexpr (ALGORITHM != EQUALIZER_ZF) {
                float denom_p = h_norm_sq + noise_p;
                if (denom_p > 1e-10f) {
                    mmse_gain += h_norm_sq / denom_p;
                    nvar_acc += (h_norm_sq * noise_p) / (denom_p * denom_p);
                }
            }
        }

        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            float inv_H_sq = 1.0f / fmaxf(H_sq, 1e-10f);
            eq.x = hhy.x * inv_H_sq / tx_scaling;
            eq.y = hhy.y * inv_H_sq / tx_scaling;
            float avg_noise = 0.0f;
            for (int p = 0; p < NOF_PORTS; p++) avg_noise += s_noise_vars[p];
            eq_noise_var = avg_noise / NOF_PORTS * inv_H_sq / (tx_scaling * tx_scaling);
        } else {
            float avg_noise = 0.0f;
            for (int p = 0; p < NOF_PORTS; p++) avg_noise += s_noise_vars[p];
            avg_noise /= NOF_PORTS;

            float denom = H_sq + avg_noise;
            float inv_denom = 1.0f / fmaxf(denom, 1e-10f);
            float gain_norm = (mmse_gain > 1e-10f) ? (1.0f / mmse_gain) : 1.0f;

            eq.x = hhy.x * inv_denom * gain_norm / tx_scaling;
            eq.y = hhy.y * inv_denom * gain_norm / tx_scaling;
            eq_noise_var = nvar_acc * gain_norm * gain_norm / (tx_scaling * tx_scaling);
        }
        eq_noise_var = (eq_noise_var > 1e-10f) ? eq_noise_var : EQ_NOISE_VAR_DEEP_FADE;
    }

    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, eq_noise_var);

    /* 16QAM soft demodulation - compile-time constants */
    float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);
    float scale = inv_noise * QAM16_SCALE;
    constexpr float L1 = QAM16_SCALE * 2.0f;

    float llr0 = eq.x * scale * 2.0f;
    float llr1 = eq.y * scale * 2.0f;
    float llr2 = (L1 - fabsf(eq.x)) * scale * 2.0f;
    float llr3 = (L1 - fabsf(eq.y)) * scale * 2.0f;

    /* Descrambling + INT8 */
    int bit_start = re_idx * 4;
    int word_idx = bit_start / 32;
    int bit_off = bit_start % 32;
    uint32_t scr_word = scramble_seq[word_idx];

    constexpr float INT8_SCALE = 6.0f;  /* Matches srsRAN: value/20.0*120 = value*6 */
    int llr_base = re_idx * 4;

    #pragma unroll
    for (int b = 0; b < 4; b++) {
        uint32_t scr = (scr_word >> (bit_off + b)) & 1;
        float llr = (b == 0) ? llr0 : (b == 1) ? llr1 : (b == 2) ? llr2 : llr3;
        llr = scr ? -llr : llr;
        llrs_int8[llr_base + b] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE)));
    }
}

/**
 * @brief Ultra-optimized 64QAM E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_ultra_e2e_64qam_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    const int* __restrict__ d_dmrs_indices)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    __shared__ int s_dmrs_indices[4];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    if (threadIdx.x < 4 && threadIdx.x < nof_dmrs_symbols) {
        s_dmrs_indices[threadIdx.x] = d_dmrs_indices[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    /* Get source RE index */
    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;
    int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

    /* Time-average all DMRS estimates (matches CPU "average" strategy) */
    cuFloatComplex h[8];
    float inv_n = 1.0f / (float)nof_dmrs_symbols;
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float sum_re = 0.0f, sum_im = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
            __half2 hd = d_dmrs_estimates_fp16[est_idx];
            sum_re += __half2float(__low2half(hd));
            sum_im += __half2float(__high2half(hd));
        }
        h[p] = make_cuFloatComplex(sum_re * inv_n, sum_im * inv_n);
    }

    /* Load received symbols */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    /* OPTIMIZED MMSE with gain normalization (matches CPU equalize_mmse_1xn.h) */
    cuFloatComplex eq;
    float eq_noise_var;

    if constexpr (NOF_PORTS == 1) {
        float h_sq = h[0].x * h[0].x + h[0].y * h[0].y;
        float noise = s_noise_vars[0];

        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            float inv_h_sq = 1.0f / fmaxf(h_sq, 1e-10f);
            eq.x = (y[0].x * h[0].x + y[0].y * h[0].y) * inv_h_sq / tx_scaling;
            eq.y = (y[0].y * h[0].x - y[0].x * h[0].y) * inv_h_sq / tx_scaling;
            eq_noise_var = noise * inv_h_sq / (tx_scaling * tx_scaling);
        } else {
            float denom = h_sq + noise;
            float inv_denom = 1.0f / fmaxf(denom, 1e-10f);
            float mmse_gain = h_sq * inv_denom;
            float gain_norm = (mmse_gain > 1e-10f) ? (1.0f / mmse_gain) : 1.0f;

            eq.x = (y[0].x * h[0].x + y[0].y * h[0].y) * inv_denom * gain_norm / tx_scaling;
            eq.y = (y[0].y * h[0].x - y[0].x * h[0].y) * inv_denom * gain_norm / tx_scaling;

            float nvar_acc = (h_sq * noise) * inv_denom * inv_denom;
            eq_noise_var = nvar_acc * gain_norm * gain_norm / (tx_scaling * tx_scaling);
        }
        eq_noise_var = (eq_noise_var > 1e-10f) ? eq_noise_var : EQ_NOISE_VAR_DEEP_FADE;
    } else {
        float H_sq = 0.0f;
        float mmse_gain = 0.0f;
        float nvar_acc = 0.0f;
        cuFloatComplex hhy = make_cuFloatComplex(0.0f, 0.0f);

        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            float h_norm_sq = h[p].x * h[p].x + h[p].y * h[p].y;
            float noise_p = s_noise_vars[p];
            H_sq += h_norm_sq;
            hhy.x += h[p].x * y[p].x + h[p].y * y[p].y;
            hhy.y += h[p].x * y[p].y - h[p].y * y[p].x;

            if constexpr (ALGORITHM != EQUALIZER_ZF) {
                float denom_p = h_norm_sq + noise_p;
                if (denom_p > 1e-10f) {
                    mmse_gain += h_norm_sq / denom_p;
                    nvar_acc += (h_norm_sq * noise_p) / (denom_p * denom_p);
                }
            }
        }

        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            float inv_H_sq = 1.0f / fmaxf(H_sq, 1e-10f);
            eq.x = hhy.x * inv_H_sq / tx_scaling;
            eq.y = hhy.y * inv_H_sq / tx_scaling;
            float avg_noise = 0.0f;
            for (int p = 0; p < NOF_PORTS; p++) avg_noise += s_noise_vars[p];
            eq_noise_var = avg_noise / NOF_PORTS * inv_H_sq / (tx_scaling * tx_scaling);
        } else {
            float avg_noise = 0.0f;
            for (int p = 0; p < NOF_PORTS; p++) avg_noise += s_noise_vars[p];
            avg_noise /= NOF_PORTS;

            float denom = H_sq + avg_noise;
            float inv_denom = 1.0f / fmaxf(denom, 1e-10f);
            float gain_norm = (mmse_gain > 1e-10f) ? (1.0f / mmse_gain) : 1.0f;

            eq.x = hhy.x * inv_denom * gain_norm / tx_scaling;
            eq.y = hhy.y * inv_denom * gain_norm / tx_scaling;
            eq_noise_var = nvar_acc * gain_norm * gain_norm / (tx_scaling * tx_scaling);
        }
        eq_noise_var = (eq_noise_var > 1e-10f) ? eq_noise_var : EQ_NOISE_VAR_DEEP_FADE;
    }

    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, eq_noise_var);

    /* 64QAM soft demodulation - piecewise linear approximation matching srsRAN CPU */
    float rcp_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);

    /* Constants matching srsRAN demodulation_mapper_qam64.cpp */
    constexpr float M_SQRT1_42 = 0.15430335f;  /* 1/sqrt(42) */
    constexpr float INTERVAL_WIDTH_01 = 2.0f * M_SQRT1_42;
    constexpr float INTERVAL_WIDTH_45 = 4.0f * M_SQRT1_42;

    /* Slopes for bits 0,1 (8 intervals) */
    constexpr float SLOPE_01[8] = {
        16.0f * M_SQRT1_42, 12.0f * M_SQRT1_42, 8.0f * M_SQRT1_42, 4.0f * M_SQRT1_42,
        4.0f * M_SQRT1_42, 8.0f * M_SQRT1_42, 12.0f * M_SQRT1_42, 16.0f * M_SQRT1_42
    };
    constexpr float INTERCEPT_01[8] = {
        24.0f/21.0f, 12.0f/21.0f, 4.0f/21.0f, 0.0f,
        0.0f, -4.0f/21.0f, -12.0f/21.0f, -24.0f/21.0f
    };

    /* Slopes for bits 2,3 (8 intervals) */
    constexpr float SLOPE_23[8] = {
        8.0f * M_SQRT1_42, 4.0f * M_SQRT1_42, 4.0f * M_SQRT1_42, 8.0f * M_SQRT1_42,
        -8.0f * M_SQRT1_42, -4.0f * M_SQRT1_42, -4.0f * M_SQRT1_42, -8.0f * M_SQRT1_42
    };
    constexpr float INTERCEPT_23[8] = {
        20.0f/21.0f, 8.0f/21.0f, 8.0f/21.0f, 12.0f/21.0f,
        12.0f/21.0f, 8.0f/21.0f, 8.0f/21.0f, 20.0f/21.0f
    };

    /* Slopes for bits 4,5 (4 intervals) */
    constexpr float SLOPE_45[4] = {
        4.0f * M_SQRT1_42, -4.0f * M_SQRT1_42, 4.0f * M_SQRT1_42, -4.0f * M_SQRT1_42
    };
    constexpr float INTERCEPT_45[4] = {
        12.0f/21.0f, -4.0f/21.0f, -4.0f/21.0f, 12.0f/21.0f
    };

    float yi = eq.x, yq = eq.y;

    /* Compute interval indices */
    int idx01_i = __float2int_rd(yi / INTERVAL_WIDTH_01) + 4;
    int idx01_q = __float2int_rd(yq / INTERVAL_WIDTH_01) + 4;
    int idx45_i = __float2int_rd(yi / INTERVAL_WIDTH_45) + 2;
    int idx45_q = __float2int_rd(yq / INTERVAL_WIDTH_45) + 2;

    /* Clamp indices */
    idx01_i = max(0, min(7, idx01_i));
    idx01_q = max(0, min(7, idx01_q));
    idx45_i = max(0, min(3, idx45_i));
    idx45_q = max(0, min(3, idx45_q));

    /* Compute LLRs using piecewise linear approximation */
    float llr0 = (SLOPE_01[idx01_i] * yi + INTERCEPT_01[idx01_i]) * rcp_noise;
    float llr1 = (SLOPE_01[idx01_q] * yq + INTERCEPT_01[idx01_q]) * rcp_noise;
    float llr2 = (SLOPE_23[idx01_i] * yi + INTERCEPT_23[idx01_i]) * rcp_noise;
    float llr3 = (SLOPE_23[idx01_q] * yq + INTERCEPT_23[idx01_q]) * rcp_noise;
    float llr4 = (SLOPE_45[idx45_i] * yi + INTERCEPT_45[idx45_i]) * rcp_noise;
    float llr5 = (SLOPE_45[idx45_q] * yq + INTERCEPT_45[idx45_q]) * rcp_noise;

    /* Descrambling + INT8 */
    int bit_start = re_idx * 6;
    int word_idx = bit_start / 32;
    int bit_off = bit_start % 32;
    uint32_t scr_word = scramble_seq[word_idx];
    uint32_t scr_word2 = scramble_seq[word_idx + 1];

    constexpr float INT8_SCALE = 6.0f;  /* Matches srsRAN: value/20.0*120 = value*6 */
    int llr_base = re_idx * 6;
    float llr_vals[6] = {llr0, llr1, llr2, llr3, llr4, llr5};

    #pragma unroll
    for (int b = 0; b < 6; b++) {
        int bit_pos = bit_off + b;
        uint32_t scr = (bit_pos < 32) ? ((scr_word >> bit_pos) & 1) : ((scr_word2 >> (bit_pos - 32)) & 1);
        float llr = scr ? -llr_vals[b] : llr_vals[b];
        llrs_int8[llr_base + b] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE)));
    }
}

/**
 * @brief Ultra-optimized 256QAM E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_ultra_e2e_256qam_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    const int* __restrict__ d_dmrs_indices)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    __shared__ int s_dmrs_indices[4];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    if (threadIdx.x < 4 && threadIdx.x < nof_dmrs_symbols) {
        s_dmrs_indices[threadIdx.x] = d_dmrs_indices[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    /* Get source RE index */
    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;
    int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

    /* Time-average all DMRS estimates (matches CPU "average" strategy) */
    cuFloatComplex h[8];
    float inv_n = 1.0f / (float)nof_dmrs_symbols;
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float sum_re = 0.0f, sum_im = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
            __half2 hd = d_dmrs_estimates_fp16[est_idx];
            sum_re += __half2float(__low2half(hd));
            sum_im += __half2float(__high2half(hd));
        }
        h[p] = make_cuFloatComplex(sum_re * inv_n, sum_im * inv_n);
    }

    /* Load received symbols */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    /* OPTIMIZED MMSE with gain normalization (matches CPU equalize_mmse_1xn.h) */
    cuFloatComplex eq;
    float eq_noise_var;

    if constexpr (NOF_PORTS == 1) {
        float h_sq = h[0].x * h[0].x + h[0].y * h[0].y;
        float noise = s_noise_vars[0];

        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            float inv_h_sq = 1.0f / fmaxf(h_sq, 1e-10f);
            eq.x = (y[0].x * h[0].x + y[0].y * h[0].y) * inv_h_sq / tx_scaling;
            eq.y = (y[0].y * h[0].x - y[0].x * h[0].y) * inv_h_sq / tx_scaling;
            eq_noise_var = noise * inv_h_sq / (tx_scaling * tx_scaling);
        } else {
            float denom = h_sq + noise;
            float inv_denom = 1.0f / fmaxf(denom, 1e-10f);
            float mmse_gain = h_sq * inv_denom;
            float gain_norm = (mmse_gain > 1e-10f) ? (1.0f / mmse_gain) : 1.0f;

            eq.x = (y[0].x * h[0].x + y[0].y * h[0].y) * inv_denom * gain_norm / tx_scaling;
            eq.y = (y[0].y * h[0].x - y[0].x * h[0].y) * inv_denom * gain_norm / tx_scaling;

            float nvar_acc = (h_sq * noise) * inv_denom * inv_denom;
            eq_noise_var = nvar_acc * gain_norm * gain_norm / (tx_scaling * tx_scaling);
        }
        eq_noise_var = (eq_noise_var > 1e-10f) ? eq_noise_var : EQ_NOISE_VAR_DEEP_FADE;
    } else {
        float H_sq = 0.0f;
        float mmse_gain = 0.0f;
        float nvar_acc = 0.0f;
        cuFloatComplex hhy = make_cuFloatComplex(0.0f, 0.0f);

        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            float h_norm_sq = h[p].x * h[p].x + h[p].y * h[p].y;
            float noise_p = s_noise_vars[p];
            H_sq += h_norm_sq;
            hhy.x += h[p].x * y[p].x + h[p].y * y[p].y;
            hhy.y += h[p].x * y[p].y - h[p].y * y[p].x;

            if constexpr (ALGORITHM != EQUALIZER_ZF) {
                float denom_p = h_norm_sq + noise_p;
                if (denom_p > 1e-10f) {
                    mmse_gain += h_norm_sq / denom_p;
                    nvar_acc += (h_norm_sq * noise_p) / (denom_p * denom_p);
                }
            }
        }

        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            float inv_H_sq = 1.0f / fmaxf(H_sq, 1e-10f);
            eq.x = hhy.x * inv_H_sq / tx_scaling;
            eq.y = hhy.y * inv_H_sq / tx_scaling;
            float avg_noise = 0.0f;
            for (int p = 0; p < NOF_PORTS; p++) avg_noise += s_noise_vars[p];
            eq_noise_var = avg_noise / NOF_PORTS * inv_H_sq / (tx_scaling * tx_scaling);
        } else {
            float avg_noise = 0.0f;
            for (int p = 0; p < NOF_PORTS; p++) avg_noise += s_noise_vars[p];
            avg_noise /= NOF_PORTS;

            float denom = H_sq + avg_noise;
            float inv_denom = 1.0f / fmaxf(denom, 1e-10f);
            float gain_norm = (mmse_gain > 1e-10f) ? (1.0f / mmse_gain) : 1.0f;

            eq.x = hhy.x * inv_denom * gain_norm / tx_scaling;
            eq.y = hhy.y * inv_denom * gain_norm / tx_scaling;
            eq_noise_var = nvar_acc * gain_norm * gain_norm / (tx_scaling * tx_scaling);
        }
        eq_noise_var = (eq_noise_var > 1e-10f) ? eq_noise_var : EQ_NOISE_VAR_DEEP_FADE;
    }

    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, eq_noise_var);

    /* 256QAM soft demodulation - compile-time constants */
    float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);
    float llr_scale = inv_noise * QAM256_SCALE * 2.0f;
    constexpr float L1 = QAM256_SCALE * 8.0f;
    constexpr float L2 = QAM256_SCALE * 4.0f;
    constexpr float L3 = QAM256_SCALE * 2.0f;

    float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);
    float llr_vals[8];
    llr_vals[0] = eq.x * llr_scale;
    llr_vals[1] = eq.y * llr_scale;
    llr_vals[2] = (L1 - abs_re) * llr_scale;
    llr_vals[3] = (L1 - abs_im) * llr_scale;
    llr_vals[4] = (L2 - fabsf(abs_re - L2 * 2.0f)) * llr_scale;
    llr_vals[5] = (L2 - fabsf(abs_im - L2 * 2.0f)) * llr_scale;
    llr_vals[6] = (L3 - fabsf(fabsf(abs_re - L2 * 2.0f) - L3 * 2.0f)) * llr_scale;
    llr_vals[7] = (L3 - fabsf(fabsf(abs_im - L2 * 2.0f) - L3 * 2.0f)) * llr_scale;

    /* Descrambling + INT8 */
    int bit_start = re_idx * 8;
    int word_idx = bit_start / 32;
    int bit_off = bit_start % 32;
    uint32_t scr_word = scramble_seq[word_idx];

    constexpr float INT8_SCALE = 6.0f;  /* Matches srsRAN: value/20.0*120 = value*6 */
    int llr_base = re_idx * 8;

    #pragma unroll
    for (int b = 0; b < 8; b++) {
        uint32_t scr = (scr_word >> (bit_off + b)) & 1;
        float llr = scr ? -llr_vals[b] : llr_vals[b];
        llrs_int8[llr_base + b] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE)));
    }
}

/* ============================================================================
 * 2-LAYER MIMO E2E KERNELS
 *
 * These kernels process 2-layer MIMO configurations using per-layer channel
 * estimates from the MIMO LSE kernels. Each RE outputs 2x the LLRs.
 *
 * Key differences from single-layer kernels:
 * - Channel estimates indexed as [dmrs_sym, prb, port, layer, 12]
 * - Uses equalize_symbol_2layer for MIMO equalization
 * - Outputs LLRs in codeword order: llrs[re][layer][bit]
 * ============================================================================ */

template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_mimo_2layer_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    __half* __restrict__ llrs_half,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    float* __restrict__ d_evm_error_sum,
    unsigned int* __restrict__ d_evm_symbol_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int mod_order,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    const int* __restrict__ d_dmrs_indices,
    const float* __restrict__ d_cfo_hz_ptr,
    const float* __restrict__ d_symbol_start_times,
    const float2* __restrict__ d_cfo_phasors,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    __shared__ int s_dmrs_indices[4];
    __shared__ float s_cfo_hz;
    __shared__ float s_sym_times[14];
    __shared__ float2 s_cfo_phasors[MIMO_CFO_DATA_PHASOR_COUNT];
    bool use_cfo_fallback =
        (d_cfo_hz_ptr != nullptr && d_cfo_phasors == nullptr && d_symbol_start_times != nullptr);
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    if (use_cfo_fallback && threadIdx.x < nof_dmrs_symbols && threadIdx.x < 4) {
        s_dmrs_indices[threadIdx.x] = d_dmrs_indices[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        s_cfo_hz = (d_cfo_hz_ptr != nullptr) ? *d_cfo_hz_ptr : 0.0f;
    }
    if (use_cfo_fallback && threadIdx.x < 14) {
        s_sym_times[threadIdx.x] = d_symbol_start_times[threadIdx.x];
    }
    if (d_cfo_phasors != nullptr) {
        for (int i = threadIdx.x; i < MIMO_CFO_DATA_PHASOR_COUNT; i += blockDim.x) {
            s_cfo_phasors[i] = d_cfo_phasors[i];
        }
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    int src_re = re_indices[re_idx];
    int ofdm_symbol = src_re / symbol_stride;
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    constexpr int NOF_LAYERS = 2;
    int re_per_layer = 12;
    int re_per_port = NOF_LAYERS * re_per_layer;
    int re_per_prb = NOF_PORTS * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    cuFloatComplex H[NOF_PORTS][2];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum[2] = {0.0f, 0.0f};
        float h_imag_sum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            if (d < nof_dmrs_symbols) {
                int base = d * re_per_dmrs_sym + prb_idx * re_per_prb + p * re_per_port;
                #pragma unroll
                for (int layer = 0; layer < NOF_LAYERS; layer++) {
                    __half2 h = d_dmrs_estimates_fp16[base + layer * re_per_layer + sc_in_prb];
                    float hr = __half2float(__low2half(h));
                    float hi = __half2float(__high2half(h));
                    if (s_cfo_hz != 0.0f) {
                        float cos_d = 1.0f;
                        float sin_d = 0.0f;
                        if (d_cfo_phasors != nullptr && (unsigned)ofdm_symbol < MAX_SYMBOLS) {
                            float2 ph = s_cfo_phasors[ofdm_symbol * MAX_DMRS_SYMBOLS + d];
                            cos_d = ph.x;
                            sin_d = ph.y;
                        } else if (use_cfo_fallback) {
                            int dmrs_sym = s_dmrs_indices[d] + start_symbol;
                            float dt = s_sym_times[ofdm_symbol] - s_sym_times[dmrs_sym];
                            sincosf(6.2831853071795864f * s_cfo_hz * dt, &sin_d, &cos_d);
                        }
                        float r2 = hr * cos_d - hi * sin_d;
                        float i2 = hr * sin_d + hi * cos_d;
                        hr = r2;
                        hi = i2;
                    }
                    h_real_sum[layer] += hr;
                    h_imag_sum[layer] += hi;
                }
            }
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        H[p][0] = make_cuFloatComplex(h_real_sum[0] * inv_n, h_imag_sum[0] * inv_n);
        H[p][1] = make_cuFloatComplex(h_real_sum[1] * inv_n, h_imag_sum[1] * inv_n);
    }

    cuFloatComplex y[NOF_PORTS];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    cuFloatComplex eq_symbols[2];
    float eq_noise_vars[2];
    equalize_symbol_2layer<NOF_PORTS, ALGORITHM>(y, H, s_noise_vars, tx_scaling, eq_symbols, eq_noise_vars);

    sinr_accumulate_warp(d_eq_noise_var_sum, d_eq_noise_var_count,
                         (eq_noise_vars[0] + eq_noise_vars[1]) * 0.5f);
    if (d_evm_error_sum && d_evm_symbol_count) {
        evm_accumulate_warp(d_evm_error_sum,
                            d_evm_symbol_count,
                            evm_error_power(eq_symbols[0], mod_order) + evm_error_power(eq_symbols[1], mod_order),
                            2u);
    }

    if (d_eq_symbols_out) {
        int eq_base = re_idx * NOF_LAYERS;
        d_eq_symbols_out[eq_base + 0] = make_float2(eq_symbols[0].x, eq_symbols[0].y);
        d_eq_symbols_out[eq_base + 1] = make_float2(eq_symbols[1].x, eq_symbols[1].y);
        d_eq_noise_var_out[eq_base + 0] = eq_noise_vars[0];
        d_eq_noise_var_out[eq_base + 1] = eq_noise_vars[1];
    }

    #pragma unroll
    for (int layer = 0; layer < NOF_LAYERS; layer++) {
        float llr_vals[8];
        soft_demod_symbol_runtime(eq_symbols[layer], eq_noise_vars[layer], mod_order, llr_vals);
        int bit_start = re_idx * (NOF_LAYERS * mod_order) + layer * mod_order;
        store_descrambled_half_llr(llrs_half, scramble_seq, bit_start, llr_vals, mod_order);
    }
}

template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 2)
kernel_mimo_3layer_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    __half* __restrict__ llrs_half,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    float* __restrict__ d_evm_error_sum,
    unsigned int* __restrict__ d_evm_symbol_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int mod_order,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    const int* __restrict__ d_dmrs_indices,
    const float* __restrict__ d_cfo_hz_ptr,
    const float* __restrict__ d_symbol_start_times,
    const float2* __restrict__ d_cfo_phasors,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    __shared__ int s_dmrs_indices[4];
    __shared__ float s_cfo_hz;
    __shared__ float s_sym_times[14];
    __shared__ float2 s_cfo_phasors[MIMO_CFO_DATA_PHASOR_COUNT];
    bool use_cfo_fallback =
        (d_cfo_hz_ptr != nullptr && d_cfo_phasors == nullptr && d_symbol_start_times != nullptr);
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    if (use_cfo_fallback && threadIdx.x < nof_dmrs_symbols && threadIdx.x < 4) {
        s_dmrs_indices[threadIdx.x] = d_dmrs_indices[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        s_cfo_hz = (d_cfo_hz_ptr != nullptr) ? *d_cfo_hz_ptr : 0.0f;
    }
    if (use_cfo_fallback && threadIdx.x < 14) {
        s_sym_times[threadIdx.x] = d_symbol_start_times[threadIdx.x];
    }
    if (d_cfo_phasors != nullptr) {
        for (int i = threadIdx.x; i < MIMO_CFO_DATA_PHASOR_COUNT; i += blockDim.x) {
            s_cfo_phasors[i] = d_cfo_phasors[i];
        }
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    int src_re = re_indices[re_idx];
    int ofdm_symbol = src_re / symbol_stride;
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    constexpr int NOF_LAYERS = 3;
    int re_per_layer = 12;
    int re_per_port = NOF_LAYERS * re_per_layer;
    int re_per_prb = NOF_PORTS * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    cuFloatComplex H[NOF_PORTS][3];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum[3] = {0.0f, 0.0f, 0.0f};
        float h_imag_sum[3] = {0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            if (d < nof_dmrs_symbols) {
                int base = d * re_per_dmrs_sym + prb_idx * re_per_prb + p * re_per_port;
                #pragma unroll
                for (int layer = 0; layer < NOF_LAYERS; layer++) {
                    __half2 h = d_dmrs_estimates_fp16[base + layer * re_per_layer + sc_in_prb];
                    float hr = __half2float(__low2half(h));
                    float hi = __half2float(__high2half(h));
                    if (s_cfo_hz != 0.0f) {
                        float cos_d = 1.0f;
                        float sin_d = 0.0f;
                        if (d_cfo_phasors != nullptr && (unsigned)ofdm_symbol < MAX_SYMBOLS) {
                            float2 ph = s_cfo_phasors[ofdm_symbol * MAX_DMRS_SYMBOLS + d];
                            cos_d = ph.x;
                            sin_d = ph.y;
                        } else if (use_cfo_fallback) {
                            int dmrs_sym = s_dmrs_indices[d] + start_symbol;
                            float dt = s_sym_times[ofdm_symbol] - s_sym_times[dmrs_sym];
                            sincosf(6.2831853071795864f * s_cfo_hz * dt, &sin_d, &cos_d);
                        }
                        float r2 = hr * cos_d - hi * sin_d;
                        float i2 = hr * sin_d + hi * cos_d;
                        hr = r2;
                        hi = i2;
                    }
                    h_real_sum[layer] += hr;
                    h_imag_sum[layer] += hi;
                }
            }
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        #pragma unroll
        for (int layer = 0; layer < NOF_LAYERS; layer++) {
            H[p][layer] = make_cuFloatComplex(h_real_sum[layer] * inv_n, h_imag_sum[layer] * inv_n);
        }
    }

    cuFloatComplex y[NOF_PORTS];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    cuFloatComplex eq_symbols[3];
    float eq_noise_vars[3];
    equalize_symbol_3layer<NOF_PORTS, ALGORITHM>(y, H, s_noise_vars, tx_scaling, eq_symbols, eq_noise_vars);

    sinr_accumulate_warp(d_eq_noise_var_sum, d_eq_noise_var_count,
                         (eq_noise_vars[0] + eq_noise_vars[1] + eq_noise_vars[2]) / 3.0f);
    if (d_evm_error_sum && d_evm_symbol_count) {
        evm_accumulate_warp(d_evm_error_sum,
                            d_evm_symbol_count,
                            evm_error_power(eq_symbols[0], mod_order) + evm_error_power(eq_symbols[1], mod_order) +
                                evm_error_power(eq_symbols[2], mod_order),
                            3u);
    }

    if (d_eq_symbols_out) {
        int eq_base = re_idx * NOF_LAYERS;
        #pragma unroll
        for (int layer = 0; layer < NOF_LAYERS; layer++) {
            d_eq_symbols_out[eq_base + layer] = make_float2(eq_symbols[layer].x, eq_symbols[layer].y);
            d_eq_noise_var_out[eq_base + layer] = eq_noise_vars[layer];
        }
    }

    #pragma unroll
    for (int layer = 0; layer < NOF_LAYERS; layer++) {
        float llr_vals[8];
        soft_demod_symbol_runtime(eq_symbols[layer], eq_noise_vars[layer], mod_order, llr_vals);
        int bit_start = re_idx * (NOF_LAYERS * mod_order) + layer * mod_order;
        store_descrambled_half_llr(llrs_half, scramble_seq, bit_start, llr_vals, mod_order);
    }
}

/**
 * @brief 2-layer MIMO QPSK E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_mimo_2layer_qpsk_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,  /* [dmrs_sym, prb, port, layer, 12] */
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    /* Get source RE index */
    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    /* Estimate buffer layout: [dmrs_sym, prb, port, layer, 12] */
    int re_per_layer = 12;
    int re_per_port = 2 * re_per_layer;  /* 2 layers */
    int re_per_prb = NOF_PORTS * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    /* Load per-layer channel estimates H[port][layer] with time averaging */
    cuFloatComplex H[8][2];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h0_real_sum = 0.0f, h0_imag_sum = 0.0f;
        float h1_real_sum = 0.0f, h1_imag_sum = 0.0f;

        #pragma unroll
        for (int d = 0; d < 4; d++) {
            if (d < nof_dmrs_symbols) {
                int base = d * re_per_dmrs_sym + prb_idx * re_per_prb + p * re_per_port;

                /* Layer 0 */
                __half2 h0_fp16 = d_dmrs_estimates_fp16[base + sc_in_prb];
                h0_real_sum += __half2float(__low2half(h0_fp16));
                h0_imag_sum += __half2float(__high2half(h0_fp16));

                /* Layer 1 */
                __half2 h1_fp16 = d_dmrs_estimates_fp16[base + re_per_layer + sc_in_prb];
                h1_real_sum += __half2float(__low2half(h1_fp16));
                h1_imag_sum += __half2float(__high2half(h1_fp16));
            }
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        H[p][0] = make_cuFloatComplex(h0_real_sum * inv_n, h0_imag_sum * inv_n);
        H[p][1] = make_cuFloatComplex(h1_real_sum * inv_n, h1_imag_sum * inv_n);
    }

    /* Load received symbols */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    /* 2-layer MIMO equalization */
    cuFloatComplex eq_symbols[2];
    float eq_noise_vars[2];
    equalize_symbol_2layer<NOF_PORTS, ALGORITHM>(y, H, s_noise_vars, tx_scaling, eq_symbols, eq_noise_vars);

    /* SINR accumulation (average of both layers) */
    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count,
                    (eq_noise_vars[0] + eq_noise_vars[1]) * 0.5f);

    if (d_eq_symbols_out) {
        int eq_base = re_idx * 2;
        d_eq_symbols_out[eq_base + 0] = make_float2(eq_symbols[0].x, eq_symbols[0].y);
        d_eq_symbols_out[eq_base + 1] = make_float2(eq_symbols[1].x, eq_symbols[1].y);
        d_eq_noise_var_out[eq_base + 0] = eq_noise_vars[0];
        d_eq_noise_var_out[eq_base + 1] = eq_noise_vars[1];
    }

    /* QPSK soft demodulation and descrambling for BOTH layers */
    constexpr int MOD_ORDER = 2;
    constexpr float INT8_SCALE = 6.0f;

    /* LLR output layout matches the PUSCH codeword: [re][layer][bit]. */
    constexpr int NOF_LAYERS = 2;

    #pragma unroll
    for (int layer = 0; layer < 2; layer++) {
        float inv_noise = 1.0f / fmaxf(eq_noise_vars[layer], 1e-10f);
        float llr_scale = inv_noise * QPSK_SCALE;
        float llr0 = eq_symbols[layer].x * llr_scale;
        float llr1 = eq_symbols[layer].y * llr_scale;

        int bit_start = re_idx * (NOF_LAYERS * MOD_ORDER) + layer * MOD_ORDER;
        int word_idx = bit_start / 32;
        int bit_off = bit_start % 32;
        uint32_t scr_word = scramble_seq[word_idx];
        uint32_t scr_word_next = scramble_seq[word_idx + 1];
        uint64_t scr_window = ((uint64_t)scr_word_next << 32) | scr_word;
        uint32_t scr0 = (scr_window >> bit_off) & 1;
        uint32_t scr1 = (scr_window >> (bit_off + 1)) & 1;

        llr0 = scr0 ? -llr0 : llr0;
        llr1 = scr1 ? -llr1 : llr1;

        int llr_base = bit_start;
        llrs_int8[llr_base + 0] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr0 * INT8_SCALE)));
        llrs_int8[llr_base + 1] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr1 * INT8_SCALE)));
    }
}

/**
 * @brief 2-layer MIMO 16QAM E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_mimo_2layer_16qam_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_layer = 12;
    int re_per_port = 2 * re_per_layer;
    int re_per_prb = NOF_PORTS * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    /* Load per-layer channel estimates */
    cuFloatComplex H[8][2];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h0_real_sum = 0.0f, h0_imag_sum = 0.0f;
        float h1_real_sum = 0.0f, h1_imag_sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            if (d < nof_dmrs_symbols) {
                int base = d * re_per_dmrs_sym + prb_idx * re_per_prb + p * re_per_port;
                __half2 h0_fp16 = d_dmrs_estimates_fp16[base + sc_in_prb];
                __half2 h1_fp16 = d_dmrs_estimates_fp16[base + re_per_layer + sc_in_prb];
                h0_real_sum += __half2float(__low2half(h0_fp16));
                h0_imag_sum += __half2float(__high2half(h0_fp16));
                h1_real_sum += __half2float(__low2half(h1_fp16));
                h1_imag_sum += __half2float(__high2half(h1_fp16));
            }
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        H[p][0] = make_cuFloatComplex(h0_real_sum * inv_n, h0_imag_sum * inv_n);
        H[p][1] = make_cuFloatComplex(h1_real_sum * inv_n, h1_imag_sum * inv_n);
    }

    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    cuFloatComplex eq_symbols[2];
    float eq_noise_vars[2];
    equalize_symbol_2layer<NOF_PORTS, ALGORITHM>(y, H, s_noise_vars, tx_scaling, eq_symbols, eq_noise_vars);

    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count,
                    (eq_noise_vars[0] + eq_noise_vars[1]) * 0.5f);

    if (d_eq_symbols_out) {
        int eq_base = re_idx * 2;
        d_eq_symbols_out[eq_base + 0] = make_float2(eq_symbols[0].x, eq_symbols[0].y);
        d_eq_symbols_out[eq_base + 1] = make_float2(eq_symbols[1].x, eq_symbols[1].y);
        d_eq_noise_var_out[eq_base + 0] = eq_noise_vars[0];
        d_eq_noise_var_out[eq_base + 1] = eq_noise_vars[1];
    }

    constexpr int MOD_ORDER = 4;
    constexpr float INT8_SCALE = 6.0f;
    constexpr int NOF_LAYERS = 2;

    #pragma unroll
    for (int layer = 0; layer < 2; layer++) {
        float inv_noise = 1.0f / fmaxf(eq_noise_vars[layer], 1e-10f);
        float llr_scale = inv_noise * QAM16_SCALE * 2.0f;
        cuFloatComplex eq = eq_symbols[layer];

        float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);
        float llr_vals[4];
        llr_vals[0] = eq.x * llr_scale;
        llr_vals[1] = eq.y * llr_scale;
        llr_vals[2] = (QAM16_SCALE * 2.0f - abs_re) * llr_scale;
        llr_vals[3] = (QAM16_SCALE * 2.0f - abs_im) * llr_scale;

        int bit_start = re_idx * (NOF_LAYERS * MOD_ORDER) + layer * MOD_ORDER;
        int word_idx = bit_start / 32;
        int bit_off = bit_start % 32;
        uint32_t scr_word = scramble_seq[word_idx];
        uint32_t scr_word_next = scramble_seq[word_idx + 1];
        uint64_t scr_window = ((uint64_t)scr_word_next << 32) | scr_word;

        int llr_base = bit_start;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint32_t scr = (scr_window >> (bit_off + b)) & 1;
            float llr = scr ? -llr_vals[b] : llr_vals[b];
            llrs_int8[llr_base + b] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE)));
        }
    }
}

/**
 * @brief 2-layer MIMO 64QAM E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_mimo_2layer_64qam_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_layer = 12;
    int re_per_port = 2 * re_per_layer;
    int re_per_prb = NOF_PORTS * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    cuFloatComplex H[8][2];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h0_real_sum = 0.0f, h0_imag_sum = 0.0f;
        float h1_real_sum = 0.0f, h1_imag_sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            if (d < nof_dmrs_symbols) {
                int base = d * re_per_dmrs_sym + prb_idx * re_per_prb + p * re_per_port;
                __half2 h0_fp16 = d_dmrs_estimates_fp16[base + sc_in_prb];
                __half2 h1_fp16 = d_dmrs_estimates_fp16[base + re_per_layer + sc_in_prb];
                h0_real_sum += __half2float(__low2half(h0_fp16));
                h0_imag_sum += __half2float(__high2half(h0_fp16));
                h1_real_sum += __half2float(__low2half(h1_fp16));
                h1_imag_sum += __half2float(__high2half(h1_fp16));
            }
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        H[p][0] = make_cuFloatComplex(h0_real_sum * inv_n, h0_imag_sum * inv_n);
        H[p][1] = make_cuFloatComplex(h1_real_sum * inv_n, h1_imag_sum * inv_n);
    }

    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    cuFloatComplex eq_symbols[2];
    float eq_noise_vars[2];
    equalize_symbol_2layer<NOF_PORTS, ALGORITHM>(y, H, s_noise_vars, tx_scaling, eq_symbols, eq_noise_vars);

    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count,
                    (eq_noise_vars[0] + eq_noise_vars[1]) * 0.5f);

    if (d_eq_symbols_out) {
        int eq_base = re_idx * 2;
        d_eq_symbols_out[eq_base + 0] = make_float2(eq_symbols[0].x, eq_symbols[0].y);
        d_eq_symbols_out[eq_base + 1] = make_float2(eq_symbols[1].x, eq_symbols[1].y);
        d_eq_noise_var_out[eq_base + 0] = eq_noise_vars[0];
        d_eq_noise_var_out[eq_base + 1] = eq_noise_vars[1];
    }

    constexpr int MOD_ORDER = 6;
    constexpr float INT8_SCALE = 6.0f;
    constexpr float L1 = QAM64_SCALE * 4.0f;
    constexpr float L2 = QAM64_SCALE * 2.0f;
    constexpr int NOF_LAYERS = 2;

    #pragma unroll
    for (int layer = 0; layer < 2; layer++) {
        float inv_noise = 1.0f / fmaxf(eq_noise_vars[layer], 1e-10f);
        float llr_scale = inv_noise * QAM64_SCALE * 2.0f;
        cuFloatComplex eq = eq_symbols[layer];

        float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);
        float llr_vals[6];
        llr_vals[0] = eq.x * llr_scale;
        llr_vals[1] = eq.y * llr_scale;
        llr_vals[2] = (L1 - abs_re) * llr_scale;
        llr_vals[3] = (L1 - abs_im) * llr_scale;
        llr_vals[4] = (L2 - fabsf(abs_re - L1)) * llr_scale;
        llr_vals[5] = (L2 - fabsf(abs_im - L1)) * llr_scale;

        int bit_start = re_idx * (NOF_LAYERS * MOD_ORDER) + layer * MOD_ORDER;
        int word_idx = bit_start / 32;
        int bit_off = bit_start % 32;
        uint32_t scr_word = scramble_seq[word_idx];
        uint32_t scr_word_next = scramble_seq[word_idx + 1];
        uint64_t scr_window = ((uint64_t)scr_word_next << 32) | scr_word;

        int llr_base = bit_start;
        #pragma unroll
        for (int b = 0; b < 6; b++) {
            uint32_t scr = (scr_window >> (bit_off + b)) & 1;
            float llr = scr ? -llr_vals[b] : llr_vals[b];
            llrs_int8[llr_base + b] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE)));
        }
    }
}

/**
 * @brief 2-layer MIMO 256QAM E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_mimo_2layer_256qam_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_layer = 12;
    int re_per_port = 2 * re_per_layer;
    int re_per_prb = NOF_PORTS * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    cuFloatComplex H[8][2];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h0_real_sum = 0.0f, h0_imag_sum = 0.0f;
        float h1_real_sum = 0.0f, h1_imag_sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            if (d < nof_dmrs_symbols) {
                int base = d * re_per_dmrs_sym + prb_idx * re_per_prb + p * re_per_port;
                __half2 h0_fp16 = d_dmrs_estimates_fp16[base + sc_in_prb];
                __half2 h1_fp16 = d_dmrs_estimates_fp16[base + re_per_layer + sc_in_prb];
                h0_real_sum += __half2float(__low2half(h0_fp16));
                h0_imag_sum += __half2float(__high2half(h0_fp16));
                h1_real_sum += __half2float(__low2half(h1_fp16));
                h1_imag_sum += __half2float(__high2half(h1_fp16));
            }
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        H[p][0] = make_cuFloatComplex(h0_real_sum * inv_n, h0_imag_sum * inv_n);
        H[p][1] = make_cuFloatComplex(h1_real_sum * inv_n, h1_imag_sum * inv_n);
    }

    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    cuFloatComplex eq_symbols[2];
    float eq_noise_vars[2];
    equalize_symbol_2layer<NOF_PORTS, ALGORITHM>(y, H, s_noise_vars, tx_scaling, eq_symbols, eq_noise_vars);

    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count,
                    (eq_noise_vars[0] + eq_noise_vars[1]) * 0.5f);

    if (d_eq_symbols_out) {
        int eq_base = re_idx * 2;
        d_eq_symbols_out[eq_base + 0] = make_float2(eq_symbols[0].x, eq_symbols[0].y);
        d_eq_symbols_out[eq_base + 1] = make_float2(eq_symbols[1].x, eq_symbols[1].y);
        d_eq_noise_var_out[eq_base + 0] = eq_noise_vars[0];
        d_eq_noise_var_out[eq_base + 1] = eq_noise_vars[1];
    }

    constexpr int MOD_ORDER = 8;
    constexpr float INT8_SCALE = 6.0f;
    constexpr float L1 = QAM256_SCALE * 8.0f;
    constexpr float L2 = QAM256_SCALE * 4.0f;
    constexpr float L3 = QAM256_SCALE * 2.0f;
    constexpr int NOF_LAYERS = 2;

    #pragma unroll
    for (int layer = 0; layer < 2; layer++) {
        float inv_noise = 1.0f / fmaxf(eq_noise_vars[layer], 1e-10f);
        float llr_scale = inv_noise * QAM256_SCALE * 2.0f;
        cuFloatComplex eq = eq_symbols[layer];

        float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);
        float llr_vals[8];
        llr_vals[0] = eq.x * llr_scale;
        llr_vals[1] = eq.y * llr_scale;
        llr_vals[2] = (L1 - abs_re) * llr_scale;
        llr_vals[3] = (L1 - abs_im) * llr_scale;
        llr_vals[4] = (L2 - fabsf(abs_re - L2 * 2.0f)) * llr_scale;
        llr_vals[5] = (L2 - fabsf(abs_im - L2 * 2.0f)) * llr_scale;
        llr_vals[6] = (L3 - fabsf(fabsf(abs_re - L2 * 2.0f) - L3 * 2.0f)) * llr_scale;
        llr_vals[7] = (L3 - fabsf(fabsf(abs_im - L2 * 2.0f) - L3 * 2.0f)) * llr_scale;

        int bit_start = re_idx * (NOF_LAYERS * MOD_ORDER) + layer * MOD_ORDER;
        int word_idx = bit_start / 32;
        int bit_off = bit_start % 32;
        uint32_t scr_word = scramble_seq[word_idx];
        uint32_t scr_word_next = scramble_seq[word_idx + 1];
        uint64_t scr_window = ((uint64_t)scr_word_next << 32) | scr_word;

        int llr_base = bit_start;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            uint32_t scr = (scr_window >> (bit_off + b)) & 1;
            float llr = scr ? -llr_vals[b] : llr_vals[b];
            llrs_int8[llr_base + b] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE)));
        }
    }
}

/* ============================================================================
 * 4-LAYER MIMO E2E KERNELS
 *
 * These kernels process 4-layer MIMO configurations using per-layer channel
 * estimates from the MIMO LSE kernels. Each RE outputs 4x the LLRs.
 *
 * Requires >= 4 RX ports (4x4 or 8x4 configurations).
 * ============================================================================ */

template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 2)
kernel_mimo_4layer_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    __half* __restrict__ llrs_half,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    float* __restrict__ d_evm_error_sum,
    unsigned int* __restrict__ d_evm_symbol_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int mod_order,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    const int* __restrict__ d_dmrs_indices,
    const float* __restrict__ d_cfo_hz_ptr,
    const float* __restrict__ d_symbol_start_times,
    const float2* __restrict__ d_cfo_phasors,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    __shared__ int s_dmrs_indices[4];
    __shared__ float s_cfo_hz;
    __shared__ float s_sym_times[14];
    __shared__ float2 s_cfo_phasors[MIMO_CFO_DATA_PHASOR_COUNT];
    bool use_cfo_fallback =
        (d_cfo_hz_ptr != nullptr && d_cfo_phasors == nullptr && d_symbol_start_times != nullptr);
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    if (use_cfo_fallback && threadIdx.x < nof_dmrs_symbols && threadIdx.x < 4) {
        s_dmrs_indices[threadIdx.x] = d_dmrs_indices[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        s_cfo_hz = (d_cfo_hz_ptr != nullptr) ? *d_cfo_hz_ptr : 0.0f;
    }
    if (use_cfo_fallback && threadIdx.x < 14) {
        s_sym_times[threadIdx.x] = d_symbol_start_times[threadIdx.x];
    }
    if (d_cfo_phasors != nullptr) {
        for (int i = threadIdx.x; i < MIMO_CFO_DATA_PHASOR_COUNT; i += blockDim.x) {
            s_cfo_phasors[i] = d_cfo_phasors[i];
        }
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    int src_re = re_indices[re_idx];
    int ofdm_symbol = src_re / symbol_stride;
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    constexpr int NOF_LAYERS = 4;
    int re_per_layer = 12;
    int re_per_port = NOF_LAYERS * re_per_layer;
    int re_per_prb = NOF_PORTS * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    cuFloatComplex H[NOF_PORTS][4];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float h_imag_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            if (d < nof_dmrs_symbols) {
                int base = d * re_per_dmrs_sym + prb_idx * re_per_prb + p * re_per_port;
                #pragma unroll
                for (int layer = 0; layer < NOF_LAYERS; layer++) {
                    __half2 h = d_dmrs_estimates_fp16[base + layer * re_per_layer + sc_in_prb];
                    float hr = __half2float(__low2half(h));
                    float hi = __half2float(__high2half(h));
                    if (s_cfo_hz != 0.0f) {
                        float cos_d = 1.0f;
                        float sin_d = 0.0f;
                        if (d_cfo_phasors != nullptr && (unsigned)ofdm_symbol < MAX_SYMBOLS) {
                            float2 ph = s_cfo_phasors[ofdm_symbol * MAX_DMRS_SYMBOLS + d];
                            cos_d = ph.x;
                            sin_d = ph.y;
                        } else if (use_cfo_fallback) {
                            int dmrs_sym = s_dmrs_indices[d] + start_symbol;
                            float dt = s_sym_times[ofdm_symbol] - s_sym_times[dmrs_sym];
                            sincosf(6.2831853071795864f * s_cfo_hz * dt, &sin_d, &cos_d);
                        }
                        float r2 = hr * cos_d - hi * sin_d;
                        float i2 = hr * sin_d + hi * cos_d;
                        hr = r2;
                        hi = i2;
                    }
                    h_real_sum[layer] += hr;
                    h_imag_sum[layer] += hi;
                }
            }
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        #pragma unroll
        for (int layer = 0; layer < NOF_LAYERS; layer++) {
            H[p][layer] = make_cuFloatComplex(h_real_sum[layer] * inv_n, h_imag_sum[layer] * inv_n);
        }
    }

    cuFloatComplex y[NOF_PORTS];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    cuFloatComplex eq_symbols[4];
    float eq_noise_vars[4];
    equalize_symbol_4layer<NOF_PORTS, ALGORITHM>(y, H, s_noise_vars, tx_scaling, eq_symbols, eq_noise_vars);

    sinr_accumulate_warp(d_eq_noise_var_sum, d_eq_noise_var_count,
                         (eq_noise_vars[0] + eq_noise_vars[1] + eq_noise_vars[2] + eq_noise_vars[3]) * 0.25f);
    if (d_evm_error_sum && d_evm_symbol_count) {
        evm_accumulate_warp(d_evm_error_sum,
                            d_evm_symbol_count,
                            evm_error_power(eq_symbols[0], mod_order) + evm_error_power(eq_symbols[1], mod_order) +
                                evm_error_power(eq_symbols[2], mod_order) + evm_error_power(eq_symbols[3], mod_order),
                            4u);
    }

    if (d_eq_symbols_out) {
        int eq_base = re_idx * NOF_LAYERS;
        #pragma unroll
        for (int layer = 0; layer < NOF_LAYERS; layer++) {
            d_eq_symbols_out[eq_base + layer] = make_float2(eq_symbols[layer].x, eq_symbols[layer].y);
            d_eq_noise_var_out[eq_base + layer] = eq_noise_vars[layer];
        }
    }

    #pragma unroll
    for (int layer = 0; layer < NOF_LAYERS; layer++) {
        float llr_vals[8];
        soft_demod_symbol_runtime(eq_symbols[layer], eq_noise_vars[layer], mod_order, llr_vals);
        int bit_start = re_idx * (NOF_LAYERS * mod_order) + layer * mod_order;
        store_descrambled_half_llr(llrs_half, scramble_seq, bit_start, llr_vals, mod_order);
    }
}

/**
 * @brief 4-layer MIMO QPSK E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_mimo_4layer_qpsk_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,  /* [dmrs_sym, prb, port, layer, 12] */
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    /* Estimate buffer layout: [dmrs_sym, prb, port, layer, 12] */
    int re_per_layer = 12;
    int re_per_port = 4 * re_per_layer;  /* 4 layers */
    int re_per_prb = NOF_PORTS * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    /* Load per-layer channel estimates H[port][layer] with time averaging */
    cuFloatComplex H[8][4];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float h_imag_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int d = 0; d < 4; d++) {
            if (d < nof_dmrs_symbols) {
                int base = d * re_per_dmrs_sym + prb_idx * re_per_prb + p * re_per_port;
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    __half2 h_fp16 = d_dmrs_estimates_fp16[base + l * re_per_layer + sc_in_prb];
                    h_real_sum[l] += __half2float(__low2half(h_fp16));
                    h_imag_sum[l] += __half2float(__high2half(h_fp16));
                }
            }
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            H[p][l] = make_cuFloatComplex(h_real_sum[l] * inv_n, h_imag_sum[l] * inv_n);
        }
    }

    /* Load received symbols */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    /* 4-layer MIMO equalization */
    cuFloatComplex eq_symbols[4];
    float eq_noise_vars[4];
    equalize_symbol_4layer<NOF_PORTS, ALGORITHM>(y, H, s_noise_vars, tx_scaling, eq_symbols, eq_noise_vars);

    /* SINR accumulation (average of all layers) */
    float avg_eq_noise = (eq_noise_vars[0] + eq_noise_vars[1] + eq_noise_vars[2] + eq_noise_vars[3]) * 0.25f;
    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, avg_eq_noise);

    if (d_eq_symbols_out) {
        int eq_base = re_idx * 4;
        #pragma unroll
        for (int layer = 0; layer < 4; layer++) {
            d_eq_symbols_out[eq_base + layer] = make_float2(eq_symbols[layer].x, eq_symbols[layer].y);
            d_eq_noise_var_out[eq_base + layer] = eq_noise_vars[layer];
        }
    }

    /* QPSK soft demodulation for all 4 layers */
    constexpr int MOD_ORDER = 2;
    constexpr float INT8_SCALE = 6.0f;
    constexpr int NOF_LAYERS = 4;

    #pragma unroll
    for (int layer = 0; layer < 4; layer++) {
        float inv_noise = 1.0f / fmaxf(eq_noise_vars[layer], 1e-10f);
        float llr_scale = inv_noise * QPSK_SCALE;
        float llr0 = eq_symbols[layer].x * llr_scale;
        float llr1 = eq_symbols[layer].y * llr_scale;

        int bit_start = re_idx * (NOF_LAYERS * MOD_ORDER) + layer * MOD_ORDER;
        int word_idx = bit_start / 32;
        int bit_off = bit_start % 32;
        uint32_t scr_word = scramble_seq[word_idx];
        uint32_t scr_word_next = scramble_seq[word_idx + 1];
        uint64_t scr_window = ((uint64_t)scr_word_next << 32) | scr_word;
        uint32_t scr0 = (scr_window >> bit_off) & 1;
        uint32_t scr1 = (scr_window >> (bit_off + 1)) & 1;

        llr0 = scr0 ? -llr0 : llr0;
        llr1 = scr1 ? -llr1 : llr1;

        int llr_base = bit_start;
        llrs_int8[llr_base + 0] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr0 * INT8_SCALE)));
        llrs_int8[llr_base + 1] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr1 * INT8_SCALE)));
    }
}

/**
 * @brief 4-layer MIMO 16QAM E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_mimo_4layer_16qam_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_layer = 12;
    int re_per_port = 4 * re_per_layer;
    int re_per_prb = NOF_PORTS * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    cuFloatComplex H[8][4];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float h_imag_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            if (d < nof_dmrs_symbols) {
                int base = d * re_per_dmrs_sym + prb_idx * re_per_prb + p * re_per_port;
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    __half2 h_fp16 = d_dmrs_estimates_fp16[base + l * re_per_layer + sc_in_prb];
                    h_real_sum[l] += __half2float(__low2half(h_fp16));
                    h_imag_sum[l] += __half2float(__high2half(h_fp16));
                }
            }
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            H[p][l] = make_cuFloatComplex(h_real_sum[l] * inv_n, h_imag_sum[l] * inv_n);
        }
    }

    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    cuFloatComplex eq_symbols[4];
    float eq_noise_vars[4];
    equalize_symbol_4layer<NOF_PORTS, ALGORITHM>(y, H, s_noise_vars, tx_scaling, eq_symbols, eq_noise_vars);

    float avg_eq_noise = (eq_noise_vars[0] + eq_noise_vars[1] + eq_noise_vars[2] + eq_noise_vars[3]) * 0.25f;
    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, avg_eq_noise);

    if (d_eq_symbols_out) {
        int eq_base = re_idx * 4;
        #pragma unroll
        for (int layer = 0; layer < 4; layer++) {
            d_eq_symbols_out[eq_base + layer] = make_float2(eq_symbols[layer].x, eq_symbols[layer].y);
            d_eq_noise_var_out[eq_base + layer] = eq_noise_vars[layer];
        }
    }

    constexpr int MOD_ORDER = 4;
    constexpr float INT8_SCALE = 6.0f;
    constexpr int NOF_LAYERS = 4;

    #pragma unroll
    for (int layer = 0; layer < 4; layer++) {
        float inv_noise = 1.0f / fmaxf(eq_noise_vars[layer], 1e-10f);
        float llr_scale = inv_noise * QAM16_SCALE * 2.0f;
        cuFloatComplex eq = eq_symbols[layer];

        float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);
        float llr_vals[4];
        llr_vals[0] = eq.x * llr_scale;
        llr_vals[1] = eq.y * llr_scale;
        llr_vals[2] = (QAM16_SCALE * 2.0f - abs_re) * llr_scale;
        llr_vals[3] = (QAM16_SCALE * 2.0f - abs_im) * llr_scale;

        int bit_start = re_idx * (NOF_LAYERS * MOD_ORDER) + layer * MOD_ORDER;
        int word_idx = bit_start / 32;
        int bit_off = bit_start % 32;
        uint32_t scr_word = scramble_seq[word_idx];
        uint32_t scr_word_next = scramble_seq[word_idx + 1];
        uint64_t scr_window = ((uint64_t)scr_word_next << 32) | scr_word;

        int llr_base = bit_start;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint32_t scr = (scr_window >> (bit_off + b)) & 1;
            float llr = scr ? -llr_vals[b] : llr_vals[b];
            llrs_int8[llr_base + b] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE)));
        }
    }
}

/**
 * @brief 4-layer MIMO 64QAM E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_mimo_4layer_64qam_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_layer = 12;
    int re_per_port = 4 * re_per_layer;
    int re_per_prb = NOF_PORTS * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    cuFloatComplex H[8][4];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float h_imag_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            if (d < nof_dmrs_symbols) {
                int base = d * re_per_dmrs_sym + prb_idx * re_per_prb + p * re_per_port;
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    __half2 h_fp16 = d_dmrs_estimates_fp16[base + l * re_per_layer + sc_in_prb];
                    h_real_sum[l] += __half2float(__low2half(h_fp16));
                    h_imag_sum[l] += __half2float(__high2half(h_fp16));
                }
            }
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            H[p][l] = make_cuFloatComplex(h_real_sum[l] * inv_n, h_imag_sum[l] * inv_n);
        }
    }

    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    cuFloatComplex eq_symbols[4];
    float eq_noise_vars[4];
    equalize_symbol_4layer<NOF_PORTS, ALGORITHM>(y, H, s_noise_vars, tx_scaling, eq_symbols, eq_noise_vars);

    float avg_eq_noise = (eq_noise_vars[0] + eq_noise_vars[1] + eq_noise_vars[2] + eq_noise_vars[3]) * 0.25f;
    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, avg_eq_noise);

    if (d_eq_symbols_out) {
        int eq_base = re_idx * 4;
        #pragma unroll
        for (int layer = 0; layer < 4; layer++) {
            d_eq_symbols_out[eq_base + layer] = make_float2(eq_symbols[layer].x, eq_symbols[layer].y);
            d_eq_noise_var_out[eq_base + layer] = eq_noise_vars[layer];
        }
    }

    constexpr int MOD_ORDER = 6;
    constexpr float INT8_SCALE = 6.0f;
    constexpr float L1 = QAM64_SCALE * 4.0f;
    constexpr float L2 = QAM64_SCALE * 2.0f;
    constexpr int NOF_LAYERS = 4;

    #pragma unroll
    for (int layer = 0; layer < 4; layer++) {
        float inv_noise = 1.0f / fmaxf(eq_noise_vars[layer], 1e-10f);
        float llr_scale = inv_noise * QAM64_SCALE * 2.0f;
        cuFloatComplex eq = eq_symbols[layer];

        float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);
        float llr_vals[6];
        llr_vals[0] = eq.x * llr_scale;
        llr_vals[1] = eq.y * llr_scale;
        llr_vals[2] = (L1 - abs_re) * llr_scale;
        llr_vals[3] = (L1 - abs_im) * llr_scale;
        llr_vals[4] = (L2 - fabsf(abs_re - L1)) * llr_scale;
        llr_vals[5] = (L2 - fabsf(abs_im - L1)) * llr_scale;

        int bit_start = re_idx * (NOF_LAYERS * MOD_ORDER) + layer * MOD_ORDER;
        int word_idx = bit_start / 32;
        int bit_off = bit_start % 32;
        uint32_t scr_word = scramble_seq[word_idx];
        uint32_t scr_word_next = scramble_seq[word_idx + 1];
        uint64_t scr_window = ((uint64_t)scr_word_next << 32) | scr_word;

        int llr_base = bit_start;
        #pragma unroll
        for (int b = 0; b < 6; b++) {
            uint32_t scr = (scr_window >> (bit_off + b)) & 1;
            float llr = scr ? -llr_vals[b] : llr_vals[b];
            llrs_int8[llr_base + b] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE)));
        }
    }
}

/**
 * @brief 4-layer MIMO 256QAM E2E kernel
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(256, 4)
kernel_mimo_4layer_256qam_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    float2* __restrict__ d_eq_symbols_out,
    float* __restrict__ d_eq_noise_var_out)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_layer = 12;
    int re_per_port = 4 * re_per_layer;
    int re_per_prb = NOF_PORTS * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;

    cuFloatComplex H[8][4];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float h_imag_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            if (d < nof_dmrs_symbols) {
                int base = d * re_per_dmrs_sym + prb_idx * re_per_prb + p * re_per_port;
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    __half2 h_fp16 = d_dmrs_estimates_fp16[base + l * re_per_layer + sc_in_prb];
                    h_real_sum[l] += __half2float(__low2half(h_fp16));
                    h_imag_sum[l] += __half2float(__high2half(h_fp16));
                }
            }
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            H[p][l] = make_cuFloatComplex(h_real_sum[l] * inv_n, h_imag_sum[l] * inv_n);
        }
    }

    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    cuFloatComplex eq_symbols[4];
    float eq_noise_vars[4];
    equalize_symbol_4layer<NOF_PORTS, ALGORITHM>(y, H, s_noise_vars, tx_scaling, eq_symbols, eq_noise_vars);

    float avg_eq_noise = (eq_noise_vars[0] + eq_noise_vars[1] + eq_noise_vars[2] + eq_noise_vars[3]) * 0.25f;
    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, avg_eq_noise);

    if (d_eq_symbols_out) {
        int eq_base = re_idx * 4;
        #pragma unroll
        for (int layer = 0; layer < 4; layer++) {
            d_eq_symbols_out[eq_base + layer] = make_float2(eq_symbols[layer].x, eq_symbols[layer].y);
            d_eq_noise_var_out[eq_base + layer] = eq_noise_vars[layer];
        }
    }

    constexpr int MOD_ORDER = 8;
    constexpr float INT8_SCALE = 6.0f;
    constexpr float L1 = QAM256_SCALE * 8.0f;
    constexpr float L2 = QAM256_SCALE * 4.0f;
    constexpr float L3 = QAM256_SCALE * 2.0f;
    constexpr int NOF_LAYERS = 4;

    #pragma unroll
    for (int layer = 0; layer < 4; layer++) {
        float inv_noise = 1.0f / fmaxf(eq_noise_vars[layer], 1e-10f);
        float llr_scale = inv_noise * QAM256_SCALE * 2.0f;
        cuFloatComplex eq = eq_symbols[layer];

        float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);
        float llr_vals[8];
        llr_vals[0] = eq.x * llr_scale;
        llr_vals[1] = eq.y * llr_scale;
        llr_vals[2] = (L1 - abs_re) * llr_scale;
        llr_vals[3] = (L1 - abs_im) * llr_scale;
        llr_vals[4] = (L2 - fabsf(abs_re - L2 * 2.0f)) * llr_scale;
        llr_vals[5] = (L2 - fabsf(abs_im - L2 * 2.0f)) * llr_scale;
        llr_vals[6] = (L3 - fabsf(fabsf(abs_re - L2 * 2.0f) - L3 * 2.0f)) * llr_scale;
        llr_vals[7] = (L3 - fabsf(fabsf(abs_im - L2 * 2.0f) - L3 * 2.0f)) * llr_scale;

        int bit_start = re_idx * (NOF_LAYERS * MOD_ORDER) + layer * MOD_ORDER;
        int word_idx = bit_start / 32;
        int bit_off = bit_start % 32;
        uint32_t scr_word = scramble_seq[word_idx];
        uint32_t scr_word_next = scramble_seq[word_idx + 1];
        uint64_t scr_window = ((uint64_t)scr_word_next << 32) | scr_word;

        int llr_base = bit_start;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            uint32_t scr = (scr_window >> (bit_off + b)) & 1;
            float llr = scr ? -llr_vals[b] : llr_vals[b];
            llrs_int8[llr_base + b] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE)));
        }
    }
}

/* ============================================================================
 * MEGA-FUSED KERNEL: LSE + Noise + EQ + Demod + Descramble in ONE pass
 *
 * This kernel eliminates ALL intermediate global memory writes by computing
 * channel estimates on-the-fly and using them immediately for equalization.
 *
 * Memory savings: ~350 KB per slot (no FP16 estimate buffer needed)
 * Kernel savings: 2 kernel launches eliminated
 *
 * Processing flow per PRB:
 *   Phase 1: Load DMRS REs → Compute LSE → Store in shared memory
 *   Phase 2: Load DATA REs → EQ using shared LSE → Demod → Descramble → Output
 * ============================================================================ */

/**
 * @brief MEGA-FUSED 64QAM kernel (most common case)
 *
 * Grid: (nof_prb), Threads: 128 (processes 12 data symbols × ~10 REs each)
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void __launch_bounds__(128, 4)
kernel_mega_fused_64qam_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const int* __restrict__ dmrs_symbol_mask_bits,  /* Which symbols are DMRS */
    const uint32_t* __restrict__ scramble_seq,
    int8_t* __restrict__ llrs_int8,
    float* __restrict__ d_noise_var_accum,
    unsigned int* __restrict__ d_noise_count,
    int nof_prb,
    int nof_data_re,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    int start_symbol,
    int nof_symbols,
    float dmrs_scaling,
    float tx_scaling,
    int slot_idx,
    uint32_t scrambling_id,
    int n_scid,
    uint32_t data_c_init)
{
    int prb_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (prb_idx >= nof_prb) return;

    int prb_start_sc = (start_prb + prb_idx) * 12;

    /* Shared memory for channel estimates: [port][6 DMRS pilots] */
    __shared__ float s_h_real[8][6];
    __shared__ float s_h_imag[8][6];
    __shared__ float s_noise_var[8];
    __shared__ int s_dmrs_symbols[4];  /* Up to 4 DMRS symbols */
    __shared__ int s_nof_dmrs;

    /* Phase 0: Identify DMRS symbol positions */
    if (tid == 0) {
        int nof_dmrs = 0;
        for (int s = 0; s < 14 && nof_dmrs < 4; s++) {
            if (dmrs_symbol_mask_bits[s]) {
                s_dmrs_symbols[nof_dmrs++] = s;
            }
        }
        s_nof_dmrs = nof_dmrs;
    }
    __syncthreads();

    int nof_dmrs = s_nof_dmrs;
    float inv_sqrt2 = 0.70710678118f;
    float h_normalizer = 1.0f / dmrs_scaling;

    /* Phase 1: Compute LSE from DMRS symbols */
    /* Each thread handles some of the 6 pilots × nof_dmrs × nof_ports work */
    int total_pilot_work = 6 * nof_dmrs * NOF_PORTS;

    /* Initialize accumulators */
    if (tid < NOF_PORTS * 6) {
        int port = tid / 6;
        int pilot = tid % 6;
        s_h_real[port][pilot] = 0.0f;
        s_h_imag[port][pilot] = 0.0f;
    }
    if (tid < NOF_PORTS) {
        s_noise_var[tid] = 0.0f;
    }
    __syncthreads();

    /* Process DMRS pilots */
    const int dmrs_sc_offset[6] = {0, 2, 4, 6, 8, 10};

    for (int work_idx = tid; work_idx < total_pilot_work; work_idx += blockDim.x) {
        int dmrs_idx = work_idx / (6 * NOF_PORTS);
        int port_pilot = work_idx % (6 * NOF_PORTS);
        int port = port_pilot / 6;
        int pilot = port_pilot % 6;

        int dmrs_symbol = s_dmrs_symbols[dmrs_idx];
        int sc = prb_start_sc + dmrs_sc_offset[pilot];

        /* Load received symbol */
        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        /* Generate DMRS pilot (simplified - using pre-computed approach) */
        /* For now, assume QPSK pilots at unit magnitude */
        uint32_t c_init = compute_dmrs_c_init_device(slot_idx, dmrs_symbol, scrambling_id, n_scid);
        uint32_t x1 = 1, x2 = c_init & 0x7FFFFFFF;
        int pilot_bit_idx = (start_prb + prb_idx) * 6 + pilot;
        int advance = NC_SKIP + pilot_bit_idx * 2;
        x1 = advance_x1_local(x1, advance);
        x2 = advance_x2_local(x2, advance);

        int c_real = (x1 ^ x2) & 1;
        x1 = step_x1_local(x1); x2 = step_x2_local(x2);
        int c_imag = (x1 ^ x2) & 1;

        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        /* LSE: h = y × conj(pilot) / dmrs_scaling */
        float h_real = (y.x * p_real + y.y * p_imag) * h_normalizer;
        float h_imag = (y.y * p_real - y.x * p_imag) * h_normalizer;

        /* Accumulate (atomic within block) */
        atomicAdd(&s_h_real[port][pilot], h_real);
        atomicAdd(&s_h_imag[port][pilot], h_imag);

        /* Noise residual */
        float h_scaled_real = h_real * dmrs_scaling;
        float h_scaled_imag = h_imag * dmrs_scaling;
        float y_ideal_real = h_scaled_real * p_real - h_scaled_imag * p_imag;
        float y_ideal_imag = h_scaled_real * p_imag + h_scaled_imag * p_real;
        float res = (y.x - y_ideal_real) * (y.x - y_ideal_real) +
                    (y.y - y_ideal_imag) * (y.y - y_ideal_imag);
        atomicAdd(&s_noise_var[port], res);
    }
    __syncthreads();

    /* Average estimates across DMRS symbols */
    float inv_nof_dmrs = 1.0f / (float)nof_dmrs;
    if (tid < NOF_PORTS * 6) {
        int port = tid / 6;
        int pilot = tid % 6;
        s_h_real[port][pilot] *= inv_nof_dmrs;
        s_h_imag[port][pilot] *= inv_nof_dmrs;
    }
    if (tid < NOF_PORTS) {
        /* Finalize noise variance */
        float nv = s_noise_var[tid] / (float)(nof_dmrs * 6);
        s_noise_var[tid] = fmaxf(nv, 0.001f);
        atomicAdd(d_noise_var_accum, nv);
    }
    if (tid == 0) {
        atomicAdd(d_noise_count, nof_dmrs * 6);
    }
    __syncthreads();

    /* Phase 2: Process DATA REs */
    /* Each PRB has 12 × (nof_symbols - nof_dmrs) data REs */
    int nof_data_symbols = nof_symbols - nof_dmrs;
    int data_re_per_prb = 12 * nof_data_symbols;
    int prb_data_re_start = prb_idx * data_re_per_prb;

    /* Pre-computed QAM constants */
    constexpr float QAM64_SCALE = 0.4082482904638631f;
    constexpr float L1 = QAM64_SCALE * 4.0f;
    constexpr float L2 = QAM64_SCALE * 2.0f;
    constexpr float INT8_SCALE = 6.0f;  /* Matches srsRAN: value/20.0*120 = value*6 */

    for (int local_re = tid; local_re < data_re_per_prb; local_re += blockDim.x) {
        /* Determine which symbol and subcarrier this is */
        int data_sym_idx = local_re / 12;
        int sc_in_prb = local_re % 12;

        /* Find actual symbol index (skip DMRS symbols) */
        int actual_symbol = start_symbol;
        int data_count = 0;
        for (int s = start_symbol; s < start_symbol + nof_symbols && data_count <= data_sym_idx; s++) {
            bool is_dmrs = false;
            for (int d = 0; d < nof_dmrs; d++) {
                if (s == s_dmrs_symbols[d]) { is_dmrs = true; break; }
            }
            if (!is_dmrs) {
                if (data_count == data_sym_idx) { actual_symbol = s; break; }
                data_count++;
            }
        }

        int sc = prb_start_sc + sc_in_prb;

        /* Interpolate channel estimate (linear from pilot positions) */
        cuFloatComplex h[8];
        int left_pilot = sc_in_prb / 2;
        int right_pilot = (left_pilot + 1 < 6) ? left_pilot + 1 : 5;
        float alpha = (sc_in_prb % 2 == 0) ? 0.0f : 0.5f;

        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            float h_r = s_h_real[p][left_pilot] * (1.0f - alpha) + s_h_real[p][right_pilot] * alpha;
            float h_i = s_h_imag[p][left_pilot] * (1.0f - alpha) + s_h_imag[p][right_pilot] * alpha;
            h[p] = make_cuFloatComplex(h_r, h_i);
        }

        /* Load received symbols */
        cuFloatComplex y[8];
        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            int grid_idx = p * grid_stride + actual_symbol * symbol_stride + sc;
            y[p] = cbf16_to_fp32(d_grid_cbf16[grid_idx]);
        }

        /* Equalization */
        cuFloatComplex eq;
        float eq_noise_var;

        if constexpr (NOF_PORTS == 1) {
            float h_sq = h[0].x * h[0].x + h[0].y * h[0].y;
            float noise = s_noise_var[0];
            float scale;
            if constexpr (ALGORITHM == EQUALIZER_ZF) {
                scale = 1.0f / fmaxf(h_sq, 1e-10f);
            } else {
                scale = 1.0f / (h_sq + noise);
            }
            eq.x = (y[0].x * h[0].x + y[0].y * h[0].y) * scale * tx_scaling;
            eq.y = (y[0].y * h[0].x - y[0].x * h[0].y) * scale * tx_scaling;
            eq_noise_var = (h_sq > 1e-10f) ? noise * scale : EQ_NOISE_VAR_DEEP_FADE;
        } else {
            float hh_real = 0.0f;
            cuFloatComplex hhy = make_cuFloatComplex(0.0f, 0.0f);
            float noise_sum = 0.0f;
            #pragma unroll
            for (int p = 0; p < NOF_PORTS; p++) {
                hh_real += h[p].x * h[p].x + h[p].y * h[p].y;
                hhy.x += h[p].x * y[p].x + h[p].y * y[p].y;
                hhy.y += h[p].x * y[p].y - h[p].y * y[p].x;
                noise_sum += s_noise_var[p];
            }
            float avg_noise = noise_sum / (float)NOF_PORTS;
            float scale;
            if constexpr (ALGORITHM == EQUALIZER_ZF) {
                scale = 1.0f / fmaxf(hh_real, 1e-10f);
            } else {
                scale = 1.0f / (hh_real + avg_noise);
            }
            eq.x = hhy.x * scale * tx_scaling;
            eq.y = hhy.y * scale * tx_scaling;
            eq_noise_var = (hh_real > 1e-10f) ? avg_noise * scale : EQ_NOISE_VAR_DEEP_FADE;
        }

        /* 64QAM soft demodulation */
        float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);
        float llr_scale = inv_noise * QAM64_SCALE * 2.0f;
        float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);

        float llr_vals[6];
        llr_vals[0] = eq.x * llr_scale;
        llr_vals[1] = eq.y * llr_scale;
        llr_vals[2] = (L1 - abs_re) * llr_scale;
        llr_vals[3] = (L1 - abs_im) * llr_scale;
        llr_vals[4] = (L2 - fabsf(abs_re - L1)) * llr_scale;
        llr_vals[5] = (L2 - fabsf(abs_im - L1)) * llr_scale;

        /* Descrambling + INT8 output */
        int global_re_idx = prb_data_re_start + local_re;
        int bit_start = global_re_idx * 6;
        int word_idx = bit_start / 32;
        int bit_off = bit_start % 32;
        uint32_t scr_word = scramble_seq[word_idx];
        uint32_t scr_word2 = scramble_seq[word_idx + 1];

        int llr_base = global_re_idx * 6;

        #pragma unroll
        for (int b = 0; b < 6; b++) {
            int bit_pos = bit_off + b;
            uint32_t scr = (bit_pos < 32) ? ((scr_word >> bit_pos) & 1) : ((scr_word2 >> (bit_pos - 32)) & 1);
            float llr = scr ? -llr_vals[b] : llr_vals[b];
            llrs_int8[llr_base + b] = (int8_t)rintf(fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE)));
        }
    }
}

/**
 * ULTRA-FAST E2E kernel with PRE-COMPUTED SCRAMBLING SEQUENCE.
 *
 * This kernel reads scrambling bits from a pre-generated buffer instead of
 * generating them on-the-fly. This eliminates O(log N) LFSR advancement per warp.
 *
 * Expected speedup: Eliminates ~1229 warp LFSR jumps, saving ~200+ µs at 100 MHz.
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void kernel_fused_e2e_precomputed_scramble_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,
    const uint32_t* __restrict__ scramble_seq,  /* Pre-computed scrambling bits */
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int mod_order,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load noise variances to shared memory */
    __shared__ float s_noise_vars[8];

    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    /* Get source RE index */
    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

    /* Time interpolation */
    cuFloatComplex h[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum = 0.0f, h_imag_sum = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
            __half2 h_fp16 = d_dmrs_estimates_fp16[est_idx];
            h_real_sum += __half2float(__low2half(h_fp16));
            h_imag_sum += __half2float(__high2half(h_fp16));
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        h[p] = make_cuFloatComplex(h_real_sum * inv_n, h_imag_sum * inv_n);
    }

    /* Load received symbols */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
    }

    /* Equalization */
    cuFloatComplex eq;
    float eq_noise_var;

    if constexpr (NOF_PORTS == 1) {
        float h_sq = h[0].x * h[0].x + h[0].y * h[0].y;
        float noise = s_noise_vars[0];
        float scale;
        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            scale = 1.0f / fmaxf(h_sq, 1e-10f);
        } else {
            scale = 1.0f / (h_sq + noise);
        }
        eq.x = (y[0].x * h[0].x + y[0].y * h[0].y) * scale;
        eq.y = (y[0].y * h[0].x - y[0].x * h[0].y) * scale;
        eq_noise_var = (h_sq > 1e-10f) ? noise * scale : EQ_NOISE_VAR_DEEP_FADE;
    } else {
        float hh_real = 0.0f;
        cuFloatComplex hhy = make_cuFloatComplex(0.0f, 0.0f);
        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            hh_real += h[p].x * h[p].x + h[p].y * h[p].y;
            hhy.x += h[p].x * y[p].x + h[p].y * y[p].y;
            hhy.y += h[p].x * y[p].y - h[p].y * y[p].x;
        }
        float noise_sum = 0.0f;
        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            noise_sum += s_noise_vars[p];
        }
        float avg_noise = noise_sum / (float)NOF_PORTS;
        float scale;
        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            scale = 1.0f / fmaxf(hh_real, 1e-10f);
        } else {
            scale = 1.0f / (hh_real + avg_noise);
        }
        eq.x = hhy.x * scale;
        eq.y = hhy.y * scale;
        eq_noise_var = (hh_real > 1e-10f) ? avg_noise * scale : EQ_NOISE_VAR_DEEP_FADE;
    }

    /* Apply TX scaling */
    eq.x *= tx_scaling;
    eq.y *= tx_scaling;

    /* Accumulate equalized noise variance for SINR calculation */
    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, eq_noise_var);

    /* Soft demodulation */
    float llr_vals[8];
    float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);

    if (mod_order == 2) {
        float scale = inv_noise * 2.8284271f  /* 2*sqrt(2) - matches CPU QPSK demod */;
        llr_vals[0] = eq.x * scale;
        llr_vals[1] = eq.y * scale;
    } else if (mod_order == 4) {
        float scale = inv_noise * 0.6324555f;
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.6324555f * 2.0f - fabsf(eq.x)) * scale * 2.0f;
        llr_vals[3] = (0.6324555f * 2.0f - fabsf(eq.y)) * scale * 2.0f;
    } else if (mod_order == 6) {
        // Proper max-log LLR computation for 64QAM
        const float s = 0.15430335f;  // 1/sqrt(42)
        const float s1 = s, s3 = 3.0f * s, s5 = 5.0f * s, s7 = 7.0f * s;
        float yi = eq.x, yq = eq.y;
        #define SQ(x) ((x)*(x))
        float d0_b0 = fminf(fminf(SQ(yi-s1),SQ(yi-s3)), fminf(SQ(yi-s5),SQ(yi-s7)));
        float d1_b0 = fminf(fminf(SQ(yi+s1),SQ(yi+s3)), fminf(SQ(yi+s5),SQ(yi+s7)));
        llr_vals[0] = (d1_b0 - d0_b0) * inv_noise;
        float d0_b1 = fminf(fminf(SQ(yq-s1),SQ(yq-s3)), fminf(SQ(yq-s5),SQ(yq-s7)));
        float d1_b1 = fminf(fminf(SQ(yq+s1),SQ(yq+s3)), fminf(SQ(yq+s5),SQ(yq+s7)));
        llr_vals[1] = (d1_b1 - d0_b1) * inv_noise;
        float d0_b2 = fminf(fminf(SQ(yi-s1),SQ(yi+s1)), fminf(SQ(yi-s3),SQ(yi+s3)));
        float d1_b2 = fminf(fminf(SQ(yi-s5),SQ(yi+s5)), fminf(SQ(yi-s7),SQ(yi+s7)));
        llr_vals[2] = (d1_b2 - d0_b2) * inv_noise;
        float d0_b3 = fminf(fminf(SQ(yq-s1),SQ(yq+s1)), fminf(SQ(yq-s3),SQ(yq+s3)));
        float d1_b3 = fminf(fminf(SQ(yq-s5),SQ(yq+s5)), fminf(SQ(yq-s7),SQ(yq+s7)));
        llr_vals[3] = (d1_b3 - d0_b3) * inv_noise;
        float d0_b4 = fminf(fminf(SQ(yi-s1),SQ(yi+s1)), fminf(SQ(yi-s5),SQ(yi+s5)));
        float d1_b4 = fminf(fminf(SQ(yi-s3),SQ(yi+s3)), fminf(SQ(yi-s7),SQ(yi+s7)));
        llr_vals[4] = (d1_b4 - d0_b4) * inv_noise;
        float d0_b5 = fminf(fminf(SQ(yq-s1),SQ(yq+s1)), fminf(SQ(yq-s5),SQ(yq+s5)));
        float d1_b5 = fminf(fminf(SQ(yq-s3),SQ(yq+s3)), fminf(SQ(yq-s7),SQ(yq+s7)));
        llr_vals[5] = (d1_b5 - d0_b5) * inv_noise;
        #undef SQ
    } else if (mod_order == 8) {
        float scale = inv_noise * 0.07669650f;
        float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.07669650f * 8.0f - abs_re) * scale * 2.0f;
        llr_vals[3] = (0.07669650f * 8.0f - abs_im) * scale * 2.0f;
        float level2 = 0.07669650f * 4.0f;
        llr_vals[4] = (level2 - fabsf(abs_re - level2 * 2.0f)) * scale * 2.0f;
        llr_vals[5] = (level2 - fabsf(abs_im - level2 * 2.0f)) * scale * 2.0f;
        float level3 = 0.07669650f * 2.0f;
        llr_vals[6] = (level3 - fabsf(fabsf(abs_re - level2 * 2.0f) - level3 * 2.0f)) * scale * 2.0f;
        llr_vals[7] = (level3 - fabsf(fabsf(abs_im - level2 * 2.0f) - level3 * 2.0f)) * scale * 2.0f;
    }

    /* Read scrambling bits from PRE-COMPUTED buffer (no LFSR computation!) */
    int bit_start = re_idx * mod_order;
    int llr_base = re_idx * mod_order;
    const float INT8_SCALE = 4.0f;

    for (int b = 0; b < mod_order; b++) {
        int bit_idx = bit_start + b;
        int word_idx = bit_idx / 32;
        int bit_offset = bit_idx % 32;
        uint32_t scr_bit = (scramble_seq[word_idx] >> bit_offset) & 1;

        float llr = scr_bit ? -llr_vals[b] : llr_vals[b];
        llr = fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE));
        llrs_int8[llr_base + b] = (int8_t)rintf(llr);
    }
}

/**
 * BATCHED version of kernel_fused_e2e_precomputed_scramble_int8.
 * Processes multiple grids in parallel using blockIdx.z for batch index.
 *
 * This eliminates the for loop over batch elements and achieves better
 * GPU utilization by launching all work in a single kernel.
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void kernel_fused_e2e_precomputed_scramble_int8_batch(
    const unsigned int* __restrict__ d_grids_cbf16,  /* [batch, port, sym, sc] contiguous */
    const __half2* __restrict__ d_dmrs_estimates_fp16,  /* [batch, dmrs_sym, prb, port, 12] */
    const int* __restrict__ re_indices,  /* Shared by all batch elements */
    const float* __restrict__ noise_vars,  /* [batch, port] */
    int8_t* __restrict__ llrs_int8,  /* [batch, nof_llrs] */
    const uint32_t* __restrict__ scramble_seq,  /* Shared by all batch elements */
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int mod_order,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    int batch_size,
    size_t grid_size_uint,  /* Size of one grid in uint32_t elements */
    int nof_llrs,  /* LLRs per slot */
    size_t estimates_per_slot)  /* FP16 estimates per slot */
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.z;

    if (batch_idx >= batch_size) return;

    /* Load noise variances to shared memory */
    __shared__ float s_noise_vars[8];

    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[batch_idx * NOF_PORTS + threadIdx.x];
    }
    __syncthreads();

    if (re_idx >= nof_re) return;

    /* Offset pointers for this batch element */
    const unsigned int* d_grid_batch = d_grids_cbf16 + batch_idx * grid_size_uint;
    const __half2* d_estimates_batch = d_dmrs_estimates_fp16 + batch_idx * estimates_per_slot;
    int8_t* llrs_batch = llrs_int8 + batch_idx * nof_llrs;

    /* Get source RE index */
    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

    /* Time interpolation */
    cuFloatComplex h[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum = 0.0f, h_imag_sum = 0.0f;
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
            __half2 h_fp16 = d_estimates_batch[est_idx];
            h_real_sum += __half2float(__low2half(h_fp16));
            h_imag_sum += __half2float(__high2half(h_fp16));
        }
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        h[p] = make_cuFloatComplex(h_real_sum * inv_n, h_imag_sum * inv_n);
    }

    /* Load received symbols */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        y[p] = cbf16_to_fp32(d_grid_batch[p * grid_stride + src_re]);
    }

    /* Equalization */
    cuFloatComplex eq;
    float eq_noise_var;

    if constexpr (NOF_PORTS == 1) {
        float h_sq = h[0].x * h[0].x + h[0].y * h[0].y;
        float noise = s_noise_vars[0];
        float scale;
        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            scale = 1.0f / fmaxf(h_sq, 1e-10f);
        } else {
            scale = 1.0f / (h_sq + noise);
        }
        eq.x = (y[0].x * h[0].x + y[0].y * h[0].y) * scale;
        eq.y = (y[0].y * h[0].x - y[0].x * h[0].y) * scale;
        eq_noise_var = (h_sq > 1e-10f) ? noise * scale : EQ_NOISE_VAR_DEEP_FADE;
    } else {
        float hh_real = 0.0f;
        cuFloatComplex hhy = make_cuFloatComplex(0.0f, 0.0f);
        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            hh_real += h[p].x * h[p].x + h[p].y * h[p].y;
            hhy.x += h[p].x * y[p].x + h[p].y * y[p].y;
            hhy.y += h[p].x * y[p].y - h[p].y * y[p].x;
        }
        float noise_sum = 0.0f;
        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            noise_sum += s_noise_vars[p];
        }
        float avg_noise = noise_sum / (float)NOF_PORTS;
        float scale;
        if constexpr (ALGORITHM == EQUALIZER_ZF) {
            scale = 1.0f / fmaxf(hh_real, 1e-10f);
        } else {
            scale = 1.0f / (hh_real + avg_noise);
        }
        eq.x = hhy.x * scale;
        eq.y = hhy.y * scale;
        eq_noise_var = (hh_real > 1e-10f) ? avg_noise * scale : EQ_NOISE_VAR_DEEP_FADE;
    }

    /* Apply TX scaling */
    eq.x *= tx_scaling;
    eq.y *= tx_scaling;

    /* Soft demodulation */
    float llr_vals[8];
    float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);

    if (mod_order == 2) {
        float scale = inv_noise * 2.8284271f  /* 2*sqrt(2) - matches CPU QPSK demod */;
        llr_vals[0] = eq.x * scale;
        llr_vals[1] = eq.y * scale;
    } else if (mod_order == 4) {
        float scale = inv_noise * 0.6324555f;
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.6324555f * 2.0f - fabsf(eq.x)) * scale * 2.0f;
        llr_vals[3] = (0.6324555f * 2.0f - fabsf(eq.y)) * scale * 2.0f;
    } else if (mod_order == 6) {
        // Proper max-log LLR computation for 64QAM
        const float s = 0.15430335f;  // 1/sqrt(42)
        const float s1 = s, s3 = 3.0f * s, s5 = 5.0f * s, s7 = 7.0f * s;
        float yi = eq.x, yq = eq.y;
        #define SQ(x) ((x)*(x))
        float d0_b0 = fminf(fminf(SQ(yi-s1),SQ(yi-s3)), fminf(SQ(yi-s5),SQ(yi-s7)));
        float d1_b0 = fminf(fminf(SQ(yi+s1),SQ(yi+s3)), fminf(SQ(yi+s5),SQ(yi+s7)));
        llr_vals[0] = (d1_b0 - d0_b0) * inv_noise;
        float d0_b1 = fminf(fminf(SQ(yq-s1),SQ(yq-s3)), fminf(SQ(yq-s5),SQ(yq-s7)));
        float d1_b1 = fminf(fminf(SQ(yq+s1),SQ(yq+s3)), fminf(SQ(yq+s5),SQ(yq+s7)));
        llr_vals[1] = (d1_b1 - d0_b1) * inv_noise;
        float d0_b2 = fminf(fminf(SQ(yi-s1),SQ(yi+s1)), fminf(SQ(yi-s3),SQ(yi+s3)));
        float d1_b2 = fminf(fminf(SQ(yi-s5),SQ(yi+s5)), fminf(SQ(yi-s7),SQ(yi+s7)));
        llr_vals[2] = (d1_b2 - d0_b2) * inv_noise;
        float d0_b3 = fminf(fminf(SQ(yq-s1),SQ(yq+s1)), fminf(SQ(yq-s3),SQ(yq+s3)));
        float d1_b3 = fminf(fminf(SQ(yq-s5),SQ(yq+s5)), fminf(SQ(yq-s7),SQ(yq+s7)));
        llr_vals[3] = (d1_b3 - d0_b3) * inv_noise;
        float d0_b4 = fminf(fminf(SQ(yi-s1),SQ(yi+s1)), fminf(SQ(yi-s5),SQ(yi+s5)));
        float d1_b4 = fminf(fminf(SQ(yi-s3),SQ(yi+s3)), fminf(SQ(yi-s7),SQ(yi+s7)));
        llr_vals[4] = (d1_b4 - d0_b4) * inv_noise;
        float d0_b5 = fminf(fminf(SQ(yq-s1),SQ(yq+s1)), fminf(SQ(yq-s5),SQ(yq+s5)));
        float d1_b5 = fminf(fminf(SQ(yq-s3),SQ(yq+s3)), fminf(SQ(yq-s7),SQ(yq+s7)));
        llr_vals[5] = (d1_b5 - d0_b5) * inv_noise;
        #undef SQ
    } else if (mod_order == 8) {
        float scale = inv_noise * 0.07669650f;
        float abs_re = fabsf(eq.x), abs_im = fabsf(eq.y);
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.07669650f * 8.0f - abs_re) * scale * 2.0f;
        llr_vals[3] = (0.07669650f * 8.0f - abs_im) * scale * 2.0f;
        float level2 = 0.07669650f * 4.0f;
        llr_vals[4] = (level2 - fabsf(abs_re - level2 * 2.0f)) * scale * 2.0f;
        llr_vals[5] = (level2 - fabsf(abs_im - level2 * 2.0f)) * scale * 2.0f;
        float level3 = 0.07669650f * 2.0f;
        llr_vals[6] = (level3 - fabsf(fabsf(abs_re - level2 * 2.0f) - level3 * 2.0f)) * scale * 2.0f;
        llr_vals[7] = (level3 - fabsf(fabsf(abs_im - level2 * 2.0f) - level3 * 2.0f)) * scale * 2.0f;
    }

    /* Read scrambling bits from PRE-COMPUTED buffer (no LFSR computation!) */
    int bit_start = re_idx * mod_order;
    int llr_base = re_idx * mod_order;
    const float INT8_SCALE = 4.0f;

    for (int b = 0; b < mod_order; b++) {
        int bit_idx = bit_start + b;
        int word_idx = bit_idx / 32;
        int bit_offset = bit_idx % 32;
        uint32_t scr_bit = (scramble_seq[word_idx] >> bit_offset) & 1;

        float llr = scr_bit ? -llr_vals[b] : llr_vals[b];
        llr = fmaxf(-120.0f, fminf(120.0f, llr * INT8_SCALE));
        llrs_batch[llr_base + b] = (int8_t)rintf(llr);
    }
}

/**
 * BATCHED version of kernel_lse_precomputed_pilots_fp16.
 * Processes channel estimation for multiple grids in parallel.
 */
__global__ void kernel_lse_precomputed_pilots_fp16_batch(
    const unsigned int* __restrict__ d_grids_cbf16,
    const uint32_t* __restrict__ d_pilot_bits,  /* Shared by all batch elements */
    const int* __restrict__ dmrs_symbol_indices,
    __half2* __restrict__ d_estimates_fp16,  /* [batch, dmrs_sym, prb, port, 12] */
    float* __restrict__ d_noise_var_accum,  /* [batch, port] */
    unsigned int* __restrict__ d_noise_count,  /* [batch] */
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling,
    int batch_size,
    size_t grid_size_uint,
    size_t estimates_per_slot,
    int precomputed_words_per_dmrs_sym)  /* uint32_t words per DMRS symbol (word-aligned) */
{
    /* Thread layout: 1 thread per PRB per port per batch */
    int prb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dmrs_sym_idx = blockIdx.y;
    int batch_port = blockIdx.z;  /* Encodes both batch_idx and port */
    int batch_idx = batch_port / nof_ports;
    int port = batch_port % nof_ports;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols ||
        batch_idx >= batch_size || port >= nof_ports) return;

    /* Offset pointers for this batch element */
    const unsigned int* d_grid_batch = d_grids_cbf16 + batch_idx * grid_size_uint;
    __half2* d_estimates_batch = d_estimates_fp16 + batch_idx * estimates_per_slot;
    float* d_noise_batch = d_noise_var_accum + batch_idx * nof_ports;

    int dmrs_symbol = dmrs_symbol_indices[dmrs_sym_idx];
    int prb_start_sc = (start_prb + prb_idx) * 12;
    const int dmrs_sc_offset[6] = {0, 2, 4, 6, 8, 10};

    float inv_sqrt2 = 0.70710678118f;
    float h_normalizer = 1.0f / dmrs_scaling;

    __half2 h_dmrs[6];
    float local_noise_sum = 0.0f;

    /* Process all 6 DMRS pilots for this PRB */
    #pragma unroll
    for (int d = 0; d < 6; d++) {
        int sc = prb_start_sc + dmrs_sc_offset[d];
        int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_batch[grid_idx]);

        /* Read pilot bits from pre-computed buffer.
         * Use word-aligned stride for DMRS symbol offset. */
        int pilot_idx = (start_prb + prb_idx) * 6 + d;
        int dmrs_word_offset = dmrs_sym_idx * precomputed_words_per_dmrs_sym;
        int word_idx_within_dmrs = pilot_idx / 16;
        int word_idx = dmrs_word_offset + word_idx_within_dmrs;
        int bit_offset = (pilot_idx % 16) * 2;
        uint32_t pilot_word = d_pilot_bits[word_idx];
        int c_real = (pilot_word >> bit_offset) & 1;
        int c_imag = (pilot_word >> (bit_offset + 1)) & 1;

        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        /* LSE: h = y × conj(pilot) / dmrs_scaling */
        float h_real = (y.x * p_real + y.y * p_imag) * h_normalizer;
        float h_imag = (y.y * p_real - y.x * p_imag) * h_normalizer;

        h_dmrs[d] = __halves2half2(__float2half(h_real), __float2half(h_imag));

        /* Compute noise residual */
        float h_scaled_real = h_real * dmrs_scaling;
        float h_scaled_imag = h_imag * dmrs_scaling;
        float y_ideal_real = h_scaled_real * p_real - h_scaled_imag * p_imag;
        float y_ideal_imag = h_scaled_real * p_imag + h_scaled_imag * p_real;
        float res_real = y.x - y_ideal_real;
        float res_imag = y.y - y_ideal_imag;
        local_noise_sum += res_real * res_real + res_imag * res_imag;
    }

    /* Frequency interpolation: output 12 estimates from 6 pilots */
    int re_per_dmrs_sym = nof_prb * nof_ports * 12;
    int out_base = dmrs_sym_idx * re_per_dmrs_sym + prb_idx * (nof_ports * 12) + port * 12;

    #pragma unroll
    for (int sc = 0; sc < 12; sc++) {
        __half2 h_out;
        if (sc % 2 == 0) {
            h_out = h_dmrs[sc / 2];
        } else {
            int left_idx = sc / 2;
            int right_idx = (left_idx + 1 < 6) ? left_idx + 1 : 5;
            __half2 h_left = h_dmrs[left_idx];
            __half2 h_right = h_dmrs[right_idx];
            __half2 h_sum = __hadd2(h_left, h_right);
            h_out = __hmul2(h_sum, __float2half2_rn(0.5f));
        }
        d_estimates_batch[out_base + sc] = h_out;
    }

    /* Atomic accumulate noise variance per batch element */
    atomicAdd(&d_noise_batch[port], local_noise_sum);

    /* Count total DMRS REs (only one thread per batch does this) */
    if (prb_idx == 0 && dmrs_sym_idx == 0 && port == 0) {
        atomicAdd(&d_noise_count[batch_idx], nof_prb * nof_dmrs_symbols * 6);
    }
}

/**
 * BATCHED version of kernel_finalize_noise_variance.
 * Finalizes noise variance for all batch elements in parallel.
 */
__global__ void kernel_finalize_noise_variance_batch(
    float* __restrict__ d_noise_vars,  /* [batch, port] */
    const unsigned int* __restrict__ d_noise_count,  /* [batch] */
    int nof_ports,
    float dmrs_scaling,
    int nof_dmrs_symbols,
    int batch_size)
{
    int batch_idx = blockIdx.x;
    int port = threadIdx.x;

    if (batch_idx >= batch_size || port >= nof_ports) return;

    unsigned int count = d_noise_count[batch_idx];
    if (count == 0) return;

    float* noise_ptr = d_noise_vars + batch_idx * nof_ports + port;
    float noise_sum = *noise_ptr;

    /* Normalize by count (2 components per complex sample) */
    float noise_var = noise_sum / (2.0f * count);

    /* Scale by DMRS scaling squared for proper noise level */
    noise_var /= (dmrs_scaling * dmrs_scaling);

    *noise_ptr = noise_var;
}

/**
 * Fused E2E kernel with on-the-fly time interpolation from FP16 estimates.
 *
 * This kernel performs:
 * 1. On-the-fly time interpolation (averaging across DMRS symbols)
 * 2. Equalization (ZF/MMSE/MMSE-IRC)
 * 3. Soft demodulation
 * 4. Descrambling
 *
 * Input: FP16 channel estimates from DMRS symbols [dmrs_sym, prb, port, 12]
 * Output: LLRs in FP16
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void kernel_fused_e2e_with_time_interp_fp16(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,  /* FP16 estimates [dmrs_sym, prb, port, 12] */
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    __half* __restrict__ llrs_half,
    const unsigned int* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int mod_order,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (re_idx >= nof_re) return;

    /* Load noise variances to shared memory */
    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    /* Get source RE index */
    int src_re = re_indices[re_idx];
    int symbol = src_re / symbol_stride;
    int subcarrier = src_re % symbol_stride;

    /* Compute relative indices for estimate lookup */
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    /* Calculate estimate layout indices */
    int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

    /* On-the-fly time interpolation: average FP16 estimates across all DMRS symbols */
    cuFloatComplex h[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum = 0.0f;
        float h_imag_sum = 0.0f;

        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
            __half2 h_fp16 = d_dmrs_estimates_fp16[est_idx];
            h_real_sum += __half2float(__low2half(h_fp16));
            h_imag_sum += __half2float(__high2half(h_fp16));
        }

        /* Average across DMRS symbols */
        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        h[p] = make_cuFloatComplex(h_real_sum * inv_n, h_imag_sum * inv_n);
    }

    /* Load received symbols from grid */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        int grid_idx = p * grid_stride + src_re;
        y[p] = cbf16_to_fp32(d_grid_cbf16[grid_idx]);
    }

    /* Equalization */
    cuFloatComplex eq;
    float eq_noise_var;
    equalize_symbol<NOF_PORTS, ALGORITHM>(y, h, s_noise_vars, tx_scaling, eq, eq_noise_var);

    /* Accumulate noise variance for SINR (atomic) */
    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, eq_noise_var);

    /* Soft demodulation */
    float llr_vals[8];
    float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);

    if (mod_order == 2) {
        float scale = inv_noise * 2.8284271f  /* 2*sqrt(2) - matches CPU QPSK demod */;
        llr_vals[0] = eq.x * scale;
        llr_vals[1] = eq.y * scale;
    } else if (mod_order == 4) {
        float scale = inv_noise * 0.6324555f;
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.6324555f * 2.0f - fabsf(eq.x)) * scale * 2.0f;
        llr_vals[3] = (0.6324555f * 2.0f - fabsf(eq.y)) * scale * 2.0f;
    } else if (mod_order == 6) {
        // 64QAM: piecewise-linear approximation matching CPU implementation
        soft_demod_64qam_piecewise(eq, inv_noise, llr_vals);
    } else if (mod_order == 8) {
        float scale = inv_noise * 0.07669650f;
        float abs_re = fabsf(eq.x);
        float abs_im = fabsf(eq.y);
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.07669650f * 8.0f - abs_re) * scale * 2.0f;
        llr_vals[3] = (0.07669650f * 8.0f - abs_im) * scale * 2.0f;
        float level2 = 0.07669650f * 4.0f;
        llr_vals[4] = (level2 - fabsf(abs_re - level2 * 2.0f)) * scale * 2.0f;
        llr_vals[5] = (level2 - fabsf(abs_im - level2 * 2.0f)) * scale * 2.0f;
        float level3 = 0.07669650f * 2.0f;
        float d_re = fabsf(abs_re - level2 * 2.0f);
        float d_im = fabsf(abs_im - level2 * 2.0f);
        llr_vals[6] = (level3 - fabsf(d_re - level3 * 2.0f)) * scale * 2.0f;
        llr_vals[7] = (level3 - fabsf(d_im - level3 * 2.0f)) * scale * 2.0f;
    }

    /* Descrambling */
    int llr_base = re_idx * mod_order;
    int word_idx = llr_base / 32;
    int bit_offset = llr_base % 32;
    unsigned int scr_word = scramble_seq[word_idx];
    unsigned int scr_word_next = scramble_seq[word_idx + 1];

    for (int b = 0; b < mod_order; b++) {
        int bit_pos = bit_offset + b;
        unsigned int scr_bit;
        if (bit_pos < 32) {
            scr_bit = (scr_word >> bit_pos) & 1;
        } else {
            scr_bit = (scr_word_next >> (bit_pos - 32)) & 1;
        }

        float llr = scr_bit ? -llr_vals[b] : llr_vals[b];
        llr = fmaxf(-65504.0f, fminf(65504.0f, llr));
        llrs_half[llr_base + b] = __float2half(llr);
    }
}

/**
 * Fused E2E kernel with direct INT8 output (eliminates FP16→INT8 conversion).
 *
 * This kernel performs the full pipeline and outputs INT8 LLRs directly:
 * 1. On-the-fly time interpolation from FP16 estimates
 * 2. Equalization (ZF/MMSE/MMSE-IRC)
 * 3. Soft demodulation with INT8 quantization
 * 4. Descrambling
 *
 * Saves ~2-3 µs by eliminating the separate FP16→INT8 conversion kernel.
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void kernel_fused_e2e_with_time_interp_int8(
    const unsigned int* __restrict__ d_grid_cbf16,
    const __half2* __restrict__ d_dmrs_estimates_fp16,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    int8_t* __restrict__ llrs_int8,  /* Direct INT8 output */
    const unsigned int* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int nof_dmrs_symbols,
    int nof_prb,
    int mod_order,
    int grid_stride,
    int symbol_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (re_idx >= nof_re) return;

    /* Load noise variances to shared memory */
    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    /* Get source RE index */
    int src_re = re_indices[re_idx];
    int subcarrier = src_re % symbol_stride;

    /* Compute relative indices for estimate lookup */
    int re_in_alloc = subcarrier - start_subcarrier;
    int prb_idx = re_in_alloc / 12;
    int sc_in_prb = re_in_alloc % 12;

    int re_per_dmrs_sym = nof_prb * NOF_PORTS * 12;

    /* On-the-fly time interpolation: average FP16 estimates */
    cuFloatComplex h[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        float h_real_sum = 0.0f;
        float h_imag_sum = 0.0f;

        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int est_idx = d * re_per_dmrs_sym + prb_idx * (NOF_PORTS * 12) + p * 12 + sc_in_prb;
            __half2 h_fp16 = d_dmrs_estimates_fp16[est_idx];
            h_real_sum += __half2float(__low2half(h_fp16));
            h_imag_sum += __half2float(__high2half(h_fp16));
        }

        float inv_n = 1.0f / (float)nof_dmrs_symbols;
        h[p] = make_cuFloatComplex(h_real_sum * inv_n, h_imag_sum * inv_n);
    }

    /* Load received symbols from grid */
    cuFloatComplex y[8];
    #pragma unroll
    for (int p = 0; p < NOF_PORTS; p++) {
        int grid_idx = p * grid_stride + src_re;
        y[p] = cbf16_to_fp32(d_grid_cbf16[grid_idx]);
    }

    /* Equalization */
    cuFloatComplex eq;
    float eq_noise_var;
    equalize_symbol<NOF_PORTS, ALGORITHM>(y, h, s_noise_vars, tx_scaling, eq, eq_noise_var);

    /* Accumulate noise variance for SINR */
    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, eq_noise_var);

    /* Soft demodulation with direct INT8 quantization */
    float llr_vals[8];
    float inv_noise = 1.0f / fmaxf(eq_noise_var, 1e-10f);

    /* INT8 scale factor (matches ldpc_llr_half_to_int8) */
    const float int8_scale = 8.0f;

    if (mod_order == 2) {
        float scale = inv_noise * 2.8284271f  /* 2*sqrt(2) - matches CPU QPSK demod */;
        llr_vals[0] = eq.x * scale;
        llr_vals[1] = eq.y * scale;
    } else if (mod_order == 4) {
        float scale = inv_noise * 0.6324555f;
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.6324555f * 2.0f - fabsf(eq.x)) * scale * 2.0f;
        llr_vals[3] = (0.6324555f * 2.0f - fabsf(eq.y)) * scale * 2.0f;
    } else if (mod_order == 6) {
        // 64QAM: piecewise-linear approximation matching CPU implementation
        soft_demod_64qam_piecewise(eq, inv_noise, llr_vals);
    } else if (mod_order == 8) {
        float scale = inv_noise * 0.07669650f;
        float abs_re = fabsf(eq.x);
        float abs_im = fabsf(eq.y);
        llr_vals[0] = eq.x * scale * 2.0f;
        llr_vals[1] = eq.y * scale * 2.0f;
        llr_vals[2] = (0.07669650f * 8.0f - abs_re) * scale * 2.0f;
        llr_vals[3] = (0.07669650f * 8.0f - abs_im) * scale * 2.0f;
        float level2 = 0.07669650f * 4.0f;
        llr_vals[4] = (level2 - fabsf(abs_re - level2 * 2.0f)) * scale * 2.0f;
        llr_vals[5] = (level2 - fabsf(abs_im - level2 * 2.0f)) * scale * 2.0f;
        float level3 = 0.07669650f * 2.0f;
        float d_re = fabsf(abs_re - level2 * 2.0f);
        float d_im = fabsf(abs_im - level2 * 2.0f);
        llr_vals[6] = (level3 - fabsf(d_re - level3 * 2.0f)) * scale * 2.0f;
        llr_vals[7] = (level3 - fabsf(d_im - level3 * 2.0f)) * scale * 2.0f;
    }

    /* Descrambling with direct INT8 output */
    int llr_base = re_idx * mod_order;
    int word_idx = llr_base / 32;
    int bit_offset = llr_base % 32;
    unsigned int scr_word = scramble_seq[word_idx];
    unsigned int scr_word_next = scramble_seq[word_idx + 1];

    for (int b = 0; b < mod_order; b++) {
        int bit_pos = bit_offset + b;
        unsigned int scr_bit;
        if (bit_pos < 32) {
            scr_bit = (scr_word >> bit_pos) & 1;
        } else {
            scr_bit = (scr_word_next >> (bit_pos - 32)) & 1;
        }

        float llr = scr_bit ? -llr_vals[b] : llr_vals[b];

        /* Quantize directly to INT8 */
        /* Note: srsRAN LLR range is [-120, 120], with ±127 reserved for fixed bits */
        float scaled = llr * int8_scale;
        scaled = fmaxf(-120.0f, fminf(120.0f, scaled));
        llrs_int8[llr_base + b] = (int8_t)rintf(scaled);
    }
}

/**
 * Compute noise variance directly from FP16 channel estimates.
 *
 * This kernel eliminates the redundant FP32 LSE computation by:
 * 1. Loading FP16 estimates for each DMRS symbol
 * 2. Averaging them on-the-fly (time domain averaging)
 * 3. Computing residual = received - h_avg × pilot
 * 4. Accumulating |residual|² with parallel reduction
 *
 * Launch: One block per port, 256 threads per block
 * Output: Per-port noise variance partial sums
 */
__global__ void kernel_noise_variance_from_fp16_estimates(
    const unsigned int* __restrict__ d_grid_cbf16,
    const unsigned int* __restrict__ d_dmrs_seq,
    const __half2* __restrict__ d_estimates_fp16,  /* FP16 estimates [dmrs_sym, prb, port, 12] */
    const int* __restrict__ dmrs_symbol_indices,
    float* __restrict__ d_noise_var_out,           /* Per-port output */
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    float dmrs_scaling)
{
    extern __shared__ float s_partial[];

    int port = blockIdx.x;
    int tid = threadIdx.x;
    int nof_threads = blockDim.x;

    if (port >= nof_ports) return;

    int total_dmrs_re = nof_prb * 6;  /* Type 1: 6 DMRS REs per PRB */
    int re_per_dmrs_sym = nof_prb * nof_ports * 12;

    float inv_sqrt2 = 0.70710678118f;
    float inv_n = 1.0f / (float)nof_dmrs_symbols;
    float local_sum = 0.0f;

    /* Each thread processes multiple DMRS REs */
    for (int dmrs_idx = tid; dmrs_idx < total_dmrs_re; dmrs_idx += nof_threads) {
        int prb_idx = dmrs_idx / 6;
        int d = dmrs_idx % 6;  /* DMRS RE index within PRB (0-5) */
        int sc_in_prb = d * 2; /* Actual subcarrier: 0,2,4,6,8,10 */

        /* Type 1 DMRS subcarrier offsets */
        const int dmrs_sc_offset[6] = {0, 2, 4, 6, 8, 10};
        int prb_start_sc = (start_prb + prb_idx) * 12;
        int sc = prb_start_sc + dmrs_sc_offset[d];

        /* Regenerate DMRS pilot (same for all symbols) */
        int pilot_idx = (start_prb + prb_idx) * 6 + d;
        int bit_idx_real = 2 * pilot_idx;
        int bit_idx_imag = 2 * pilot_idx + 1;

        int word_idx_real = bit_idx_real / 32;
        int bit_pos_real = 31 - (bit_idx_real % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_real = (d_dmrs_seq[word_idx_real] >> bit_pos_real) & 1;

        int word_idx_imag = bit_idx_imag / 32;
        int bit_pos_imag = 31 - (bit_idx_imag % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_imag = (d_dmrs_seq[word_idx_imag] >> bit_pos_imag) & 1;

        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        /* Average FP16 channel estimates across all DMRS symbols */
        float h_avg_real = 0.0f;
        float h_avg_imag = 0.0f;

        for (int sym = 0; sym < nof_dmrs_symbols; sym++) {
            /* FP16 estimate index: [dmrs_sym, prb, port, 12] */
            int est_idx = sym * re_per_dmrs_sym + prb_idx * (nof_ports * 12) + port * 12 + sc_in_prb;
            __half2 h_fp16 = d_estimates_fp16[est_idx];
            h_avg_real += __half2float(__low2half(h_fp16));
            h_avg_imag += __half2float(__high2half(h_fp16));
        }
        h_avg_real *= inv_n;
        h_avg_imag *= inv_n;

        /* Scale by dmrs_scaling to match CPU's scaled_estimates */
        h_avg_real *= dmrs_scaling;
        h_avg_imag *= dmrs_scaling;

        /* For each DMRS symbol, compute residual and accumulate */
        for (int sym = 0; sym < nof_dmrs_symbols; sym++) {
            int dmrs_symbol = dmrs_symbol_indices[sym];

            /* Load received symbol from grid */
            int grid_idx = port * grid_stride + dmrs_symbol * symbol_stride + sc;
            cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

            /* Compute ideal received: y_ideal = h_avg × pilot */
            float y_ideal_real = h_avg_real * p_real - h_avg_imag * p_imag;
            float y_ideal_imag = h_avg_real * p_imag + h_avg_imag * p_real;

            /* Residual = received - ideal */
            float res_real = y.x - y_ideal_real;
            float res_imag = y.y - y_ideal_imag;

            /* Accumulate |residual|² */
            local_sum += res_real * res_real + res_imag * res_imag;
        }
    }

    /* Store to shared memory for reduction */
    s_partial[tid] = local_sum;
    __syncthreads();

    /* Parallel reduction */
    for (int stride = nof_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }

    /* Thread 0 writes final result with scaling */
    if (tid == 0) {
        float total_sum = s_partial[0];

        /* Same finalization logic as kernel_noise_variance_finalize */
        int total_dmrs_re_all = nof_dmrs_symbols * total_dmrs_re;
        float divisor = (total_dmrs_re_all > 1) ? (float)(total_dmrs_re_all - 1) : 1.0f;
        float noise_var = total_sum / divisor;

        /* Dynamic scaling to match CPU */
        float N = (float)nof_dmrs_symbols;
        float beta_sq = dmrs_scaling * dmrs_scaling;
        float base_scale = beta_sq / sqrtf(N);

        /* Small allocation regularization */
        float alloc_factor = (total_dmrs_re_all < 50) ? 1.2f : 1.0f;

        float noise_var_scaled = noise_var * base_scale * alloc_factor;
        d_noise_var_out[port] = fmaxf(noise_var_scaled, 0.001f);
    }
}

/**
 * OPTIMIZED noise variance kernel that reads from pre-stored DMRS received symbols.
 *
 * This version eliminates grid re-reads by using the received symbols stored
 * during the DMRS LSE kernel. ~2x faster than kernel_noise_variance_from_fp16_estimates.
 *
 * Launch: One block per port, 256 threads per block
 * Input: d_dmrs_received [dmrs_sym, prb, port, 6] - stored during LSE
 * Output: Per-port noise variance
 */
__global__ void kernel_noise_variance_from_stored_dmrs(
    const cuFloatComplex* __restrict__ d_dmrs_received,  /* Stored received DMRS [dmrs_sym, prb, port, nof_dmrs_re_per_prb] */
    const unsigned int* __restrict__ d_dmrs_seq,
    const __half2* __restrict__ d_estimates_fp16,        /* FP16 estimates [dmrs_sym, prb, port, 12] */
    float* __restrict__ d_noise_var_out,                 /* Per-port output */
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int nof_dmrs_re_per_prb,  /* 6 for Type 1, 4 for Type 2 */
    int start_prb,
    float dmrs_scaling)
{
    extern __shared__ float s_partial[];

    int port = blockIdx.x;
    int tid = threadIdx.x;
    int nof_threads = blockDim.x;

    if (port >= nof_ports) return;

    int total_dmrs_re = nof_prb * nof_dmrs_re_per_prb;
    int re_per_dmrs_sym = nof_prb * nof_ports * 12;
    int recv_per_dmrs_sym = nof_prb * nof_ports * nof_dmrs_re_per_prb;

    float inv_sqrt2 = 0.70710678118f;
    float inv_n = 1.0f / (float)nof_dmrs_symbols;
    float local_sum = 0.0f;

    /* DMRS Type 1: RE positions 0,2,4,6,8,10 (sc = d*2)
     * DMRS Type 2: RE positions 0,1,6,7 (sc = type2_sc_map[d]) */
    const int type2_sc_map[4] = {0, 1, 6, 7};

    /* Each thread processes multiple DMRS REs */
    for (int dmrs_idx = tid; dmrs_idx < total_dmrs_re; dmrs_idx += nof_threads) {
        int prb_idx = dmrs_idx / nof_dmrs_re_per_prb;
        int d = dmrs_idx % nof_dmrs_re_per_prb;  /* DMRS RE index within PRB */

        /* Subcarrier position within PRB for channel estimate lookup */
        int sc_in_prb = (nof_dmrs_re_per_prb == 6) ? (d * 2) : type2_sc_map[d];

        /* Regenerate DMRS pilot (same for all symbols) */
        int pilot_idx = (start_prb + prb_idx) * nof_dmrs_re_per_prb + d;
        int bit_idx_real = 2 * pilot_idx;
        int bit_idx_imag = 2 * pilot_idx + 1;

        int word_idx_real = bit_idx_real / 32;
        int bit_pos_real = 31 - (bit_idx_real % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_real = (d_dmrs_seq[word_idx_real] >> bit_pos_real) & 1;

        int word_idx_imag = bit_idx_imag / 32;
        int bit_pos_imag = 31 - (bit_idx_imag % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_imag = (d_dmrs_seq[word_idx_imag] >> bit_pos_imag) & 1;

        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;

        /* Average FP16 channel estimates across all DMRS symbols */
        float h_avg_real = 0.0f;
        float h_avg_imag = 0.0f;

        for (int sym = 0; sym < nof_dmrs_symbols; sym++) {
            /* FP16 estimate index: [dmrs_sym, prb, port, 12] */
            int est_idx = sym * re_per_dmrs_sym + prb_idx * (nof_ports * 12) + port * 12 + sc_in_prb;
            __half2 h_fp16 = d_estimates_fp16[est_idx];
            h_avg_real += __half2float(__low2half(h_fp16));
            h_avg_imag += __half2float(__high2half(h_fp16));
        }
        h_avg_real *= inv_n;
        h_avg_imag *= inv_n;

        /* Scale by dmrs_scaling to match CPU's scaled_estimates */
        h_avg_real *= dmrs_scaling;
        h_avg_imag *= dmrs_scaling;

        /* For each DMRS symbol, compute residual using STORED received symbols */
        for (int sym = 0; sym < nof_dmrs_symbols; sym++) {
            /* Load received symbol from stored buffer (no grid read!) */
            int recv_idx = sym * recv_per_dmrs_sym + prb_idx * (nof_ports * nof_dmrs_re_per_prb) + port * nof_dmrs_re_per_prb + d;
            cuFloatComplex y = d_dmrs_received[recv_idx];

            /* Compute ideal received: y_ideal = h_avg × pilot */
            float y_ideal_real = h_avg_real * p_real - h_avg_imag * p_imag;
            float y_ideal_imag = h_avg_real * p_imag + h_avg_imag * p_real;

            /* Residual = received - ideal */
            float res_real = y.x - y_ideal_real;
            float res_imag = y.y - y_ideal_imag;

            /* Accumulate |residual|² */
            local_sum += res_real * res_real + res_imag * res_imag;
        }
    }

    /* Store to shared memory for reduction */
    s_partial[tid] = local_sum;
    __syncthreads();

    /* Parallel reduction */
    for (int stride = nof_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }

    /* Thread 0 writes final result with scaling */
    if (tid == 0) {
        float total_sum = s_partial[0];

        /* Same finalization logic as original kernel */
        int total_dmrs_re_all = nof_dmrs_symbols * total_dmrs_re;
        float divisor = (total_dmrs_re_all > 1) ? (float)(total_dmrs_re_all - 1) : 1.0f;
        float noise_var = total_sum / divisor;

        /* Dynamic scaling to match CPU */
        float N = (float)nof_dmrs_symbols;
        float beta_sq = dmrs_scaling * dmrs_scaling;
        float base_scale = beta_sq / sqrtf(N);

        /* Small allocation regularization */
        float alloc_factor = (total_dmrs_re_all < 50) ? 1.2f : 1.0f;

        float noise_var_scaled = noise_var * base_scale * alloc_factor;
        d_noise_var_out[port] = fmaxf(noise_var_scaled, 0.001f);
    }
}

/* ============================================================================
 * ORIGINAL CUDA Kernels (kept for compatibility)
 * ============================================================================ */

/**
 * Generate DMRS pilots and compute LSE channel estimates at DMRS positions.
 *
 * This kernel processes one PRB per thread block, with threads handling
 * different ports in parallel. For Type 1 DMRS, each PRB has 6 DMRS REs.
 *
 * Input grid layout: [port, symbol, subcarrier] in cbf16 format
 * Output: LSE estimates at DMRS positions [dmrs_symbol, port, prb, dmrs_re_in_prb]
 */
__global__ void kernel_dmrs_lse_type1(
    const unsigned int* __restrict__ d_grid_cbf16,
    const unsigned int* __restrict__ d_dmrs_seq,  /* Packed Gold sequence bits */
    cuFloatComplex* __restrict__ d_lse_estimates,
    int nof_prb,
    int nof_ports,
    int grid_stride,           /* nof_symbols * nof_subcarriers */
    int symbol_stride,         /* nof_subcarriers */
    int start_prb,
    int dmrs_symbol_idx,       /* Which symbol in the slot (absolute) */
    int start_symbol,          /* First symbol of allocation (for relative indexing) */
    float dmrs_scaling)
{
    int prb_idx = blockIdx.x;
    int port = threadIdx.x;

    if (prb_idx >= nof_prb || port >= nof_ports) return;

    /* Type 1 DMRS: REs at subcarriers 0, 2, 4, 6, 8, 10 within PRB */
    int prb_start_sc = (start_prb + prb_idx) * 12;

    /* DMRS RE indices within PRB for Type 1 */
    const int dmrs_sc_offset[6] = {0, 2, 4, 6, 8, 10};

    /* Match CPU channel estimation scaling exactly:
     * CPU generates pilots with amplitude 1/sqrt(2), giving |pilot|² = 0.5.
     * The received DMRS signal is: y = H × pilot_tx + noise
     * where pilot_tx = dmrs_scaling × (1/sqrt(2)) × QPSK.
     *
     * CPU LSE: h_raw = y × pilot* = H × dmrs_scaling × |pilot|² = H × dmrs_scaling × 0.5
     * CPU then divides by dmrs_scaling: h_cpu = H × 0.5
     *
     * The CPU MMSE formula is designed for this scaling, with |h|² = |H|² × 0.25.
     * The 0.5 factor affects both signal and noise variance such that SINR is correct.
     *
     * GPU must match: h = H × 0.5
     */
    float inv_sqrt2 = 0.70710678118f;  /* Unit amplitude, no dmrs_scaling */
    float h_normalizer = 1.0f / dmrs_scaling;  /* Divide by dmrs_scaling like CPU */

    #pragma unroll
    for (int d = 0; d < 6; d++) {
        int sc = prb_start_sc + dmrs_sc_offset[d];

        /* Load received symbol from grid.
         * Grid is uploaded with ABSOLUTE symbol indexing (0..13 for entire slot),
         * so use dmrs_symbol_idx directly without any offset. */
        int grid_idx = port * grid_stride + dmrs_symbol_idx * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        /* Generate DMRS pilot from Gold sequence.
         * Pilot index must account for start_prb offset since the sequence
         * starts from point A (RB 0) per 3GPP TS 38.211 Section 6.4.1.1.1.
         * CPU code uses prg.advance(start_prb * 6 * 2) to skip bits.
         */
        int pilot_idx = (start_prb + prb_idx) * 6 + d;
        int bit_idx_real = 2 * pilot_idx;
        int bit_idx_imag = 2 * pilot_idx + 1;

        int word_idx_real = bit_idx_real / 32;
        int bit_pos_real = 31 - (bit_idx_real % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_real = (d_dmrs_seq[word_idx_real] >> bit_pos_real) & 1;

        int word_idx_imag = bit_idx_imag / 32;
        int bit_pos_imag = 31 - (bit_idx_imag % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_imag = (d_dmrs_seq[word_idx_imag] >> bit_pos_imag) & 1;

        /* QPSK mapping: (1-2c) / sqrt(2) - unit amplitude like CPU */
        float p_real = (1 - 2 * c_real) * inv_sqrt2;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2;
        cuFloatComplex pilot = make_cuFloatComplex(p_real, p_imag);

        /* LSE: h_raw = y × conj(pilot) = H × dmrs_scaling × |pilot|² + noise_term
         *            = H × dmrs_scaling × 0.5 + noise_term
         * Divide by dmrs_scaling to match CPU: h = H × 0.5
         */
        cuFloatComplex h_raw = cuCmulf_fast(y, cuConjf_fast(pilot));
        cuFloatComplex h = make_cuFloatComplex(h_raw.x * h_normalizer,
                                               h_raw.y * h_normalizer);

        /* Store LSE estimate */
        /* Layout: [prb, port, dmrs_re] for coalesced access in interpolation */
        int out_idx = prb_idx * (nof_ports * 6) + port * 6 + d;
        d_lse_estimates[out_idx] = h;
    }
}

/**
 * Generate DMRS pilots and compute LSE channel estimates for LOW-PAPR DMRS.
 *
 * This kernel is used when transform precoding is enabled (MSG3/DFT-s-OFDM).
 * Low-PAPR DMRS uses pre-computed complex pilot symbols (Zadoff-Chu or lookup table)
 * instead of Gold sequence bits.
 *
 * For transform precoding DMRS (TS 38.211 Section 6.4.1.1.2):
 * - All 12 subcarriers per PRB carry DMRS (no data, no CDM)
 * - But pilots are mapped to even subcarriers (0, 2, 4, 6, 8, 10) per PRB
 * - Sequence length M_zc = nof_prb * 6
 *
 * Input grid layout: [port, symbol, subcarrier] in cbf16 format
 * Output: LSE estimates at DMRS positions [dmrs_symbol, port, prb, dmrs_re_in_prb]
 */
__global__ void kernel_dmrs_lse_type1_low_papr(
    const unsigned int* __restrict__ d_grid_cbf16,
    const cuFloatComplex* __restrict__ d_pilots,  /* Pre-computed low-PAPR pilots [M_zc] */
    cuFloatComplex* __restrict__ d_lse_estimates,
    int nof_prb,
    int nof_ports,
    int grid_stride,           /* nof_symbols * nof_subcarriers */
    int symbol_stride,         /* nof_subcarriers */
    int start_prb,
    int dmrs_symbol_idx,       /* Which symbol in the slot (absolute) */
    int start_symbol,          /* First symbol of allocation (for relative indexing) */
    float dmrs_scaling)
{
    int prb_idx = blockIdx.x;
    int port = threadIdx.x;

    if (prb_idx >= nof_prb || port >= nof_ports) return;

    /* Type 1 DMRS: REs at subcarriers 0, 2, 4, 6, 8, 10 within PRB */
    int prb_start_sc = (start_prb + prb_idx) * 12;

    /* DMRS RE indices within PRB for Type 1 */
    const int dmrs_sc_offset[6] = {0, 2, 4, 6, 8, 10};

    /* Normalization to match CPU scaling (see kernel_dmrs_lse_type1 for details) */
    float h_normalizer = 1.0f / dmrs_scaling;

    #pragma unroll
    for (int d = 0; d < 6; d++) {
        int sc = prb_start_sc + dmrs_sc_offset[d];

        /* Load received symbol from grid.
         * Grid is uploaded with ABSOLUTE symbol indexing (0..13 for entire slot),
         * so use dmrs_symbol_idx directly without any offset. */
        int grid_idx = port * grid_stride + dmrs_symbol_idx * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        /* Get low-PAPR pilot symbol.
         * For transform precoding, pilot index is within the allocation (no start_prb offset)
         * since the low-PAPR sequence is generated for the allocated PRBs only.
         * Pilot index = prb_idx * 6 + d (where d is DMRS RE index within PRB).
         */
        int pilot_idx = prb_idx * 6 + d;
        cuFloatComplex pilot = d_pilots[pilot_idx];

        /* LSE: h_raw = y × conj(pilot) = H × dmrs_scaling × |pilot|² + noise_term
         * Low-PAPR pilots have |pilot|² = 1 (unit amplitude).
         * Divide by dmrs_scaling to normalize like CPU does.
         */
        cuFloatComplex h_raw = cuCmulf_fast(y, cuConjf_fast(pilot));
        cuFloatComplex h = make_cuFloatComplex(h_raw.x * h_normalizer,
                                               h_raw.y * h_normalizer);

        /* Store LSE estimate */
        /* Layout: [prb, port, dmrs_re] for coalesced access in interpolation */
        int out_idx = prb_idx * (nof_ports * 6) + port * 6 + d;
        d_lse_estimates[out_idx] = h;
    }
}

/**
 * Average LSE channel estimates across all DMRS symbols (time-domain averaging).
 *
 * This mirrors the CPU algorithm which averages h_est FIRST before computing noise.
 * CPU: scaled_estimates = Σ h_est[sym] / (beta × nof_dmrs_symbols)
 *
 * Since the LSE kernel already divided by dmrs_scaling (beta), we only need to
 * divide by nof_dmrs_symbols here to complete the time-domain averaging.
 *
 * Input: LSE estimates for each DMRS symbol [dmrs_sym, prb, port, dmrs_re]
 * Output: Averaged estimates [prb, port, dmrs_re] (single buffer for all RE)
 */
__global__ void kernel_average_lse_estimates(
    const cuFloatComplex* __restrict__ d_lse_all_symbols,  /* [nof_dmrs_sym, nof_prb * nof_ports * 6] */
    cuFloatComplex* __restrict__ d_lse_averaged,           /* [nof_prb * nof_ports * 6] */
    int nof_dmrs_symbols,
    int nof_prb,
    int nof_ports,
    int total_dmrs_re_per_symbol,  /* nof_prb * 6 for Type 1 */
    float beta)                    /* dmrs_scaling factor from CPU (unused - LSE already applied it) */
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nof_prb * nof_ports * 6;
    if (idx >= total) return;

    /* Scale factor: 1 / nof_dmrs_symbols (LSE kernel already divided by beta) */
    float scale = 1.0f / (float)nof_dmrs_symbols;

    /* Accumulate across all DMRS symbols */
    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
    for (int d = 0; d < nof_dmrs_symbols; d++) {
        int sym_offset = d * (nof_ports * total_dmrs_re_per_symbol);
        cuFloatComplex h = d_lse_all_symbols[sym_offset + idx];
        sum.x += h.x;
        sum.y += h.y;
    }

    /* Apply scaling */
    d_lse_averaged[idx] = make_cuFloatComplex(sum.x * scale, sum.y * scale);
}

/**
 * Compute noise variance from DMRS residuals using TIME-AVERAGED channel estimates.
 *
 * This mirrors the CPU algorithm exactly:
 * 1. Use averaged h_est (computed by kernel_average_lse_estimates)
 * 2. For each DMRS symbol: residual = rx - avg_h_est × pilot
 * 3. Accumulate |residual|² across all DMRS symbols and REs
 *
 * Using averaged h_est prevents the noise cancellation problem that occurs
 * when using per-symbol estimates.
 *
 * Launch with one block per port, threads cooperatively reduce.
 */
__global__ void kernel_noise_variance_from_dmrs(
    const unsigned int* __restrict__ d_grid_cbf16,
    const unsigned int* __restrict__ d_dmrs_seq,
    const cuFloatComplex* __restrict__ d_lse_averaged,   /* AVERAGED estimates across DMRS symbols */
    float* __restrict__ d_noise_var_partial,             /* Per-block partial sums */
    int nof_prb,
    int nof_ports,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    int dmrs_symbol_idx,
    int start_symbol,        /* First symbol of allocation (for relative indexing) */
    float dmrs_scaling,
    int total_dmrs_re)    /* nof_prb * 6 for Type 1 */
{
    extern __shared__ float s_partial[];

    int port = blockIdx.x;
    int tid = threadIdx.x;
    int nof_threads = blockDim.x;

    if (port >= nof_ports) return;

    /* Reconstruct the ideal received signal using AVERAGED channel estimates.
     *
     * CPU algorithm (from port_channel_estimator_average_impl.cpp lines 446-491):
     * - First averages h_est across all DMRS symbols with scaling: avg_h = beta × Σ h / N
     *   where h_lse = H × 0.5 (after dividing by beta in LSE)
     *   So: avg_h = beta × H × 0.5
     * - Then computes: predicted_obs = avg_h × pilot_unit (NOT scaled by 2!)
     * - And residual = rx_pilots - predicted_obs
     *
     * For received signal: y = H × pilot_tx = H × beta × pilot_unit
     * predicted = avg_h × pilot_unit = beta × H × 0.5 × pilot_unit
     * residual = y - predicted = H × beta × pilot_unit - beta × H × 0.5 × pilot_unit
     *          = 0.5 × beta × H × pilot_unit
     *
     * The CPU intentionally has a residual that's 0.5 × signal, not zero.
     * This residual's power scales with signal power, giving a practical SINR
     * estimate (around 5-7 dB in ZMQ). We match this exactly.
     */
    float inv_sqrt2 = 0.70710678118f;  /* Unit pilot amplitude */
    float pilot_scale = 1.0f;  /* NO compensation - match CPU which leaves 0.5× signal residual */
    float local_sum = 0.0f;

    /* Each thread processes multiple DMRS REs */
    for (int dmrs_idx = tid; dmrs_idx < total_dmrs_re; dmrs_idx += nof_threads) {
        int prb_idx = dmrs_idx / 6;
        int d = dmrs_idx % 6;

        /* Type 1 DMRS: REs at subcarriers 0, 2, 4, 6, 8, 10 */
        const int dmrs_sc_offset[6] = {0, 2, 4, 6, 8, 10};
        int prb_start_sc = (start_prb + prb_idx) * 12;
        int sc = prb_start_sc + dmrs_sc_offset[d];

        /* Load received symbol.
         * Grid is uploaded with ABSOLUTE symbol indexing (0..13 for entire slot),
         * so use dmrs_symbol_idx directly without any offset. */
        int grid_idx = port * grid_stride + dmrs_symbol_idx * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        /* Regenerate DMRS pilot (accounting for start_prb offset) */
        int pilot_idx = (start_prb + prb_idx) * 6 + d;
        int bit_idx_real = 2 * pilot_idx;
        int bit_idx_imag = 2 * pilot_idx + 1;

        int word_idx_real = bit_idx_real / 32;
        int bit_pos_real = 31 - (bit_idx_real % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_real = (d_dmrs_seq[word_idx_real] >> bit_pos_real) & 1;

        int word_idx_imag = bit_idx_imag / 32;
        int bit_pos_imag = 31 - (bit_idx_imag % 32);  // MSB-first to match gold_sequence_generate_kernel
        int c_imag = (d_dmrs_seq[word_idx_imag] >> bit_pos_imag) & 1;

        /* QPSK with pilot_scale to reconstruct transmitted pilot amplitude */
        float p_real = (1 - 2 * c_real) * inv_sqrt2 * pilot_scale;
        float p_imag = (1 - 2 * c_imag) * inv_sqrt2 * pilot_scale;
        cuFloatComplex pilot = make_cuFloatComplex(p_real, p_imag);

        /* Load AVERAGED channel estimate and scale by beta (dmrs_scaling).
         *
         * CPU algorithm (port_channel_estimator_average_impl.cpp lines 447-449):
         *   scaling_factor = beta / nof_lse_symbols
         *   scaled_estimates = estimates × scaling_factor
         *
         * For time-domain average (nof_lse_symbols=1), this gives:
         *   scaled_estimates = filtered_estimates × beta
         *
         * Our d_lse_averaged = (Σ y × conj(p) / beta) / N = (Σ y × conj(p)) / (beta × N)
         * We need to multiply by beta to match CPU's scaled_estimates.
         */
        int lse_idx = prb_idx * (nof_ports * 6) + port * 6 + d;
        cuFloatComplex h_avg_raw = d_lse_averaged[lse_idx];
        /* Scale by dmrs_scaling to match CPU's scaled_estimates = estimates × beta */
        cuFloatComplex h_avg = make_cuFloatComplex(h_avg_raw.x * dmrs_scaling, h_avg_raw.y * dmrs_scaling);

        /* Compute ideal received using scaled estimate:
         * y_ideal = h_avg × pilot
         *
         * With h_avg = (Σ y × conj(p)) / N and pilot = unit amplitude:
         * For unity channel y ≈ beta × pilot, so:
         * y_ideal = (Σ beta × pilot × conj(pilot)) / N × pilot = beta × |p|² / N × pilot = beta × pilot / N
         *
         * The actual received is: y = beta × pilot + noise
         * residual = y - y_ideal = beta × pilot - beta × pilot / N = beta × pilot × (1 - 1/N) + noise
         *
         * For N > 1, there's a residual signal component. This matches CPU behavior.
         */
        cuFloatComplex y_ideal = cuCmulf_fast(h_avg, pilot);

        /* Residual = received - ideal ≈ noise */
        float res_real = y.x - y_ideal.x;
        float res_imag = y.y - y_ideal.y;

        /* Accumulate |residual|^2 */
        local_sum += res_real * res_real + res_imag * res_imag;
    }

    /* Store to shared memory */
    s_partial[tid] = local_sum;
    __syncthreads();

    /* Parallel reduction in shared memory */
    for (int stride = nof_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }

    /* Thread 0 writes result */
    if (tid == 0) {
        d_noise_var_partial[port] = s_partial[0];
    }
}

__global__ void kernel_noise_variance_from_low_papr_dmrs(
    const unsigned int* __restrict__ d_grid_cbf16,
    const cuFloatComplex* __restrict__ d_pilots,
    const cuFloatComplex* __restrict__ d_lse_averaged,
    float* __restrict__ d_noise_var_partial,
    int nof_prb,
    int nof_ports,
    int grid_stride,
    int symbol_stride,
    int start_prb,
    int dmrs_symbol_idx,
    float dmrs_scaling,
    int total_dmrs_re)
{
    extern __shared__ float s_partial[];

    int port = blockIdx.x;
    int tid = threadIdx.x;
    int nof_threads = blockDim.x;

    if (port >= nof_ports) return;

    const int dmrs_sc_offset[6] = {0, 2, 4, 6, 8, 10};
    float local_sum = 0.0f;

    for (int dmrs_idx = tid; dmrs_idx < total_dmrs_re; dmrs_idx += nof_threads) {
        int prb_idx = dmrs_idx / 6;
        int d = dmrs_idx - prb_idx * 6;
        int sc = (start_prb + prb_idx) * 12 + dmrs_sc_offset[d];

        int grid_idx = port * grid_stride + dmrs_symbol_idx * symbol_stride + sc;
        cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

        int pilot_idx = prb_idx * 6 + d;
        cuFloatComplex pilot = d_pilots[pilot_idx];

        int lse_idx = prb_idx * (nof_ports * 6) + port * 6 + d;
        cuFloatComplex h_avg_raw = d_lse_averaged[lse_idx];
        cuFloatComplex h_avg = make_cuFloatComplex(h_avg_raw.x * dmrs_scaling, h_avg_raw.y * dmrs_scaling);
        cuFloatComplex y_ideal = cuCmulf_fast(h_avg, pilot);

        float res_real = y.x - y_ideal.x;
        float res_imag = y.y - y_ideal.y;
        local_sum += res_real * res_real + res_imag * res_imag;
    }

    s_partial[tid] = local_sum;
    __syncthreads();

    for (int stride = nof_threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_noise_var_partial[port] = s_partial[0];
    }
}

/**
 * Finalize noise variance: divide accumulated sum by count with dynamic scaling.
 *
 * The scaling factor accounts for differences between GPU and CPU noise estimation:
 * - GPU: residual = y - (h_avg × dmrs_scaling × pilot)
 * - CPU: residual = y - (h_lse × dmrs_scaling² / N × pilot)
 *
 * The residual magnitudes differ, requiring a configuration-dependent scaling factor
 * to match the CPU's noise variance estimate and achieve consistent SINR.
 */
__global__ void kernel_noise_variance_finalize(
    float* __restrict__ d_noise_vars,
    const float* __restrict__ d_partial_sums,
    int nof_ports,
    int nof_dmrs_symbols,
    int total_dmrs_re_per_symbol,
    float dmrs_scaling)
{
    int port = threadIdx.x;
    if (port >= nof_ports) return;

    /* Sum across all DMRS symbols */
    float total_sum = 0.0f;
    for (int d = 0; d < nof_dmrs_symbols; d++) {
        total_sum += d_partial_sums[d * nof_ports + port];
    }

    /* Average: divide by total DMRS REs.
     * Use (total_dmrs_re - 1) to match CPU's unbiased variance estimator. */
    int total_dmrs_re = nof_dmrs_symbols * total_dmrs_re_per_symbol;
    float divisor = (total_dmrs_re > 1) ? (float)(total_dmrs_re - 1) : 1.0f;
    float noise_var = total_sum / divisor;

    /* Dynamic scaling based on DMRS configuration.
     *
     * The GPU and CPU use different formulas for the predicted signal, leading to
     * different residual magnitudes. The scaling factor corrects for this difference.
     *
     * GPU residual factor: (1 - 0.5/N) where N = nof_dmrs_symbols
     * CPU residual factor: (1 - beta × 0.5/N) where beta = dmrs_scaling
     *
     * For beta ≈ 1.414 (2 CDM groups):
     * - N=1: GPU/CPU ratio ≈ 2.9, but empirically 2.0 works better
     * - N=2: GPU/CPU ratio ≈ 1.35
     *
     * Use a scaling formula that interpolates between these cases:
     * scale = 2.0 / sqrt(N) provides reasonable balance.
     *
     * Additionally, for very small allocations (few DMRS REs), the statistical
     * noise in the variance estimate is higher, so we may need more regularization.
     */
    float N = (float)nof_dmrs_symbols;
    /* Dynamic noise variance scaling to match CPU behavior.
     *
     * The CPU uses DIFFERENT scaling for equalization vs noise estimation:
     * - Equalization: h = pilots_lse / beta / N
     * - Noise estimation: h = pilots_lse × beta / N
     *
     * The noise h is beta² larger than eq h. Since GPU uses the same scaling
     * (both divided by beta and N), the GPU noise variance is beta² times smaller.
     *
     * To compensate, we multiply noise_var by dmrs_scaling² (beta²).
     *
     * Additional factors:
     * - 2.0 / sqrt(N) provides empirical regularization
     */
    float beta_sq = dmrs_scaling * dmrs_scaling;
    /* Reduced scale factor to better match CPU noise variance estimation.
     * Previous formula (2.0 * beta_sq / sqrt(N)) was overestimating noise variance.
     * New formula: beta_sq / sqrt(N) to match CPU behavior more closely. */
    float base_scale = beta_sq / sqrtf(N);

    /* For small allocations (< 50 DMRS REs total), increase scaling slightly
     * to provide more regularization and improve decode stability. */
    float alloc_factor = 1.0f;
    if (total_dmrs_re < 50) {
        alloc_factor = 1.2f;  /* 20% more regularization for small allocations */
    }

    float noise_var_scaled = noise_var * base_scale * alloc_factor;

    /* Apply minimum floor to prevent numerical issues with clean channels. */
    d_noise_vars[port] = fmaxf(noise_var_scaled, 0.001f);
}

__global__ void kernel_noise_variance_finalize_low_papr(
    float* __restrict__ d_noise_vars,
    const float* __restrict__ d_partial_sums,
    int nof_ports,
    int nof_dmrs_symbols,
    int total_dmrs_re_per_symbol)
{
    int port = threadIdx.x;
    if (port >= nof_ports) return;

    float total_sum = 0.0f;
    for (int d = 0; d < nof_dmrs_symbols; d++) {
        total_sum += d_partial_sums[d * nof_ports + port];
    }

    int total_dmrs_re = nof_dmrs_symbols * total_dmrs_re_per_symbol;
    float divisor = (total_dmrs_re > 1) ? static_cast<float>(total_dmrs_re - 1) : 1.0f;
    float noise_var = total_sum / divisor;

    /* The averaged LSE includes the same DMRS noise that is present in the
     * residual. Correct the variance loss from subtracting an N-symbol mean. */
    if (nof_dmrs_symbols > 1) {
        noise_var *= static_cast<float>(nof_dmrs_symbols) / static_cast<float>(nof_dmrs_symbols - 1);
    }

    d_noise_vars[port] = fmaxf(noise_var, 1e-8f);
}

/**
 * Frequency interpolation: expand DMRS estimates to all subcarriers within PRB.
 *
 * Type 1 DMRS: estimates at SC 0,2,4,6,8,10 → interpolate to 1,3,5,7,9,11
 *
 * Input: LSE estimates at DMRS positions [prb, port, dmrs_re_in_prb]
 * Output: Full estimates for all subcarriers [port, prb, sc_in_prb]
 */
__global__ void kernel_freq_interpolate_type1(
    const cuFloatComplex* __restrict__ d_lse_estimates,
    cuFloatComplex* __restrict__ d_full_estimates,
    int nof_prb,
    int nof_ports)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nof_prb * nof_ports * 12;  /* All REs */
    if (idx >= total) return;

    int prb = idx / (nof_ports * 12);
    int port_sc = idx % (nof_ports * 12);
    int port = port_sc / 12;
    int sc_in_prb = port_sc % 12;

    /* Load LSE estimates for this PRB and port */
    int lse_base = prb * (nof_ports * 6) + port * 6;

    cuFloatComplex h;

    if (sc_in_prb % 2 == 0) {
        /* DMRS position - direct copy */
        int dmrs_idx = sc_in_prb / 2;
        h = d_lse_estimates[lse_base + dmrs_idx];
    } else {
        /* Interpolate between adjacent DMRS positions */
        int left_idx = sc_in_prb / 2;
        int right_idx = left_idx + 1;
        if (right_idx >= 6) right_idx = 5;  /* Edge case */

        cuFloatComplex h_left = d_lse_estimates[lse_base + left_idx];
        cuFloatComplex h_right = d_lse_estimates[lse_base + right_idx];

        /* Linear interpolation (alpha = 0.5 for midpoint) */
        h.x = 0.5f * (h_left.x + h_right.x);
        h.y = 0.5f * (h_left.y + h_right.y);
    }

    /* Store to output: [port, prb, sc_in_prb] layout */
    int out_idx = port * (nof_prb * 12) + prb * 12 + sc_in_prb;
    d_full_estimates[out_idx] = h;
}

/**
 * Frequency-domain smoothing kernel (FP16 in-place).
 *
 * Applies a 15-tap raised cosine FIR filter to denoise LSE channel estimates
 * across frequency, matching the CPU port_channel_estimator_helpers filter.
 *
 * The filter operates on DMRS pilot positions (6 per PRB for Type 1), then
 * re-interpolates to all 12 subcarriers per PRB.
 *
 * Grid:  (nof_prb, nof_dmrs_symbols)
 * Block: (nof_ports)
 *
 * Each thread handles one (prb, port) pair for one DMRS symbol, applying
 * the filter across 6 pilot positions and writing 12 smoothed subcarriers.
 */
__global__ void kernel_fd_smoothing_fp16(
    __half2* __restrict__ d_estimates,  /* [dmrs_sym, prb, port, 12] in-place */
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols,
    int dmrs_type)
{
    int prb_idx = blockIdx.x;
    int dmrs_sym_idx = blockIdx.y;
    int port = threadIdx.x;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols || port >= nof_ports) return;

    const int HALF_FILTER = FD_SMOOTH_FILTER_LEN / 2;  /* 7 */
    int re_per_dmrs_sym = nof_prb * nof_ports * 12;
    int base = dmrs_sym_idx * re_per_dmrs_sym + prb_idx * (nof_ports * 12) + port * 12;
    int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);
    int total_pilots = nof_prb * nof_pilots;
    int pilot_base = prb_idx * nof_pilots;

    /* Apply FIR filter to each of the 6 pilot positions */
    float smoothed_re[6], smoothed_im[6];

    #pragma unroll
    for (int p = 0; p < 6; p++) {
        if (p >= nof_pilots) break;
        float sum_re = 0.0f;
        float sum_im = 0.0f;
        float coef_sum = 0.0f;
        int global_pilot = pilot_base + p;

        for (int t = 0; t < FD_SMOOTH_FILTER_LEN; t++) {
            int src_pilot = global_pilot + (t - HALF_FILTER);

            /* Skip out-of-range taps (don't clamp) */
            if (src_pilot < 0 || src_pilot >= total_pilots) continue;

            /* Read from the appropriate PRB's estimates */
            int src_prb = src_pilot / nof_pilots;
            int src_pos = src_pilot % nof_pilots;
            int src_idx = dmrs_sym_idx * re_per_dmrs_sym +
                          src_prb * (nof_ports * 12) + port * 12 +
                          pusch_dmrs_sc_offset(dmrs_type, 0, src_pos);

            __half2 h_src = d_estimates[src_idx];
            float r = __half2float(__low2half(h_src));
            float i = __half2float(__high2half(h_src));

            sum_re += d_rc_filter[t] * r;
            sum_im += d_rc_filter[t] * i;
            coef_sum += d_rc_filter[t];
        }

        /* Renormalize to compensate for truncated taps at band edges */
        if (coef_sum > 1e-6f) {
            float inv_coef = 1.0f / coef_sum;
            sum_re *= inv_coef;
            sum_im *= inv_coef;
        }

        smoothed_re[p] = sum_re;
        smoothed_im[p] = sum_im;
    }

    /* Write back smoothed estimates for all 12 subcarriers */
    #pragma unroll
    for (int sc = 0; sc < 12; sc++) {
        float r, i_val;
        int left = 0;
        int right = 0;
        int left_sc = pusch_dmrs_sc_offset(dmrs_type, 0, 0);
        int right_sc = left_sc;
        bool exact = false;
        for (int p = 0; p < nof_pilots; ++p) {
            int pilot_sc = pusch_dmrs_sc_offset(dmrs_type, 0, p);
            if (pilot_sc == sc) {
                left = right = p;
                exact = true;
                break;
            }
            if (pilot_sc < sc) {
                left = p;
                left_sc = pilot_sc;
            } else {
                right = p;
                right_sc = pilot_sc;
                break;
            }
        }
        if (exact || sc <= left_sc || right_sc <= left_sc) {
            r = smoothed_re[left];
            i_val = smoothed_im[left];
        } else {
            float alpha = (float)(sc - left_sc) / (float)(right_sc - left_sc);
            r = (1.0f - alpha) * smoothed_re[left] + alpha * smoothed_re[right];
            i_val = (1.0f - alpha) * smoothed_im[left] + alpha * smoothed_im[right];
        }
        d_estimates[base + sc] = __halves2half2(__float2half(r), __float2half(i_val));
    }
}

__global__ void kernel_mimo_fd_smoothing_fp16(
    __half2* __restrict__ d_estimates,  /* [dmrs_sym, prb, port, layer, 12] in-place */
    int nof_prb,
    int nof_ports,
    int nof_layers,
    int nof_dmrs_symbols,
    int dmrs_type)
{
    int prb_idx = blockIdx.x;
    int dmrs_sym_idx = blockIdx.y;
    int layer = blockIdx.z;
    int port = threadIdx.x;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols || port >= nof_ports || layer >= nof_layers) return;

    const int HALF_FILTER = FD_SMOOTH_FILTER_LEN / 2;
    int re_per_layer = 12;
    int re_per_port = nof_layers * re_per_layer;
    int re_per_prb = nof_ports * re_per_port;
    int re_per_dmrs_sym = nof_prb * re_per_prb;
    int base = dmrs_sym_idx * re_per_dmrs_sym + prb_idx * re_per_prb + port * re_per_port + layer * re_per_layer;
    int nof_pilots = pusch_dmrs_pilots_per_cdm(dmrs_type);
    int total_pilots = nof_prb * nof_pilots;
    int pilot_base = prb_idx * nof_pilots;
    int cdm_offset = layer / 2;

    float smoothed_re[6], smoothed_im[6];

    #pragma unroll
    for (int p = 0; p < 6; p++) {
        if (p >= nof_pilots) break;
        float sum_re = 0.0f;
        float sum_im = 0.0f;
        float coef_sum = 0.0f;
        int global_pilot = pilot_base + p;

        for (int t = 0; t < FD_SMOOTH_FILTER_LEN; t++) {
            int src_pilot = global_pilot + (t - HALF_FILTER);
            if (src_pilot < 0 || src_pilot >= total_pilots) continue;

            int src_prb = src_pilot / nof_pilots;
            int src_pos = src_pilot % nof_pilots;
            int src_sc = pusch_dmrs_sc_offset(dmrs_type, cdm_offset, src_pos);
            int src_idx = dmrs_sym_idx * re_per_dmrs_sym +
                          src_prb * re_per_prb + port * re_per_port + layer * re_per_layer + src_sc;

            __half2 h_src = d_estimates[src_idx];
            sum_re += d_rc_filter[t] * __half2float(__low2half(h_src));
            sum_im += d_rc_filter[t] * __half2float(__high2half(h_src));
            coef_sum += d_rc_filter[t];
        }

        if (coef_sum > 1e-6f) {
            float inv_coef = 1.0f / coef_sum;
            sum_re *= inv_coef;
            sum_im *= inv_coef;
        }

        smoothed_re[p] = sum_re;
        smoothed_im[p] = sum_im;
    }

    __half2 smoothed_pilots[6];
    #pragma unroll
    for (int p = 0; p < 6; p++) {
        if (p >= nof_pilots) break;
        smoothed_pilots[p] = __halves2half2(__float2half(smoothed_re[p]), __float2half(smoothed_im[p]));
    }

    #pragma unroll
    for (int sc = 0; sc < 12; sc++) {
        d_estimates[base + sc] =
            interpolate_dmrs_pilots_half2(smoothed_pilots, nof_pilots, dmrs_type, cdm_offset, sc);
    }
}

/**
 * Frequency-domain smoothing kernel (FP32 cuFloatComplex, in-place).
 *
 * Same algorithm as kernel_fd_smoothing_fp16 but for the separate-kernel path
 * which uses cuFloatComplex with layout [dmrs_sym][port][prb * 12].
 *
 * Grid:  (nof_prb, nof_dmrs_symbols)
 * Block: (nof_ports)
 */
__global__ void kernel_fd_smoothing_fp32(
    cuFloatComplex* __restrict__ d_estimates,  /* [dmrs_sym][port][nof_prb * 12] in-place */
    int nof_prb,
    int nof_ports,
    int nof_dmrs_symbols)
{
    int prb_idx = blockIdx.x;
    int dmrs_sym_idx = blockIdx.y;
    int port = threadIdx.x;

    if (prb_idx >= nof_prb || dmrs_sym_idx >= nof_dmrs_symbols || port >= nof_ports) return;

    const int HALF_FILTER = FD_SMOOTH_FILTER_LEN / 2;  /* 7 */
    int re_per_sym = nof_ports * nof_prb * 12;
    int port_base = dmrs_sym_idx * re_per_sym + port * (nof_prb * 12) + prb_idx * 12;
    int total_pilots = nof_prb * 6;
    int pilot_base = prb_idx * 6;

    /* Apply FIR filter to each of the 6 pilot positions */
    float smoothed_re[6], smoothed_im[6];

    #pragma unroll
    for (int p = 0; p < 6; p++) {
        float sum_re = 0.0f;
        float sum_im = 0.0f;
        float coef_sum = 0.0f;
        int global_pilot = pilot_base + p;

        for (int t = 0; t < FD_SMOOTH_FILTER_LEN; t++) {
            int src_pilot = global_pilot + (t - HALF_FILTER);

            /* Skip out-of-range taps (don't clamp) */
            if (src_pilot < 0 || src_pilot >= total_pilots) continue;

            int src_prb = src_pilot / 6;
            int src_pos = src_pilot % 6;
            int src_idx = dmrs_sym_idx * re_per_sym +
                          port * (nof_prb * 12) + src_prb * 12 + src_pos * 2;

            cuFloatComplex h_src = d_estimates[src_idx];
            sum_re += d_rc_filter[t] * h_src.x;
            sum_im += d_rc_filter[t] * h_src.y;
            coef_sum += d_rc_filter[t];
        }

        /* Renormalize to compensate for truncated taps at band edges */
        if (coef_sum > 1e-6f) {
            float inv_coef = 1.0f / coef_sum;
            sum_re *= inv_coef;
            sum_im *= inv_coef;
        }

        smoothed_re[p] = sum_re;
        smoothed_im[p] = sum_im;
    }

    /* Write back smoothed estimates for all 12 subcarriers */
    #pragma unroll
    for (int sc = 0; sc < 12; sc++) {
        float r, i_val;
        if (sc % 2 == 0) {
            r = smoothed_re[sc / 2];
            i_val = smoothed_im[sc / 2];
        } else {
            int left = sc / 2;
            int right = (left + 1 < 6) ? left + 1 : 5;
            r = 0.5f * (smoothed_re[left] + smoothed_re[right]);
            i_val = 0.5f * (smoothed_im[left] + smoothed_im[right]);
        }
        d_estimates[port_base + sc] = make_cuFloatComplex(r, i_val);
    }
}

/**
 * Time interpolation: interpolate estimates from DMRS symbols to data symbols.
 *
 * For symbols between two DMRS symbols: linear interpolation
 * For symbols before first DMRS: copy from first DMRS
 * For symbols after last DMRS: copy from last DMRS
 *
 * Input: Frequency-interpolated estimates for each DMRS symbol
 * Output: Estimates for all symbols in the allocation
 */
/**
 * Time interpolation kernel with optional scaling.
 *
 * CPU algorithm applies: total_scaling = 1 / (beta × N) to the LSE
 * Our LSE kernel already applies 1/beta, so we need to apply 1/N here.
 *
 * The scale factor ensures GPU channel estimates match CPU magnitude:
 * - GPU without scale: h = y × conj(p) / beta
 * - GPU with scale: h = y × conj(p) / beta × scale = y × conj(p) / (beta × N)
 * - CPU: h = y × conj(p) / (beta × N)
 */
__global__ void kernel_time_interpolate(
    const cuFloatComplex* __restrict__ d_dmrs_estimates,  /* [dmrs_sym_idx, port, nof_re] */
    cuFloatComplex* __restrict__ d_full_estimates,        /* [symbol, port, nof_re] */
    const int* __restrict__ dmrs_symbol_indices,          /* Array of DMRS symbol indices */
    int nof_dmrs_symbols,
    int nof_symbols,
    int nof_ports,
    int nof_re_per_symbol,
    int start_symbol,
    float output_scale)   /* Scale factor: 1/nof_dmrs_symbols to match CPU averaging */
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nof_symbols * nof_ports * nof_re_per_symbol;
    if (idx >= total) return;

    int sym_rel = idx / (nof_ports * nof_re_per_symbol);
    int port_re = idx % (nof_ports * nof_re_per_symbol);
    int port = port_re / nof_re_per_symbol;
    int re = port_re % nof_re_per_symbol;

    /* Time-average all DMRS estimates (matches CPU "average" strategy) */
    int est_idx = port * nof_re_per_symbol + re;
    int stride = nof_ports * nof_re_per_symbol;

    cuFloatComplex h = make_cuFloatComplex(0.0f, 0.0f);
    for (int d = 0; d < nof_dmrs_symbols; d++) {
        cuFloatComplex hd = d_dmrs_estimates[d * stride + est_idx];
        h.x += hd.x;
        h.y += hd.y;
    }
    float inv_n = output_scale / (float)nof_dmrs_symbols;
    h.x *= inv_n;
    h.y *= inv_n;

    /* Store output: [symbol, port, re] */
    int out_idx = sym_rel * (nof_ports * nof_re_per_symbol) + port * nof_re_per_symbol + re;
    d_full_estimates[out_idx] = h;
}

/**
 * Fused E2E kernel with channel estimate lookup.
 *
 * This version looks up pre-computed channel estimates from GPU memory
 * instead of receiving them as input. Supports ZF, MMSE, and MMSE-IRC.
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void kernel_pusch_e2e_with_estimates(
    const unsigned int* __restrict__ d_grid_cbf16,
    const cuFloatComplex* __restrict__ d_estimates,  /* GPU-computed estimates */
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    __half* __restrict__ llrs_half,
    const unsigned int* __restrict__ scramble_seq,
    float* __restrict__ d_eq_noise_var_sum,         /* Accumulator for SINR */
    unsigned int* __restrict__ d_eq_noise_var_count, /* RE counter for SINR */
    int nof_re,
    int mod_order,
    int grid_stride,           /* nof_symbols * nof_subcarriers */
    int symbol_stride,         /* grid_nof_subcarriers (for decoding src_re) */
    int est_stride,            /* nof_re_per_symbol for estimates */
    int start_symbol,          /* allocation start symbol */
    int start_subcarrier,      /* allocation start subcarrier (start_prb * 12) */
    float tx_scaling)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (re_idx >= nof_re) return;

    /* Load noise variances to shared memory */
    __shared__ float s_noise_vars[8];
    if (threadIdx.x < NOF_PORTS) {
        s_noise_vars[threadIdx.x] = noise_vars[threadIdx.x];
    }
    __syncthreads();

    /* Get source RE index (in full grid) */
    int src_re = re_indices[re_idx];

    /* Decode symbol and subcarrier from src_re */
    int symbol = src_re / symbol_stride;
    int subcarrier = src_re % symbol_stride;

    /* Compute relative indices for estimate lookup */
    int symbol_rel = symbol - start_symbol;
    int re_in_symbol = subcarrier - start_subcarrier;

    /* Load channel data for all ports.
     * Grid is uploaded with ALL slot symbols (absolute indexing 0..13),
     * so use src_re directly for grid lookup.
     * Estimates are GPU-computed with relative indexing, so use symbol_rel/re_in_symbol. */
    cuFloatComplex y[8], h[8];

    /* Validate estimate indexing bounds */
    if (re_in_symbol < 0 || re_in_symbol >= est_stride || symbol_rel < 0) {
        /* Out of bounds - use identity channel to prevent crash */
        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            h[p] = make_cuFloatComplex(1.0f, 0.0f);
            y[p] = make_cuFloatComplex(0.0f, 0.0f);
        }
    } else {
        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            int grid_idx = p * grid_stride + src_re;
            y[p] = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

            /* Load estimate from GPU-computed buffer */
            /* Estimate layout: [symbol_rel, port, re_in_symbol] */
            int est_idx = symbol_rel * (NOF_PORTS * est_stride) + p * est_stride + re_in_symbol;
            h[p] = d_estimates[est_idx];
        }
    }

    /* ---- Equalization (ZF, MMSE, or MMSE-IRC) ---- */
    cuFloatComplex eq;
    float eq_noise_var;
    equalize_symbol<NOF_PORTS, ALGORITHM>(y, h, s_noise_vars, tx_scaling, eq, eq_noise_var);

    /* Accumulate post-equalization noise variance for SINR calculation. */
    sinr_accumulate(d_eq_noise_var_sum, d_eq_noise_var_count, eq_noise_var);

    float inv_noise = 1.0f / eq_noise_var;

    /* ---- Soft Demodulation ---- */
    float llr_vals[8];

    if (mod_order == 2) {
        float scale = 2.0f * 1.41421356f * inv_noise;
        llr_vals[0] = eq.x * scale;
        llr_vals[1] = eq.y * scale;
    } else if (mod_order == 4) {
        /* 16QAM soft demodulation - piecewise linear to match CPU implementation.
         * M_SQRT1_10 = 1/sqrt(10) = 0.31622776
         * GAIN_FIRST = 4 * M_SQRT1_10 = 1.2649
         * THRESHOLD = 2 * M_SQRT1_10 = 0.6324555
         */
        const float M_SQRT1_10 = 0.31622776601683794f;
        const float GAIN_FIRST = 4.0f * M_SQRT1_10;
        const float THRESHOLD = 2.0f * M_SQRT1_10;
        const float CONST_0_8 = 0.8f;

        float abs_re = fabsf(eq.x);
        float abs_im = fabsf(eq.y);

        /* Bits 0,1: piecewise linear with threshold at 2*M_SQRT1_10 */
        float l_first_re = GAIN_FIRST * eq.x;
        float l_first_im = GAIN_FIRST * eq.y;

        /* Second interval: 2*first - copysign(0.8, symbol) */
        float l_second_re = 2.0f * l_first_re - copysignf(CONST_0_8, eq.x);
        float l_second_im = 2.0f * l_first_im - copysignf(CONST_0_8, eq.y);

        /* Select based on threshold */
        float l_01_re = (abs_re > THRESHOLD) ? l_second_re : l_first_re;
        float l_01_im = (abs_im > THRESHOLD) ? l_second_im : l_first_im;

        /* Bits 2,3: 0.8 - |first| */
        float l_23_re = CONST_0_8 - fabsf(l_first_re);
        float l_23_im = CONST_0_8 - fabsf(l_first_im);

        /* Scale by 1/noise_var */
        llr_vals[0] = l_01_re * inv_noise;
        llr_vals[1] = l_01_im * inv_noise;
        llr_vals[2] = l_23_re * inv_noise;
        llr_vals[3] = l_23_im * inv_noise;

    } else if (mod_order == 6) {
        // 64QAM: piecewise-linear approximation matching CPU implementation
        soft_demod_64qam_piecewise(eq, inv_noise, llr_vals);
    } else if (mod_order == 8) {
        // 256QAM: piecewise-linear approximation matching CPU implementation
        soft_demod_256qam_piecewise(eq, inv_noise, llr_vals);
    }

    /* ---- Descrambling ---- */
    int llr_base = re_idx * mod_order;
    int word_idx = llr_base / 32;
    int bit_offset = llr_base % 32;
    unsigned int scr_word = scramble_seq[word_idx];
    unsigned int scr_word_next = scramble_seq[word_idx + 1];

    for (int b = 0; b < mod_order; b++) {
        int bit_pos = bit_offset + b;
        unsigned int scr_bit;
        if (bit_pos < 32) {
            scr_bit = (scr_word >> bit_pos) & 1;
        } else {
            scr_bit = (scr_word_next >> (bit_pos - 32)) & 1;
        }

        float llr = scr_bit ? -llr_vals[b] : llr_vals[b];
        llr = fmaxf(-65504.0f, fminf(65504.0f, llr));
        llrs_half[llr_base + b] = __float2half(llr);
    }
}

/* ============================================================================
 * API Implementation
 * ============================================================================ */

nr_ldpc_status_t pusch_e2e_create(pusch_e2e_handle_t* handle)
{
    if (!handle) return NR_LDPC_ERROR_INVALID_CONFIG;

    /* Initialize LFSR jump tables for this module (once per process) */
    cudaError_t jump_err = initialize_pusch_e2e_jump_tables();
    if (jump_err != cudaSuccess) {
        fprintf(stderr, "[OCUDU PHY CUDA] Failed to initialize PUSCH E2E LFSR jump tables: %s\n",
                cudaGetErrorString(jump_err));
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    pusch_e2e_ctx* ctx = new (std::nothrow) pusch_e2e_ctx();
    if (!ctx) return NR_LDPC_ERROR_ALLOC_FAILED;

    ctx->configured = false;
    ctx->d_dmrs_pilots = nullptr;
    ctx->dmrs_pilots_capacity = 0;
    ctx->d_ch_estimates = nullptr;
    ctx->ch_estimates_capacity = 0;
    ctx->d_dmrs_estimates = nullptr;
    ctx->dmrs_estimates_capacity = 0;
    ctx->nof_estimates = 0;
    ctx->d_noise_vars = nullptr;
    ctx->noise_vars_capacity = 0;
    ctx->d_noise_var_partial = nullptr;
    ctx->noise_var_partial_capacity = 0;
    ctx->d_eq_noise_var_sum = nullptr;
    ctx->d_eq_noise_var_count = nullptr;
    ctx->d_evm_error_sum = nullptr;
    ctx->d_evm_symbol_count = nullptr;
    ctx->h_sinr_eq_block = nullptr;
    ctx->h_evm_error_sum = nullptr;
    ctx->h_evm_symbol_count = nullptr;
    ctx->d_rsrp_accum = nullptr;
    ctx->d_epre_accum = nullptr;
    ctx->d_sinr_noise_accum = nullptr;
    ctx->d_sinr_rsrp_accum = nullptr;
    ctx->d_dmrs_c_inits = nullptr;
    ctx->d_sinr_epre_result = nullptr;
    ctx->h_sinr_epre_result = nullptr;
    ctx->d_estimates_fp16 = nullptr;
    ctx->estimates_fp16_capacity = 0;

    /* Pre-allocated buffers for per-slot reuse (avoid malloc in hot path!) */
    ctx->d_dmrs_indices = nullptr;
    ctx->dmrs_indices_capacity = 0;
    ctx->d_freq_interp_estimates = nullptr;
    ctx->freq_interp_capacity = 0;
    ctx->d_lse_averaged = nullptr;
    ctx->lse_averaged_capacity = 0;
    ctx->d_llrs_half_temp = nullptr;
    ctx->llrs_half_temp_capacity = 0;
    ctx->d_dmrs_received = nullptr;
    ctx->dmrs_received_capacity = 0;
    ctx->d_noise_count = nullptr;
    ctx->d_scrambling_seq = nullptr;
    ctx->scrambling_seq_capacity = 0;
    ctx->cached_data_scrambling_c_init = UINT32_MAX;
    ctx->cached_data_scrambling_bits = 0;
    ctx->d_dmrs_pilot_bits = nullptr;
    ctx->dmrs_pilot_bits_capacity = 0;
    ctx->d_dmrs_mask = nullptr;
    ctx->dmrs_mask_capacity = 0;

    /* Low-PAPR DMRS pilots for transform precoding */
    ctx->d_low_papr_pilots = nullptr;
    ctx->low_papr_pilots_capacity = 0;
    ctx->cached_low_papr_nof_prb = -1;
    ctx->cached_low_papr_n_rs_id = -1;
    ctx->d_deprecode_symbols = nullptr;
    ctx->d_deprecode_noise_vars = nullptr;
    ctx->deprecode_symbols_capacity = 0;
    ctx->deprecode_noise_capacity = 0;
#ifdef OCUDU_PHY_CUDA_ENABLE_VKFFT
    ctx->transform_deprecoder_backend = PUSCH_DEPRECODER_AUTO;
#else
    ctx->transform_deprecoder_backend = PUSCH_DEPRECODER_CUSTOM;
#endif
    ctx->vkfft_auto_min_dft_size = 864;  /* 72 PRB measured crossover on DGX Spark. */
    const char* deprecoder_env = std::getenv("OCUDU_PHY_CUDA_PUSCH_TRANSFORM_DEPRECODER");
    if (deprecoder_env) {
        if (std::strcmp(deprecoder_env, "vkfft") == 0) {
            ctx->transform_deprecoder_backend = PUSCH_DEPRECODER_VKFFT;
        } else if (std::strcmp(deprecoder_env, "auto") == 0) {
            ctx->transform_deprecoder_backend = PUSCH_DEPRECODER_AUTO;
        }
    }
    const char* vkfft_min_env = std::getenv("OCUDU_PHY_CUDA_PUSCH_VKFFT_MIN_DFT_SIZE");
    if (vkfft_min_env) {
        int threshold = std::atoi(vkfft_min_env);
        if (threshold > 0 && threshold <= PUSCH_DEPRECODE_MAX_DFT_SIZE) {
            ctx->vkfft_auto_min_dft_size = threshold;
        }
    }
#ifdef OCUDU_PHY_CUDA_ENABLE_VKFFT
    std::memset(ctx->vkfft_plans, 0, sizeof(ctx->vkfft_plans));
    ctx->vkfft_buffer_ptr = nullptr;
    ctx->vkfft_device = 0;
    ctx->vkfft_stream = nullptr;
    ctx->vkfft_preplanned = false;
#endif
    memset(ctx->deprecoding_fft_factors, 0, sizeof(ctx->deprecoding_fft_factors));
    memset(ctx->deprecoding_fft_nof_factors, 0, sizeof(ctx->deprecoding_fft_nof_factors));
    for (int n = 12; n <= PUSCH_DEPRECODE_MAX_DFT_SIZE; n += 12) {
        int nof_factors = 0;
        uint64_t packed = pack_deprecoding_fft_factors(n, &nof_factors);
        if (nof_factors > 0) {
            ctx->deprecoding_fft_factors[n] = packed;
            ctx->deprecoding_fft_nof_factors[n] = nof_factors;
        }
    }

    /* OFDM symbol start times for CFO compensation */
    ctx->d_symbol_start_times = nullptr;
    ctx->symbol_times_scs_khz = 0;
    ctx->d_mimo_cfo_phasors = nullptr;
    memset(ctx->h_symbol_start_times, 0, sizeof(ctx->h_symbol_start_times));

    /* Precomputed DMRS pilots for all 14 symbols × all slots */
    ctx->d_precomputed_dmrs_pilots = nullptr;
    ctx->precomputed_dmrs_pilots_capacity = 0;
    ctx->precomputed_nof_slots = 0;
    ctx->precomputed_pilots_per_symbol = 0;
    ctx->precomputed_words_per_slot = 0;
    ctx->precomputed_words_per_sym = 0;
    ctx->dmrs_pilots_precomputed = false;
    ctx->cached_dmrs_symbol_mask = -1;  /* Force first upload */
    ctx->cached_nof_dmrs_symbols = 0;
    ctx->cached_dmrs_scrambling_id = UINT32_MAX;  /* Force first DMRS precompute */
    ctx->cached_dmrs_n_scid = -1;
    ctx->cached_dmrs_nof_prb = -1;
    ctx->cached_dmrs_type = -1;

    /* Allocate consolidated per-iteration accumulator block (single memset zeros all).
     * Layout: noise_count(4) | rsrp(32) | epre(32) | sinr_noise(32) | sinr_rsrp(32) |
     *         eq_sum(4) | eq_count(4) | evm_sum(4) | evm_count(4) |
     *         ta_cfo_accum(24) | cv_noise_accum(32) | cv_done_counter(4). */
    {
        size_t off = 0;
        /* noise_count: unsigned int */
        off += sizeof(unsigned int);                        /* 4 */
        size_t off_rsrp = off;
        off += 8 * sizeof(float);                           /* +32 */
        size_t off_epre = off;
        off += 8 * sizeof(float);                           /* +32 */
        size_t off_sinr_noise = off;
        off += 8 * sizeof(float);                           /* +32 */
        size_t off_sinr_rsrp = off;
        off += 8 * sizeof(float);                           /* +32 */
        size_t off_eq_sum = off;
        off += sizeof(float);                               /* +4 */
        size_t off_eq_count = off;
        off += sizeof(unsigned int);                        /* +4 = 140 */
        size_t off_evm_sum = off;
        off += sizeof(float);                               /* +4 = 144 */
        size_t off_evm_count = off;
        off += sizeof(unsigned int);                        /* +4 = 148 */
        size_t off_ta_cfo = off;
        off += 6 * sizeof(float);                           /* +24 = 164 (ta_r, ta_i, cfo_r, cfo_i, rsrp, epre) */
        size_t off_cv_noise = off;
        off += 8 * sizeof(float);                           /* +32 = 196 (per-port CV noise accum) */
        size_t off_cv_done = off;
        off += sizeof(unsigned int);                        /* +4 = 200 (CV last-block done counter) */

        ctx->accum_block_size = off;
        cudaError_t aerr = cudaMalloc(&ctx->d_accum_block, off);
        if (aerr != cudaSuccess) {
            delete ctx;
            return NR_LDPC_ERROR_ALLOC_FAILED;
        }
        cudaMemset(ctx->d_accum_block, 0, off);

        /* Set pointer aliases into the contiguous block */
        char* base = (char*)ctx->d_accum_block;
        ctx->d_noise_count      = (unsigned int*)(base);
        ctx->d_rsrp_accum       = (float*)(base + off_rsrp);
        ctx->d_epre_accum       = (float*)(base + off_epre);
        ctx->d_sinr_noise_accum = (float*)(base + off_sinr_noise);
        ctx->d_sinr_rsrp_accum  = (float*)(base + off_sinr_rsrp);
        ctx->d_eq_noise_var_sum = (float*)(base + off_eq_sum);
        ctx->d_eq_noise_var_count = (unsigned int*)(base + off_eq_count);
        ctx->d_evm_error_sum = (float*)(base + off_evm_sum);
        ctx->d_evm_symbol_count = (unsigned int*)(base + off_evm_count);
        ctx->d_ta_cfo_accum     = (float*)(base + off_ta_cfo);
        ctx->d_cv_noise_accum   = (float*)(base + off_cv_noise);
        ctx->d_cv_done_counter  = (unsigned int*)(base + off_cv_done);
    }

    ctx->deprecode_symbols_capacity = PUSCH_DEPRECODE_MAX_SYMBOLS * PUSCH_DEPRECODE_MAX_DFT_SIZE;
    cudaError_t derr = cudaMalloc(&ctx->d_deprecode_symbols,
                                  ctx->deprecode_symbols_capacity * sizeof(cuFloatComplex));
    if (derr != cudaSuccess) {
        cudaFree(ctx->d_accum_block);
        delete ctx;
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }
    ctx->deprecode_noise_capacity = PUSCH_DEPRECODE_MAX_SYMBOLS;
    derr = cudaMalloc(&ctx->d_deprecode_noise_vars, ctx->deprecode_noise_capacity * sizeof(float));
    if (derr != cudaSuccess) {
        cudaFree(ctx->d_deprecode_symbols);
        cudaFree(ctx->d_accum_block);
        delete ctx;
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }
#ifdef OCUDU_PHY_CUDA_ENABLE_VKFFT
    ctx->vkfft_buffer_ptr = ctx->d_deprecode_symbols;
#endif

    /* Allocate pinned host memory for async SINR/EVM readback (single contiguous alloc) */
    const size_t sinr_evm_host_bytes = 2 * sizeof(float) + 2 * sizeof(unsigned int);
    cudaError_t err = cudaMallocHost(&ctx->h_sinr_eq_block, sinr_evm_host_bytes);
    if (err == cudaSuccess) {
        char* host_base = (char*)ctx->h_sinr_eq_block;
        ctx->h_sinr_noise_var_sum = (float*)host_base;
        ctx->h_sinr_count = (unsigned int*)(host_base + sizeof(float));
        ctx->h_evm_error_sum = (float*)(host_base + sizeof(float) + sizeof(unsigned int));
        ctx->h_evm_symbol_count = (unsigned int*)(host_base + 2 * sizeof(float) + sizeof(unsigned int));
    } else {
        ctx->h_sinr_eq_block = nullptr;
        ctx->h_sinr_noise_var_sum = nullptr;
        ctx->h_sinr_count = nullptr;
        ctx->h_evm_error_sum = nullptr;
        ctx->h_evm_symbol_count = nullptr;
    }
    /* Initialize host SINR/EVM buffers to safe defaults (avoid stale reads) */
    if (ctx->h_sinr_eq_block) {
        memset(ctx->h_sinr_eq_block, 0, sinr_evm_host_bytes);
    }

    err = cudaMalloc(&ctx->d_sinr_epre_result, 5 * sizeof(float));
    if (err != cudaSuccess) { ctx->d_sinr_epre_result = nullptr; }
    err = cudaMallocHost(&ctx->h_sinr_epre_result, 5 * sizeof(float));
    if (err != cudaSuccess) { ctx->h_sinr_epre_result = nullptr; }
    if (ctx->h_sinr_epre_result) {
        for (int i = 0; i < 5; i++) ctx->h_sinr_epre_result[i] = -INFINITY;
    }

    /* Allocate OFDM symbol start times for CFO compensation */
    err = cudaMalloc(&ctx->d_symbol_start_times, 14 * sizeof(float));
    if (err != cudaSuccess) { ctx->d_symbol_start_times = nullptr; }
    err = cudaMalloc(&ctx->d_mimo_cfo_phasors, MIMO_CFO_PHASOR_COUNT * sizeof(float2));
    if (err != cudaSuccess) { ctx->d_mimo_cfo_phasors = nullptr; }

    /* External eq-output (disabled by default) */
    ctx->d_eq_symbols_out = nullptr;
    ctx->d_eq_noise_var_out = nullptr;

    /* Create scramblers */
    nr_ldpc_status_t status = scrambler_create(&ctx->data_scrambler);
    if (status != NR_LDPC_SUCCESS) {
        delete ctx;
        return status;
    }

    status = scrambler_create(&ctx->dmrs_scrambler);
    if (status != NR_LDPC_SUCCESS) {
        scrambler_destroy(ctx->data_scrambler);
        delete ctx;
        return status;
    }

    *handle = ctx;
    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t pusch_e2e_preplan_transform_deprecoder(pusch_e2e_handle_t handle,
                                                         cudaStream_t stream)
{
    if (!handle) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

#ifdef OCUDU_PHY_CUDA_ENABLE_VKFFT
    if ((handle->transform_deprecoder_backend != PUSCH_DEPRECODER_VKFFT) &&
        (handle->transform_deprecoder_backend != PUSCH_DEPRECODER_AUTO)) {
        return NR_LDPC_SUCCESS;
    }
    if (handle->vkfft_preplanned && handle->vkfft_stream == stream) {
        return NR_LDPC_SUCCESS;
    }

    int cuda_device = 0;
    cudaError_t cerr = cudaGetDevice(&cuda_device);
    if (cerr != cudaSuccess) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }
    CUresult cu_res = cuDeviceGet(&handle->vkfft_device, cuda_device);
    if (cu_res != CUDA_SUCCESS) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    handle->vkfft_stream = stream;
    for (int n = 12; n <= PUSCH_DEPRECODE_MAX_DFT_SIZE; n += 12) {
        if (handle->deprecoding_fft_nof_factors[n] <= 0) {
            continue;
        }
        if ((handle->transform_deprecoder_backend == PUSCH_DEPRECODER_AUTO) &&
            (n < handle->vkfft_auto_min_dft_size)) {
            continue;
        }

        auto* plan = new (std::nothrow) pusch_vkfft_deprecoder_plan();
        if (!plan) {
            return NR_LDPC_ERROR_ALLOC_FAILED;
        }
        std::memset(plan, 0, sizeof(*plan));
        plan->dft_size = n;
        plan->buffer_size = static_cast<pfUINT>(PUSCH_DEPRECODE_MAX_SYMBOLS * n * sizeof(cuFloatComplex));

        VkFFTConfiguration configuration = {};
        configuration.FFTdim = 1;
        configuration.size[0] = static_cast<pfUINT>(n);
        configuration.device = &handle->vkfft_device;
        configuration.stream = &handle->vkfft_stream;
        configuration.num_streams = 1;
        configuration.numberBatches = PUSCH_DEPRECODE_MAX_SYMBOLS;
        configuration.bufferSize = &plan->buffer_size;
        configuration.buffer = &handle->vkfft_buffer_ptr;
        configuration.makeInversePlanOnly = 1;
        configuration.normalize = 0;
        configuration.disableReorderFourStep = 1;
        configuration.useLUT = 1;
        configuration.aimThreads = 128;

        VkFFTResult result = initializeVkFFT(&plan->app, configuration);
        if (result != VKFFT_SUCCESS) {
            delete plan;
            if (handle->transform_deprecoder_backend == PUSCH_DEPRECODER_VKFFT) {
                return NR_LDPC_ERROR_CUDA_FAILED;
            }
            continue;
        }
        plan->initialized = true;
        handle->vkfft_plans[n] = plan;
    }

    handle->vkfft_preplanned = true;
    return NR_LDPC_SUCCESS;
#else
    (void)stream;
    if (handle->transform_deprecoder_backend == PUSCH_DEPRECODER_VKFFT) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    return NR_LDPC_SUCCESS;
#endif
}

void pusch_e2e_destroy(pusch_e2e_handle_t handle)
{
    if (handle) {
#ifdef OCUDU_PHY_CUDA_ENABLE_VKFFT
        for (int n = 0; n <= PUSCH_DEPRECODE_MAX_DFT_SIZE; ++n) {
            if (handle->vkfft_plans[n]) {
                deleteVkFFT(&handle->vkfft_plans[n]->app);
                delete handle->vkfft_plans[n];
            }
        }
#endif
        if (handle->d_dmrs_pilots) cudaFree(handle->d_dmrs_pilots);
        if (handle->d_ch_estimates) cudaFree(handle->d_ch_estimates);
        if (handle->d_dmrs_estimates) cudaFree(handle->d_dmrs_estimates);
        if (handle->d_noise_vars) cudaFree(handle->d_noise_vars);
        if (handle->d_noise_var_partial) cudaFree(handle->d_noise_var_partial);
        /* Free consolidated accumulator block (contains d_noise_count, d_rsrp_accum,
         * d_epre_accum, d_sinr_noise_accum, d_sinr_rsrp_accum, d_eq_noise_var_sum,
         * d_eq_noise_var_count as aliases — do NOT free them individually) */
        if (handle->d_accum_block) cudaFree(handle->d_accum_block);
        if (handle->h_sinr_eq_block) cudaFreeHost(handle->h_sinr_eq_block);
        if (handle->d_dmrs_c_inits) cudaFree(handle->d_dmrs_c_inits);
        if (handle->d_sinr_epre_result) cudaFree(handle->d_sinr_epre_result);
        if (handle->h_sinr_epre_result) cudaFreeHost(handle->h_sinr_epre_result);
        /* Free pre-allocated per-slot buffers */
        if (handle->d_dmrs_indices) cudaFree(handle->d_dmrs_indices);
        if (handle->d_freq_interp_estimates) cudaFree(handle->d_freq_interp_estimates);
        if (handle->d_lse_averaged) cudaFree(handle->d_lse_averaged);
        if (handle->d_llrs_half_temp) cudaFree(handle->d_llrs_half_temp);
        if (handle->d_estimates_fp16) cudaFree(handle->d_estimates_fp16);
        if (handle->d_dmrs_received) cudaFree(handle->d_dmrs_received);
        /* d_noise_count: alias into d_accum_block unless batch path reallocated it */
        if (handle->d_noise_count && handle->d_accum_block &&
            !((char*)handle->d_noise_count >= (char*)handle->d_accum_block &&
              (char*)handle->d_noise_count < (char*)handle->d_accum_block + handle->accum_block_size)) {
            cudaFree(handle->d_noise_count);
        }
        if (handle->d_scrambling_seq) cudaFree(handle->d_scrambling_seq);
        if (handle->d_dmrs_pilot_bits) cudaFree(handle->d_dmrs_pilot_bits);
        if (handle->d_dmrs_mask) cudaFree(handle->d_dmrs_mask);
        if (handle->d_precomputed_dmrs_pilots) cudaFree(handle->d_precomputed_dmrs_pilots);
        if (handle->d_low_papr_pilots) cudaFree(handle->d_low_papr_pilots);
        if (handle->d_deprecode_symbols) cudaFree(handle->d_deprecode_symbols);
        if (handle->d_deprecode_noise_vars) cudaFree(handle->d_deprecode_noise_vars);
        if (handle->d_symbol_start_times) cudaFree(handle->d_symbol_start_times);
        if (handle->d_mimo_cfo_phasors) cudaFree(handle->d_mimo_cfo_phasors);
        if (handle->data_scrambler) scrambler_destroy(handle->data_scrambler);
        if (handle->dmrs_scrambler) scrambler_destroy(handle->dmrs_scrambler);
        delete handle;
    }
}

void pusch_e2e_set_eq_output(pusch_e2e_handle_t handle,
                              float2* d_eq_symbols_out,
                              float* d_eq_noise_var_out)
{
    if (!handle) return;
    handle->d_eq_symbols_out = d_eq_symbols_out;
    handle->d_eq_noise_var_out = d_eq_noise_var_out;
}

const void* pusch_e2e_get_estimates_fp16(pusch_e2e_handle_t handle)
{
    return handle ? handle->d_estimates_fp16 : nullptr;
}

nr_ldpc_status_t pusch_e2e_configure(pusch_e2e_handle_t handle,
                                      const pusch_e2e_config_t* cfg)
{
    if (!handle || !cfg) return NR_LDPC_ERROR_INVALID_CONFIG;

    /* Validate nof_tx_layers: E2E GPU kernels currently only support single-layer.
     * For 2/4-layer MIMO, the caller should use the batch GPU path (CPU equalization
     * + GPU soft demod), which leverages the multi-layer kernels in equalization.cu. */
    int nof_tx_layers = cfg->nof_tx_layers;
    if (nof_tx_layers == 0) nof_tx_layers = 1;  /* Default to 1 if not set */
    if (nof_tx_layers != 1 && nof_tx_layers != 2 && nof_tx_layers != 3 && nof_tx_layers != 4) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (cfg->dmrs_type != DMRS_TYPE_1 && cfg->dmrs_type != DMRS_TYPE_2) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (pusch_e2e_get_dmrs_symbol_indices_checked(cfg->dmrs_symbol_mask, nullptr) <= 0) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    /* Multi-layer requires appropriate port configuration */
    if (nof_tx_layers == 2 && cfg->nof_rx_ports < 2) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (nof_tx_layers == 3 && cfg->nof_rx_ports < 3) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (nof_tx_layers == 4 && cfg->nof_rx_ports < 4) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    handle->config = *cfg;
    handle->config.nof_tx_layers = nof_tx_layers;  /* Store validated value */
    handle->configured = true;

    /* Configure data scrambler
     * Per TS 38.211 Section 6.3.1.1 (PUSCH) and 7.3.1.1 (PDSCH):
     * c_init = n_RNTI × 2^15 + q × 2^14 + n_ID
     * Slot number is NOT used for data scrambling (only for DMRS). */
    nr_scrambling_config_t scr_cfg = {};
    scr_cfg.n_RNTI = cfg->rnti;
    scr_cfg.n_ID = cfg->n_id;
    scr_cfg.q = 0;
    scr_cfg.n_s = 0;  // Not used for data scrambling per spec

    nr_ldpc_status_t status = scrambler_configure(handle->data_scrambler, &scr_cfg);
    if (status != NR_LDPC_SUCCESS) return status;

    /* ================================================================
     * PRE-COMPUTE SCRAMBLING SEQUENCE AT CONFIGURE TIME
     *
     * Since c_init = (rnti << 15) + n_id is fixed per configuration,
     * we can pre-generate the entire scrambling sequence once here.
     * This eliminates ~1229 O(log N) LFSR jumps per slot at runtime!
     *
     * Expected savings: ~280 µs per slot at 100 MHz 64QAM
     * ================================================================ */
    {
        /* Calculate max data REs for this configuration. Keep this tied to the
         * actual DMRS mask so single-DMRS MIMO allocations do not under-size the
         * cached scrambling sequence. */
        int nof_dmrs_symbols_for_capacity =
            pusch_e2e_get_dmrs_symbol_indices_checked(cfg->dmrs_symbol_mask, nullptr);
        int dmrs_re_per_prb = dmrs_get_re_per_prb(cfg->dmrs_type) * cfg->nof_cdm_groups_without_data;
        if (dmrs_re_per_prb > 12) return NR_LDPC_ERROR_INVALID_CONFIG;
        int data_re_per_prb =
            cfg->nof_symbols * 12 - nof_dmrs_symbols_for_capacity * dmrs_re_per_prb;
        if (data_re_per_prb <= 0) return NR_LDPC_ERROR_INVALID_CONFIG;
        int max_data_re = cfg->nof_prb * data_re_per_prb;
        int nof_scramble_bits = max_data_re * nof_tx_layers * cfg->mod_order;
        int nof_scramble_words = (nof_scramble_bits + 31) / 32;
        size_t scramble_seq_size = nof_scramble_words * sizeof(uint32_t);

        /* Allocate if needed */
        if (scramble_seq_size > handle->scrambling_seq_capacity) {
            if (handle->d_scrambling_seq) cudaFree(handle->d_scrambling_seq);
            handle->d_scrambling_seq = nullptr;
            handle->scrambling_seq_capacity = 0;
            handle->cached_data_scrambling_c_init = UINT32_MAX;
            handle->cached_data_scrambling_bits = 0;
            cudaError_t err = cudaMalloc(&handle->d_scrambling_seq, scramble_seq_size);
            if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
            handle->scrambling_seq_capacity = scramble_seq_size;
        }

        /* Pre-generate scrambling sequence when the cached prefix is not sufficient. */
        uint32_t data_c_init = (cfg->rnti << 15) + cfg->n_id;
        if (handle->cached_data_scrambling_c_init != data_c_init ||
            handle->cached_data_scrambling_bits < nof_scramble_bits) {
            int scr_gen_threads = 256;
            int scr_gen_blocks = (nof_scramble_words + scr_gen_threads - 1) / scr_gen_threads;

            cudaStream_t precompute_stream = nullptr;
            bool owns_precompute_stream =
                (cudaStreamCreateWithFlags(&precompute_stream, cudaStreamNonBlocking) == cudaSuccess);
            if (!owns_precompute_stream) precompute_stream = nullptr;

            kernel_generate_scrambling_sequence<<<scr_gen_blocks, scr_gen_threads, 0, precompute_stream>>>(
                handle->d_scrambling_seq, data_c_init, nof_scramble_bits);
            nr_ldpc_status_t sync_status =
                sync_pusch_precompute_stream(precompute_stream, "scrambling-sequence precompute");
            if (owns_precompute_stream) cudaStreamDestroy(precompute_stream);
            if (sync_status != NR_LDPC_SUCCESS) return sync_status;

            handle->cached_data_scrambling_c_init = data_c_init;
            handle->cached_data_scrambling_bits = nof_scramble_bits;
        }
    }

    /* ================================================================
     * PRE-COMPUTE DMRS PILOTS FOR ALL 14 OFDM SYMBOLS × ALL SLOTS
     *
     * Since scrambling_id and n_scid are fixed per configuration,
     * we pre-generate pilots for ALL 14 symbol positions per slot.
     * This means ANY dmrs_symbol_mask works at runtime without
     * mask matching — the runtime path just gathers the needed
     * symbols via D2D copies.
     *
     * DMRS c_init depends ONLY on scrambling_id, n_scid, slot, and
     * symbol index (TS 38.211 6.4.1.1.1.1).  Skip the expensive
     * 280-kernel precompute when only RNTI/n_id/mod_order changed.
     *
     * Layout: [slot_idx * 14 * words_per_sym + ofdm_symbol * words_per_sym]
     * Memory: ~115 KB for 273 PRB (trivial).
     * ================================================================ */
    if (cfg->use_low_papr_dmrs != 0) {
        /* Transform-precoded PUSCH (Msg3 / DFT-s-OFDM) uses low-PAPR DMRS
         * pilots, not the pseudo-random Gold DMRS table below.  Avoid the
         * 20 slots x 14 symbols precompute in the timing-critical Msg3 path. */
        int M_zc = cfg->nof_prb * 6;
        if (M_zc <= 0) return NR_LDPC_ERROR_INVALID_CONFIG;

        size_t low_papr_size = (size_t)M_zc * sizeof(cuFloatComplex);
        if (low_papr_size > handle->low_papr_pilots_capacity) {
            if (handle->d_low_papr_pilots) cudaFree(handle->d_low_papr_pilots);
            cudaError_t err = cudaMalloc(&handle->d_low_papr_pilots, low_papr_size);
            if (err != cudaSuccess) {
                handle->d_low_papr_pilots = nullptr;
                handle->low_papr_pilots_capacity = 0;
                handle->cached_low_papr_nof_prb = -1;
                handle->cached_low_papr_n_rs_id = -1;
                return NR_LDPC_ERROR_ALLOC_FAILED;
            }
            handle->low_papr_pilots_capacity = low_papr_size;
            handle->cached_low_papr_nof_prb = -1;
            handle->cached_low_papr_n_rs_id = -1;
        }

        if (handle->cached_low_papr_nof_prb != cfg->nof_prb ||
            handle->cached_low_papr_n_rs_id != cfg->n_rs_id) {
            cudaStream_t precompute_stream = nullptr;
            bool owns_precompute_stream =
                (cudaStreamCreateWithFlags(&precompute_stream, cudaStreamNonBlocking) == cudaSuccess);
            if (!owns_precompute_stream) precompute_stream = nullptr;

            int u = cfg->n_rs_id % NOF_LOW_PAPR_GROUPS;
            if (u < 0) u += NOF_LOW_PAPR_GROUPS;
            int pilot_threads = 256;
            int pilot_blocks = (M_zc + pilot_threads - 1) / pilot_threads;
            kernel_generate_low_papr_pilots<<<pilot_blocks, pilot_threads, 0, precompute_stream>>>(
                handle->d_low_papr_pilots, u, M_zc);

            nr_ldpc_status_t sync_status =
                sync_pusch_precompute_stream(precompute_stream, "low-PAPR DMRS-pilot precompute");
            if (owns_precompute_stream) cudaStreamDestroy(precompute_stream);
            if (sync_status != NR_LDPC_SUCCESS) return sync_status;

            handle->cached_low_papr_nof_prb = cfg->nof_prb;
            handle->cached_low_papr_n_rs_id = cfg->n_rs_id;
        }
    } else if (cfg->scrambling_id != handle->cached_dmrs_scrambling_id ||
        cfg->n_scid        != handle->cached_dmrs_n_scid ||
        cfg->nof_prb       != handle->cached_dmrs_nof_prb ||
        (int)cfg->dmrs_type != handle->cached_dmrs_type)
    {
        int pilots_per_symbol = cfg->nof_prb * dmrs_get_re_per_prb(cfg->dmrs_type);
        int bits_per_sym = pilots_per_symbol * 2;   /* 2 bits per pilot (QPSK) */
        int words_per_sym = (bits_per_sym + 31) / 32;
        int words_per_slot = words_per_sym * 14;    /* all 14 OFDM symbols */
        int total_words = words_per_slot * MAX_SLOTS_PER_FRAME;
        size_t total_size = total_words * sizeof(uint32_t);

        /* Allocate if needed */
        if (total_size > handle->precomputed_dmrs_pilots_capacity) {
            if (handle->d_precomputed_dmrs_pilots) cudaFree(handle->d_precomputed_dmrs_pilots);
            cudaError_t err = cudaMalloc(&handle->d_precomputed_dmrs_pilots, total_size);
            if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
            handle->precomputed_dmrs_pilots_capacity = total_size;
        }

        /* Pre-generate DMRS pilots for all 14 symbols × all slots */
        int pilot_gen_threads = 256;
        int nof_threads_needed = (pilots_per_symbol + 31) / 32;
        int pilot_gen_blocks = (nof_threads_needed + pilot_gen_threads - 1) / pilot_gen_threads;

        {
            int total_pairs = MAX_SLOTS_PER_FRAME * 14;  /* 20 slots × 14 symbols = 280 */
            dim3 grid(pilot_gen_blocks, total_pairs);
            dim3 block(pilot_gen_threads);
            cudaStream_t precompute_stream = nullptr;
            bool owns_precompute_stream =
                (cudaStreamCreateWithFlags(&precompute_stream, cudaStreamNonBlocking) == cudaSuccess);
            if (!owns_precompute_stream) precompute_stream = nullptr;

            kernel_generate_dmrs_pilots_batched<<<grid, block, 0, precompute_stream>>>(
                handle->d_precomputed_dmrs_pilots,
                pilots_per_symbol, words_per_sym, words_per_slot,
                MAX_SLOTS_PER_FRAME, cfg->scrambling_id, cfg->n_scid);
            nr_ldpc_status_t sync_status = sync_pusch_precompute_stream(precompute_stream, "DMRS-pilot precompute");
            if (owns_precompute_stream) cudaStreamDestroy(precompute_stream);
            if (sync_status != NR_LDPC_SUCCESS) return sync_status;
        }

        /* Ensure d_dmrs_pilot_bits is large enough for D2D gather at runtime
         * (MAX_DMRS_SYMBOLS contiguous symbols for CV kernel) */
        size_t gather_size = (size_t)MAX_DMRS_SYMBOLS * words_per_sym * sizeof(uint32_t);
        if (gather_size > handle->dmrs_pilot_bits_capacity) {
            if (handle->d_dmrs_pilot_bits) cudaFree(handle->d_dmrs_pilot_bits);
            cudaError_t err = cudaMalloc(&handle->d_dmrs_pilot_bits, gather_size);
            if (err == cudaSuccess) {
                handle->dmrs_pilot_bits_capacity = gather_size;
            } else {
                handle->d_dmrs_pilot_bits = nullptr;
                handle->dmrs_pilot_bits_capacity = 0;
            }
        }

        /* Store precomputation metadata */
        handle->precomputed_nof_slots = MAX_SLOTS_PER_FRAME;
        handle->precomputed_pilots_per_symbol = pilots_per_symbol;
        handle->precomputed_words_per_slot = words_per_slot;
        handle->precomputed_words_per_sym = words_per_sym;
        handle->dmrs_pilots_precomputed = true;

        handle->cached_dmrs_scrambling_id = cfg->scrambling_id;
        handle->cached_dmrs_n_scid = cfg->n_scid;
        handle->cached_dmrs_nof_prb = cfg->nof_prb;
        handle->cached_dmrs_type = (int)cfg->dmrs_type;
    }

    /* ================================================================
     * PRECOMPUTE OFDM SYMBOL START TIMES FOR CFO COMPENSATION
     *
     * These are the start-of-FFT-window times (seconds from slot start)
     * for each of the 14 OFDM symbols. Only recompute when SCS changes.
     * Used by the E2E kernel to compute phase rotation dt between DMRS
     * and data symbols: exp(j * 2π * cfo * dt).
     * ================================================================ */
    if (cfg->scs_khz > 0 && cfg->scs_khz != handle->symbol_times_scs_khz &&
        handle->d_symbol_start_times) {
        int mu = 0;
        if (cfg->scs_khz == 30) mu = 1;
        else if (cfg->scs_khz == 60) mu = 2;
        else if (cfg->scs_khz == 120) mu = 3;
        float scs_hz = (float)cfg->scs_khz * 1000.0f;
        int half_slot_sym = 7 * (1 << mu);

        float t = 0.0f;
        for (int j = 0; j < 14; j++) {
            int cp_kappa = 144 >> mu;
            if (j == 0 || j == half_slot_sym) cp_kappa += 16;
            float cp_s = (float)cp_kappa / 30720000.0f;
            handle->h_symbol_start_times[j] = t + cp_s;  /* time at start of FFT window */
            t = handle->h_symbol_start_times[j] + 1.0f / scs_hz;  /* end of FFT window */
        }
        cudaMemcpy(handle->d_symbol_start_times, handle->h_symbol_start_times,
                   14 * sizeof(float), cudaMemcpyHostToDevice);
        handle->symbol_times_scs_khz = cfg->scs_khz;
    }

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t pusch_e2e_update_slot_config(pusch_e2e_handle_t handle,
                                               int nof_prb,
                                               int start_prb,
                                               int slot_idx,
                                               int dmrs_symbol_mask)
{
    if (!handle || !handle->configured) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (pusch_e2e_get_dmrs_symbol_indices_checked(dmrs_symbol_mask, nullptr) <= 0) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    /* Update per-slot fields in the handle's config.
     * These values change per PUSCH transmission but don't require
     * a full reconfiguration (which would trigger GPU synchronization). */
    handle->config.nof_prb = nof_prb;
    handle->config.start_prb = start_prb;
    handle->config.slot_idx = slot_idx;
    handle->config.dmrs_symbol_mask = dmrs_symbol_mask;

    /* NOTE: For loopback tests, we use slot=0 to match the transmitter.
     * The scrambling was already configured correctly at configure time with
     * slot=0, so we don't need to reconfigure it here. In real deployments,
     * this would need to be updated per-slot. */

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t pusch_e2e_process_with_chest(
    pusch_e2e_handle_t handle,
    const void* d_grid_cbf16,
    const float* d_noise_vars,
    void* d_llrs_half,
    const int* d_re_indices,
    int nof_data_re,
    cudaStream_t stream)
{
    if (!handle || !handle->configured) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (!d_grid_cbf16 || !d_noise_vars || !d_llrs_half || !d_re_indices) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    const pusch_e2e_config_t& cfg = handle->config;

    /* Calculate dimensions */
    int nof_dmrs_re_per_symbol = cfg.nof_prb * dmrs_get_re_per_prb(cfg.dmrs_type);
    int nof_re_per_symbol = cfg.nof_prb * 12;
    int grid_stride = cfg.grid_nof_symbols * cfg.grid_nof_subcarriers;
    int symbol_stride = cfg.grid_nof_subcarriers;

    int dmrs_symbol_indices[MAX_DMRS_SYMBOLS];
    int nof_dmrs_symbols = pusch_e2e_get_dmrs_symbol_indices_checked(cfg.dmrs_symbol_mask, dmrs_symbol_indices);
    if (nof_dmrs_symbols <= 0) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    cudaError_t err;
    nr_ldpc_status_t status;

    /* Allocate device memory for DMRS LSE estimates */
    size_t lse_size = nof_dmrs_symbols * nof_dmrs_re_per_symbol * cfg.nof_rx_ports * sizeof(cuFloatComplex);
    if (lse_size > handle->dmrs_estimates_capacity) {
        if (handle->d_dmrs_estimates) cudaFree(handle->d_dmrs_estimates);
        err = cudaMalloc(&handle->d_dmrs_estimates, lse_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->dmrs_estimates_capacity = lse_size;
    }

    /* Allocate device memory for full channel estimates */
    size_t est_size = cfg.nof_symbols * nof_re_per_symbol * cfg.nof_rx_ports * sizeof(cuFloatComplex);
    if (est_size > handle->ch_estimates_capacity) {
        if (handle->d_ch_estimates) cudaFree(handle->d_ch_estimates);
        err = cudaMalloc(&handle->d_ch_estimates, est_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->ch_estimates_capacity = est_size;
    }
    handle->nof_estimates = cfg.nof_symbols * nof_re_per_symbol * cfg.nof_rx_ports;

    /* ================================================================
     * Phase 1: Generate DMRS pilots and compute LSE for each DMRS symbol
     * ================================================================ */

    /* Use pre-allocated buffers (avoid per-slot malloc - HUGE efficiency win!) */
    size_t dmrs_indices_needed = nof_dmrs_symbols * sizeof(int);
    if (dmrs_indices_needed > handle->dmrs_indices_capacity) {
        if (handle->d_dmrs_indices) cudaFree(handle->d_dmrs_indices);
        err = cudaMalloc(&handle->d_dmrs_indices, dmrs_indices_needed);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->dmrs_indices_capacity = dmrs_indices_needed;
    }
    int* d_dmrs_indices = handle->d_dmrs_indices;
    /* CRITICAL FIX: Use synchronous cudaMemcpy - dmrs_symbol_indices is a local
     * stack variable that may be overwritten before async copy completes. */
    cudaMemcpy(d_dmrs_indices, dmrs_symbol_indices, nof_dmrs_symbols * sizeof(int),
               cudaMemcpyHostToDevice);

    size_t freq_interp_size = nof_dmrs_symbols * nof_re_per_symbol * cfg.nof_rx_ports * sizeof(cuFloatComplex);
    if (freq_interp_size > handle->freq_interp_capacity) {
        if (handle->d_freq_interp_estimates) cudaFree(handle->d_freq_interp_estimates);
        err = cudaMalloc(&handle->d_freq_interp_estimates, freq_interp_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->freq_interp_capacity = freq_interp_size;
    }
    cuFloatComplex* d_freq_interp_estimates = handle->d_freq_interp_estimates;

    /* CRITICAL FIX: cfg.slot_idx is absolute slot number, but DMRS c_init uses slot within frame */
    int slot_in_frame_for_dmrs = cfg.slot_idx % MAX_SLOTS_PER_FRAME;

    for (int d = 0; d < nof_dmrs_symbols; d++) {
        int dmrs_sym = dmrs_symbol_indices[d];

        /* Compute c_init for this DMRS symbol - use slot within frame per 3GPP TS 38.211 */
        uint32_t c_init = dmrs_compute_c_init(slot_in_frame_for_dmrs, dmrs_sym, cfg.scrambling_id, cfg.n_scid);

        /* Configure DMRS scrambler with direct c_init value */
        status = scrambler_configure_c_init(handle->dmrs_scrambler, c_init);

        /* Generate sequence bits (2 bits per DMRS RE for QPSK).
         * Must generate enough bits to cover the offset from point A (RB 0).
         * The kernel accesses bits at position (start_prb + prb_idx) * 6 * 2.
         */
        int nof_seq_bits = (cfg.start_prb + cfg.nof_prb) * 6 * 2;
        if (status != NR_LDPC_SUCCESS) {
            return status;  /* Pre-allocated buffers persist */
        }

        status = scrambler_generate_sequence(handle->dmrs_scrambler, nof_seq_bits, stream);
        if (status != NR_LDPC_SUCCESS) {
            return status;  /* Pre-allocated buffers persist */
        }

        const unsigned int* d_dmrs_seq = scrambler_get_sequence_ptr(handle->dmrs_scrambler);

        /* Output buffer for this DMRS symbol's LSE estimates */
        cuFloatComplex* d_lse_out = handle->d_dmrs_estimates +
                                     d * (cfg.nof_rx_ports * nof_dmrs_re_per_symbol);

        /* Launch LSE kernel */
        if (cfg.dmrs_type == DMRS_TYPE_1) {
            int blocks = cfg.nof_prb;
            int threads = cfg.nof_rx_ports;
            kernel_dmrs_lse_type1<<<blocks, threads, 0, stream>>>(
                static_cast<const unsigned int*>(d_grid_cbf16),
                d_dmrs_seq,
                d_lse_out,
                cfg.nof_prb,
                cfg.nof_rx_ports,
                grid_stride,
                symbol_stride,
                cfg.start_prb,
                dmrs_sym,
                cfg.start_symbol,
                cfg.dmrs_scaling);
        }

        /* Frequency interpolation for this DMRS symbol */
        cuFloatComplex* d_freq_out = d_freq_interp_estimates +
                                      d * (cfg.nof_rx_ports * nof_re_per_symbol);

        int total_re = cfg.nof_prb * cfg.nof_rx_ports * 12;
        int threads_fi = 256;
        int blocks_fi = (total_re + threads_fi - 1) / threads_fi;
        kernel_freq_interpolate_type1<<<blocks_fi, threads_fi, 0, stream>>>(
            d_lse_out,
            d_freq_out,
            cfg.nof_prb,
            cfg.nof_rx_ports);
    }

    /* ================================================================
     * Phase 1b: Frequency-domain smoothing
     * ================================================================ */
    if (cfg.nof_prb > 1) {
        dim3 smooth_grid(cfg.nof_prb, nof_dmrs_symbols);
        kernel_fd_smoothing_fp32<<<smooth_grid, cfg.nof_rx_ports, 0, stream>>>(
            d_freq_interp_estimates,
            cfg.nof_prb,
            cfg.nof_rx_ports,
            nof_dmrs_symbols);
    }

    /* ================================================================
     * Phase 2: Time interpolation to all data symbols
     * ================================================================ */

    int total_time = cfg.nof_symbols * cfg.nof_rx_ports * nof_re_per_symbol;
    int threads_ti = 256;
    int blocks_ti = (total_time + threads_ti - 1) / threads_ti;

    /* Scale factor for time interpolation.
     * We keep the full magnitude (1.0) for equalization stability, and compensate
     * for the scaling mismatch in the noise variance calculation instead.
     * This avoids reducing channel estimate SNR by N² which would hurt weak signals. */
    float time_interp_scale = 1.0f;

    kernel_time_interpolate<<<blocks_ti, threads_ti, 0, stream>>>(
        d_freq_interp_estimates,
        handle->d_ch_estimates,
        d_dmrs_indices,
        nof_dmrs_symbols,
        cfg.nof_symbols,
        cfg.nof_rx_ports,
        nof_re_per_symbol,
        cfg.start_symbol,
        time_interp_scale);

    /* ================================================================
     * Phase 3: Generate data scrambling sequence
     * ================================================================ */

    int num_llrs = nof_data_re * cfg.mod_order;
    status = scrambler_generate_sequence(handle->data_scrambler, num_llrs, stream);
    if (status != NR_LDPC_SUCCESS) {
        /* Pre-allocated, persist */
        /* Pre-allocated, persist */
        return status;
    }

    const unsigned int* d_scramble_seq = scrambler_get_sequence_ptr(handle->data_scrambler);

    /* ================================================================
     * Phase 4: Fused equalization + demodulation + descrambling
     * ================================================================ */

    int threads_e2e = 256;
    int blocks_e2e = (nof_data_re + threads_e2e - 1) / threads_e2e;

    /* Reset SINR accumulators before kernel launch */
    cudaMemsetAsync(handle->d_eq_noise_var_sum, 0, 2 * sizeof(float) + 2 * sizeof(unsigned int), stream);

    /* Dispatch kernel based on nof_rx_ports and equalizer_algorithm */
    #define LAUNCH_E2E_KERNEL(PORTS, ALG) \
        kernel_pusch_e2e_with_estimates<PORTS, ALG><<<blocks_e2e, threads_e2e, 0, stream>>>( \
            static_cast<const unsigned int*>(d_grid_cbf16), \
            handle->d_ch_estimates, \
            d_re_indices, \
            d_noise_vars, \
            static_cast<__half*>(d_llrs_half), \
            d_scramble_seq, \
            handle->d_eq_noise_var_sum, \
            handle->d_eq_noise_var_count, \
            nof_data_re, cfg.mod_order, grid_stride, symbol_stride, nof_re_per_symbol, \
            cfg.start_symbol, cfg.start_prb * 12, cfg.tx_scaling)

    #define DISPATCH_ALGORITHM(PORTS) \
        switch (cfg.equalizer_algorithm) { \
            case EQUALIZER_ZF: LAUNCH_E2E_KERNEL(PORTS, EQUALIZER_ZF); break; \
            case EQUALIZER_MMSE: LAUNCH_E2E_KERNEL(PORTS, EQUALIZER_MMSE); break; \
            case EQUALIZER_MMSE_IRC: default: LAUNCH_E2E_KERNEL(PORTS, EQUALIZER_MMSE_IRC); break; \
        }

    switch (cfg.nof_rx_ports) {
        case 1: DISPATCH_ALGORITHM(1); break;
        case 2: DISPATCH_ALGORITHM(2); break;
        case 4: DISPATCH_ALGORITHM(4); break;
        case 8: DISPATCH_ALGORITHM(8); break;
        default:
            /* Pre-allocated, persist */
            /* Pre-allocated, persist */
            return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    #undef LAUNCH_E2E_KERNEL
    #undef DISPATCH_ALGORITHM

    /* Cleanup temporary buffers */
    /* Pre-allocated, persist */
    /* Pre-allocated, persist */

    err = cudaGetLastError();
    return (err == cudaSuccess) ? NR_LDPC_SUCCESS : NR_LDPC_ERROR_CUDA_FAILED;
}

const void* pusch_e2e_get_estimates(pusch_e2e_handle_t handle)
{
    if (!handle) return nullptr;
    return handle->d_ch_estimates;
}

int pusch_e2e_get_nof_estimates(pusch_e2e_handle_t handle)
{
    if (!handle) return 0;
    return handle->nof_estimates;
}

static nr_ldpc_status_t pusch_e2e_process_full_gpu_impl(
    pusch_e2e_handle_t handle,
    const void* d_grid_cbf16,
    void* d_llrs_half,
    const int* d_re_indices,
    int nof_data_re,
    cudaStream_t stream,
    bool run_demod)
{
    if (!handle || !handle->configured) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (!d_grid_cbf16 || !d_re_indices || (run_demod && !d_llrs_half)) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    const pusch_e2e_config_t& cfg = handle->config;

    /* MIMO support: 1, 2, or 4 layers are supported.
     * - 1 layer: MRC combining (existing path)
     * - 2 layers: 2x2 MIMO with ZF/MMSE equalization
     * - 4 layers: 4x4 MIMO with ZF/MMSE equalization
     * Layer count must not exceed port count. */
    if (cfg.nof_tx_layers > cfg.nof_rx_ports) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (cfg.nof_tx_layers != 1 && cfg.nof_tx_layers != 2 && cfg.nof_tx_layers != 3 && cfg.nof_tx_layers != 4) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    /* Calculate dimensions */
    int nof_dmrs_re_per_symbol = cfg.nof_prb * dmrs_get_re_per_prb(cfg.dmrs_type);
    int nof_re_per_symbol = cfg.nof_prb * 12;
    int grid_stride = cfg.grid_nof_symbols * cfg.grid_nof_subcarriers;
    int symbol_stride = cfg.grid_nof_subcarriers;

    int dmrs_symbol_indices[MAX_DMRS_SYMBOLS];
    int nof_dmrs_symbols = pusch_e2e_get_dmrs_symbol_indices_checked(cfg.dmrs_symbol_mask, dmrs_symbol_indices);
    if (nof_dmrs_symbols <= 0) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    cudaError_t err;
    nr_ldpc_status_t status;

    /* Allocate device memory for DMRS LSE estimates */
    size_t lse_size = nof_dmrs_symbols * nof_dmrs_re_per_symbol * cfg.nof_rx_ports * sizeof(cuFloatComplex);
    if (lse_size > handle->dmrs_estimates_capacity) {
        if (handle->d_dmrs_estimates) cudaFree(handle->d_dmrs_estimates);
        err = cudaMalloc(&handle->d_dmrs_estimates, lse_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->dmrs_estimates_capacity = lse_size;
    }

    /* Allocate device memory for full channel estimates */
    size_t est_size = cfg.nof_symbols * nof_re_per_symbol * cfg.nof_rx_ports * sizeof(cuFloatComplex);
    if (est_size > handle->ch_estimates_capacity) {
        if (handle->d_ch_estimates) cudaFree(handle->d_ch_estimates);
        err = cudaMalloc(&handle->d_ch_estimates, est_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->ch_estimates_capacity = est_size;
    }
    handle->nof_estimates = cfg.nof_symbols * nof_re_per_symbol * cfg.nof_rx_ports;

    /* Allocate device memory for GPU-computed noise variances */
    size_t noise_var_size = cfg.nof_rx_ports * sizeof(float);
    if (noise_var_size > handle->noise_vars_capacity) {
        if (handle->d_noise_vars) cudaFree(handle->d_noise_vars);
        err = cudaMalloc(&handle->d_noise_vars, noise_var_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        cudaMemset(handle->d_noise_vars, 0, noise_var_size);  /* Zero once at alloc */
        handle->noise_vars_capacity = noise_var_size;
    }

    /* Allocate device memory for partial noise variance sums */
    size_t partial_size = nof_dmrs_symbols * cfg.nof_rx_ports * sizeof(float);
    if (partial_size > handle->noise_var_partial_capacity) {
        if (handle->d_noise_var_partial) cudaFree(handle->d_noise_var_partial);
        err = cudaMalloc(&handle->d_noise_var_partial, partial_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->noise_var_partial_capacity = partial_size;
    }

    /* Use pre-allocated buffers (avoid per-slot malloc - HUGE efficiency win!) */
    size_t dmrs_indices_needed = nof_dmrs_symbols * sizeof(int);
    if (dmrs_indices_needed > handle->dmrs_indices_capacity) {
        if (handle->d_dmrs_indices) cudaFree(handle->d_dmrs_indices);
        err = cudaMalloc(&handle->d_dmrs_indices, dmrs_indices_needed);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->dmrs_indices_capacity = dmrs_indices_needed;
    }
    int* d_dmrs_indices = handle->d_dmrs_indices;
    /* CRITICAL FIX: Use synchronous cudaMemcpy - dmrs_symbol_indices is a local
     * stack variable that may be overwritten before async copy completes. */
    cudaMemcpy(d_dmrs_indices, dmrs_symbol_indices, nof_dmrs_symbols * sizeof(int),
               cudaMemcpyHostToDevice);

    /* Use pre-allocated buffer for frequency-interpolated DMRS estimates */
    size_t freq_interp_size = nof_dmrs_symbols * nof_re_per_symbol * cfg.nof_rx_ports * sizeof(cuFloatComplex);
    if (freq_interp_size > handle->freq_interp_capacity) {
        if (handle->d_freq_interp_estimates) cudaFree(handle->d_freq_interp_estimates);
        err = cudaMalloc(&handle->d_freq_interp_estimates, freq_interp_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->freq_interp_capacity = freq_interp_size;
    }
    cuFloatComplex* d_freq_interp_estimates = handle->d_freq_interp_estimates;

    /* ================================================================
     * Phase 1a: Generate DMRS and compute LSE for ALL symbols first
     * (Mirroring CPU algorithm: collect all estimates before averaging)
     * ================================================================ */

    /* Store DMRS sequence pointers for reuse in noise variance estimation.
     * Keep this on the stack to avoid allocator jitter in the Msg3 hot path. */
    unsigned int* h_dmrs_seqs[MAX_DMRS_SYMBOLS] = {};

    /* CRITICAL FIX: cfg.slot_idx is absolute slot number, but DMRS c_init uses slot within frame */
    int slot_in_frame_for_dmrs = cfg.slot_idx % MAX_SLOTS_PER_FRAME;

    /* Low-PAPR DMRS path for transform precoding (MSG3/DFT-s-OFDM) */
    bool use_low_papr = (cfg.use_low_papr_dmrs != 0);
    cuFloatComplex* d_low_papr_pilots = nullptr;

    if (use_low_papr) {
        /* Allocate buffer for low-PAPR pilot sequence if needed.
         * For transform precoding, DMRS has 6 REs per PRB (no CDM, all pilots).
         * Sequence length M_zc = nof_prb * 6. */
        int M_zc = cfg.nof_prb * 6;
        size_t low_papr_size = M_zc * sizeof(cuFloatComplex);

        if (low_papr_size > handle->low_papr_pilots_capacity) {
            if (handle->d_low_papr_pilots) cudaFree(handle->d_low_papr_pilots);
            err = cudaMalloc(&handle->d_low_papr_pilots, low_papr_size);
            if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
            handle->low_papr_pilots_capacity = low_papr_size;
            handle->cached_low_papr_nof_prb = -1;
            handle->cached_low_papr_n_rs_id = -1;
        }
        d_low_papr_pilots = handle->d_low_papr_pilots;

        if (handle->cached_low_papr_nof_prb != cfg.nof_prb ||
            handle->cached_low_papr_n_rs_id != cfg.n_rs_id) {
            /* Generate low-PAPR pilots: u = n_rs_id % 30 */
            int u = cfg.n_rs_id % NOF_LOW_PAPR_GROUPS;
            if (u < 0) u += NOF_LOW_PAPR_GROUPS;
            int pilot_threads = 256;
            int pilot_blocks = (M_zc + pilot_threads - 1) / pilot_threads;
            kernel_generate_low_papr_pilots<<<pilot_blocks, pilot_threads, 0, stream>>>(
                d_low_papr_pilots, u, M_zc);
            handle->cached_low_papr_nof_prb = cfg.nof_prb;
            handle->cached_low_papr_n_rs_id = cfg.n_rs_id;
        }
    }

    for (int d = 0; d < nof_dmrs_symbols; d++) {
        int dmrs_sym = dmrs_symbol_indices[d];

        if (!use_low_papr) {
            /* Standard pseudo-random Gold sequence DMRS (TS 38.211 Section 6.4.1.1.1) */

            /* Compute c_init for this DMRS symbol - use slot within frame per 3GPP TS 38.211 */
            uint32_t c_init = dmrs_compute_c_init(slot_in_frame_for_dmrs, dmrs_sym, cfg.scrambling_id, cfg.n_scid);

            /* Configure DMRS scrambler with direct c_init value */
            status = scrambler_configure_c_init(handle->dmrs_scrambler, c_init);
            if (status != NR_LDPC_SUCCESS) {
                return status;
            }

            /* Generate sequence bits (accounting for start_prb offset from point A) */
            int nof_seq_bits = (cfg.start_prb + cfg.nof_prb) * 6 * 2;
            status = scrambler_generate_sequence(handle->dmrs_scrambler, nof_seq_bits, stream);
            if (status != NR_LDPC_SUCCESS) {
                return status;
            }

            h_dmrs_seqs[d] = (unsigned int*)scrambler_get_sequence_ptr(handle->dmrs_scrambler);
        } else {
            /* Low-PAPR DMRS uses same sequence for all symbols (only depends on n_rs_id).
             * Set to NULL to indicate the LSE kernel should use the complex pilot buffer. */
            h_dmrs_seqs[d] = nullptr;
        }

        /* Output buffer for this DMRS symbol's LSE estimates */
        cuFloatComplex* d_lse_out = handle->d_dmrs_estimates +
                                     d * (cfg.nof_rx_ports * nof_dmrs_re_per_symbol);

        /* Launch LSE kernel */
        if (cfg.dmrs_type == DMRS_TYPE_1) {
            int blocks = cfg.nof_prb;
            int threads = cfg.nof_rx_ports;

            if (!use_low_papr) {
                /* Standard pseudo-random DMRS LSE */
                kernel_dmrs_lse_type1<<<blocks, threads, 0, stream>>>(
                    static_cast<const unsigned int*>(d_grid_cbf16),
                    h_dmrs_seqs[d],
                    d_lse_out,
                    cfg.nof_prb,
                    cfg.nof_rx_ports,
                    grid_stride,
                    symbol_stride,
                    cfg.start_prb,
                    dmrs_sym,
                    cfg.start_symbol,
                    cfg.dmrs_scaling);
            } else {
                /* Low-PAPR DMRS LSE - uses complex pilot buffer directly.
                 * For transform precoding, all 6 REs per PRB carry pilots (no CDM). */
                kernel_dmrs_lse_type1_low_papr<<<blocks, threads, 0, stream>>>(
                    static_cast<const unsigned int*>(d_grid_cbf16),
                    d_low_papr_pilots,
                    d_lse_out,
                    cfg.nof_prb,
                    cfg.nof_rx_ports,
                    grid_stride,
                    symbol_stride,
                    cfg.start_prb,
                    dmrs_sym,
                    cfg.start_symbol,
                    cfg.dmrs_scaling);
            }
        }

        /* Frequency interpolation for this DMRS symbol */
        cuFloatComplex* d_freq_out = d_freq_interp_estimates +
                                      d * (cfg.nof_rx_ports * nof_re_per_symbol);

        int total_re = cfg.nof_prb * cfg.nof_rx_ports * 12;
        int threads_fi = 256;
        int blocks_fi = (total_re + threads_fi - 1) / threads_fi;
        kernel_freq_interpolate_type1<<<blocks_fi, threads_fi, 0, stream>>>(
            d_lse_out,
            d_freq_out,
            cfg.nof_prb,
            cfg.nof_rx_ports);
    }

    /* ================================================================
     * Phase 1b: Average LSE estimates across ALL DMRS symbols
     * (Mirroring CPU algorithm: avg_h = beta × Σ h_est / nof_dmrs_symbols)
     * ================================================================ */

    /* Use pre-allocated buffer for averaged LSE estimates */
    size_t avg_lse_size = nof_dmrs_re_per_symbol * cfg.nof_rx_ports * sizeof(cuFloatComplex);
    if (avg_lse_size > handle->lse_averaged_capacity) {
        if (handle->d_lse_averaged) cudaFree(handle->d_lse_averaged);
        err = cudaMalloc(&handle->d_lse_averaged, avg_lse_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->lse_averaged_capacity = avg_lse_size;
    }
    cuFloatComplex* d_lse_averaged = handle->d_lse_averaged;

    /* Launch averaging kernel */
    int avg_total = cfg.nof_prb * cfg.nof_rx_ports * 6;
    int avg_threads = 256;
    int avg_blocks = (avg_total + avg_threads - 1) / avg_threads;
    kernel_average_lse_estimates<<<avg_blocks, avg_threads, 0, stream>>>(
        handle->d_dmrs_estimates,
        d_lse_averaged,
        nof_dmrs_symbols,
        cfg.nof_prb,
        cfg.nof_rx_ports,
        nof_dmrs_re_per_symbol,
        cfg.dmrs_scaling);

    /* ================================================================
     * Phase 1c: Estimate noise variance using AVERAGED channel estimates
     * (Mirroring CPU algorithm: compute residuals with averaged h_est)
     * ================================================================ */

    /* Use slot within frame for DMRS c_init (3GPP TS 38.211 Section 6.4.1.1.1.1) */
    int slot_in_frame_for_dmrs_nv = cfg.slot_idx % MAX_SLOTS_PER_FRAME;

    for (int d = 0; d < nof_dmrs_symbols; d++) {
        int dmrs_sym = dmrs_symbol_indices[d];

        /* Launch noise variance estimation kernel using AVERAGED estimates */
        if (cfg.dmrs_type == DMRS_TYPE_1) {
            int nv_threads = 256;
            int nv_blocks = cfg.nof_rx_ports;
            int shared_mem = nv_threads * sizeof(float);
            float* d_partial_out = handle->d_noise_var_partial + d * cfg.nof_rx_ports;

            if (!use_low_papr) {
                /* Regenerate DMRS sequence for this symbol (needed for noise estimation) */
                uint32_t c_init = dmrs_compute_c_init(slot_in_frame_for_dmrs_nv, dmrs_sym, cfg.scrambling_id, cfg.n_scid);
                status = scrambler_configure_c_init(handle->dmrs_scrambler, c_init);
                if (status != NR_LDPC_SUCCESS) {
                    return status;
                }

                int nof_seq_bits = (cfg.start_prb + cfg.nof_prb) * 6 * 2;
                status = scrambler_generate_sequence(handle->dmrs_scrambler, nof_seq_bits, stream);
                if (status != NR_LDPC_SUCCESS) {
                    return status;
                }

                const unsigned int* d_dmrs_seq = scrambler_get_sequence_ptr(handle->dmrs_scrambler);

                kernel_noise_variance_from_dmrs<<<nv_blocks, nv_threads, shared_mem, stream>>>(
                    static_cast<const unsigned int*>(d_grid_cbf16),
                    d_dmrs_seq,
                    d_lse_averaged,  /* Use AVERAGED estimates, not per-symbol */
                    d_partial_out,
                    cfg.nof_prb,
                    cfg.nof_rx_ports,
                    grid_stride,
                    symbol_stride,
                    cfg.start_prb,
                    dmrs_sym,
                    cfg.start_symbol,
                    cfg.dmrs_scaling,
                    nof_dmrs_re_per_symbol);
            } else {
                kernel_noise_variance_from_low_papr_dmrs<<<nv_blocks, nv_threads, shared_mem, stream>>>(
                    static_cast<const unsigned int*>(d_grid_cbf16),
                    d_low_papr_pilots,
                    d_lse_averaged,
                    d_partial_out,
                    cfg.nof_prb,
                    cfg.nof_rx_ports,
                    grid_stride,
                    symbol_stride,
                    cfg.start_prb,
                    dmrs_sym,
                    cfg.dmrs_scaling,
                    nof_dmrs_re_per_symbol);
            }
        }
    }

    /* Finalize noise variance (average across all DMRS symbols) */
    if (!use_low_papr) {
        kernel_noise_variance_finalize<<<1, cfg.nof_rx_ports, 0, stream>>>(
            handle->d_noise_vars,
            handle->d_noise_var_partial,
            cfg.nof_rx_ports,
            nof_dmrs_symbols,
            nof_dmrs_re_per_symbol,
            cfg.dmrs_scaling);
    } else {
        kernel_noise_variance_finalize_low_papr<<<1, cfg.nof_rx_ports, 0, stream>>>(
            handle->d_noise_vars,
            handle->d_noise_var_partial,
            cfg.nof_rx_ports,
            nof_dmrs_symbols,
            nof_dmrs_re_per_symbol);
    }

    /* ================================================================
     * Phase 2: Time interpolation to all data symbols
     * ================================================================ */

    int total_time = cfg.nof_symbols * cfg.nof_rx_ports * nof_re_per_symbol;
    int threads_ti = 256;
    int blocks_ti = (total_time + threads_ti - 1) / threads_ti;

    /* Scale factor for time interpolation.
     * We keep the full magnitude (1.0) for equalization stability, and compensate
     * for the scaling mismatch in the noise variance calculation instead.
     * This avoids reducing channel estimate SNR by N² which would hurt weak signals. */
    float time_interp_scale = 1.0f;

    kernel_time_interpolate<<<blocks_ti, threads_ti, 0, stream>>>(
        d_freq_interp_estimates,
        handle->d_ch_estimates,
        d_dmrs_indices,
        nof_dmrs_symbols,
        cfg.nof_symbols,
        cfg.nof_rx_ports,
        nof_re_per_symbol,
        cfg.start_symbol,
        time_interp_scale);

    if (!run_demod) {
        err = cudaGetLastError();
        return (err == cudaSuccess) ? NR_LDPC_SUCCESS : NR_LDPC_ERROR_CUDA_FAILED;
    }

    /* ================================================================
     * Phase 3: Generate data scrambling sequence
     * ================================================================ */

    int num_llrs = nof_data_re * cfg.mod_order;
    status = scrambler_generate_sequence(handle->data_scrambler, num_llrs, stream);
    if (status != NR_LDPC_SUCCESS) {
        /* Pre-allocated, persist */
        /* Pre-allocated, persist */
        return status;
    }

    const unsigned int* d_scramble_seq = scrambler_get_sequence_ptr(handle->data_scrambler);

    /* ================================================================
     * Phase 4: Fused equalization + demodulation + descrambling
     *          (using GPU-computed noise variances!)
     * ================================================================ */

    int threads_e2e = 256;
    int blocks_e2e = (nof_data_re + threads_e2e - 1) / threads_e2e;

    /* Reset SINR accumulators before kernel launch */
    cudaMemsetAsync(handle->d_eq_noise_var_sum, 0, sizeof(float) + sizeof(unsigned int), stream);

    /* Dispatch kernel based on nof_rx_ports and equalizer_algorithm */
    #define LAUNCH_E2E_KERNEL_FULL(PORTS, ALG) \
        kernel_pusch_e2e_with_estimates<PORTS, ALG><<<blocks_e2e, threads_e2e, 0, stream>>>( \
            static_cast<const unsigned int*>(d_grid_cbf16), \
            handle->d_ch_estimates, \
            d_re_indices, \
            handle->d_noise_vars, \
            static_cast<__half*>(d_llrs_half), \
            d_scramble_seq, \
            handle->d_eq_noise_var_sum, \
            handle->d_eq_noise_var_count, \
            nof_data_re, cfg.mod_order, grid_stride, symbol_stride, nof_re_per_symbol, \
            cfg.start_symbol, cfg.start_prb * 12, cfg.tx_scaling)

    #define DISPATCH_ALGORITHM_FULL(PORTS) \
        switch (cfg.equalizer_algorithm) { \
            case EQUALIZER_ZF: LAUNCH_E2E_KERNEL_FULL(PORTS, EQUALIZER_ZF); break; \
            case EQUALIZER_MMSE: LAUNCH_E2E_KERNEL_FULL(PORTS, EQUALIZER_MMSE); break; \
            case EQUALIZER_MMSE_IRC: default: LAUNCH_E2E_KERNEL_FULL(PORTS, EQUALIZER_MMSE_IRC); break; \
        }

    switch (cfg.nof_rx_ports) {
        case 1: DISPATCH_ALGORITHM_FULL(1); break;
        case 2: DISPATCH_ALGORITHM_FULL(2); break;
        case 4: DISPATCH_ALGORITHM_FULL(4); break;
        case 8: DISPATCH_ALGORITHM_FULL(8); break;
        default:
            /* Pre-allocated, persist */
            /* Pre-allocated, persist */
            return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    #undef LAUNCH_E2E_KERNEL_FULL
    #undef DISPATCH_ALGORITHM_FULL

    /* Cleanup temporary buffers */
    /* Pre-allocated, persist */
    /* Pre-allocated, persist */

    err = cudaGetLastError();
    return (err == cudaSuccess) ? NR_LDPC_SUCCESS : NR_LDPC_ERROR_CUDA_FAILED;
}

nr_ldpc_status_t pusch_e2e_process_full_gpu(
    pusch_e2e_handle_t handle,
    const void* d_grid_cbf16,
    void* d_llrs_half,
    const int* d_re_indices,
    int nof_data_re,
    cudaStream_t stream)
{
    return pusch_e2e_process_full_gpu_impl(
        handle, d_grid_cbf16, d_llrs_half, d_re_indices, nof_data_re, stream, true);
}

const float* pusch_e2e_get_noise_vars(pusch_e2e_handle_t handle)
{
    if (!handle) return nullptr;
    return handle->d_noise_vars;
}

float pusch_e2e_get_sinr_db(pusch_e2e_handle_t handle, cudaStream_t stream)
{
    if (!handle || !handle->d_eq_noise_var_sum || !handle->d_eq_noise_var_count) {
        return -INFINITY;
    }

    /* Check if stream is complete without blocking.
     * This avoids the ~2 second delay on first CUDA call due to context init. */
    cudaError_t status = cudaStreamQuery(stream);
    if (status == cudaErrorNotReady) {
        /* Stream not complete yet - return infinity to indicate no valid SINR */
        return INFINITY;
    }

    /* Copy accumulated values from device */
    float noise_var_sum = 0.0f;
    unsigned int count = 0;

    cudaMemcpy(&noise_var_sum, handle->d_eq_noise_var_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count, handle->d_eq_noise_var_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (count == 0 || noise_var_sum <= 0.0f) {
        return -INFINITY;
    }

    /* Compute mean noise variance */
    float mean_noise_var = noise_var_sum / static_cast<float>(count);

    /* SINR = 10*log10(signal_power / noise_power).
     * After MMSE equalization with gain normalization, signal power is ~1.
     * So SINR = 10*log10(1 / mean_noise_var) = -10*log10(mean_noise_var). */
    float sinr_db = -10.0f * log10f(mean_noise_var);

    return sinr_db;
}

void pusch_e2e_sinr_async_launch(pusch_e2e_handle_t handle, cudaStream_t stream)
{
    if (!handle || !handle->d_eq_noise_var_sum || !handle->d_eq_noise_var_count) {
        return;
    }
    if (!handle->h_sinr_noise_var_sum || !handle->h_sinr_count) {
        return;
    }

    /* Single 16-byte D2H copy for adjacent SINR and EVM accumulators in d_accum_block. */
    cudaMemcpyAsync(handle->h_sinr_eq_block, handle->d_eq_noise_var_sum,
                    2 * sizeof(float) + 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);

    /* Launch async D2H copy of DMRS-based SINR/EPRE/RSRP/TA/CFO result (20 bytes) */
    if (handle->d_sinr_epre_result && handle->h_sinr_epre_result) {
        cudaMemcpyAsync(handle->h_sinr_epre_result, handle->d_sinr_epre_result,
                        5 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
}

float pusch_e2e_sinr_get_result(pusch_e2e_handle_t handle)
{
    /* Primary: post-equalization SINR = -10*log10(mean(eq_noise_var))
     * Matches CPU demodulator SINR. Per-RE eq_noise_var captures both
     * thermal noise and channel estimation error at data positions,
     * giving accurate SINR under frequency-selective fading. */
    if (handle && handle->h_sinr_noise_var_sum && handle->h_sinr_count) {
        float noise_var_sum = *handle->h_sinr_noise_var_sum;
        unsigned int count = *handle->h_sinr_count;
        if (count > 0 && noise_var_sum > 0.0f) {
            float mean_noise_var = noise_var_sum / static_cast<float>(count);
            return -10.0f * log10f(mean_noise_var);
        }
    }

    /* Fallback: DMRS pilot-ratio SINR (may over-report under fading) */
    if (handle && handle->h_sinr_epre_result) {
        float sinr_db = handle->h_sinr_epre_result[0];
        if (std::isfinite(sinr_db)) {
            return sinr_db;
        }
    }

    return -INFINITY;
}

float pusch_e2e_epre_get_result(pusch_e2e_handle_t handle)
{
    if (!handle || !handle->h_sinr_epre_result) {
        return -INFINITY;
    }
    return handle->h_sinr_epre_result[1];
}

float pusch_e2e_rsrp_get_result(pusch_e2e_handle_t handle)
{
    if (!handle || !handle->h_sinr_epre_result) {
        return -INFINITY;
    }
    return handle->h_sinr_epre_result[2];
}

float pusch_e2e_ta_get_result(pusch_e2e_handle_t handle)
{
    if (!handle || !handle->h_sinr_epre_result) {
        return NAN;
    }
    return handle->h_sinr_epre_result[3];
}

float pusch_e2e_cfo_get_result(pusch_e2e_handle_t handle)
{
    if (!handle || !handle->h_sinr_epre_result) {
        return NAN;
    }
    return handle->h_sinr_epre_result[4];
}

float pusch_e2e_evm_get_result(pusch_e2e_handle_t handle)
{
    if (!handle || !handle->config.enable_evm_metric || !handle->h_evm_error_sum || !handle->h_evm_symbol_count) {
        return NAN;
    }

    float error_sum = *handle->h_evm_error_sum;
    unsigned int count = *handle->h_evm_symbol_count;
    if (count == 0 || !std::isfinite(error_sum) || error_sum < 0.0f) {
        return NAN;
    }

    return std::sqrt(error_sum / static_cast<float>(count));
}

void pusch_e2e_print_diagnostics(pusch_e2e_handle_t handle, cudaStream_t stream)
{
    if (!handle || !handle->configured) {
        fprintf(stderr, "[GPU E2E DIAG] Handle not configured\n");
        return;
    }

    cudaStreamSynchronize(stream);

    const pusch_e2e_config_t& cfg = handle->config;

    /* Print configuration */
    fprintf(stderr, "[GPU E2E DIAG] Configuration:\n");
    fprintf(stderr, "  nof_prb=%d, start_prb=%d, nof_symbols=%d, start_symbol=%d\n",
            cfg.nof_prb, cfg.start_prb, cfg.nof_symbols, cfg.start_symbol);
    fprintf(stderr, "  grid: %dx%d, nof_ports=%d, dmrs_mask=0x%x\n",
            cfg.grid_nof_subcarriers, cfg.grid_nof_symbols, cfg.nof_rx_ports, cfg.dmrs_symbol_mask);
    fprintf(stderr, "  dmrs_scaling=%.4f, tx_scaling=%.4f\n", cfg.dmrs_scaling, cfg.tx_scaling);

    /* Compute and print scaling factor used for noise variance */
    int nof_dmrs_symbols = __builtin_popcount(cfg.dmrs_symbol_mask);
    int nof_dmrs_re_per_symbol = cfg.nof_prb * 6;  /* Type 1 DMRS */
    int total_dmrs_re = nof_dmrs_symbols * nof_dmrs_re_per_symbol;
    float N = (float)nof_dmrs_symbols;
    float base_scale = 4.0f / sqrtf(N);  /* 2x original (4.0 instead of 2.0) */
    float alloc_factor = (total_dmrs_re < 50) ? 1.2f : 1.0f;
    float effective_scale = base_scale * alloc_factor;
    fprintf(stderr, "  nof_dmrs_symbols=%d, nof_dmrs_re=%d, noise_scale=%.3f (base=%.3f, alloc=%.3f)\n",
            nof_dmrs_symbols, total_dmrs_re, effective_scale, base_scale, alloc_factor);

    /* Print raw LSE estimates from the FP16 buffer used by the E2E kernel. */
    if (handle->d_estimates_fp16 && nof_dmrs_re_per_symbol > 0) {
        /* FP16 buffer layout: [dmrs_sym, prb, port, 12 subcarriers] */
        int sample_count = std::min(12, cfg.nof_prb * 12);  /* First PRB worth of estimates */
        std::vector<__half2> h_lse_fp16(sample_count);
        cudaMemcpy(h_lse_fp16.data(), handle->d_estimates_fp16,
                   sample_count * sizeof(__half2), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[GPU E2E DIAG] First %d FP16 channel estimates (DMRS sym 0, port 0):\n", sample_count);
        float sum_mag = 0.0f;
        for (int i = 0; i < sample_count; i++) {
            float h_real = __half2float(h_lse_fp16[i].x);
            float h_imag = __half2float(h_lse_fp16[i].y);
            float mag = sqrtf(h_real * h_real + h_imag * h_imag);
            sum_mag += mag;
            if (i < 6) {
                fprintf(stderr, "  [%d] (%.4f, %.4f) mag=%.4f\n", i, h_real, h_imag, mag);
            }
        }
        fprintf(stderr, "  fp16_avg_mag=%.4f (first %d samples)\n", sum_mag / sample_count, sample_count);
    }

    /* Get noise variances */
    if (handle->d_noise_vars) {
        std::vector<float> h_noise_vars(cfg.nof_rx_ports);
        cudaMemcpy(h_noise_vars.data(), handle->d_noise_vars,
                   cfg.nof_rx_ports * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[GPU E2E DIAG] Pre-EQ noise variances per port:\n");
        for (int p = 0; p < cfg.nof_rx_ports; p++) {
            fprintf(stderr, "  port %d: %.6f (%.2f dB)\n", p, h_noise_vars[p],
                    -10.0f * log10f(h_noise_vars[p]));
        }
    }

    /* Get post-EQ noise variance accumulators */
    if (handle->d_eq_noise_var_sum && handle->d_eq_noise_var_count) {
        float noise_var_sum = 0.0f;
        unsigned int count = 0;
        cudaMemcpy(&noise_var_sum, handle->d_eq_noise_var_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&count, handle->d_eq_noise_var_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[GPU E2E DIAG] Post-EQ: sum=%.6f, count=%u, mean=%.6f\n",
                noise_var_sum, count, count > 0 ? noise_var_sum / count : 0.0f);
    }

    /* Sample a few channel estimates to check magnitude */
    if (handle->d_ch_estimates && handle->nof_estimates > 0) {
        int sample_count = std::min(10, handle->nof_estimates);
        std::vector<cuFloatComplex> h_samples(sample_count);
        cudaMemcpy(h_samples.data(), handle->d_ch_estimates,
                   sample_count * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[GPU E2E DIAG] First %d channel estimate samples:\n", sample_count);
        float sum_mag = 0.0f;
        for (int i = 0; i < sample_count; i++) {
            float mag = sqrtf(h_samples[i].x * h_samples[i].x + h_samples[i].y * h_samples[i].y);
            sum_mag += mag;
            fprintf(stderr, "  [%d] (%.4f, %.4f) mag=%.4f\n", i, h_samples[i].x, h_samples[i].y, mag);
        }
        fprintf(stderr, "  avg_mag=%.4f\n", sum_mag / sample_count);
    }

    /* For SMALL allocations only: sample equalized symbols by computing a few manually.
     * This helps inspect constellation quality. Only do for nof_prb <= 5 to limit output. */
    if (cfg.nof_prb <= 5 && handle->d_ch_estimates && handle->nof_estimates >= 12) {
        /* Sample first symbol's data (skip DMRS positions).
         * For Type 1 DMRS, data REs are at subcarriers 1,3,5,7,9,11 in DMRS symbols,
         * or all 12 in non-DMRS symbols. Symbol 0 is usually data if start_symbol=0. */
        int first_data_symbol = cfg.start_symbol;
        int sample_re_count = std::min(6, cfg.nof_prb * 12);

        std::vector<cuFloatComplex> h_first(sample_re_count);
        std::vector<float> h_grid(sample_re_count * 2);  /* cbf16 packed as uint32 */

        /* Get channel estimates for first symbol */
        cudaMemcpy(h_first.data(), handle->d_ch_estimates,
                   sample_re_count * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

        fprintf(stderr, "[GPU E2E DIAG] First %d equalized symbol samples (manual compute):\n", sample_re_count);
        fprintf(stderr, "  nof_prb=%d, start_prb=%d, first_data_symbol=%d\n",
                cfg.nof_prb, cfg.start_prb, first_data_symbol);
    }
}

/* ============================================================================
 * Transform Deprecoding (IDFT) Support for Full GPU Path
 * ============================================================================ */

__device__ __forceinline__ cuFloatComplex cmul_e2e(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x * b.x - a.y * b.y,
                               a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ cuFloatComplex twiddle_idft_cycles_e2e(float cycles) {
    float angle = 2.0f * 3.14159265359f * cycles;
    float c, s;
    __sincosf(angle, &s, &c);
    return make_cuFloatComplex(c, s);
}

static uint64_t pack_deprecoding_fft_factors(int n, int* nof_factors)
{
    uint64_t packed = 0;
    int count = 0;
    auto push_factor = [&](int factor) {
        packed |= (static_cast<uint64_t>(factor) << (4 * count));
        ++count;
        n /= factor;
    };

    while ((n % 5) == 0) {
        push_factor(5);
    }
    while ((n % 4) == 0) {
        push_factor(4);
    }
    while ((n % 3) == 0) {
        push_factor(3);
    }
    while ((n % 2) == 0) {
        push_factor(2);
    }

    if (n != 1) {
        *nof_factors = 0;
        return 0;
    }
    *nof_factors = count;
    return packed;
}

__device__ __forceinline__ cuFloatComplex mixed_radix_butterfly_idft_e2e(
    const cuFloatComplex* __restrict__ in,
    int radix,
    int n0)
{
    if (radix == 2) {
        return (n0 == 0) ? make_cuFloatComplex(in[0].x + in[1].x, in[0].y + in[1].y) :
                           make_cuFloatComplex(in[0].x - in[1].x, in[0].y - in[1].y);
    }

    if (radix == 3) {
        constexpr float c = -0.5f;
        constexpr float s = 0.8660254037844386f;
        if (n0 == 0) {
            return make_cuFloatComplex(in[0].x + in[1].x + in[2].x,
                                       in[0].y + in[1].y + in[2].y);
        }
        cuFloatComplex w1 = (n0 == 1) ? make_cuFloatComplex(c, s) : make_cuFloatComplex(c, -s);
        cuFloatComplex w2 = (n0 == 1) ? make_cuFloatComplex(c, -s) : make_cuFloatComplex(c, s);
        cuFloatComplex a = cmul_e2e(w1, in[1]);
        cuFloatComplex b = cmul_e2e(w2, in[2]);
        return make_cuFloatComplex(in[0].x + a.x + b.x, in[0].y + a.y + b.y);
    }

    if (radix == 4) {
        if (n0 == 0) {
            return make_cuFloatComplex(in[0].x + in[1].x + in[2].x + in[3].x,
                                       in[0].y + in[1].y + in[2].y + in[3].y);
        }
        if (n0 == 1) {
            return make_cuFloatComplex(in[0].x - in[1].y - in[2].x + in[3].y,
                                       in[0].y + in[1].x - in[2].y - in[3].x);
        }
        if (n0 == 2) {
            return make_cuFloatComplex(in[0].x - in[1].x + in[2].x - in[3].x,
                                       in[0].y - in[1].y + in[2].y - in[3].y);
        }
        return make_cuFloatComplex(in[0].x + in[1].y - in[2].x - in[3].y,
                                   in[0].y - in[1].x - in[2].y + in[3].x);
    }

    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
    cuFloatComplex w = make_cuFloatComplex(1.0f, 0.0f);
    float angle = 2.0f * 3.14159265359f * static_cast<float>(n0) / static_cast<float>(radix);
    float s, c;
    __sincosf(angle, &s, &c);
    cuFloatComplex w_step = make_cuFloatComplex(c, s);

    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        if (i >= radix) {
            break;
        }
        cuFloatComplex x = cmul_e2e(w, in[i]);
        sum.x += x.x;
        sum.y += x.y;
        w = cmul_e2e(w, w_step);
    }
    return sum;
}

/**
 * Fused E2E kernel with GPU channel estimates AND transform deprecoding.
 *
 * Processing flow per symbol:
 * 1. Equalize REs using GPU-computed channel estimates
 * 2. Apply a shared-memory mixed-radix IDFT across DFT-size subcarriers
 * 3. Soft demodulate + descramble
 *
 * Each block processes one OFDM symbol. Threads cooperatively compute IDFT.
 */
template <int NOF_PORTS, int ALGORITHM>
__global__ void kernel_pusch_e2e_with_estimates_deprecode(
    const unsigned int* __restrict__ d_grid_cbf16,
    const cuFloatComplex* __restrict__ d_estimates,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    __half* __restrict__ llrs_half,
    uint32_t scramble_c_init,  /* c_init for on-the-fly LFSR (matches non-transform path) */
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int mod_order,
    int grid_stride,
    int symbol_stride,         /* nof_subcarriers (for decoding src_re) */
    int est_stride,            /* nof_re_per_symbol for estimates */
    int start_symbol,          /* allocation start symbol */
    int start_subcarrier,      /* allocation start subcarrier (start_prb * 12) */
    float tx_scaling,
    int dft_size,
    uint64_t packed_fft_factors,
    int nof_fft_factors,
    int nof_symbols)
{
    extern __shared__ char shared_mem[];
    cuFloatComplex* s_eq_symbols = reinterpret_cast<cuFloatComplex*>(shared_mem);
    cuFloatComplex* s_fft_tmp = s_eq_symbols + dft_size;
    float* s_eq_noise_sum = reinterpret_cast<float*>(s_fft_tmp + dft_size);
    unsigned int* s_eq_noise_count = reinterpret_cast<unsigned int*>(s_eq_noise_sum + blockDim.x);

    int symbol_idx = blockIdx.x;
    if (symbol_idx >= nof_symbols) return;

    int tid = threadIdx.x;
    int nof_threads = blockDim.x;

    /* Load noise variances to shared memory */
    __shared__ float s_noise_vars[8];
    if (tid < NOF_PORTS) {
        s_noise_vars[tid] = noise_vars[tid];
    }
    __syncthreads();

    /* Calculate RE range for this symbol */
    int re_per_symbol = dft_size;
    int re_start = symbol_idx * re_per_symbol;
    int re_end = min(re_start + re_per_symbol, nof_re);
    int nof_re_this_symbol = re_end - re_start;

    /* Phase 1: Equalize all REs for this symbol */
    float local_eq_noise_sum = 0.0f;
    unsigned int local_eq_noise_count = 0;
    for (int local_re = tid; local_re < nof_re_this_symbol; local_re += nof_threads) {
        int global_re_idx = re_start + local_re;
        int src_re = re_indices[global_re_idx];

        /* Decode symbol and subcarrier from src_re */
        int symbol = src_re / symbol_stride;
        int subcarrier = src_re % symbol_stride;

        /* Compute relative indices for estimate lookup */
        int symbol_rel = symbol - start_symbol;
        int re_in_symbol = subcarrier - start_subcarrier;

        /* Load channel data for all ports.
         * Grid is uploaded with ALL slot symbols (absolute indexing 0..13),
         * so use src_re directly for grid lookup.
         * Estimates are GPU-computed with relative indexing, so use symbol_rel/re_in_symbol. */
        cuFloatComplex y[8], h[8];

        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            int grid_idx = p * grid_stride + src_re;
            y[p] = cbf16_to_fp32(d_grid_cbf16[grid_idx]);

            /* Estimate layout: [symbol_rel, port, re_in_symbol] */
            int est_idx = symbol_rel * (NOF_PORTS * est_stride) + p * est_stride + re_in_symbol;
            h[p] = d_estimates[est_idx];
        }

        /* Equalization (ZF, MMSE, or MMSE-IRC) */
        cuFloatComplex eq;
        float eq_noise_var;
        equalize_symbol<NOF_PORTS, ALGORITHM>(y, h, s_noise_vars, tx_scaling, eq, eq_noise_var);
        if (sinr_noise_var_is_valid(eq_noise_var)) {
            local_eq_noise_sum += eq_noise_var;
            ++local_eq_noise_count;
        }

        /* Store equalized symbol to shared memory for IDFT. */
        s_eq_symbols[local_re] = eq;
    }

    /* Zero-pad if symbol has fewer REs */
    for (int local_re = nof_re_this_symbol + tid; local_re < dft_size; local_re += nof_threads) {
        s_eq_symbols[local_re] = make_cuFloatComplex(0.0f, 0.0f);
    }

    s_eq_noise_sum[tid] = local_eq_noise_sum;
    s_eq_noise_count[tid] = local_eq_noise_count;
    __syncthreads();

    for (int stride = nof_threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_eq_noise_sum[tid] += s_eq_noise_sum[tid + stride];
            s_eq_noise_count[tid] += s_eq_noise_count[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float s_symbol_eq_noise_var;
    if (tid == 0) {
        float avg_eq_noise_var;
        if (s_eq_noise_count[0] > 0) {
            avg_eq_noise_var = s_eq_noise_sum[0] / static_cast<float>(s_eq_noise_count[0]);
            atomicAdd(d_eq_noise_var_sum, s_eq_noise_sum[0]);
            atomicAdd(d_eq_noise_var_count, s_eq_noise_count[0]);
        } else {
            avg_eq_noise_var = 0.0f;
            #pragma unroll
            for (int p = 0; p < NOF_PORTS; p++) {
                avg_eq_noise_var += s_noise_vars[p];
            }
            avg_eq_noise_var /= (NOF_PORTS * tx_scaling * tx_scaling);
        }
        s_symbol_eq_noise_var = fmaxf(avg_eq_noise_var, 1e-10f);
    }
    __syncthreads();

    /* Phase 2: Apply transform deprecoding with a shared-memory mixed-radix
     * inverse FFT. The recursive Cooley-Tukey layout is applied iteratively:
     *
     *   in[p, k1 + M*k2] -> out[p + P*n0, k1]
     *
     * where L = R*M is the remaining transform length and P is the product of
     * previous radices. This keeps output in natural time order and avoids a
     * separate permutation pass.
     */
    cuFloatComplex* stage_in = s_eq_symbols;
    cuFloatComplex* stage_out = s_fft_tmp;
    int prefix_count = 1;
    int remaining = dft_size;

    for (int stage = 0; stage < nof_fft_factors; ++stage) {
        int radix = static_cast<int>((packed_fft_factors >> (4 * stage)) & 0xf);
        int next_remaining = remaining / radix;

        for (int idx = tid; idx < dft_size; idx += nof_threads) {
            int k1 = idx % next_remaining;
            int new_prefix = idx / next_remaining;
            int n0 = new_prefix / prefix_count;
            int p = new_prefix - n0 * prefix_count;
            int in_base = p * remaining + k1;

            cuFloatComplex inputs[5];
            #pragma unroll
            for (int k2 = 0; k2 < 5; ++k2) {
                inputs[k2] = (k2 < radix) ? stage_in[in_base + next_remaining * k2] :
                                            make_cuFloatComplex(0.0f, 0.0f);
            }

            cuFloatComplex sum = mixed_radix_butterfly_idft_e2e(inputs, radix, n0);
            if ((n0 != 0) && (k1 != 0)) {
                cuFloatComplex tw = twiddle_idft_cycles_e2e(
                    static_cast<float>(n0 * k1) / static_cast<float>(remaining));
                sum = cmul_e2e(tw, sum);
            }
            stage_out[idx] = sum;
        }
        __syncthreads();

        cuFloatComplex* tmp = stage_in;
        stage_in = stage_out;
        stage_out = tmp;
        prefix_count *= radix;
        remaining = next_remaining;
    }

    for (int out_idx = tid; out_idx < dft_size; out_idx += nof_threads) {
        float scale = rsqrtf((float)dft_size);
        cuFloatComplex x = stage_in[out_idx];
        s_eq_symbols[out_idx] = make_cuFloatComplex(x.x * scale, x.y * scale);
    }
    __syncthreads();

    /* Phase 3: Soft demodulation + descrambling */
    /* Unitary transform deprecoding preserves the average equalized noise power. */
    float inv_noise = 1.0f / s_symbol_eq_noise_var;

    for (int local_re = tid; local_re < nof_re_this_symbol; local_re += nof_threads) {
        int global_re_idx = re_start + local_re;
        cuFloatComplex eq = s_eq_symbols[local_re];

        /* Soft demodulation */
        float llr_vals[8];

        if (mod_order == 2) {
            float scale = 2.0f * 1.41421356f * inv_noise;
            llr_vals[0] = eq.x * scale;
            llr_vals[1] = eq.y * scale;
        } else if (mod_order == 4) {
            float scale = inv_noise * 0.6324555f;
            float abs_re = fabsf(eq.x);
            float abs_im = fabsf(eq.y);
            llr_vals[0] = eq.x * scale * 2.0f;
            llr_vals[1] = eq.y * scale * 2.0f;
            llr_vals[2] = (0.6324555f * 2.0f - abs_re) * scale * 2.0f;
            llr_vals[3] = (0.6324555f * 2.0f - abs_im) * scale * 2.0f;
        } else if (mod_order == 6) {
            /* 64QAM max-log MAP soft demapping.
             * Uses closed-form distance to decision boundaries.
             * 1/sqrt(42) = 0.154303349962092
             */
            const float scale64 = 0.154303349962092f;
            float scale = inv_noise * scale64;
            float abs_re = fabsf(eq.x);
            float abs_im = fabsf(eq.y);
            /* Bit 0,1: linear in symbol */
            llr_vals[0] = eq.x * scale * 4.0f * 2.0f;
            llr_vals[1] = eq.y * scale * 4.0f * 2.0f;
            /* Bit 2,3: distance from 4M level */
            llr_vals[2] = (scale64 * 4.0f - abs_re) * scale * 2.0f * 2.0f;
            llr_vals[3] = (scale64 * 4.0f - abs_im) * scale * 2.0f * 2.0f;
            /* Bit 4,5: nested distance from 2M level */
            float level2 = scale64 * 2.0f;
            llr_vals[4] = (level2 - fabsf(abs_re - scale64 * 4.0f)) * scale * 2.0f * 2.0f;
            llr_vals[5] = (level2 - fabsf(abs_im - scale64 * 4.0f)) * scale * 2.0f * 2.0f;
        } else if (mod_order == 8) {
            float scale = inv_noise * 0.07669650f;
            float abs_re = fabsf(eq.x);
            float abs_im = fabsf(eq.y);
            llr_vals[0] = eq.x * scale * 2.0f;
            llr_vals[1] = eq.y * scale * 2.0f;
            llr_vals[2] = (abs_re - 0.07669650f * 8.0f) * scale * 2.0f;
            llr_vals[3] = (abs_im - 0.07669650f * 8.0f) * scale * 2.0f;
            float level2 = 0.07669650f * 4.0f;
            llr_vals[4] = (fabsf(abs_re - level2 * 2.0f) - level2) * scale * 2.0f;
            llr_vals[5] = (fabsf(abs_im - level2 * 2.0f) - level2) * scale * 2.0f;
            float level3 = 0.07669650f * 2.0f;
            float d_re = fabsf(abs_re - level2 * 2.0f);
            float d_im = fabsf(abs_im - level2 * 2.0f);
            llr_vals[6] = (fabsf(d_re - level3 * 2.0f) - level3) * scale * 2.0f;
            llr_vals[7] = (fabsf(d_im - level3 * 2.0f) - level3) * scale * 2.0f;
        }

        /* Descrambling - use on-the-fly LFSR (matches non-transform path) */
        int llr_base = global_re_idx * mod_order;

        /* Advance LFSRs to starting bit position for this RE.
         * NC_SKIP (1600) is already accounted for in the advance functions. */
        int bit_start = llr_base + NC_SKIP;
        uint32_t x1 = advance_x1_local(1, bit_start);
        uint32_t x2 = advance_x2_local(scramble_c_init, bit_start);

        for (int b = 0; b < mod_order; b++) {
            /* Generate scrambling bit on-the-fly */
            uint32_t scr_bit = (x1 ^ x2) & 1;

            float llr = scr_bit ? -llr_vals[b] : llr_vals[b];
            llr = fmaxf(-65504.0f, fminf(65504.0f, llr));
            llrs_half[llr_base + b] = __float2half(llr);

            /* Step LFSRs for next bit */
            x1 = step_x1_local(x1);
            x2 = step_x2_local(x2);
        }
    }
}

#ifdef OCUDU_PHY_CUDA_ENABLE_VKFFT
template <int NOF_PORTS, int ALGORITHM>
__global__ void kernel_pusch_e2e_deprecode_equalize_gather(
    const unsigned int* __restrict__ d_grid_cbf16,
    const cuFloatComplex* __restrict__ d_estimates,
    const int* __restrict__ re_indices,
    const float* __restrict__ noise_vars,
    cuFloatComplex* __restrict__ transform_symbols,
    float* __restrict__ symbol_noise_vars,
    float* __restrict__ d_eq_noise_var_sum,
    unsigned int* __restrict__ d_eq_noise_var_count,
    int nof_re,
    int grid_stride,
    int symbol_stride,
    int est_stride,
    int start_symbol,
    int start_subcarrier,
    float tx_scaling,
    int dft_size,
    int nof_symbols)
{
    extern __shared__ char shared_mem[];
    float* s_eq_noise_sum = reinterpret_cast<float*>(shared_mem);
    unsigned int* s_eq_noise_count = reinterpret_cast<unsigned int*>(s_eq_noise_sum + blockDim.x);

    int symbol_idx = blockIdx.x;
    if (symbol_idx >= nof_symbols) return;

    int tid = threadIdx.x;
    int nof_threads = blockDim.x;

    __shared__ float s_noise_vars[8];
    if (tid < NOF_PORTS) {
        s_noise_vars[tid] = noise_vars[tid];
    }
    __syncthreads();

    int re_start = symbol_idx * dft_size;
    int re_end = min(re_start + dft_size, nof_re);
    int nof_re_this_symbol = re_end - re_start;
    float local_eq_noise_sum = 0.0f;
    unsigned int local_eq_noise_count = 0;

    for (int local_re = tid; local_re < nof_re_this_symbol; local_re += nof_threads) {
        int global_re_idx = re_start + local_re;
        int src_re = re_indices[global_re_idx];
        int symbol = src_re / symbol_stride;
        int subcarrier = src_re % symbol_stride;
        int symbol_rel = symbol - start_symbol;
        int re_in_symbol = subcarrier - start_subcarrier;

        cuFloatComplex y[8], h[8];
        #pragma unroll
        for (int p = 0; p < NOF_PORTS; p++) {
            y[p] = cbf16_to_fp32(d_grid_cbf16[p * grid_stride + src_re]);
            h[p] = d_estimates[symbol_rel * (NOF_PORTS * est_stride) + p * est_stride + re_in_symbol];
        }

        cuFloatComplex eq;
        float eq_noise_var;
        equalize_symbol<NOF_PORTS, ALGORITHM>(y, h, s_noise_vars, tx_scaling, eq, eq_noise_var);
        if (sinr_noise_var_is_valid(eq_noise_var)) {
            local_eq_noise_sum += eq_noise_var;
            ++local_eq_noise_count;
        }

        transform_symbols[re_start + local_re] = eq;
    }

    for (int local_re = nof_re_this_symbol + tid; local_re < dft_size; local_re += nof_threads) {
        transform_symbols[re_start + local_re] = make_cuFloatComplex(0.0f, 0.0f);
    }

    s_eq_noise_sum[tid] = local_eq_noise_sum;
    s_eq_noise_count[tid] = local_eq_noise_count;
    __syncthreads();

    for (int stride = nof_threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_eq_noise_sum[tid] += s_eq_noise_sum[tid + stride];
            s_eq_noise_count[tid] += s_eq_noise_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float avg_eq_noise_var;
        if (s_eq_noise_count[0] > 0) {
            avg_eq_noise_var = s_eq_noise_sum[0] / static_cast<float>(s_eq_noise_count[0]);
            atomicAdd(d_eq_noise_var_sum, s_eq_noise_sum[0]);
            atomicAdd(d_eq_noise_var_count, s_eq_noise_count[0]);
        } else {
            avg_eq_noise_var = 0.0f;
            #pragma unroll
            for (int p = 0; p < NOF_PORTS; p++) {
                avg_eq_noise_var += s_noise_vars[p];
            }
            avg_eq_noise_var /= (NOF_PORTS * tx_scaling * tx_scaling);
        }
        symbol_noise_vars[symbol_idx] = fmaxf(avg_eq_noise_var, 1e-10f);
    }
}

__global__ void kernel_pusch_e2e_deprecode_demod_descramble(
    const cuFloatComplex* __restrict__ transform_symbols,
    const float* __restrict__ symbol_noise_vars,
    __half* __restrict__ llrs_half,
    uint32_t scramble_c_init,
    int nof_re,
    int mod_order,
    int dft_size)
{
    int global_re_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_re_idx >= nof_re) return;

    int symbol_idx = global_re_idx / dft_size;
    cuFloatComplex eq = transform_symbols[global_re_idx];
    float scale = rsqrtf(static_cast<float>(dft_size));
    eq.x *= scale;
    eq.y *= scale;

    float inv_noise = 1.0f / symbol_noise_vars[symbol_idx];
    float llr_vals[8];

    if (mod_order == 2) {
        float llr_scale = 2.0f * 1.41421356f * inv_noise;
        llr_vals[0] = eq.x * llr_scale;
        llr_vals[1] = eq.y * llr_scale;
    } else if (mod_order == 4) {
        float llr_scale = inv_noise * 0.6324555f;
        float abs_re = fabsf(eq.x);
        float abs_im = fabsf(eq.y);
        llr_vals[0] = eq.x * llr_scale * 2.0f;
        llr_vals[1] = eq.y * llr_scale * 2.0f;
        llr_vals[2] = (0.6324555f * 2.0f - abs_re) * llr_scale * 2.0f;
        llr_vals[3] = (0.6324555f * 2.0f - abs_im) * llr_scale * 2.0f;
    } else if (mod_order == 6) {
        const float scale64 = 0.154303349962092f;
        float llr_scale = inv_noise * scale64;
        float abs_re = fabsf(eq.x);
        float abs_im = fabsf(eq.y);
        llr_vals[0] = eq.x * llr_scale * 4.0f * 2.0f;
        llr_vals[1] = eq.y * llr_scale * 4.0f * 2.0f;
        llr_vals[2] = (scale64 * 4.0f - abs_re) * llr_scale * 2.0f * 2.0f;
        llr_vals[3] = (scale64 * 4.0f - abs_im) * llr_scale * 2.0f * 2.0f;
        float level2 = scale64 * 2.0f;
        llr_vals[4] = (level2 - fabsf(abs_re - scale64 * 4.0f)) * llr_scale * 2.0f * 2.0f;
        llr_vals[5] = (level2 - fabsf(abs_im - scale64 * 4.0f)) * llr_scale * 2.0f * 2.0f;
    } else if (mod_order == 8) {
        float llr_scale = inv_noise * 0.07669650f;
        float abs_re = fabsf(eq.x);
        float abs_im = fabsf(eq.y);
        llr_vals[0] = eq.x * llr_scale * 2.0f;
        llr_vals[1] = eq.y * llr_scale * 2.0f;
        llr_vals[2] = (abs_re - 0.07669650f * 8.0f) * llr_scale * 2.0f;
        llr_vals[3] = (abs_im - 0.07669650f * 8.0f) * llr_scale * 2.0f;
        float level2 = 0.07669650f * 4.0f;
        llr_vals[4] = (fabsf(abs_re - level2 * 2.0f) - level2) * llr_scale * 2.0f;
        llr_vals[5] = (fabsf(abs_im - level2 * 2.0f) - level2) * llr_scale * 2.0f;
        float level3 = 0.07669650f * 2.0f;
        float d_re = fabsf(abs_re - level2 * 2.0f);
        float d_im = fabsf(abs_im - level2 * 2.0f);
        llr_vals[6] = (fabsf(d_re - level3 * 2.0f) - level3) * llr_scale * 2.0f;
        llr_vals[7] = (fabsf(d_im - level3 * 2.0f) - level3) * llr_scale * 2.0f;
    }

    int llr_base = global_re_idx * mod_order;
    int bit_start = llr_base + NC_SKIP;
    uint32_t x1 = advance_x1_local(1, bit_start);
    uint32_t x2 = advance_x2_local(scramble_c_init, bit_start);

    for (int b = 0; b < mod_order; b++) {
        uint32_t scr_bit = (x1 ^ x2) & 1;
        float llr = scr_bit ? -llr_vals[b] : llr_vals[b];
        llr = fmaxf(-65504.0f, fminf(65504.0f, llr));
        llrs_half[llr_base + b] = __float2half(llr);
        x1 = step_x1_local(x1);
        x2 = step_x2_local(x2);
    }
}

static nr_ldpc_status_t launch_vkfft_deprecoder(
    pusch_e2e_handle_t handle,
    const void* d_grid_cbf16,
    void* d_llrs_half,
    const int* d_re_indices,
    int nof_data_re,
    int dft_size,
    int nof_ofdm_symbols,
    int grid_stride,
    int symbol_stride,
    int est_stride,
    int start_subcarrier,
    uint32_t scramble_c_init,
    cudaStream_t stream)
{
    if (!handle->vkfft_preplanned || handle->vkfft_stream != stream ||
        !handle->vkfft_plans[dft_size] || !handle->vkfft_plans[dft_size]->initialized) {
        if (handle->transform_deprecoder_backend == PUSCH_DEPRECODER_VKFFT) {
            return NR_LDPC_ERROR_INVALID_CONFIG;
        }
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (nof_ofdm_symbols > PUSCH_DEPRECODE_MAX_SYMBOLS ||
        static_cast<size_t>(nof_ofdm_symbols * dft_size) > handle->deprecode_symbols_capacity) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    const pusch_e2e_config_t& cfg = handle->config;
    int threads = min(dft_size, 256);
    size_t shared_mem_size = threads * (sizeof(float) + sizeof(unsigned int));

    #define LAUNCH_GATHER_KERNEL(PORTS, ALG) \
        kernel_pusch_e2e_deprecode_equalize_gather<PORTS, ALG><<<nof_ofdm_symbols, threads, shared_mem_size, stream>>>( \
            static_cast<const unsigned int*>(d_grid_cbf16), \
            handle->d_ch_estimates, d_re_indices, handle->d_noise_vars, \
            handle->d_deprecode_symbols, handle->d_deprecode_noise_vars, \
            handle->d_eq_noise_var_sum, handle->d_eq_noise_var_count, \
            nof_data_re, grid_stride, symbol_stride, est_stride, \
            cfg.start_symbol, start_subcarrier, cfg.tx_scaling, dft_size, nof_ofdm_symbols)

    #define DISPATCH_GATHER_ALGORITHM(PORTS) \
        switch (cfg.equalizer_algorithm) { \
            case EQUALIZER_ZF: LAUNCH_GATHER_KERNEL(PORTS, EQUALIZER_ZF); break; \
            case EQUALIZER_MMSE: LAUNCH_GATHER_KERNEL(PORTS, EQUALIZER_MMSE); break; \
            case EQUALIZER_MMSE_IRC: default: LAUNCH_GATHER_KERNEL(PORTS, EQUALIZER_MMSE_IRC); break; \
        }

    switch (cfg.nof_rx_ports) {
        case 1: DISPATCH_GATHER_ALGORITHM(1); break;
        case 2: DISPATCH_GATHER_ALGORITHM(2); break;
        case 4: DISPATCH_GATHER_ALGORITHM(4); break;
        case 8: DISPATCH_GATHER_ALGORITHM(8); break;
        default:
            return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    #undef LAUNCH_GATHER_KERNEL
    #undef DISPATCH_GATHER_ALGORITHM

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    VkFFTLaunchParams launch_params = {};
    launch_params.buffer = &handle->vkfft_buffer_ptr;
    VkFFTResult fft_result = VkFFTAppend(&handle->vkfft_plans[dft_size]->app, 1, &launch_params);
    if (fft_result != VKFFT_SUCCESS) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    int demod_threads = 256;
    int demod_blocks = (nof_data_re + demod_threads - 1) / demod_threads;
    kernel_pusch_e2e_deprecode_demod_descramble<<<demod_blocks, demod_threads, 0, stream>>>(
        handle->d_deprecode_symbols,
        handle->d_deprecode_noise_vars,
        static_cast<__half*>(d_llrs_half),
        scramble_c_init,
        nof_data_re,
        cfg.mod_order,
        dft_size);

    err = cudaGetLastError();
    return (err == cudaSuccess) ? NR_LDPC_SUCCESS : NR_LDPC_ERROR_CUDA_FAILED;
}
#endif

nr_ldpc_status_t pusch_e2e_process_full_gpu_with_deprecoding(
    pusch_e2e_handle_t handle,
    const void* d_grid_cbf16,
    void* d_llrs_half,
    const int* d_re_indices,
    int nof_data_re,
    int dft_size,
    int nof_ofdm_symbols,
    cudaStream_t stream)
{
    /* First run the CE/noise portion of the full GPU path. Transform mode
     * must not generate normal CP-OFDM LLRs only to overwrite them below. */
    nr_ldpc_status_t status = pusch_e2e_process_full_gpu_impl(
        handle, d_grid_cbf16, d_llrs_half, d_re_indices, nof_data_re, stream, false);

    if (status != NR_LDPC_SUCCESS) {
        return status;
    }

    /* Now launch the transform deprecoder. The previous call already set up:
     * - Channel estimates in handle->d_ch_estimates
     * - Noise variances in handle->d_noise_vars
     *
     * The transform kernel emits the final LLRs directly.
     */

    const pusch_e2e_config_t& cfg = handle->config;
    int grid_stride = cfg.grid_nof_symbols * cfg.grid_nof_subcarriers;
    int symbol_stride = cfg.grid_nof_subcarriers;
    int est_stride = cfg.nof_prb * 12;  /* nof_re_per_symbol */
    int start_subcarrier = cfg.start_prb * 12;
    if (dft_size < 0 || dft_size > PUSCH_DEPRECODE_MAX_DFT_SIZE) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    uint64_t packed_fft_factors = handle->deprecoding_fft_factors[dft_size];
    int nof_fft_factors = handle->deprecoding_fft_nof_factors[dft_size];
    if ((packed_fft_factors == 0) || (nof_fft_factors <= 0)) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    /* Match the main E2E data descrambler c_init exactly. Do not use
     * scrambler_get_c_init() here: scrambler_configure() masks n_ID to 10 bits
     * for legacy TB usage, while the resident PUSCH path uses the full
     * configured data n_id when generating the on-device sequence. */
    uint32_t scramble_c_init = (static_cast<uint32_t>(cfg.rnti) << 15) + static_cast<uint32_t>(cfg.n_id);

#ifdef OCUDU_PHY_CUDA_ENABLE_VKFFT
    bool try_vkfft =
        (handle->transform_deprecoder_backend == PUSCH_DEPRECODER_VKFFT) ||
        ((handle->transform_deprecoder_backend == PUSCH_DEPRECODER_AUTO) &&
         (dft_size >= handle->vkfft_auto_min_dft_size));
    if (try_vkfft) {
        nr_ldpc_status_t vkfft_status = launch_vkfft_deprecoder(handle,
                                                                d_grid_cbf16,
                                                                d_llrs_half,
                                                                d_re_indices,
                                                                nof_data_re,
                                                                dft_size,
                                                                nof_ofdm_symbols,
                                                                grid_stride,
                                                                symbol_stride,
                                                                est_stride,
                                                                start_subcarrier,
                                                                scramble_c_init,
                                                                stream);
        if (vkfft_status == NR_LDPC_SUCCESS ||
            handle->transform_deprecoder_backend == PUSCH_DEPRECODER_VKFFT) {
            return vkfft_status;
        }
    }
#else
    if (handle->transform_deprecoder_backend == PUSCH_DEPRECODER_VKFFT) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
#endif

    /* Launch one block per OFDM symbol */
    int threads = min(dft_size, 256);
    /* Shared memory: two complex work buffers plus per-thread noise reductions. */
    size_t shared_mem_size = 2 * dft_size * sizeof(cuFloatComplex) +
                             threads * (sizeof(float) + sizeof(unsigned int));

    /* Dispatch kernel based on nof_rx_ports and equalizer_algorithm */
    #define LAUNCH_DEPRECODE_KERNEL(PORTS, ALG) \
        cudaFuncSetAttribute(kernel_pusch_e2e_with_estimates_deprecode<PORTS, ALG>, \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, \
                             static_cast<int>(shared_mem_size)); \
        kernel_pusch_e2e_with_estimates_deprecode<PORTS, ALG><<<nof_ofdm_symbols, threads, shared_mem_size, stream>>>( \
            static_cast<const unsigned int*>(d_grid_cbf16), \
            handle->d_ch_estimates, \
            d_re_indices, \
            handle->d_noise_vars, \
            static_cast<__half*>(d_llrs_half), \
            scramble_c_init, \
            handle->d_eq_noise_var_sum, handle->d_eq_noise_var_count, \
            nof_data_re, cfg.mod_order, grid_stride, symbol_stride, est_stride, \
            cfg.start_symbol, start_subcarrier, cfg.tx_scaling, \
            dft_size, packed_fft_factors, nof_fft_factors, nof_ofdm_symbols)

    #define DISPATCH_DEPRECODE_ALGORITHM(PORTS) \
        switch (cfg.equalizer_algorithm) { \
            case EQUALIZER_ZF: LAUNCH_DEPRECODE_KERNEL(PORTS, EQUALIZER_ZF); break; \
            case EQUALIZER_MMSE: LAUNCH_DEPRECODE_KERNEL(PORTS, EQUALIZER_MMSE); break; \
            case EQUALIZER_MMSE_IRC: default: LAUNCH_DEPRECODE_KERNEL(PORTS, EQUALIZER_MMSE_IRC); break; \
        }

    switch (cfg.nof_rx_ports) {
        case 1: DISPATCH_DEPRECODE_ALGORITHM(1); break;
        case 2: DISPATCH_DEPRECODE_ALGORITHM(2); break;
        case 4: DISPATCH_DEPRECODE_ALGORITHM(4); break;
        case 8: DISPATCH_DEPRECODE_ALGORITHM(8); break;
        default:
            return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    #undef LAUNCH_DEPRECODE_KERNEL
    #undef DISPATCH_DEPRECODE_ALGORITHM

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? NR_LDPC_SUCCESS : NR_LDPC_ERROR_CUDA_FAILED;
}

/* ============================================================================
 * GPU Warm-up for Kernel JIT Compilation
 * ============================================================================ */

nr_ldpc_status_t pusch_e2e_warmup(pusch_e2e_handle_t handle, cudaStream_t stream)
{
    if (!handle) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    /* Configure a minimal PUSCH allocation to trigger kernel compilation.
     * We use the smallest valid configuration:
     * - 1 PRB (12 REs)
     * - 2 DMRS symbols (symbols 2 and 11)
     * - 12 data symbols
     * - QPSK modulation
     * - 1 RX port
     */
    pusch_e2e_config_t warmup_cfg = {};
    warmup_cfg.nof_prb = 1;
    warmup_cfg.start_prb = 0;
    warmup_cfg.nof_symbols = 14;
    warmup_cfg.start_symbol = 0;
    warmup_cfg.dmrs_symbol_mask = 0x0804;  /* Symbols 2 and 11 */
    warmup_cfg.dmrs_type = DMRS_TYPE_1;
    warmup_cfg.nof_cdm_groups_without_data = 2;
    warmup_cfg.mod_order = 2;  /* QPSK */
    warmup_cfg.nof_rx_ports = 1;
    warmup_cfg.n_id = 0;
    warmup_cfg.n_scid = 0;
    warmup_cfg.slot_idx = 0;
    warmup_cfg.dmrs_scaling = 1.0f;
    warmup_cfg.tx_scaling = 1.0f;
    warmup_cfg.grid_nof_subcarriers = 12;
    warmup_cfg.grid_nof_symbols = 14;
    warmup_cfg.equalizer_algorithm = EQUALIZER_MMSE;
    warmup_cfg.scrambling_id = 0;
    warmup_cfg.rnti = 0;

    /* Configure the handle */
    nr_ldpc_status_t status = pusch_e2e_configure(handle, &warmup_cfg);
    if (status != NR_LDPC_SUCCESS) {
        return status;
    }

    /* Allocate minimal dummy buffers */
    size_t grid_size = 14 * 12 * sizeof(unsigned int);  /* cbf16 packed as uint32 */
    size_t re_indices_size = 12 * 12 * sizeof(int);  /* 12 data symbols * 12 REs */
    size_t llrs_size = 12 * 12 * 2 * sizeof(__half); /* QPSK: 2 bits per RE, half precision */

    void* d_grid = nullptr;
    int* d_re_indices = nullptr;
    void* d_llrs = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_grid, grid_size);
    if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;

    err = cudaMalloc(&d_re_indices, re_indices_size);
    if (err != cudaSuccess) {
        cudaFree(d_grid);
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }

    err = cudaMalloc(&d_llrs, llrs_size);
    if (err != cudaSuccess) {
        cudaFree(d_grid);
        cudaFree(d_re_indices);
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }

    /* Zero-fill the buffers */
    cudaMemsetAsync(d_grid, 0, grid_size, stream);
    cudaMemsetAsync(d_re_indices, 0, re_indices_size, stream);
    cudaMemsetAsync(d_llrs, 0, llrs_size, stream);

    /* Run the E2E pipeline - this triggers kernel JIT compilation */
    status = pusch_e2e_process_full_gpu_optimized(handle,
                                        d_grid,
                                        d_llrs,
                                        d_re_indices,
                                        12 * 12,  /* nof_re */
                                        stream);

    /* Wait for completion */
    cudaError_t sync_err = cudaStreamSynchronize(stream);

    /* Clean up */
    cudaFree(d_grid);
    cudaFree(d_re_indices);
    cudaFree(d_llrs);

    /* Mark handle as unconfigured so it will be re-configured for actual use */
    handle->configured = false;

    if (status != NR_LDPC_SUCCESS) {
        return status;
    }
    if (sync_err != cudaSuccess) {
        std::fprintf(stderr,
                     "[OCUDU PHY CUDA] PUSCH E2E warmup process failed: %s\n",
                     cudaGetErrorString(sync_err));
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t pusch_e2e_warmup_transform_deprecoding(pusch_e2e_handle_t handle,
                                                         cudaStream_t stream,
                                                         int nof_prb,
                                                         int nof_rx_ports)
{
    if (!handle) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    nof_prb = std::max(1, std::min(nof_prb, PUSCH_DEPRECODE_MAX_DFT_SIZE / 12));
    while (nof_prb > 1 && handle->deprecoding_fft_nof_factors[nof_prb * 12] <= 0) {
        --nof_prb;
    }
    if (handle->deprecoding_fft_nof_factors[nof_prb * 12] <= 0) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (nof_rx_ports <= 1) {
        nof_rx_ports = 1;
    } else if (nof_rx_ports <= 2) {
        nof_rx_ports = 2;
    } else if (nof_rx_ports <= 4) {
        nof_rx_ports = 4;
    } else {
        nof_rx_ports = 8;
    }

    const int nof_symbols = 14;
    const int dmrs_symbol_mask = 0x0004;
    const int dft_size = nof_prb * 12;
    const int nof_data_symbols = nof_symbols - 1;
    const int nof_data_re = nof_data_symbols * dft_size;

    pusch_e2e_config_t warmup_cfg = {};
    warmup_cfg.nof_prb = nof_prb;
    warmup_cfg.start_prb = 0;
    warmup_cfg.nof_symbols = nof_symbols;
    warmup_cfg.start_symbol = 0;
    warmup_cfg.dmrs_symbol_mask = dmrs_symbol_mask;
    warmup_cfg.dmrs_type = DMRS_TYPE_1;
    warmup_cfg.nof_cdm_groups_without_data = 2;
    warmup_cfg.mod_order = 2;
    warmup_cfg.nof_rx_ports = nof_rx_ports;
    warmup_cfg.nof_tx_layers = 1;
    warmup_cfg.n_id = 0;
    warmup_cfg.n_scid = 0;
    warmup_cfg.slot_idx = 0;
    warmup_cfg.dmrs_scaling = 1.0f;
    warmup_cfg.tx_scaling = 1.0f;
    warmup_cfg.grid_nof_subcarriers = dft_size;
    warmup_cfg.grid_nof_symbols = nof_symbols;
    warmup_cfg.equalizer_algorithm = EQUALIZER_MMSE;
    warmup_cfg.scrambling_id = 0;
    warmup_cfg.rnti = 0;
    warmup_cfg.scs_khz = 30;
    warmup_cfg.use_low_papr_dmrs = 1;
    warmup_cfg.n_rs_id = 0;
    warmup_cfg.enable_evm_metric = 0;

    (void)cudaGetLastError();
    nr_ldpc_status_t status = pusch_e2e_configure(handle, &warmup_cfg);
    if (status != NR_LDPC_SUCCESS) {
        handle->configured = false;
        return status;
    }

    size_t grid_size = (size_t)nof_rx_ports * nof_symbols * dft_size * sizeof(unsigned int);
    size_t re_indices_size = (size_t)nof_data_re * sizeof(int);
    size_t llrs_size = (size_t)nof_data_re * warmup_cfg.mod_order * sizeof(__half);

    void* d_grid = nullptr;
    int* d_re_indices = nullptr;
    void* d_llrs = nullptr;

    cudaError_t err = cudaMalloc(&d_grid, grid_size);
    if (err != cudaSuccess) {
        handle->configured = false;
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }
    err = cudaMalloc(&d_re_indices, re_indices_size);
    if (err != cudaSuccess) {
        cudaFree(d_grid);
        handle->configured = false;
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }
    err = cudaMalloc(&d_llrs, llrs_size);
    if (err != cudaSuccess) {
        cudaFree(d_re_indices);
        cudaFree(d_grid);
        handle->configured = false;
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }

    cudaMemsetAsync(d_grid, 0, grid_size, stream);
    cudaMemsetAsync(d_re_indices, 0, re_indices_size, stream);
    cudaMemsetAsync(d_llrs, 0, llrs_size, stream);

    status = pusch_e2e_process_full_gpu_with_deprecoding(
        handle, d_grid, d_llrs, d_re_indices, nof_data_re, dft_size, nof_data_symbols, stream);
    cudaError_t sync_err = cudaStreamSynchronize(stream);

    cudaFree(d_llrs);
    cudaFree(d_re_indices);
    cudaFree(d_grid);
    handle->configured = false;

    if (status != NR_LDPC_SUCCESS) {
        (void)cudaGetLastError();
        return status;
    }
    if (sync_err != cudaSuccess) {
        (void)cudaGetLastError();
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    (void)cudaGetLastError();
    return NR_LDPC_SUCCESS;
}

/* ============================================================================
 * OPTIMIZED API - Uses fused FP16 kernels for maximum performance
 * ============================================================================ */

nr_ldpc_status_t pusch_e2e_process_full_gpu_optimized(
    pusch_e2e_handle_t handle,
    const void* d_grid_cbf16,
    void* d_llrs_half,
    const int* d_re_indices,
    int nof_data_re,
    cudaStream_t stream)
{
    if (!handle || !handle->configured) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (!d_grid_cbf16 || !d_llrs_half || !d_re_indices) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    const pusch_e2e_config_t& cfg = handle->config;

    /* MIMO support: 1, 2, or 4 layers are supported.
     * Layer count must not exceed port count. */
    if (cfg.nof_tx_layers > cfg.nof_rx_ports) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (cfg.nof_tx_layers != 1 && cfg.nof_tx_layers != 2 && cfg.nof_tx_layers != 4) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    /* Calculate dimensions */
    int nof_re_per_symbol = cfg.nof_prb * 12;
    int grid_stride = cfg.grid_nof_symbols * cfg.grid_nof_subcarriers;
    int symbol_stride = cfg.grid_nof_subcarriers;

    int dmrs_symbol_indices[MAX_DMRS_SYMBOLS];
    int nof_dmrs_symbols = pusch_e2e_get_dmrs_symbol_indices_checked(cfg.dmrs_symbol_mask, dmrs_symbol_indices);
    if (nof_dmrs_symbols <= 0) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    cudaError_t err;

    /* ================================================================
     * ULTRA-OPTIMIZED PATH: Fused LSE+NoiseVar + On-the-fly Scrambling
     *
     * This path eliminates:
     * - Separate noise variance kernel (fused into LSE)
     * - Data scrambling sequence generation kernel (on-the-fly in E2E)
     * - DMRS received buffer (noise computed inline)
     *
     * Kernel launches: DMRS gen → Fused LSE+Noise → Finalize → E2E
     * (4 kernels vs 5 in previous optimized path)
     * ================================================================ */

    /* Allocate FP16 estimate buffer [dmrs_sym, prb, port, layer, 12] for MIMO */
    int nof_layers_fused = (cfg.nof_tx_layers > 0) ? cfg.nof_tx_layers : 1;
    size_t fp16_est_size = nof_dmrs_symbols * nof_re_per_symbol * cfg.nof_rx_ports * nof_layers_fused * sizeof(__half2);
    if (fp16_est_size > handle->estimates_fp16_capacity) {
        if (handle->d_estimates_fp16) cudaFree(handle->d_estimates_fp16);
        err = cudaMalloc(&handle->d_estimates_fp16, fp16_est_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->estimates_fp16_capacity = fp16_est_size;
    }

    /* Allocate noise variance buffer (reused as accumulator) */
    size_t noise_size = cfg.nof_rx_ports * sizeof(float);
    if (noise_size > handle->noise_vars_capacity) {
        if (handle->d_noise_vars) cudaFree(handle->d_noise_vars);
        err = cudaMalloc(&handle->d_noise_vars, noise_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->noise_vars_capacity = noise_size;
    }

    /* d_noise_count is now part of d_accum_block (allocated in pusch_e2e_create) */

    /* Upload DMRS symbol indices to device (cached — skip if mask unchanged) */
    size_t dmrs_indices_needed = nof_dmrs_symbols * sizeof(int);
    if (dmrs_indices_needed > handle->dmrs_indices_capacity) {
        if (handle->d_dmrs_indices) cudaFree(handle->d_dmrs_indices);
        err = cudaMalloc(&handle->d_dmrs_indices, dmrs_indices_needed);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->dmrs_indices_capacity = dmrs_indices_needed;
        handle->cached_dmrs_symbol_mask = -1;  /* Force re-upload after realloc */
    }
    if (cfg.dmrs_symbol_mask != handle->cached_dmrs_symbol_mask) {
        cudaMemcpyAsync(handle->d_dmrs_indices, dmrs_symbol_indices,
                        nof_dmrs_symbols * sizeof(int), cudaMemcpyHostToDevice, stream);
        handle->cached_dmrs_symbol_mask = cfg.dmrs_symbol_mask;
        handle->cached_nof_dmrs_symbols = nof_dmrs_symbols;
    }

    /* ================================================================
     * ULTRA-OPTIMIZED: On-the-fly DMRS + Fused LSE + Noise Variance
     *
     * This eliminates the separate DMRS sequence generation kernel!
     * DMRS pilots are computed on-the-fly using LFSR advancement.
     *
     * Kernel launches: Ultra-Fused LSE → Finalize → E2E (3 kernels!)
     * ================================================================ */
    /* Zero ALL per-iteration accumulators in a single memset (140 bytes).
     * Covers: d_noise_count, d_rsrp_accum, d_epre_accum, d_sinr_noise_accum,
     * d_sinr_rsrp_accum, d_eq_noise_var_sum, d_eq_noise_var_count.
     * d_noise_vars is NOT in this block — it needs a separate mid-processing re-zero
     * before the CV kernel (line ~9955). */
    cudaMemsetAsync(handle->d_accum_block, 0, handle->accum_block_size, stream);

    /* Allocate d_dmrs_c_inits buffer if needed (small: MAX_DMRS_SYMBOLS × 4 bytes) */
    if (!handle->d_dmrs_c_inits) {
        cudaMalloc(&handle->d_dmrs_c_inits, MAX_DMRS_SYMBOLS * sizeof(uint32_t));
    }

    {
        int slot_in_frame = cfg.slot_idx % MAX_SLOTS_PER_FRAME;

        /* Compute and upload c_init values for all DMRS symbols */
        uint32_t h_dmrs_c_inits[MAX_DMRS_SYMBOLS];
        for (int d = 0; d < nof_dmrs_symbols; d++) {
            int dmrs_sym = dmrs_symbol_indices[d];
            h_dmrs_c_inits[d] = dmrs_compute_c_init(slot_in_frame, dmrs_sym, cfg.scrambling_id, cfg.n_scid);
        }
        cudaMemcpyAsync(handle->d_dmrs_c_inits, h_dmrs_c_inits,
                         nof_dmrs_symbols * sizeof(uint32_t),
                         cudaMemcpyHostToDevice, stream);

        /* Single 2D launch for all DMRS symbols — eliminates per-symbol launch overhead */
        dim3 lse_grid(cfg.nof_prb, nof_dmrs_symbols);
        kernel_fused_lse_freq_interp_noise_fp16<<<lse_grid, cfg.nof_rx_ports, 0, stream>>>(
            static_cast<const unsigned int*>(d_grid_cbf16),
            handle->d_dmrs_c_inits,
            handle->d_dmrs_indices,
            handle->d_estimates_fp16,
            handle->d_noise_vars,
            handle->d_noise_count,
            handle->d_rsrp_accum,
            handle->d_epre_accum,
            cfg.nof_prb,
            cfg.nof_rx_ports,
            nof_dmrs_symbols,
            grid_stride,
            symbol_stride,
            cfg.start_prb,
            cfg.dmrs_scaling,
            (int)cfg.dmrs_type);
    }

    /* ================================================================
     * Phase 1b: Frequency-domain smoothing
     *
     * Apply raised cosine FIR filter across frequency to denoise the
     * LSE channel estimates. Operates on pilot positions and
     * re-interpolates to all 12 subcarriers. Must run before noise
     * estimation so cross-validation sees smoothed estimates.
     * ================================================================ */
    if (cfg.nof_prb > 1) {
        dim3 smooth_grid(cfg.nof_prb, nof_dmrs_symbols);
        kernel_fd_smoothing_fp16<<<smooth_grid, cfg.nof_rx_ports, 0, stream>>>(
            handle->d_estimates_fp16,
            cfg.nof_prb,
            cfg.nof_rx_ports,
            nof_dmrs_symbols,
            (int)cfg.dmrs_type);
    }

    /* ================================================================
     * Phase 2: Cross-validation noise estimation + fused SINR/EPRE
     *
     * The LSE residuals give near-zero noise because h = y / pilot, so
     * |y - h*pilot| ≈ 0. Use cross-validation across DMRS symbols instead:
     * variance of h estimates gives true noise variance.
     *
     * SINR residual computation is fused into the CV kernel at pilot
     * positions (sc % 2 == 0) to eliminate a separate SINR kernel launch.
     * ================================================================ */

    /* Resolve pilot bits for fused SINR (needed before CV kernel).
     * With all-14-symbols precomputation, no mask matching is needed — just
     * gather the DMRS symbols for this slot's mask via D2D copies. */
    int sinr_words_per_dmrs_sym = 0;
    uint32_t* sinr_pilot_bits = nullptr;
    if (handle->d_sinr_epre_result && handle->d_epre_accum && handle->d_sinr_noise_accum) {
        int sinr_slot_in_frame = cfg.slot_idx % MAX_SLOTS_PER_FRAME;
        int sinr_max_pilot_prb = cfg.start_prb + cfg.nof_prb;
        int sinr_precomputed_max_prb = handle->precomputed_pilots_per_symbol / dmrs_get_re_per_prb(cfg.dmrs_type);
        bool sinr_can_precompute = handle->dmrs_pilots_precomputed &&
                                   sinr_slot_in_frame < handle->precomputed_nof_slots &&
                                   sinr_max_pilot_prb <= sinr_precomputed_max_prb &&
                                   handle->d_dmrs_pilot_bits;

        if (sinr_can_precompute) {
            /* Gather needed DMRS symbols from all-14 layout via single kernel launch */
            sinr_words_per_dmrs_sym = handle->precomputed_words_per_sym;
            {
                const uint32_t* slot_base = handle->d_precomputed_dmrs_pilots +
                                            sinr_slot_in_frame * handle->precomputed_words_per_slot;
                int gather_threads = ((sinr_words_per_dmrs_sym + 31) / 32) * 32;
                if (gather_threads < 32) gather_threads = 32;
                kernel_gather_dmrs_pilot_bits<<<nof_dmrs_symbols, gather_threads, 0, stream>>>(
                    handle->d_dmrs_pilot_bits, slot_base, handle->d_dmrs_indices,
                    sinr_words_per_dmrs_sym);
            }
            sinr_pilot_bits = handle->d_dmrs_pilot_bits;
        } else if (handle->d_dmrs_c_inits) {
            /* Fallback: generate pilots via kernel (should not happen in normal operation) */
            int nof_pilots_per_sym = cfg.nof_prb * dmrs_get_re_per_prb(cfg.dmrs_type);
            int nof_pilot_words = (nof_pilots_per_sym * nof_dmrs_symbols * 2 + 31) / 32;
            size_t pilot_bits_size = nof_pilot_words * sizeof(uint32_t);
            if (pilot_bits_size > handle->dmrs_pilot_bits_capacity) {
                if (handle->d_dmrs_pilot_bits) cudaFree(handle->d_dmrs_pilot_bits);
                err = cudaMalloc(&handle->d_dmrs_pilot_bits, pilot_bits_size);
                if (err == cudaSuccess) {
                    handle->dmrs_pilot_bits_capacity = pilot_bits_size;
                } else {
                    handle->d_dmrs_pilot_bits = nullptr;
                    handle->dmrs_pilot_bits_capacity = 0;
                }
            }
            if (handle->d_dmrs_pilot_bits) {
                sinr_words_per_dmrs_sym = (nof_pilots_per_sym * 2 + 31) / 32;
                for (int d = 0; d < nof_dmrs_symbols; d++) {
                    int sinr_slot_for_dmrs = cfg.slot_idx % MAX_SLOTS_PER_FRAME;
                    uint64_t term1 = (1ULL << 17) * (14ULL * sinr_slot_for_dmrs + dmrs_symbol_indices[d] + 1) * (2ULL * cfg.scrambling_id + 1);
                    uint64_t term2 = 2ULL * cfg.scrambling_id + cfg.n_scid;
                    uint32_t c_init = static_cast<uint32_t>((term1 + term2) & 0x7FFFFFFF);
                    int nof_threads_needed = (nof_pilots_per_sym + 31) / 32;
                    int pilot_gen_threads = 256;
                    int pilot_gen_blocks = (nof_threads_needed + pilot_gen_threads - 1) / pilot_gen_threads;
                    uint32_t* pilot_bits_offset = handle->d_dmrs_pilot_bits + d * sinr_words_per_dmrs_sym;
                    kernel_generate_dmrs_pilots<<<pilot_gen_blocks, pilot_gen_threads, 0, stream>>>(
                        pilot_bits_offset, c_init, nof_pilots_per_sym);
                }
                sinr_pilot_bits = handle->d_dmrs_pilot_bits;
            }
        }
    }

    /* The normal two-DMRS CSI path overwrites all five result fields:
     * SINR/EPRE/RSRP in the CV/finalize kernel and TA/CFO in
     * kernel_compute_ta_and_cfo. Keep the reset for reduced metrics paths
     * where some fields may otherwise retain stale values. */
    const bool sinr_epre_result_fully_overwritten =
        sinr_pilot_bits && cfg.scs_khz > 0 && nof_dmrs_symbols >= 2;
    static const bool skip_redundant_result_reset =
        !pusch_e2e_env_flag_disabled("OCUDU_PUSCH_SKIP_REDUNDANT_RESULT_RESET");
    if (handle->d_sinr_epre_result &&
        !(skip_redundant_result_reset && sinr_epre_result_fully_overwritten)) {
        kernel_reset_sinr_epre_result<<<1, 1, 0, stream>>>(handle->d_sinr_epre_result);
    }

    static const bool enable_cfo_phasors =
        !pusch_e2e_env_flag_disabled("OCUDU_PUSCH_CFO_PHASORS");
    const bool compute_ta_cfo = sinr_pilot_bits && handle->d_sinr_epre_result && cfg.scs_khz > 0;
    const bool use_cfo_phasors =
        compute_ta_cfo && enable_cfo_phasors && cfg.compensate_cfo && nof_dmrs_symbols >= 2 &&
        handle->d_mimo_cfo_phasors != nullptr && handle->d_symbol_start_times != nullptr;
    float2* cfo_phasors = use_cfo_phasors ? handle->d_mimo_cfo_phasors : nullptr;

    /* TA + CFO via fused phase estimation kernel — launched BEFORE CV noise
     * so that CFO estimate (d_sinr_epre_result[4]) is available for coherent
     * de-rotation in kernel_cross_validation_noise_variance. */
    if (compute_ta_cfo) {
        int pilot_spacing = (cfg.dmrs_type == DMRS_TYPE_1) ? 2 : 3;
        float pilot_spacing_hz = (float)pilot_spacing * (float)cfg.scs_khz * 1000.0f;
        int nof_pilots = cfg.nof_prb * dmrs_get_re_per_prb(cfg.dmrs_type);
        /* 3 pilot arrays × nof_pilots × 2 floats + 8 warps × 4 accumulators for reduction */
        size_t ta_cfo_smem = nof_pilots * 6 * sizeof(float) + 8 * 4 * sizeof(float);

        /* Compute time gap between first and last DMRS symbols (for CFO).
         * Matches CPU port_channel_estimator_average_impl::initialize_symbol_start_epochs():
         *   epoch[j] - epoch[j-1] = cp_j_seconds * SCS_Hz + 1.0  (in SCS periods)
         *   delta_t = (epoch[last] - epoch[first]) / SCS_Hz       (in seconds)
         * Normal CP: cp_len_kappa(j) = (144 >> mu) for regular, + 16 for j=0 or j=7*2^mu.
         * cp_seconds = cp_len_kappa * kappa * T_c = cp_len_kappa / 30720000. */
        float* d_cfo_ptr = nullptr;
        float delta_t_s = 0.0f;
        if (nof_dmrs_symbols >= 2) {
            int mu = 0;
            if (cfg.scs_khz == 30) mu = 1;
            else if (cfg.scs_khz == 60) mu = 2;
            else if (cfg.scs_khz == 120) mu = 3;
            float scs_hz = (float)cfg.scs_khz * 1000.0f;
            int half_slot_sym = 7 * (1 << mu);  /* Extended CP symbol index */
            for (int j = dmrs_symbol_indices[0] + 1; j <= dmrs_symbol_indices[nof_dmrs_symbols - 1]; j++) {
                int cp_kappa = 144 >> mu;
                if (j == 0 || j == half_slot_sym) cp_kappa += 16;
                float cp_s = (float)cp_kappa / 30720000.0f;
                delta_t_s += cp_s + 1.0f / scs_hz;
            }
            d_cfo_ptr = handle->d_sinr_epre_result + 4;
        }

        /* d_ta_cfo_accum already zeroed by d_accum_block memset above.
         * Launch one block per port for parallel execution across SMs. */
        kernel_compute_ta_and_cfo<<<cfg.nof_rx_ports, 256, ta_cfo_smem, stream>>>(
            handle->d_estimates_fp16,
            handle->d_sinr_epre_result + 3,   /* TA output */
            d_cfo_ptr,                         /* CFO output (NULL if < 2 DMRS) */
            handle->d_ta_cfo_accum,            /* cross-port accumulators */
            cfg.nof_prb, cfg.nof_rx_ports, nof_dmrs_symbols,
            pilot_spacing_hz,
            delta_t_s,
            handle->d_symbol_start_times,
            handle->d_dmrs_indices,
            cfo_phasors,
            cfg.start_symbol,
            (int)cfg.dmrs_type);
    }

    /* d_cv_noise_accum and d_cv_done_counter are already zeroed by the d_accum_block
     * memset above — no separate re-zero of d_noise_vars needed for the CV path. */

    if (nof_dmrs_symbols >= 2) {
        int cv_threads = 12 * cfg.nof_rx_ports;
        int cv_blocks = cfg.nof_prb;

        const float* cv_cfo_ptr = (cfg.compensate_cfo && nof_dmrs_symbols >= 2) ? (handle->d_sinr_epre_result + 4) : nullptr;

        /* Fused CV + finalize: last block normalizes noise and computes SINR/EPRE.
         * Eliminates separate cudaMemsetAsync + kernel_finalize_cross_validation_noise launch. */
        kernel_cross_validation_noise_variance<<<cv_blocks, cv_threads, 0, stream>>>(
            handle->d_estimates_fp16,
            handle->d_cv_noise_accum,
            static_cast<const unsigned int*>(d_grid_cbf16),
            sinr_pilot_bits,
            handle->d_dmrs_indices,
            sinr_pilot_bits ? handle->d_sinr_noise_accum : nullptr,
            sinr_pilot_bits ? handle->d_sinr_rsrp_accum : nullptr,
            cfg.nof_prb,
            cfg.nof_rx_ports,
            nof_dmrs_symbols,
            grid_stride,
            symbol_stride,
            cfg.start_prb,
            cfg.dmrs_scaling,
            sinr_words_per_dmrs_sym,
            cv_cfo_ptr,
            handle->d_symbol_start_times,
            cfo_phasors,
            /* Fused finalize params */
            handle->d_noise_vars,
            handle->d_cv_done_counter,
            sinr_pilot_bits ? handle->d_epre_accum : nullptr,
            sinr_pilot_bits ? handle->d_sinr_epre_result : nullptr,
            cfg.noise_mode,
            (int)cfg.dmrs_type);
    } else {
        /* Single DMRS symbol — cannot cross-validate across time. Match the CPU
         * estimator by computing pilot-residual noise from the smoothed channel
         * estimates instead of using the raw LSE self-residual/floor path. */
        if (sinr_pilot_bits) {
            dim3 residual_grid(cfg.nof_prb);
            int residual_threads = cfg.nof_rx_ports * dmrs_get_re_per_prb(cfg.dmrs_type);
            kernel_compute_residual_sinr_epre_v3<<<residual_grid, residual_threads, 0, stream>>>(
                static_cast<const unsigned int*>(d_grid_cbf16),
                handle->d_estimates_fp16,
                sinr_pilot_bits,
                handle->d_dmrs_indices,
                handle->d_sinr_noise_accum,
                handle->d_sinr_rsrp_accum,
                cfg.nof_prb,
                cfg.nof_rx_ports,
                nof_dmrs_symbols,
                grid_stride,
                symbol_stride,
                cfg.start_prb,
                cfg.dmrs_scaling,
                sinr_words_per_dmrs_sym,
                (int)cfg.dmrs_type);
            kernel_finalize_sinr_epre<<<1, 1, 0, stream>>>(
                handle->d_sinr_noise_accum,
                handle->d_sinr_rsrp_accum,
                handle->d_epre_accum,
                handle->d_noise_vars,
                handle->d_sinr_epre_result,
                cfg.nof_prb,
                cfg.nof_rx_ports,
                nof_dmrs_symbols,
                cfg.dmrs_scaling,
                (int)cfg.dmrs_type);
        } else {
            cudaMemsetAsync(handle->d_noise_vars, 0, cfg.nof_rx_ports * sizeof(float), stream);
            kernel_finalize_noise_variance<<<1, cfg.nof_rx_ports, 0, stream>>>(
                handle->d_noise_vars,
                handle->d_noise_count,
                cfg.nof_rx_ports,
                cfg.dmrs_scaling,
                nof_dmrs_symbols);
        }
    }

    /* ================================================================
     * Phase 3: OPTIMIZED E2E kernel with WARP-COOPERATIVE scrambling
     *
     * Optimizations over previous version:
     * 1. Warp-cooperative scrambling: Lane 0 generates bits for 32 threads
     * 2. Block-level noise variance reduction (fewer global atomics)
     * 3. Template on NOF_DMRS for compile-time loop unrolling
     * 4. Vectorized half2 LLR writes
     * ================================================================ */
    /* Compute data scrambling c_init per TS 38.211 section 6.3.1.1:
     * c_init = n_RNTI * 2^15 + q * 2^14 + n_ID
     * Note: Slot index is NOT part of PUSCH/PDSCH data scrambling c_init.
     * Slot is only used for DMRS sequence generation (different section). */
    uint32_t data_c_init = (cfg.rnti << 15) + (0 << 14) + cfg.n_id;
    int nof_data_bits = nof_data_re * cfg.mod_order;
    static const bool enable_scramble_cache =
        !pusch_e2e_env_flag_disabled("OCUDU_PUSCH_SCRAMBLE_CACHE");
    static const unsigned scramble_cache_min_re =
        pusch_e2e_env_unsigned("OCUDU_PUSCH_SCRAMBLE_CACHE_MIN_RE", 10000);
    const uint32_t* data_scramble_seq =
        (enable_scramble_cache && nof_data_re >= static_cast<int>(scramble_cache_min_re) &&
         handle->d_scrambling_seq &&
         handle->cached_data_scrambling_c_init == data_c_init &&
         handle->cached_data_scrambling_bits >= nof_data_bits) ? handle->d_scrambling_seq : nullptr;

    int threads_e2e = 256;
    int blocks_e2e = (nof_data_re + threads_e2e - 1) / threads_e2e;

    /* d_eq_noise_var_sum and d_eq_noise_var_count already zeroed by d_accum_block memset */
    const float* e2e_cfo_ptr =
        (cfg.compensate_cfo && nof_dmrs_symbols >= 2) ? (handle->d_sinr_epre_result + 4) : nullptr;
    const float2* e2e_cfo_phasors = (e2e_cfo_ptr != nullptr) ? cfo_phasors : nullptr;

    /* Use optimized warp-cooperative kernel with templated DMRS count and time interp mode.
     * TIME_AVG=true: average all DMRS estimates (default, matches CPU).
     * TIME_AVG=false: linear interpolation between bracketing DMRS symbols. */
    #define LAUNCH_WARP_COOP_E2E(PORTS, ALG, DMRS, TAVG, MOD) \
        kernel_fused_e2e_warp_cooperative_fp16<PORTS, ALG, DMRS, TAVG, MOD><<<blocks_e2e, threads_e2e, 0, stream>>>( \
            static_cast<const unsigned int*>(d_grid_cbf16), \
            handle->d_estimates_fp16, \
            d_re_indices, \
            handle->d_noise_vars, \
            static_cast<__half*>(d_llrs_half), \
            data_scramble_seq, \
            data_c_init, \
            handle->d_eq_noise_var_sum, \
            handle->d_eq_noise_var_count, \
            cfg.enable_evm_metric ? handle->d_evm_error_sum : nullptr, \
            cfg.enable_evm_metric ? handle->d_evm_symbol_count : nullptr, \
            nof_data_re, cfg.nof_prb, \
            grid_stride, symbol_stride, cfg.start_symbol, cfg.start_prb * 12, cfg.tx_scaling, \
            handle->d_dmrs_indices, \
            e2e_cfo_ptr, \
            handle->d_symbol_start_times, \
            e2e_cfo_phasors, \
            handle->d_eq_symbols_out, \
            handle->d_eq_noise_var_out)

    #define DISPATCH_MOD(PORTS, ALG, DMRS, TAVG) \
        switch (cfg.mod_order) { \
            case 2: LAUNCH_WARP_COOP_E2E(PORTS, ALG, DMRS, TAVG, 2); break; \
            case 4: LAUNCH_WARP_COOP_E2E(PORTS, ALG, DMRS, TAVG, 4); break; \
            case 6: LAUNCH_WARP_COOP_E2E(PORTS, ALG, DMRS, TAVG, 6); break; \
            case 8: LAUNCH_WARP_COOP_E2E(PORTS, ALG, DMRS, TAVG, 8); break; \
            default: return NR_LDPC_ERROR_INVALID_CONFIG; \
        }

    /* Dispatch based on DMRS count for compile-time unrolling */
    #define DISPATCH_DMRS(PORTS, ALG, TAVG) \
        switch (nof_dmrs_symbols) { \
            case 1: DISPATCH_MOD(PORTS, ALG, 1, TAVG); break; \
            case 2: DISPATCH_MOD(PORTS, ALG, 2, TAVG); break; \
            case 3: DISPATCH_MOD(PORTS, ALG, 3, TAVG); break; \
            case 4: default: DISPATCH_MOD(PORTS, ALG, 4, TAVG); break; \
        }

    #define DISPATCH_ALG(PORTS, TAVG) \
        switch (cfg.equalizer_algorithm) { \
            case EQUALIZER_ZF: DISPATCH_DMRS(PORTS, EQUALIZER_ZF, TAVG); break; \
            case EQUALIZER_MMSE: DISPATCH_DMRS(PORTS, EQUALIZER_MMSE, TAVG); break; \
            case EQUALIZER_MMSE_IRC: default: DISPATCH_DMRS(PORTS, EQUALIZER_MMSE_IRC, TAVG); break; \
        }

    #define DISPATCH_PORTS(TAVG) \
        switch (cfg.nof_rx_ports) { \
            case 1: DISPATCH_ALG(1, TAVG); break; \
            case 2: DISPATCH_ALG(2, TAVG); break; \
            case 4: DISPATCH_ALG(4, TAVG); break; \
            case 8: DISPATCH_ALG(8, TAVG); break; \
            default: return NR_LDPC_ERROR_INVALID_CONFIG; \
        }

    if (cfg.time_interp_mode == 1) {
        DISPATCH_PORTS(false);  /* Linear interpolation */
    } else {
        DISPATCH_PORTS(true);   /* Time averaging (default) */
    }

    #undef LAUNCH_WARP_COOP_E2E
    #undef DISPATCH_MOD
    #undef DISPATCH_DMRS
    #undef DISPATCH_ALG
    #undef DISPATCH_PORTS

    err = cudaGetLastError();
    return (err == cudaSuccess) ? NR_LDPC_SUCCESS : NR_LDPC_ERROR_CUDA_FAILED;
}

nr_ldpc_status_t pusch_e2e_process_full_gpu_optimized_mimo_half(
    pusch_e2e_handle_t handle,
    const void* d_grid_cbf16,
    void* d_llrs_half,
    const int* d_re_indices,
    int nof_data_re,
    cudaStream_t stream)
{
    if (!handle || !handle->configured) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (!d_grid_cbf16 || !d_llrs_half || !d_re_indices) return NR_LDPC_ERROR_INVALID_CONFIG;

    const pusch_e2e_config_t& cfg = handle->config;
    if ((cfg.nof_tx_layers != 2 && cfg.nof_tx_layers != 3 && cfg.nof_tx_layers != 4) ||
        cfg.nof_tx_layers > cfg.nof_rx_ports) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (cfg.mod_order != 2 && cfg.mod_order != 4 && cfg.mod_order != 6 && cfg.mod_order != 8) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    int nof_re_per_symbol = cfg.nof_prb * 12;
    int grid_stride = cfg.grid_nof_symbols * cfg.grid_nof_subcarriers;
    int symbol_stride = cfg.grid_nof_subcarriers;

    int dmrs_symbol_indices[MAX_DMRS_SYMBOLS];
    int nof_dmrs_symbols = pusch_e2e_get_dmrs_symbol_indices_checked(cfg.dmrs_symbol_mask, dmrs_symbol_indices);
    if (nof_dmrs_symbols <= 0) return NR_LDPC_ERROR_INVALID_CONFIG;

    cudaError_t err;
    int nof_layers = cfg.nof_tx_layers;
    size_t fp16_est_size = nof_dmrs_symbols * nof_re_per_symbol * cfg.nof_rx_ports * nof_layers * sizeof(__half2);
    if (fp16_est_size > handle->estimates_fp16_capacity) {
        if (handle->d_estimates_fp16) cudaFree(handle->d_estimates_fp16);
        err = cudaMalloc(&handle->d_estimates_fp16, fp16_est_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->estimates_fp16_capacity = fp16_est_size;
    }

    size_t noise_size = cfg.nof_rx_ports * sizeof(float);
    if (noise_size > handle->noise_vars_capacity) {
        if (handle->d_noise_vars) cudaFree(handle->d_noise_vars);
        err = cudaMalloc(&handle->d_noise_vars, noise_size);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->noise_vars_capacity = noise_size;
    }
    if (handle->d_sinr_epre_result) {
        kernel_reset_sinr_epre_result<<<1, 1, 0, stream>>>(handle->d_sinr_epre_result);
    }

    if (!handle->d_noise_count) {
        err = cudaMalloc(&handle->d_noise_count, sizeof(unsigned int));
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
    }

    size_t dmrs_indices_needed = nof_dmrs_symbols * sizeof(int);
    if (dmrs_indices_needed > handle->dmrs_indices_capacity) {
        if (handle->d_dmrs_indices) cudaFree(handle->d_dmrs_indices);
        err = cudaMalloc(&handle->d_dmrs_indices, dmrs_indices_needed);
        if (err != cudaSuccess) return NR_LDPC_ERROR_ALLOC_FAILED;
        handle->dmrs_indices_capacity = dmrs_indices_needed;
    }
    cudaMemcpy(handle->d_dmrs_indices, dmrs_symbol_indices, dmrs_indices_needed, cudaMemcpyHostToDevice);

    int slot_in_frame = cfg.slot_idx % MAX_SLOTS_PER_FRAME;
    int max_pilot_prb = cfg.start_prb + cfg.nof_prb;
    int precomputed_max_prb = handle->precomputed_pilots_per_symbol / dmrs_get_re_per_prb(cfg.dmrs_type);
    bool can_use_precomputed = handle->dmrs_pilots_precomputed &&
                               slot_in_frame < handle->precomputed_nof_slots &&
                               max_pilot_prb <= precomputed_max_prb &&
                               handle->d_dmrs_pilot_bits;

    uint32_t* d_pilot_bits_for_slot = nullptr;
    int words_per_dmrs_sym_stride = 0;
    if (can_use_precomputed) {
        words_per_dmrs_sym_stride = handle->precomputed_words_per_sym;
        const uint32_t* slot_base = handle->d_precomputed_dmrs_pilots +
                                    slot_in_frame * handle->precomputed_words_per_slot;
        int gather_threads = ((words_per_dmrs_sym_stride + 31) / 32) * 32;
        if (gather_threads < 32) gather_threads = 32;
        kernel_gather_dmrs_pilot_bits<<<nof_dmrs_symbols, gather_threads, 0, stream>>>(
            handle->d_dmrs_pilot_bits, slot_base, handle->d_dmrs_indices, words_per_dmrs_sym_stride);
        d_pilot_bits_for_slot = handle->d_dmrs_pilot_bits;
    } else {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    dim3 lse_grid(cfg.nof_prb, nof_dmrs_symbols);
    int lse_threads = cfg.nof_rx_ports;
    if (cfg.nof_tx_layers == 2) {
        kernel_mimo_lse_2layer_fp16<<<lse_grid, lse_threads, 0, stream>>>(
            static_cast<const unsigned int*>(d_grid_cbf16),
            d_pilot_bits_for_slot,
            handle->d_dmrs_indices,
            handle->d_estimates_fp16,
            cfg.nof_prb,
            cfg.nof_rx_ports,
            nof_dmrs_symbols,
            grid_stride,
            symbol_stride,
            cfg.start_prb,
            cfg.dmrs_scaling,
            words_per_dmrs_sym_stride,
            (int)cfg.dmrs_type);
    } else if (cfg.nof_tx_layers == 3) {
        kernel_mimo_lse_3layer_fp16<<<lse_grid, lse_threads, 0, stream>>>(
            static_cast<const unsigned int*>(d_grid_cbf16),
            d_pilot_bits_for_slot,
            handle->d_dmrs_indices,
            handle->d_estimates_fp16,
            cfg.nof_prb,
            cfg.nof_rx_ports,
            nof_dmrs_symbols,
            grid_stride,
            symbol_stride,
            cfg.start_prb,
            cfg.dmrs_scaling,
            words_per_dmrs_sym_stride,
            (int)cfg.dmrs_type);
    } else {
        kernel_mimo_lse_4layer_fp16<<<lse_grid, lse_threads, 0, stream>>>(
            static_cast<const unsigned int*>(d_grid_cbf16),
            d_pilot_bits_for_slot,
            handle->d_dmrs_indices,
            handle->d_estimates_fp16,
            cfg.nof_prb,
            cfg.nof_rx_ports,
            nof_dmrs_symbols,
            grid_stride,
            symbol_stride,
            cfg.start_prb,
            cfg.dmrs_scaling,
            words_per_dmrs_sym_stride,
            (int)cfg.dmrs_type);
    }

    if (cfg.nof_prb > 1) {
        dim3 smooth_grid(cfg.nof_prb, nof_dmrs_symbols, cfg.nof_tx_layers);
        kernel_mimo_fd_smoothing_fp16<<<smooth_grid, cfg.nof_rx_ports, 0, stream>>>(
            handle->d_estimates_fp16,
            cfg.nof_prb,
            cfg.nof_rx_ports,
            cfg.nof_tx_layers,
            nof_dmrs_symbols,
            (int)cfg.dmrs_type);
    }

    /* Reset the contiguous MIMO accumulator tail once:
     * eq SINR sum/count, EVM sum/count, TA/CFO accum, CV noise accum, CV done counter.
     * d_noise_vars is fully overwritten by kernel_mimo_dmrs_residual_noise_variance. */
    const size_t mimo_accum_reset_bytes =
        2 * sizeof(float) + 2 * sizeof(unsigned int) + 6 * sizeof(float) +
        8 * sizeof(float) + sizeof(unsigned int);
    cudaMemsetAsync(handle->d_eq_noise_var_sum, 0, mimo_accum_reset_bytes, stream);

    static const bool enable_mimo_csi_metrics =
        !pusch_e2e_env_flag_disabled("OCUDU_PUSCH_MIMO_CSI_METRICS");
    const bool compute_mimo_csi =
        enable_mimo_csi_metrics && handle->d_sinr_epre_result && handle->d_ta_cfo_accum && cfg.scs_khz > 0;
    const float* mimo_cfo_ptr =
        (compute_mimo_csi && cfg.compensate_cfo && nof_dmrs_symbols >= 2) ? (handle->d_sinr_epre_result + 4) : nullptr;
    static const bool enable_mimo_cfo_phasors =
        !pusch_e2e_env_flag_disabled("OCUDU_PUSCH_CFO_PHASORS");
    const bool use_mimo_cfo_phasors =
        enable_mimo_cfo_phasors && mimo_cfo_ptr != nullptr &&
        handle->d_mimo_cfo_phasors != nullptr && handle->d_symbol_start_times != nullptr;
    float2* mimo_cfo_phasors = use_mimo_cfo_phasors ? handle->d_mimo_cfo_phasors : nullptr;

    if (compute_mimo_csi) {
        int pilot_spacing = (cfg.dmrs_type == DMRS_TYPE_1) ? 2 : 3;
        float pilot_spacing_hz = (float)pilot_spacing * (float)cfg.scs_khz * 1000.0f;
        float delta_t_s = 0.0f;
        if (nof_dmrs_symbols >= 2) {
            int mu = 0;
            if (cfg.scs_khz == 30) mu = 1;
            else if (cfg.scs_khz == 60) mu = 2;
            else if (cfg.scs_khz == 120) mu = 3;
            float scs_hz = (float)cfg.scs_khz * 1000.0f;
            int half_slot_sym = 7 * (1 << mu);
            for (int j = dmrs_symbol_indices[0] + 1; j <= dmrs_symbol_indices[nof_dmrs_symbols - 1]; j++) {
                int cp_kappa = 144 >> mu;
                if (j == 0 || j == half_slot_sym) cp_kappa += 16;
                float cp_s = (float)cp_kappa / 30720000.0f;
                delta_t_s += cp_s + 1.0f / scs_hz;
            }
        }

        kernel_compute_mimo_csi_from_estimates<<<1, 256, 0, stream>>>(
            static_cast<const unsigned int*>(d_grid_cbf16),
            handle->d_estimates_fp16,
            handle->d_dmrs_indices,
            handle->d_sinr_epre_result,
            handle->d_ta_cfo_accum,
            cfg.nof_prb,
            cfg.nof_rx_ports,
            cfg.nof_tx_layers,
            nof_dmrs_symbols,
            grid_stride,
            symbol_stride,
            cfg.start_prb,
            pilot_spacing_hz,
            delta_t_s,
            handle->d_symbol_start_times,
            mimo_cfo_phasors,
            cfg.start_symbol,
            cfg.dmrs_scaling,
            (int)cfg.dmrs_type);
    }

    int nof_cdm = (cfg.nof_tx_layers + 1) / 2;
    int nof_dmrs_pilots_per_cdm = dmrs_get_re_per_prb(cfg.dmrs_type);
    dim3 residual_grid(cfg.nof_prb, cfg.nof_rx_ports);
    kernel_mimo_dmrs_residual_noise_variance<<<residual_grid, nof_dmrs_pilots_per_cdm * nof_cdm, 0, stream>>>(
        static_cast<const unsigned int*>(d_grid_cbf16),
        d_pilot_bits_for_slot,
        handle->d_dmrs_indices,
        handle->d_estimates_fp16,
        handle->d_cv_noise_accum,
        cfg.nof_prb,
        cfg.nof_rx_ports,
        cfg.nof_tx_layers,
        nof_dmrs_symbols,
        grid_stride,
        symbol_stride,
        cfg.start_prb,
        cfg.dmrs_scaling,
        words_per_dmrs_sym_stride,
        handle->d_noise_vars,
        handle->d_cv_done_counter,
        mimo_cfo_ptr,
        handle->d_symbol_start_times,
        mimo_cfo_phasors,
        (int)cfg.dmrs_type);

    if (!handle->d_scrambling_seq) return NR_LDPC_ERROR_INVALID_CONFIG;

    int threads_e2e = 256;
    int blocks_e2e = (nof_data_re + threads_e2e - 1) / threads_e2e;

    #define LAUNCH_MIMO_HALF_2L(PORTS, ALG) \
        kernel_mimo_2layer_fp16<PORTS, ALG><<<blocks_e2e, threads_e2e, 0, stream>>>( \
            static_cast<const unsigned int*>(d_grid_cbf16), \
            handle->d_estimates_fp16, d_re_indices, handle->d_noise_vars, \
            static_cast<__half*>(d_llrs_half), handle->d_scrambling_seq, \
            handle->d_eq_noise_var_sum, handle->d_eq_noise_var_count, \
            cfg.enable_evm_metric ? handle->d_evm_error_sum : nullptr, \
            cfg.enable_evm_metric ? handle->d_evm_symbol_count : nullptr, \
            nof_data_re, nof_dmrs_symbols, cfg.nof_prb, cfg.mod_order, \
            grid_stride, symbol_stride, cfg.start_symbol, cfg.start_prb * 12, cfg.tx_scaling, \
            handle->d_dmrs_indices, \
            mimo_cfo_ptr, \
            handle->d_symbol_start_times, \
            mimo_cfo_phasors, \
            handle->d_eq_symbols_out, handle->d_eq_noise_var_out)

    #define LAUNCH_MIMO_HALF_4L(PORTS, ALG) \
        kernel_mimo_4layer_fp16<PORTS, ALG><<<blocks_e2e, threads_e2e, 0, stream>>>( \
            static_cast<const unsigned int*>(d_grid_cbf16), \
            handle->d_estimates_fp16, d_re_indices, handle->d_noise_vars, \
            static_cast<__half*>(d_llrs_half), handle->d_scrambling_seq, \
            handle->d_eq_noise_var_sum, handle->d_eq_noise_var_count, \
            cfg.enable_evm_metric ? handle->d_evm_error_sum : nullptr, \
            cfg.enable_evm_metric ? handle->d_evm_symbol_count : nullptr, \
            nof_data_re, nof_dmrs_symbols, cfg.nof_prb, cfg.mod_order, \
            grid_stride, symbol_stride, cfg.start_symbol, cfg.start_prb * 12, cfg.tx_scaling, \
            handle->d_dmrs_indices, \
            mimo_cfo_ptr, \
            handle->d_symbol_start_times, \
            mimo_cfo_phasors, \
            handle->d_eq_symbols_out, handle->d_eq_noise_var_out)

    #define LAUNCH_MIMO_HALF_3L(PORTS, ALG) \
        kernel_mimo_3layer_fp16<PORTS, ALG><<<blocks_e2e, threads_e2e, 0, stream>>>( \
            static_cast<const unsigned int*>(d_grid_cbf16), \
            handle->d_estimates_fp16, d_re_indices, handle->d_noise_vars, \
            static_cast<__half*>(d_llrs_half), handle->d_scrambling_seq, \
            handle->d_eq_noise_var_sum, handle->d_eq_noise_var_count, \
            cfg.enable_evm_metric ? handle->d_evm_error_sum : nullptr, \
            cfg.enable_evm_metric ? handle->d_evm_symbol_count : nullptr, \
            nof_data_re, nof_dmrs_symbols, cfg.nof_prb, cfg.mod_order, \
            grid_stride, symbol_stride, cfg.start_symbol, cfg.start_prb * 12, cfg.tx_scaling, \
            handle->d_dmrs_indices, \
            mimo_cfo_ptr, \
            handle->d_symbol_start_times, \
            mimo_cfo_phasors, \
            handle->d_eq_symbols_out, handle->d_eq_noise_var_out)

    #define DISPATCH_ALG_2L(PORTS) \
        switch (cfg.equalizer_algorithm) { \
            case EQUALIZER_ZF: LAUNCH_MIMO_HALF_2L(PORTS, EQUALIZER_ZF); break; \
            case EQUALIZER_MMSE: LAUNCH_MIMO_HALF_2L(PORTS, EQUALIZER_MMSE); break; \
            case EQUALIZER_MMSE_IRC: default: LAUNCH_MIMO_HALF_2L(PORTS, EQUALIZER_MMSE_IRC); break; \
        }

    #define DISPATCH_ALG_4L(PORTS) \
        switch (cfg.equalizer_algorithm) { \
            case EQUALIZER_ZF: LAUNCH_MIMO_HALF_4L(PORTS, EQUALIZER_ZF); break; \
            case EQUALIZER_MMSE: LAUNCH_MIMO_HALF_4L(PORTS, EQUALIZER_MMSE); break; \
            case EQUALIZER_MMSE_IRC: default: LAUNCH_MIMO_HALF_4L(PORTS, EQUALIZER_MMSE_IRC); break; \
        }

    #define DISPATCH_ALG_3L(PORTS) \
        switch (cfg.equalizer_algorithm) { \
            case EQUALIZER_ZF: LAUNCH_MIMO_HALF_3L(PORTS, EQUALIZER_ZF); break; \
            case EQUALIZER_MMSE: LAUNCH_MIMO_HALF_3L(PORTS, EQUALIZER_MMSE); break; \
            case EQUALIZER_MMSE_IRC: default: LAUNCH_MIMO_HALF_3L(PORTS, EQUALIZER_MMSE_IRC); break; \
        }

    if (cfg.nof_tx_layers == 2) {
        switch (cfg.nof_rx_ports) {
            case 2: DISPATCH_ALG_2L(2); break;
            case 4: DISPATCH_ALG_2L(4); break;
            case 8: DISPATCH_ALG_2L(8); break;
            default: return NR_LDPC_ERROR_INVALID_CONFIG;
        }
    } else if (cfg.nof_tx_layers == 3) {
        switch (cfg.nof_rx_ports) {
            case 4: DISPATCH_ALG_3L(4); break;
            case 8: DISPATCH_ALG_3L(8); break;
            default: return NR_LDPC_ERROR_INVALID_CONFIG;
        }
    } else {
        switch (cfg.nof_rx_ports) {
            case 4: DISPATCH_ALG_4L(4); break;
            case 8: DISPATCH_ALG_4L(8); break;
            default: return NR_LDPC_ERROR_INVALID_CONFIG;
        }
    }

    #undef LAUNCH_MIMO_HALF_2L
    #undef LAUNCH_MIMO_HALF_3L
    #undef LAUNCH_MIMO_HALF_4L
    #undef DISPATCH_ALG_2L
    #undef DISPATCH_ALG_3L
    #undef DISPATCH_ALG_4L

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[OCUDU PHY CUDA] E2E MIMO FP16 kernel error: %s (%d)\n",
                cudaGetErrorString(err), static_cast<int>(err));
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    return NR_LDPC_SUCCESS;
}
