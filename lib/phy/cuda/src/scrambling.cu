/**
 * @file scrambling.cu
 * @brief 5G NR Scrambling Implementation per 3GPP TS 38.211
 *
 * Implements Gold sequence generation and scrambling/descrambling
 * for PDSCH/PUSCH using CUDA acceleration.
 *
 * Gold sequence: c(n) = (x1(n+Nc) + x2(n+Nc)) mod 2
 * where:
 *   x1(n+31) = (x1(n+3) + x1(n)) mod 2
 *   x2(n+31) = (x2(n+3) + x2(n+2) + x2(n+1) + x2(n)) mod 2
 *   Nc = 1600
 *
 * OPTIMIZATION: Uses matrix exponentiation for O(log N) LFSR advancement
 * instead of O(N) sequential stepping. Pre-computed jump tables stored
 * in constant memory enable 50-100x speedup for large sequences.
 */

#include "scrambling.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <new>
#include <vector>

// ============================================================================
// Constants
// ============================================================================

#define NC_SKIP 1600  // Pre-skip cycles for Gold sequence
#define LFSR_BITS 31  // LFSR state size
#define MAX_JUMP_POWER 24  // Supports up to 2^24 = 16M bit advancement

// ============================================================================
// LFSR Jump Tables (Matrix Exponentiation)
// ============================================================================
//
// For an LFSR, state transition is linear over GF(2):
//   new_state = M × old_state  (matrix multiply mod 2)
//
// To advance N steps: state_N = M^N × state_0
// Using binary exponentiation: pre-compute M^1, M^2, M^4, M^8, ...
// Then compose: M^N = M^(2^a) × M^(2^b) × ... for each set bit in N
//
// Each matrix row is stored as a 32-bit mask indicating which input bits
// XOR together to produce that output bit.
//
// Memory: 2 LFSRs × 25 powers × 31 rows × 4 bytes = 6.2 KB constant memory

__constant__ uint32_t d_x1_jump[MAX_JUMP_POWER + 1][LFSR_BITS];
__constant__ uint32_t d_x2_jump[MAX_JUMP_POWER + 1][LFSR_BITS];

// Flag to track if jump tables are initialized
static bool g_jump_tables_initialized = false;
static bool g_jump_tables_available = true;

// ============================================================================
// Host: Matrix Operations for Jump Table Computation
// ============================================================================

/**
 * @brief Multiply two 31×31 GF(2) matrices
 * Each matrix is 31 uint32_t values (rows), bits are columns
 */
static void matrix_multiply_gf2(const uint32_t A[LFSR_BITS],
                                 const uint32_t B[LFSR_BITS],
                                 uint32_t C[LFSR_BITS]) {
    // First, transpose B for efficient column access
    uint32_t B_T[LFSR_BITS] = {0};
    for (int i = 0; i < LFSR_BITS; i++) {
        for (int j = 0; j < LFSR_BITS; j++) {
            if (B[j] & (1u << i)) {
                B_T[i] |= (1u << j);
            }
        }
    }

    // C[i][j] = popcount(A[i] & B_T[j]) mod 2
    for (int i = 0; i < LFSR_BITS; i++) {
        C[i] = 0;
        for (int j = 0; j < LFSR_BITS; j++) {
            uint32_t dot = A[i] & B_T[j];
            // Count bits, if odd then set bit j
            int count = __builtin_popcount(dot);
            if (count & 1) {
                C[i] |= (1u << j);
            }
        }
    }
}

/**
 * @brief Initialize the base transition matrix for x1 LFSR
 * x1(n+31) = x1(n+3) XOR x1(n)
 * State stored as: bit i = x1(n+i)
 * After one step: new[i] = old[i+1] for i<30, new[30] = old[0] XOR old[3]
 */
static void init_x1_base_matrix(uint32_t M[LFSR_BITS]) {
    for (int i = 0; i < LFSR_BITS; i++) {
        M[i] = 0;
    }
    // new[i] = old[i+1] for i = 0..29
    for (int i = 0; i < LFSR_BITS - 1; i++) {
        M[i] = 1u << (i + 1);
    }
    // new[30] = old[0] XOR old[3]
    M[LFSR_BITS - 1] = (1u << 0) | (1u << 3);
}

/**
 * @brief Initialize the base transition matrix for x2 LFSR
 * x2(n+31) = x2(n+3) XOR x2(n+2) XOR x2(n+1) XOR x2(n)
 */
static void init_x2_base_matrix(uint32_t M[LFSR_BITS]) {
    for (int i = 0; i < LFSR_BITS; i++) {
        M[i] = 0;
    }
    // new[i] = old[i+1] for i = 0..29
    for (int i = 0; i < LFSR_BITS - 1; i++) {
        M[i] = 1u << (i + 1);
    }
    // new[30] = old[0] XOR old[1] XOR old[2] XOR old[3]
    M[LFSR_BITS - 1] = (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3);
}

/**
 * @brief Compute and upload jump tables to GPU constant memory
 * Pre-computes M^(2^p) for p = 0..MAX_JUMP_POWER for both LFSRs
 */
static cudaError_t initialize_jump_tables() {
    if (g_jump_tables_initialized) {
        return cudaSuccess;
    }

    // CUDA runtime calls such as cudaSetDeviceFlags may leave a non-fatal
    // error in the per-thread runtime slot after a context is already active.
    // Clear it before the synchronous constant-memory uploads below so stale
    // state does not make scrambler initialization fail on otherwise usable
    // discrete GPUs.
    (void)cudaGetLastError();

    uint32_t h_x1_jump[MAX_JUMP_POWER + 1][LFSR_BITS];
    uint32_t h_x2_jump[MAX_JUMP_POWER + 1][LFSR_BITS];

    // Initialize base matrices (M^1)
    init_x1_base_matrix(h_x1_jump[0]);
    init_x2_base_matrix(h_x2_jump[0]);

    // Compute M^(2^p) = M^(2^(p-1)) × M^(2^(p-1)) for p = 1..MAX_JUMP_POWER
    for (int p = 1; p <= MAX_JUMP_POWER; p++) {
        matrix_multiply_gf2(h_x1_jump[p-1], h_x1_jump[p-1], h_x1_jump[p]);
        matrix_multiply_gf2(h_x2_jump[p-1], h_x2_jump[p-1], h_x2_jump[p]);
    }

    // Upload to constant memory
    cudaError_t err;
    err = cudaMemcpyToSymbol(d_x1_jump, h_x1_jump, sizeof(h_x1_jump));
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        g_jump_tables_available = false;
        g_jump_tables_initialized = true;
        std::fprintf(stderr,
                     "[OCUDU PHY CUDA] LFSR jump-table upload failed (%s); using host-generated scrambling sequences\n",
                     cudaGetErrorString(err));
        return cudaSuccess;
    }

    err = cudaMemcpyToSymbol(d_x2_jump, h_x2_jump, sizeof(h_x2_jump));
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        g_jump_tables_available = false;
        g_jump_tables_initialized = true;
        std::fprintf(stderr,
                     "[OCUDU PHY CUDA] LFSR jump-table upload failed (%s); using host-generated scrambling sequences\n",
                     cudaGetErrorString(err));
        return cudaSuccess;
    }

    g_jump_tables_available = true;
    g_jump_tables_initialized = true;
    return cudaSuccess;
}

static inline uint32_t advance_x1_host(uint32_t x1) {
    uint32_t new_bit = ((x1 >> 3) ^ x1) & 1;
    return (x1 >> 1) | (new_bit << 30);
}

static inline uint32_t advance_x2_host(uint32_t x2) {
    uint32_t new_bit = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
    return (x2 >> 1) | (new_bit << 30);
}

static void generate_gold_sequence_host(std::vector<uint32_t>& sequence,
                                        uint32_t c_init,
                                        int num_words,
                                        int bit_offset) {
    uint32_t x1 = 1;
    uint32_t x2 = c_init & 0x7FFFFFFF;

    const int total_advance = NC_SKIP + bit_offset;
    for (int i = 0; i != total_advance; ++i) {
        x1 = advance_x1_host(x1);
        x2 = advance_x2_host(x2);
    }

    sequence.resize(num_words);
    for (int word_idx = 0; word_idx != num_words; ++word_idx) {
        uint32_t word = 0;
        for (int b = 0; b != 32; ++b) {
            uint32_t c_bit = (x1 ^ x2) & 1;
            word |= (c_bit << (31 - b));
            x1 = advance_x1_host(x1);
            x2 = advance_x2_host(x2);
        }
        sequence[word_idx] = word;
    }
}

// ============================================================================
// Device: Fast LFSR Advancement using Jump Tables
// ============================================================================

/**
 * @brief Apply a pre-computed jump matrix to LFSR state
 * new_state[i] = popcount(state & jump[i]) mod 2
 */
__device__ __forceinline__ uint32_t apply_jump_matrix(uint32_t state,
                                                        const uint32_t jump[LFSR_BITS]) {
    uint32_t result = 0;
    #pragma unroll
    for (int i = 0; i < LFSR_BITS; i++) {
        uint32_t masked = state & jump[i];
        // If odd number of bits set, set bit i in result
        if (__popc(masked) & 1) {
            result |= (1u << i);
        }
    }
    return result;
}

/**
 * @brief Advance x1 LFSR by N steps using matrix exponentiation - O(log N)
 */
__device__ uint32_t advance_x1_fast(uint32_t x1, int n) {
    // Apply jump matrices for each set bit in n
    // Only check up to highest set bit (saves iterations for small n)
    while (n) {
        int p = __ffs(n) - 1;  // Find lowest set bit (0-indexed)
        x1 = apply_jump_matrix(x1, d_x1_jump[p]);
        n &= (n - 1);  // Clear lowest set bit
    }
    return x1;
}

/**
 * @brief Advance x2 LFSR by N steps using matrix exponentiation - O(log N)
 */
__device__ uint32_t advance_x2_fast(uint32_t x2, int n) {
    // Apply jump matrices for each set bit in n
    // Only check up to highest set bit (saves iterations for small n)
    while (n) {
        int p = __ffs(n) - 1;  // Find lowest set bit (0-indexed)
        x2 = apply_jump_matrix(x2, d_x2_jump[p]);
        n &= (n - 1);  // Clear lowest set bit
    }
    return x2;
}

// ============================================================================
// Device: Legacy O(N) Functions (kept for small N where overhead > benefit)
// ============================================================================

/**
 * @brief Advance LFSR x1 by one step
 * x1(n+31) = (x1(n+3) + x1(n)) mod 2
 */
__device__ __forceinline__ uint32_t advance_x1(uint32_t x1) {
    uint32_t new_bit = ((x1 >> 3) ^ x1) & 1;
    return (x1 >> 1) | (new_bit << 30);
}

/**
 * @brief Advance LFSR x2 by one step
 * x2(n+31) = (x2(n+3) + x2(n+2) + x2(n+1) + x2(n)) mod 2
 */
__device__ __forceinline__ uint32_t advance_x2(uint32_t x2) {
    uint32_t new_bit = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
    return (x2 >> 1) | (new_bit << 30);
}

// ============================================================================
// Kernels
// ============================================================================

/**
 * @brief Generate Gold sequence in packed uint32_t format - FAST VERSION
 *
 * Uses matrix exponentiation for O(log N) LFSR advancement instead of O(N).
 * Each thread generates 32 consecutive bits (one word).
 *
 * IMPORTANT: The bit ordering must match the CPU's pseudo_random_generator_impl.cpp.
 * The CPU uses MSB-first ordering within bytes, with byte swapping for 64-bit writes.
 * To match this, we use MSB-first ordering: bit 0 goes to position 31 (MSB),
 * bit 1 goes to position 30, etc. This matches how the CPU extracts bits using
 * (c & (1U << 31U)) and shifts left.
 *
 * @param d_sequence Output sequence buffer
 * @param c_init Initialization value for x2 LFSR
 * @param num_words Number of 32-bit words to generate
 * @param bit_offset Starting bit offset (skipped bits before generation)
 */
__global__ void __launch_bounds__(256, 4) gold_sequence_generate_kernel(
    uint32_t* __restrict__ d_sequence,
    uint32_t c_init,
    int num_words,
    int bit_offset
) {
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (word_idx >= num_words) return;

    // Initialize LFSRs
    // x1(0) = 1, x1(n) = 0 for n = 1,...,30
    uint32_t x1 = 1;
    // x2 initialized from c_init (31 bits)
    uint32_t x2 = c_init & 0x7FFFFFFF;

    // Advance by Nc + bit_offset + word_idx * 32 steps
    // Using O(log N) matrix exponentiation instead of O(N) loop!
    int total_advance = NC_SKIP + bit_offset + word_idx * 32;

    x1 = advance_x1_fast(x1, total_advance);
    x2 = advance_x2_fast(x2, total_advance);

    // Generate 32 bits for this word (still O(32) but unavoidable)
    // Use MSB-first bit ordering to match CPU's pseudo_random_generator_impl.cpp:
    // - CPU extracts bit n using (c & (1U << 31U)) then shifts left with (c << 1U)
    // - So bit 0 is at position 31 (MSB), bit 1 at position 30, etc.
    // This ensures GPU scrambling sequence matches CPU bit-for-bit.
    uint32_t word = 0;
    #pragma unroll
    for (int b = 0; b < 32; b++) {
        // c(n) = (x1(n) XOR x2(n))
        uint32_t c_bit = (x1 ^ x2) & 1;
        // MSB-first: bit b maps to position (31 - b)
        word |= (c_bit << (31 - b));

        // Advance LFSRs by 1 step (fast for single step)
        x1 = advance_x1(x1);
        x2 = advance_x2(x2);
    }

    d_sequence[word_idx] = word;
}

/**
 * @brief Scramble bits using XOR (TX path)
 *
 * output[i] = input[i] XOR sequence[i]
 * Operates on packed uint32_t words for efficiency.
 */
__global__ void scramble_bits_kernel(
    const uint32_t* __restrict__ d_input,
    const uint32_t* __restrict__ d_sequence,
    uint32_t* __restrict__ d_output,
    int num_words
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    d_output[idx] = d_input[idx] ^ d_sequence[idx];
}

/**
 * @brief Scramble bits in-place using XOR (TX path)
 */
__global__ void scramble_bits_inplace_kernel(
    uint32_t* __restrict__ d_bits,
    const uint32_t* __restrict__ d_sequence,
    int num_words
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    d_bits[idx] ^= d_sequence[idx];
}

/**
 * @brief Descramble soft LLRs (RX path)
 *
 * For soft decoding, flip sign of LLR where scrambling bit is 1.
 * output[i] = (sequence[i] == 1) ? -input[i] : input[i]
 *
 * Uses MSB-first bit indexing to match gold_sequence_generate_kernel:
 * - Gold sequence stores bit b at position (31 - b) (MSB-first within words)
 * - To get sequence bit i, access word i/32, bit position (31 - i%32)
 */
__global__ void descramble_llr_kernel(
    const float* __restrict__ d_input_llrs,
    const uint32_t* __restrict__ d_sequence,
    float* __restrict__ d_output_llrs,
    int num_bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bits) return;

    // MSB-first bit indexing to match gold sequence generator
    int word_idx = idx / 32;
    int bit_pos = 31 - (idx % 32);
    uint32_t seq_bit = (d_sequence[word_idx] >> bit_pos) & 1;

    float llr = d_input_llrs[idx];
    d_output_llrs[idx] = seq_bit ? -llr : llr;
}

/**
 * @brief Descramble soft LLRs in-place (RX path)
 */
__global__ void descramble_llr_inplace_kernel(
    float* __restrict__ d_llrs,
    const uint32_t* __restrict__ d_sequence,
    int num_bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bits) return;

    // MSB-first bit indexing to match gold sequence generator
    int word_idx = idx / 32;
    int bit_pos = 31 - (idx % 32);
    uint32_t seq_bit = (d_sequence[word_idx] >> bit_pos) & 1;

    if (seq_bit) {
        d_llrs[idx] = -d_llrs[idx];
    }
}

/**
 * @brief Descramble half-precision (fp16) LLRs in-place (RX path)
 *
 * For fp16, flip sign by XORing the sign bit (bit 15) directly.
 * This is more efficient than negation for fp16.
 */
__global__ void descramble_llr_half_inplace_kernel(
    __half* __restrict__ d_llrs,
    const uint32_t* __restrict__ d_sequence,
    int num_bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bits) return;

    // MSB-first bit indexing to match gold sequence generator
    int word_idx = idx / 32;
    int bit_pos = 31 - (idx % 32);
    uint32_t seq_bit = (d_sequence[word_idx] >> bit_pos) & 1;

    if (seq_bit) {
        // Flip sign bit of fp16 value (bit 15)
        unsigned short bits = __half_as_ushort(d_llrs[idx]);
        d_llrs[idx] = __ushort_as_half(bits ^ 0x8000);
    }
}

// ============================================================================
// Context Structure
// ============================================================================

struct scrambler_ctx {
    nr_scrambling_config_t config;
    uint32_t c_init;

    // Pre-generated sequence
    uint32_t* d_sequence;
    size_t sequence_alloc_words;
    int sequence_length_bits;
    int sequence_offset;      // Offset at which sequence was generated

    // Current bit offset (for advance/set_offset)
    int current_offset;

    // Configuration flags
    bool configured;
    bool sequence_generated;
};

// ============================================================================
// Host API Implementation
// ============================================================================

extern "C" {

nr_ldpc_status_t scrambler_create(scrambler_handle_t* handle) {
    if (!handle) return NR_LDPC_ERROR_INVALID_CONFIG;

    // Initialize jump tables on first scrambler creation
    cudaError_t err = initialize_jump_tables();
    if (err != cudaSuccess) {
        fprintf(stderr, "[OCUDU PHY CUDA] Failed to initialize LFSR jump tables: %s\n",
                cudaGetErrorString(err));
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    scrambler_ctx* ctx = new (std::nothrow) scrambler_ctx();
    if (!ctx) return NR_LDPC_ERROR_ALLOC_FAILED;

    ctx->d_sequence = nullptr;
    ctx->sequence_alloc_words = 0;
    ctx->sequence_length_bits = 0;
    ctx->sequence_offset = 0;
    ctx->current_offset = 0;
    ctx->c_init = 0;
    ctx->configured = false;
    ctx->sequence_generated = false;

    *handle = ctx;
    return NR_LDPC_SUCCESS;
}

void scrambler_destroy(scrambler_handle_t handle) {
    if (!handle) return;

    scrambler_ctx* ctx = handle;
    if (ctx->d_sequence) {
        cudaFree(ctx->d_sequence);
    }
    delete ctx;
}

nr_ldpc_status_t scrambler_configure(scrambler_handle_t handle,
                                     const nr_scrambling_config_t* cfg) {
    if (!handle || !cfg) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;
    ctx->config = *cfg;

    // Compute c_init = n_RNTI * 2^15 + q * 2^14 + n_s * 2^4 + n_ID
    // Per 3GPP TS 38.211 section 6.3.1.1
    uint32_t new_c_init = ((uint32_t)cfg->n_RNTI << 15) |
                          ((uint32_t)(cfg->q & 1) << 14) |
                          ((uint32_t)(cfg->n_s & 0x1F) << 4) |  // CRITICAL FIX: Include slot number!
                          (cfg->n_ID & 0x3FF);

    // Only regenerate sequence if c_init actually changed (allows caching)
    if (!ctx->configured || ctx->c_init != new_c_init) {
        ctx->c_init = new_c_init;
        ctx->sequence_generated = false;  // Need to regenerate sequence
    }

    // ALWAYS reset offset on configure - TX and RX both start from offset 0
    // This is critical: TX scrambles from 0, RX must descramble from 0 too
    ctx->current_offset = 0;
    ctx->configured = true;

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t scrambler_configure_c_init(scrambler_handle_t handle,
                                             uint32_t c_init) {
    if (!handle) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;

    // Only regenerate sequence if c_init actually changed (allows caching)
    if (!ctx->configured || ctx->c_init != c_init) {
        ctx->c_init = c_init;
        ctx->sequence_generated = false;  // Need to regenerate sequence
    }

    // ALWAYS reset offset on configure - TX and RX both start from offset 0
    ctx->current_offset = 0;
    ctx->configured = true;

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t scrambler_generate_sequence(scrambler_handle_t handle,
                                              int num_bits,
                                              cudaStream_t stream) {
    if (!handle) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (!handle->configured) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (num_bits <= 0) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;

    // Check if we can reuse cached sequence
    // IMPORTANT: Also check that the offset matches! If configure() was called,
    // current_offset is reset to 0 but sequence_offset may be non-zero from a
    // previous operation. We must regenerate if the offsets don't match.
    if (ctx->sequence_generated &&
        ctx->sequence_length_bits >= num_bits &&
        ctx->sequence_offset == ctx->current_offset) {
        // Sequence already generated with sufficient length and matching offset
        return NR_LDPC_SUCCESS;
    }

    // Add +1 padding word to handle kernel's unconditional word_idx+1 read.
    // The kernel_pusch_e2e_cbf16 always reads scramble_seq[word_idx + 1] for
    // cross-word bit spans, even when it might not be needed. This padding
    // ensures the read is always within bounds.
    int num_words = (num_bits + 31) / 32 + 1;

    // Allocate or reallocate sequence buffer if needed
    if ((size_t)num_words > ctx->sequence_alloc_words) {
        if (ctx->d_sequence) {
            cudaFree(ctx->d_sequence);
        }
        cudaError_t err = cudaMalloc(&ctx->d_sequence, num_words * sizeof(uint32_t));
        if (err != cudaSuccess) {
            ctx->d_sequence = nullptr;
            ctx->sequence_alloc_words = 0;
            return NR_LDPC_ERROR_ALLOC_FAILED;
        }
        ctx->sequence_alloc_words = num_words;
    }

    if (g_jump_tables_available) {
        // Generate sequence with current offset - NOW O(log N) per thread!
        int block_size = 256;
        int num_blocks = (num_words + block_size - 1) / block_size;

        gold_sequence_generate_kernel<<<num_blocks, block_size, 0, stream>>>(
            ctx->d_sequence, ctx->c_init, num_words, ctx->current_offset);
    } else {
        std::vector<uint32_t> h_sequence;
        generate_gold_sequence_host(h_sequence, ctx->c_init, num_words, ctx->current_offset);
        cudaError_t err =
            cudaMemcpy(ctx->d_sequence, h_sequence.data(), num_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return NR_LDPC_ERROR_CUDA_FAILED;
        }
    }

    ctx->sequence_length_bits = num_bits;
    ctx->sequence_offset = ctx->current_offset;
    ctx->sequence_generated = true;

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t scrambler_generate_sequence_with_offset(scrambler_handle_t handle,
                                                          int num_bits,
                                                          int offset,
                                                          cudaStream_t stream) {
    if (!handle) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (!handle->configured) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (num_bits <= 0) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (offset < 0) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;
    // Add +1 padding word for kernel's unconditional word_idx+1 read
    int num_words = (num_bits + 31) / 32 + 1;

    // Allocate or reallocate sequence buffer if needed
    if ((size_t)num_words > ctx->sequence_alloc_words) {
        if (ctx->d_sequence) {
            cudaFree(ctx->d_sequence);
        }
        cudaError_t err = cudaMalloc(&ctx->d_sequence, num_words * sizeof(uint32_t));
        if (err != cudaSuccess) {
            ctx->d_sequence = nullptr;
            ctx->sequence_alloc_words = 0;
            return NR_LDPC_ERROR_ALLOC_FAILED;
        }
        ctx->sequence_alloc_words = num_words;
    }

    if (g_jump_tables_available) {
        // Generate sequence with explicit offset
        int block_size = 256;
        int num_blocks = (num_words + block_size - 1) / block_size;

        gold_sequence_generate_kernel<<<num_blocks, block_size, 0, stream>>>(
            ctx->d_sequence, ctx->c_init, num_words, offset);
    } else {
        std::vector<uint32_t> h_sequence;
        generate_gold_sequence_host(h_sequence, ctx->c_init, num_words, offset);
        cudaError_t err =
            cudaMemcpy(ctx->d_sequence, h_sequence.data(), num_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return NR_LDPC_ERROR_CUDA_FAILED;
        }
    }

    ctx->sequence_length_bits = num_bits;
    ctx->sequence_offset = offset;
    ctx->sequence_generated = true;

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t scrambler_advance(scrambler_handle_t handle, int count) {
    if (!handle) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (count < 0) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;
    ctx->current_offset += count;

    // Invalidate cached sequence if offset changed
    ctx->sequence_generated = false;

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t scrambler_set_offset(scrambler_handle_t handle, int offset) {
    if (!handle) return NR_LDPC_ERROR_INVALID_CONFIG;
    if (offset < 0) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;
    ctx->current_offset = offset;

    // Invalidate cached sequence if offset changed
    ctx->sequence_generated = false;

    return NR_LDPC_SUCCESS;
}

int scrambler_get_offset(scrambler_handle_t handle) {
    if (!handle) return 0;
    return handle->current_offset;
}

nr_ldpc_status_t scrambler_scramble(scrambler_handle_t handle,
                                    const uint32_t* d_input,
                                    uint32_t* d_output,
                                    int num_bits,
                                    cudaStream_t stream) {
    if (!handle || !d_input || !d_output) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;

    // Generate sequence if needed, if length changed, or if offset changed
    // The offset check is critical: after configure(), current_offset is 0 but
    // the cached sequence might have been generated at a different offset.
    if (!ctx->sequence_generated ||
        ctx->sequence_length_bits < num_bits ||
        ctx->sequence_offset != ctx->current_offset) {
        nr_ldpc_status_t status = scrambler_generate_sequence(handle, num_bits, stream);
        if (status != NR_LDPC_SUCCESS) return status;
    }

    int num_words = (num_bits + 31) / 32;
    int block_size = 256;
    int num_blocks = (num_words + block_size - 1) / block_size;

    scramble_bits_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_input, ctx->d_sequence, d_output, num_words);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t scrambler_scramble_inplace(scrambler_handle_t handle,
                                            uint32_t* d_bits,
                                            int num_bits,
                                            cudaStream_t stream) {
    if (!handle || !d_bits) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;

    // Generate sequence if needed, if length changed, or if offset changed
    if (!ctx->sequence_generated ||
        ctx->sequence_length_bits < num_bits ||
        ctx->sequence_offset != ctx->current_offset) {
        nr_ldpc_status_t status = scrambler_generate_sequence(handle, num_bits, stream);
        if (status != NR_LDPC_SUCCESS) return status;
    }

    int num_words = (num_bits + 31) / 32;
    int block_size = 256;
    int num_blocks = (num_words + block_size - 1) / block_size;

    scramble_bits_inplace_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_bits, ctx->d_sequence, num_words);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t scrambler_descramble_llr(scrambler_handle_t handle,
                                          const float* d_input_llrs,
                                          float* d_output_llrs,
                                          int num_bits,
                                          cudaStream_t stream) {
    if (!handle || !d_input_llrs || !d_output_llrs) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;

    // Generate sequence if needed, if length changed, or if offset changed
    if (!ctx->sequence_generated ||
        ctx->sequence_length_bits < num_bits ||
        ctx->sequence_offset != ctx->current_offset) {
        nr_ldpc_status_t status = scrambler_generate_sequence(handle, num_bits, stream);
        if (status != NR_LDPC_SUCCESS) return status;
    }

    int block_size = 256;
    int num_blocks = (num_bits + block_size - 1) / block_size;

    descramble_llr_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_input_llrs, ctx->d_sequence, d_output_llrs, num_bits);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t scrambler_descramble_llr_inplace(scrambler_handle_t handle,
                                                  float* d_llrs,
                                                  int num_bits,
                                                  cudaStream_t stream) {
    if (!handle || !d_llrs) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;

    // Generate sequence if needed, if length changed, or if offset changed
    if (!ctx->sequence_generated ||
        ctx->sequence_length_bits < num_bits ||
        ctx->sequence_offset != ctx->current_offset) {
        nr_ldpc_status_t status = scrambler_generate_sequence(handle, num_bits, stream);
        if (status != NR_LDPC_SUCCESS) return status;
    }

    int block_size = 256;
    int num_blocks = (num_bits + block_size - 1) / block_size;

    descramble_llr_inplace_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_llrs, ctx->d_sequence, num_bits);

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t scrambler_descramble_llr_half_inplace(scrambler_handle_t handle,
                                                        void* d_llrs_half,
                                                        int num_bits,
                                                        cudaStream_t stream) {
    if (!handle || !d_llrs_half) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;
    __half* d_llrs = static_cast<__half*>(d_llrs_half);

    // Generate sequence if needed, if length changed, or if offset changed
    if (!ctx->sequence_generated ||
        ctx->sequence_length_bits < num_bits ||
        ctx->sequence_offset != ctx->current_offset) {
        nr_ldpc_status_t status = scrambler_generate_sequence(handle, num_bits, stream);
        if (status != NR_LDPC_SUCCESS) return status;
    }

    int block_size = 256;
    int num_blocks = (num_bits + block_size - 1) / block_size;

    descramble_llr_half_inplace_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_llrs, ctx->d_sequence, num_bits);

    return NR_LDPC_SUCCESS;
}

uint32_t scrambler_get_c_init(scrambler_handle_t handle) {
    if (!handle || !handle->configured) return 0;
    return handle->c_init;
}

const unsigned int* scrambler_get_sequence_ptr(scrambler_handle_t handle) {
    if (!handle || !handle->sequence_generated) return nullptr;
    return handle->d_sequence;
}

nr_ldpc_status_t scrambler_preallocate_sequence(scrambler_handle_t handle,
                                                  int max_bits) {
    if (!handle || max_bits <= 0) return NR_LDPC_ERROR_INVALID_CONFIG;

    scrambler_ctx* ctx = handle;
    // Add +1 padding word for kernel's unconditional word_idx+1 read
    int num_words = (max_bits + 31) / 32 + 1;

    // Allocate or reallocate sequence buffer if needed
    if ((size_t)num_words > ctx->sequence_alloc_words) {
        if (ctx->d_sequence) {
            cudaFree(ctx->d_sequence);
            ctx->d_sequence = nullptr;
        }

        cudaError_t err = cudaMalloc(&ctx->d_sequence, num_words * sizeof(uint32_t));
        if (err != cudaSuccess) {
            ctx->d_sequence = nullptr;
            ctx->sequence_alloc_words = 0;
            return NR_LDPC_ERROR_ALLOC_FAILED;
        }
        ctx->sequence_alloc_words = num_words;
    }

    // Note: We don't mark sequence_generated = true here because we haven't
    // actually generated the sequence yet. The caller should call
    // scrambler_generate_sequence() after this.

    return NR_LDPC_SUCCESS;
}

}  // extern "C"
