// PHASE 4 OPTIMIZATION: Template Specialization for LDPC Decoder
// Compile-time constants for base graph parameters enable:
// 1. Perfect loop unrolling (compiler knows exact iteration counts)
// 2. Zero runtime branching (no if (BG == 1) checks)
// 3. Optimal register allocation (compiler knows exact array sizes)
// 4. Better instruction scheduling
//
// Target: 50-100 μs improvement from eliminating branch overhead

#ifndef LDPC_DECODER_SPECIALIZED_CUH
#define LDPC_DECODER_SPECIALIZED_CUH

#include <cuda_fp16.h>

// ============================================================================
// LDPC Configuration Templates for 5G NR Base Graphs
// ============================================================================

template<int BG, int Z>
struct LdpcConfig;

// BG1 Z=384 specialization (273 PRB - MOST COMMON!)
template<>
struct LdpcConfig<1, 384> {
    static constexpr int M = 46;           // Number of parity check rows
    static constexpr int N = 68;           // Total columns (Kb + parity)
    static constexpr int Kb = 22;          // Information bit columns
    static constexpr int max_deg = 19;     // Maximum row degree
    static constexpr int nnz = 316;        // Non-zero entries in base graph

    // Optimization hints
    static constexpr int blocks_per_sm = 4;       // Target occupancy
    static constexpr int threads_per_block = 384; // Must equal Z for this kernel

    // Memory requirements
    static constexpr int shared_mem_bytes = 2176 + 1; // CRC buffer + fence flag
};

// BG1 Z=256 specialization (176 PRB)
template<>
struct LdpcConfig<1, 256> {
    static constexpr int M = 46;
    static constexpr int N = 68;
    static constexpr int Kb = 22;
    static constexpr int max_deg = 19;
    static constexpr int nnz = 316;

    static constexpr int blocks_per_sm = 4;
    static constexpr int threads_per_block = 256;
    static constexpr int shared_mem_bytes = 2176 + 1;
};

// BG1 Z=128 specialization (88 PRB)
template<>
struct LdpcConfig<1, 128> {
    static constexpr int M = 46;
    static constexpr int N = 68;
    static constexpr int Kb = 22;
    static constexpr int max_deg = 19;
    static constexpr int nnz = 316;

    static constexpr int blocks_per_sm = 4;
    static constexpr int threads_per_block = 128;
    static constexpr int shared_mem_bytes = 2176 + 1;
};

// BG2 Z=384 specialization
template<>
struct LdpcConfig<2, 384> {
    static constexpr int M = 42;           // BG2 has 42 rows
    static constexpr int N = 52;           // BG2 has 52 columns
    static constexpr int Kb = 10;          // BG2 has 10 information columns
    static constexpr int max_deg = 10;     // BG2 maximum degree
    static constexpr int nnz = 197;        // BG2 non-zero entries

    static constexpr int blocks_per_sm = 4;
    static constexpr int threads_per_block = 384;
    static constexpr int shared_mem_bytes = 2176 + 1;
};

// BG2 Z=256 specialization
template<>
struct LdpcConfig<2, 256> {
    static constexpr int M = 42;
    static constexpr int N = 52;
    static constexpr int Kb = 10;
    static constexpr int max_deg = 10;
    static constexpr int nnz = 197;

    static constexpr int blocks_per_sm = 4;
    static constexpr int threads_per_block = 256;
    static constexpr int shared_mem_bytes = 2176 + 1;
};

// BG2 Z=128 specialization
template<>
struct LdpcConfig<2, 128> {
    static constexpr int M = 42;
    static constexpr int N = 52;
    static constexpr int Kb = 10;
    static constexpr int max_deg = 10;
    static constexpr int nnz = 197;

    static constexpr int blocks_per_sm = 4;
    static constexpr int threads_per_block = 128;
    static constexpr int shared_mem_bytes = 2176 + 1;
};

// ============================================================================
// Compile-Time Utility Functions
// ============================================================================

// Check if a BG/Z combination has a specialization
template<int BG, int Z>
constexpr bool has_specialization() {
    return (BG == 1 && (Z == 384 || Z == 256 || Z == 128)) ||
           (BG == 2 && (Z == 384 || Z == 256 || Z == 128));
}

// NOTE: Compile-time dispatch functions would go here
// Currently not used - kept for future optimization phases

#endif // LDPC_DECODER_SPECIALIZED_CUH
