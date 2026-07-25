/**
 * @file polar_benchmark.cu
 * @brief CUDA polar primitive latency benchmark.
 */

#include "polar.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(call)                                                                               \
    do {                                                                                            \
        cudaError_t _err = (call);                                                                  \
        if (_err != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

#define POLAR_OK(call)                                                                              \
    do {                                                                                            \
        nr_ldpc_status_t _st = (call);                                                              \
        if (_st != NR_LDPC_SUCCESS) {                                                               \
            std::fprintf(stderr, "Polar API error %s:%d: status=%d\n", __FILE__, __LINE__, (int)_st);       \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

struct bench_case_t {
    const char* name;
    int K;
    int E;
    int n_max;
    polar_ibil_t ibil;
};

static std::vector<uint8_t> make_input_u(const polar_code_config_t& cfg, uint32_t seed)
{
    std::vector<uint8_t> input(cfg.N, 0);
    uint32_t state = seed;
    for (int i = 0; i < cfg.N; ++i) {
        state = 1664525U * state + 1013904223U;
        if (!cfg.frozen_mask[i]) {
            input[i] = (state >> 31) & 1U;
        }
    }
    return input;
}

template <typename Fn>
static bool measure(cudaStream_t stream, int warmup, int iterations, Fn&& fn, float& us)
{
    for (int i = 0; i < warmup; ++i) {
        POLAR_OK(fn());
    }
    CUDA_OK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    CUDA_OK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        POLAR_OK(fn());
    }
    CUDA_OK(cudaEventRecord(stop, stream));
    CUDA_OK(cudaEventSynchronize(stop));
    float ms = 0.0F;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    us = ms * 1000.0F / static_cast<float>(iterations);
    return true;
}

static bool run_case(const bench_case_t& bc, int warmup, int iterations)
{
    polar_code_config_t cfg{};
    POLAR_OK(polar_code_configure(&cfg, bc.K, bc.E, bc.n_max, bc.ibil));

    polar_handle_t handle = nullptr;
    POLAR_OK(polar_create(&handle));
    POLAR_OK(polar_configure(handle, &cfg));

    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreate(&stream));

    std::vector<uint8_t> input = make_input_u(cfg, 0x76543210U + static_cast<uint32_t>(bc.K * 131 + bc.E));
    std::vector<__half> llr(cfg.E);
    for (int i = 0; i < cfg.E; ++i) {
        llr[i] = __float2half((i & 1) ? -16.0F : 16.0F);
    }

    uint8_t* d_input = nullptr;
    uint8_t* d_rm = nullptr;
    __half* d_llr = nullptr;
    uint8_t* d_decoded = nullptr;

    CUDA_OK(cudaMalloc(&d_input, cfg.N));
    CUDA_OK(cudaMalloc(&d_rm, cfg.E));
    CUDA_OK(cudaMalloc(&d_llr, cfg.E * sizeof(__half)));
    CUDA_OK(cudaMalloc(&d_decoded, cfg.N));
    CUDA_OK(cudaMemcpyAsync(d_input, input.data(), cfg.N, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_llr, llr.data(), cfg.E * sizeof(__half), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaStreamSynchronize(stream));

    POLAR_OK(polar_encode_rate_match_u8(handle, d_rm, d_input, stream));
    POLAR_OK(polar_rate_dematch_decode_half(handle, d_decoded, d_llr, stream));
    CUDA_OK(cudaStreamSynchronize(stream));

    float tx_us = 0.0F;
    float rx_us = 0.0F;

    if (!measure(stream, warmup, iterations, [&]() {
            return polar_encode_rate_match_u8(handle, d_rm, d_input, stream);
        }, tx_us) ||
        !measure(stream, warmup, iterations, [&]() {
            return polar_rate_dematch_decode_half(handle, d_decoded, d_llr, stream);
        }, rx_us)) {
        return false;
    }

    std::printf("%-16s %4d %5d %5d %4d %8.3f %8.3f\n", bc.name, cfg.K, cfg.E, cfg.N, cfg.ibil, tx_us, rx_us);

    cudaFree(d_decoded);
    cudaFree(d_llr);
    cudaFree(d_rm);
    cudaFree(d_input);
    cudaStreamDestroy(stream);
    polar_destroy(handle);
    return true;
}

int main(int argc, char** argv)
{
    int iterations = 5000;
    int warmup = 200;
    if (argc > 1) {
        iterations = std::max(1, std::atoi(argv[1]));
    }
    if (argc > 2) {
        warmup = std::max(0, std::atoi(argv[2]));
    }

    const bench_case_t cases[] = {
        {"BCH", 56, 864, 9, POLAR_IBIL_NOT_PRESENT},
        {"DCI", 40, 100, 9, POLAR_IBIL_NOT_PRESENT},
        {"UCI-rep", 20, 256, 10, POLAR_IBIL_PRESENT},
        {"UCI-punc", 18, 45, 10, POLAR_IBIL_PRESENT},
        {"UCI-short", 18, 38, 10, POLAR_IBIL_PRESENT},
        {"UCI-1024", 64, 1024, 10, POLAR_IBIL_PRESENT},
    };

    std::printf("Polar CUDA primitive latency, iterations=%d warmup=%d\n", iterations, warmup);
    std::printf("%-16s %4s %5s %5s %4s %8s %8s\n",
                "case",
                "K",
                "E",
                "N",
                "ibil",
                "tx_us",
                "rx_us");
    bool ok = true;
    for (const auto& bc : cases) {
        ok = run_case(bc, warmup, iterations) && ok;
    }
    return ok ? 0 : 1;
}
