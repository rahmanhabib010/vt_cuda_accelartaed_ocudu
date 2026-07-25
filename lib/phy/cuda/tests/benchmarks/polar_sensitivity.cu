/**
 * @file polar_sensitivity.cu
 * @brief CUDA polar noisy-LLR sensitivity comparison against a CPU reference path.
 */

#include "polar.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
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

struct sweep_case_t {
    const char* name;
    int K;
    int E;
    int n_max;
    polar_ibil_t ibil;
};

static std::vector<uint8_t> cpu_encode(const std::vector<uint8_t>& input, int N)
{
    std::vector<uint8_t> bits(input.begin(), input.begin() + N);
    for (int step = 1; step < N; step <<= 1) {
        int group = step << 1;
        for (int i = 0; i < N; ++i) {
            if ((i % group) < step) {
                bits[i] ^= bits[i + step];
            }
        }
    }
    return bits;
}

static int channel_interleaver_T(int E)
{
    int S = 1;
    int T = 1;
    while (S < E) {
        S += ++T;
    }
    return T;
}

static int channel_interleaver_input_for_output(int out_idx, int E)
{
    int T = channel_interleaver_T(E);
    int out = 0;
    for (int r = 0; r < T; ++r) {
        int in = r;
        for (int c = 0; c < T - r; ++c) {
            if (in >= E) {
                break;
            }
            if (out == out_idx) {
                return in;
            }
            ++out;
            in += (T - c);
        }
    }
    return 0;
}

static int channel_interleaver_output_for_input(int in_idx, int E)
{
    int T = channel_interleaver_T(E);
    int out = 0;
    for (int r = 0; r < T; ++r) {
        int in = r;
        for (int c = 0; c < T - r; ++c) {
            if (in >= E) {
                break;
            }
            if (in == in_idx) {
                return out;
            }
            ++out;
            in += (T - c);
        }
    }
    return 0;
}

static std::vector<uint8_t> cpu_rate_match(const polar_code_config_t& cfg, const std::vector<uint8_t>& input)
{
    std::vector<uint8_t> output(cfg.E);
    for (int out_idx = 0; out_idx < cfg.E; ++out_idx) {
        int e_idx = (cfg.ibil == POLAR_IBIL_PRESENT) ? channel_interleaver_input_for_output(out_idx, cfg.E) : out_idx;
        int y_idx = 0;
        if (cfg.E >= cfg.N) {
            y_idx = e_idx % cfg.N;
        } else if (16 * cfg.K <= 7 * cfg.E) {
            y_idx = cfg.N - cfg.E + e_idx;
        } else {
            y_idx = e_idx;
        }
        output[out_idx] = input[cfg.block_interleaver[y_idx]];
    }
    return output;
}

static float clamp_llr(float v)
{
    return std::min(std::max(v, -127.0F), 127.0F);
}

static std::vector<float> cpu_rate_dematch(const polar_code_config_t& cfg, const std::vector<float>& input)
{
    std::vector<float> output(cfg.N, 0.0F);
    for (int y_idx = 0; y_idx < cfg.N; ++y_idx) {
        float v = 0.0F;
        if (cfg.E >= cfg.N) {
            float sum = 0.0F;
            for (int e_idx = y_idx; e_idx < cfg.E; e_idx += cfg.N) {
                int in_idx = (cfg.ibil == POLAR_IBIL_PRESENT) ? channel_interleaver_output_for_input(e_idx, cfg.E) : e_idx;
                sum += input[in_idx];
            }
            v = clamp_llr(sum);
        } else if (16 * cfg.K <= 7 * cfg.E) {
            if (y_idx >= cfg.N - cfg.E) {
                int e_idx = y_idx - (cfg.N - cfg.E);
                int in_idx = (cfg.ibil == POLAR_IBIL_PRESENT) ? channel_interleaver_output_for_input(e_idx, cfg.E) : e_idx;
                v = input[in_idx];
            }
        } else {
            if (y_idx < cfg.E) {
                int in_idx = (cfg.ibil == POLAR_IBIL_PRESENT) ? channel_interleaver_output_for_input(y_idx, cfg.E) : y_idx;
                v = input[in_idx];
            } else {
                v = 127.0F;
            }
        }
        output[cfg.block_interleaver[y_idx]] = v;
    }
    return output;
}

static float soft_xor(float x, float y)
{
    float mag = std::min(std::fabs(x), std::fabs(y));
    return (x * y < 0.0F) ? -mag : mag;
}

static void cpu_encode_inplace(std::vector<uint8_t>& bits, int offset, int len)
{
    for (int step = 1; step < len; step <<= 1) {
        int group = step << 1;
        for (int i = 0; i < len; ++i) {
            if ((i % group) < step) {
                bits[offset + i] ^= bits[offset + i + step];
            }
        }
    }
}

static void cpu_ssc_decode_node(const polar_code_config_t& cfg,
                                int start,
                                int stage,
                                const std::vector<float>& alpha_in,
                                std::vector<uint8_t>& decoded,
                                std::vector<uint8_t>& beta_out)
{
    int len = 1 << stage;
    uint8_t rate = cfg.node_rate[stage * POLAR_MAX_N + start];
    if (rate == POLAR_NODE_RATE_0) {
        std::fill(decoded.begin() + start, decoded.begin() + start + len, 0);
        beta_out.assign(len, 0);
        return;
    }
    if (rate == POLAR_NODE_RATE_1) {
        beta_out.assign(len, 0);
        for (int i = 0; i < len; ++i) {
            uint8_t bit = static_cast<uint8_t>(alpha_in[i] <= 0.0F);
            decoded[start + i] = bit;
            beta_out[i] = bit;
        }
        cpu_encode_inplace(decoded, start, len);
        return;
    }
    if (stage == 0) {
        uint8_t bit = cfg.frozen_mask[start] ? 0U : static_cast<uint8_t>(alpha_in[0] <= 0.0F);
        decoded[start] = bit;
        beta_out.assign(1, bit);
        return;
    }

    int half = 1 << (stage - 1);
    std::vector<float> left_alpha(half);
    std::vector<float> right_alpha(half);
    std::vector<uint8_t> left_beta;
    std::vector<uint8_t> right_beta;
    for (int i = 0; i < half; ++i) {
        left_alpha[i] = soft_xor(alpha_in[i], alpha_in[half + i]);
    }
    cpu_ssc_decode_node(cfg, start, stage - 1, left_alpha, decoded, left_beta);
    for (int i = 0; i < half; ++i) {
        right_alpha[i] = clamp_llr(left_beta[i] ? (alpha_in[half + i] - alpha_in[i]) : (alpha_in[half + i] + alpha_in[i]));
    }
    cpu_ssc_decode_node(cfg, start + half, stage - 1, right_alpha, decoded, right_beta);

    beta_out.assign(len, 0);
    for (int i = 0; i < half; ++i) {
        beta_out[i] = left_beta[i] ^ right_beta[i];
        beta_out[half + i] = right_beta[i];
    }
}

static std::vector<uint8_t> cpu_decode(const polar_code_config_t& cfg, const std::vector<float>& dematched_llr)
{
    std::vector<uint8_t> decoded(cfg.N, 0);
    std::vector<uint8_t> beta;
    cpu_ssc_decode_node(cfg, 0, cfg.n, dematched_llr, decoded, beta);
    return decoded;
}

static std::vector<uint8_t> make_input_u(const polar_code_config_t& cfg, std::mt19937& rng)
{
    std::bernoulli_distribution bits(0.5);
    std::vector<uint8_t> input(cfg.N, 0);
    for (int i = 0; i < cfg.N; ++i) {
        if (!cfg.frozen_mask[i]) {
            input[i] = bits(rng) ? 1U : 0U;
        }
    }
    return input;
}

static bool equal_bits(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, int len)
{
    for (int i = 0; i < len; ++i) {
        if ((a[i] & 1U) != (b[i] & 1U)) {
            return false;
        }
    }
    return true;
}

static bool run_case(const sweep_case_t& sc, int trials, const std::vector<float>& snrs_db)
{
    polar_code_config_t cfg{};
    POLAR_OK(polar_code_configure(&cfg, sc.K, sc.E, sc.n_max, sc.ibil));

    polar_handle_t handle = nullptr;
    POLAR_OK(polar_create(&handle));
    POLAR_OK(polar_configure(handle, &cfg));
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreate(&stream));

    __half* d_input_llr = nullptr;
    uint8_t* d_decoded = nullptr;
    CUDA_OK(cudaMalloc(&d_input_llr, cfg.E * sizeof(__half)));
    CUDA_OK(cudaMalloc(&d_decoded, cfg.N));

    std::printf("%-12s K=%3d E=%4d N=%4d ibil=%d trials=%d\n", sc.name, cfg.K, cfg.E, cfg.N, cfg.ibil, trials);
    std::printf("  %8s %10s %10s %10s %12s\n", "snr_db", "cpu_fer", "gpu_fer", "delta", "disagree");

    std::mt19937 rng(0x5eed1234U + static_cast<uint32_t>(sc.K * 17 + sc.E));
    std::normal_distribution<float> normal(0.0F, 1.0F);
    bool ok = true;

    for (float snr_db : snrs_db) {
        float snr_linear = std::pow(10.0F, snr_db / 10.0F);
        float sigma2 = 1.0F / (2.0F * snr_linear);
        float sigma = std::sqrt(sigma2);
        int cpu_err = 0;
        int gpu_err = 0;
        int disagree = 0;

        for (int t = 0; t < trials; ++t) {
            std::vector<uint8_t> input_u = make_input_u(cfg, rng);
            std::vector<uint8_t> encoded = cpu_encode(input_u, cfg.N);
            std::vector<uint8_t> rate_matched = cpu_rate_match(cfg, encoded);
            std::vector<float> llr(cfg.E);
            std::vector<__half> h_llr(cfg.E);
            for (int i = 0; i < cfg.E; ++i) {
                float symbol = rate_matched[i] ? -1.0F : 1.0F;
                float y = symbol + sigma * normal(rng);
                llr[i] = clamp_llr(2.0F * y / sigma2);
                h_llr[i] = __float2half(llr[i]);
            }

            std::vector<float> cpu_dematched = cpu_rate_dematch(cfg, llr);
            std::vector<uint8_t> cpu_decoded = cpu_decode(cfg, cpu_dematched);

            CUDA_OK(cudaMemcpyAsync(d_input_llr, h_llr.data(), cfg.E * sizeof(__half), cudaMemcpyHostToDevice, stream));
            POLAR_OK(polar_rate_dematch_decode_half(handle, d_decoded, d_input_llr, stream));
            CUDA_OK(cudaStreamSynchronize(stream));

            std::vector<uint8_t> gpu_decoded(cfg.N);
            CUDA_OK(cudaMemcpy(gpu_decoded.data(), d_decoded, cfg.N, cudaMemcpyDeviceToHost));

            bool cpu_bad = !equal_bits(cpu_decoded, input_u, cfg.N);
            bool gpu_bad = !equal_bits(gpu_decoded, input_u, cfg.N);
            bool diff = !equal_bits(cpu_decoded, gpu_decoded, cfg.N);
            cpu_err += cpu_bad ? 1 : 0;
            gpu_err += gpu_bad ? 1 : 0;
            disagree += diff ? 1 : 0;
        }

        float cpu_fer = static_cast<float>(cpu_err) / static_cast<float>(trials);
        float gpu_fer = static_cast<float>(gpu_err) / static_cast<float>(trials);
        float delta = gpu_fer - cpu_fer;
        float disagree_rate = static_cast<float>(disagree) / static_cast<float>(trials);
        std::printf("  %8.2f %10.4f %10.4f %+10.4f %12.4f\n", snr_db, cpu_fer, gpu_fer, delta, disagree_rate);
        if (delta > 0.03F && gpu_err > cpu_err + 3) {
            ok = false;
        }
    }

    cudaFree(d_decoded);
    cudaFree(d_input_llr);
    cudaStreamDestroy(stream);
    polar_destroy(handle);
    return ok;
}

int main(int argc, char** argv)
{
    int trials = 1000;
    if (argc > 1) {
        trials = std::max(1, std::atoi(argv[1]));
    }

    const sweep_case_t cases[] = {
        {"BCH", 56, 864, 9, POLAR_IBIL_NOT_PRESENT},
        {"DCI", 40, 100, 9, POLAR_IBIL_NOT_PRESENT},
        {"UCI-rep", 20, 256, 10, POLAR_IBIL_PRESENT},
        {"UCI-punc", 18, 45, 10, POLAR_IBIL_PRESENT},
        {"UCI-short", 18, 38, 10, POLAR_IBIL_PRESENT},
    };
    const std::vector<float> snrs_db = {-2.0F, 0.0F, 2.0F, 4.0F, 6.0F};

    bool ok = true;
    for (const auto& sc : cases) {
        ok = run_case(sc, trials, snrs_db) && ok;
    }
    return ok ? 0 : 1;
}
