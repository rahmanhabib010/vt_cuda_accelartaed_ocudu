/**
 * @file test_polar.cu
 * @brief OCUDU PHY CUDA polar primitive correctness test.
 */

#include "polar.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        cudaError_t _err = (call);                                                                  \
        if (_err != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

#define POLAR_CHECK(call)                                                                           \
    do {                                                                                            \
        nr_ldpc_status_t _st = (call);                                                              \
        if (_st != NR_LDPC_SUCCESS) {                                                               \
            std::fprintf(stderr, "Polar API error %s:%d: status=%d\n", __FILE__, __LINE__, (int)_st);       \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

struct test_case_t {
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

static void cpu_sc_decode_node(const polar_code_config_t& cfg,
                               int start,
                               int stage,
                               const std::vector<float>& alpha_in,
                               std::vector<uint8_t>& decoded,
                               std::vector<uint8_t>& beta_out)
{
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
    cpu_sc_decode_node(cfg, start, stage - 1, left_alpha, decoded, left_beta);

    for (int i = 0; i < half; ++i) {
        right_alpha[i] = clamp_llr(left_beta[i] ? (alpha_in[half + i] - alpha_in[i]) : (alpha_in[half + i] + alpha_in[i]));
    }
    cpu_sc_decode_node(cfg, start + half, stage - 1, right_alpha, decoded, right_beta);

    beta_out.assign(2 * half, 0);
    for (int i = 0; i < half; ++i) {
        beta_out[i] = left_beta[i] ^ right_beta[i];
        beta_out[half + i] = right_beta[i];
    }
}

static std::vector<uint8_t> cpu_sc_decode(const polar_code_config_t& cfg, const std::vector<float>& llr)
{
    std::vector<uint8_t> decoded(cfg.N, 0);
    std::vector<uint8_t> beta;
    cpu_sc_decode_node(cfg, 0, cfg.n, llr, decoded, beta);
    return decoded;
}

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

static bool compare_bits(const char* label, const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected, int len)
{
    for (int i = 0; i < len; ++i) {
        if ((actual[i] & 1U) != (expected[i] & 1U)) {
            std::fprintf(stderr, "%s mismatch at %d: got=%u expected=%u\n", label, i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

static bool run_case(const test_case_t& tc)
{
    polar_code_config_t cfg{};
    POLAR_CHECK(polar_code_configure(&cfg, tc.K, tc.E, tc.n_max, tc.ibil));

    polar_handle_t handle = nullptr;
    POLAR_CHECK(polar_create(&handle));
    POLAR_CHECK(polar_configure(handle, &cfg));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<uint8_t> input_u = make_input_u(cfg, 0x12345678U + static_cast<uint32_t>(tc.K * 97 + tc.E));
    std::vector<uint8_t> ref_encoded = cpu_encode(input_u, cfg.N);
    std::vector<uint8_t> ref_rm = cpu_rate_match(cfg, ref_encoded);

    std::vector<float> rate_matched_llr(cfg.E);
    std::vector<__half> h_rate_matched_llr(cfg.E);
    for (int i = 0; i < cfg.E; ++i) {
        rate_matched_llr[i] = ref_rm[i] ? -20.0F : 20.0F;
        h_rate_matched_llr[i] = __float2half(rate_matched_llr[i]);
    }
    std::vector<float> ref_dematched = cpu_rate_dematch(cfg, rate_matched_llr);
    std::vector<uint8_t> ref_decoded = cpu_sc_decode(cfg, ref_dematched);

    uint8_t* d_input = nullptr;
    uint8_t* d_rm = nullptr;
    __half* d_rm_llr = nullptr;
    uint8_t* d_decoded = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, cfg.N));
    CUDA_CHECK(cudaMalloc(&d_rm, cfg.E));
    CUDA_CHECK(cudaMalloc(&d_rm_llr, cfg.E * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_decoded, cfg.N));

    CUDA_CHECK(cudaMemcpyAsync(d_input, input_u.data(), cfg.N, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_rm_llr, h_rate_matched_llr.data(), cfg.E * sizeof(__half), cudaMemcpyHostToDevice, stream));

    POLAR_CHECK(polar_encode_rate_match_u8(handle, d_rm, d_input, stream));
    POLAR_CHECK(polar_rate_dematch_decode_half(handle, d_decoded, d_rm_llr, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<uint8_t> got_rm(cfg.E);
    std::vector<uint8_t> got_decoded(cfg.N);

    CUDA_CHECK(cudaMemcpy(got_rm.data(), d_rm, cfg.E, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_decoded.data(), d_decoded, cfg.N, cudaMemcpyDeviceToHost));

    bool ok = compare_bits("encode_rate_match", got_rm, ref_rm, cfg.E) &&
              compare_bits("decode_cpu_reference", got_decoded, ref_decoded, cfg.N) &&
              compare_bits("decode_input", got_decoded, input_u, cfg.N);

    std::printf("%-18s K=%4d E=%4d N=%4d ibil=%d : %s\n", tc.name, cfg.K, cfg.E, cfg.N, cfg.ibil, ok ? "PASS" : "FAIL");

    cudaFree(d_decoded);
    cudaFree(d_rm_llr);
    cudaFree(d_rm);
    cudaFree(d_input);
    cudaStreamDestroy(stream);
    polar_destroy(handle);
    return ok;
}


static uint32_t uci_crc_checksum(const std::vector<uint8_t>& bits, int crc_order)
{
    uint32_t highbit = 1U << crc_order;
    uint32_t poly = (crc_order == 6) ? 0x61U : 0xe21U;
    uint32_t remainder = 0;
    for (uint8_t bit : bits) {
        remainder = (remainder << 1U) | static_cast<uint32_t>(bit & 1U);
        if ((remainder & highbit) != 0U) {
            remainder ^= poly;
        }
    }
    for (int i = 0; i != crc_order; ++i) {
        remainder <<= 1U;
        if ((remainder & highbit) != 0U) {
            remainder ^= poly;
        }
    }
    return remainder & (highbit - 1U);
}

static std::vector<uint8_t> make_uci_deallocated_bits(int payload_bits, int filler_bits, int crc_order, uint32_t seed)
{
    std::vector<uint8_t> bits;
    bits.reserve(filler_bits + payload_bits + crc_order);
    for (int i = 0; i != filler_bits; ++i) {
        bits.push_back(0);
    }
    uint32_t state = seed;
    for (int i = 0; i != payload_bits; ++i) {
        state = 1664525U * state + 1013904223U;
        bits.push_back((state >> 31) & 1U);
    }
    uint32_t crc = uci_crc_checksum(bits, crc_order);
    for (int i = crc_order - 1; i >= 0; --i) {
        bits.push_back((crc >> i) & 1U);
    }
    return bits;
}

static std::vector<uint8_t> make_uci_input_u(const polar_code_config_t& cfg, const std::vector<uint8_t>& deallocated)
{
    std::vector<uint8_t> input(cfg.N, 0);
    uint8_t y0 = 0;
    uint8_t y1 = 0;
    uint8_t y2 = 0;
    uint8_t y3 = 0;
    uint8_t y4 = 0;
    int msg_idx = 0;
    int pc_idx = 0;
    for (int i = 0; i != cfg.N; ++i) {
        uint8_t tmp = y0;
        y0 = y1;
        y1 = y2;
        y2 = y3;
        y3 = y4;
        y4 = tmp;
        if (cfg.frozen_mask[i] != 0U) {
            continue;
        }
        if (pc_idx < cfg.n_pc && i == cfg.pc_set[pc_idx]) {
            input[i] = y0;
            ++pc_idx;
        } else {
            input[i] = deallocated[msg_idx++];
            y0 ^= input[i];
        }
    }
    return input;
}

static bool run_uci_case(const char* name, int payload_bits, int filler_bits, int E, int crc_order)
{
    polar_code_config_t cfg{};
    int K = payload_bits + filler_bits + crc_order;
    POLAR_CHECK(polar_code_configure(&cfg, K, E, 10, POLAR_IBIL_PRESENT));

    polar_handle_t handle = nullptr;
    POLAR_CHECK(polar_create(&handle));
    POLAR_CHECK(polar_configure(handle, &cfg));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<uint8_t> deallocated = make_uci_deallocated_bits(payload_bits, filler_bits, crc_order, 0xabc000U + E + K);
    std::vector<uint8_t> input_u = make_uci_input_u(cfg, deallocated);
    std::vector<uint8_t> ref_rm = cpu_rate_match(cfg, cpu_encode(input_u, cfg.N));
    std::vector<__half> h_llr(cfg.E);
    for (int i = 0; i != cfg.E; ++i) {
        h_llr[i] = __float2half(ref_rm[i] ? -20.0F : 20.0F);
    }

    __half* d_llr = nullptr;
    polar_uci_decode_result_t* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_llr, cfg.E * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(polar_uci_decode_result_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_llr, h_llr.data(), cfg.E * sizeof(__half), cudaMemcpyHostToDevice, stream));
    POLAR_CHECK(polar_uci_rate_dematch_decode_crc_half(handle, d_result, d_llr, payload_bits, filler_bits, crc_order, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    polar_uci_decode_result_t result{};
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(result), cudaMemcpyDeviceToHost));
    bool ok = result.decoded != 0 && result.status != 0 && result.nof_bits == payload_bits;
    for (int i = 0; ok && i != payload_bits; ++i) {
        uint8_t expected = deallocated[filler_bits + i] & 1U;
        if ((result.payload[i] & 1U) != expected) {
            std::fprintf(stderr, "%s payload mismatch at %d: got=%u expected=%u\n", name, i, result.payload[i], expected);
            ok = false;
        }
    }
    std::printf("%-18s K=%4d E=%4d N=%4d crc=%2d filler=%d : %s\n", name, cfg.K, cfg.E, cfg.N, crc_order, filler_bits, ok ? "PASS" : "FAIL");

    cudaFree(d_result);
    cudaFree(d_llr);
    cudaStreamDestroy(stream);
    polar_destroy(handle);
    return ok;
}

static bool run_uci_segmented_case(const char* name, int payload_bits, int total_E, int crc_order)
{
    constexpr int codeblocks = 2;
    int cb_payload_bits[codeblocks] = {payload_bits / codeblocks, (payload_bits + 1) / codeblocks};
    int cb_filler_bits[codeblocks]  = {payload_bits % codeblocks, 0};
    int cb_E[codeblocks]            = {total_E / codeblocks, total_E - (total_E / codeblocks)};

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<__half> h_llr(total_E);
    polar_handle_t handles[codeblocks] = {};
    std::vector<std::vector<uint8_t>> expected_payload(codeblocks);

    int llr_offset = 0;
    for (int cb = 0; cb != codeblocks; ++cb) {
        polar_code_config_t cfg{};
        int K = cb_payload_bits[cb] + cb_filler_bits[cb] + crc_order;
        POLAR_CHECK(polar_code_configure(&cfg, K, cb_E[cb], 10, POLAR_IBIL_PRESENT));
        POLAR_CHECK(polar_create(&handles[cb]));
        POLAR_CHECK(polar_configure(handles[cb], &cfg));

        std::vector<uint8_t> deallocated = make_uci_deallocated_bits(
            cb_payload_bits[cb], cb_filler_bits[cb], crc_order, 0xdef000U + total_E + payload_bits + cb);
        expected_payload[cb].assign(deallocated.begin() + cb_filler_bits[cb],
                                    deallocated.begin() + cb_filler_bits[cb] + cb_payload_bits[cb]);
        std::vector<uint8_t> input_u = make_uci_input_u(cfg, deallocated);
        std::vector<uint8_t> ref_rm = cpu_rate_match(cfg, cpu_encode(input_u, cfg.N));
        for (int i = 0; i != cfg.E; ++i) {
            h_llr[llr_offset + i] = __float2half(ref_rm[i] ? -20.0F : 20.0F);
        }
        llr_offset += cfg.E;
    }

    __half* d_llr = nullptr;
    polar_uci_decode_result_t* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_llr, total_E * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_result, codeblocks * sizeof(polar_uci_decode_result_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_llr, h_llr.data(), total_E * sizeof(__half), cudaMemcpyHostToDevice, stream));

    llr_offset = 0;
    for (int cb = 0; cb != codeblocks; ++cb) {
        POLAR_CHECK(polar_uci_rate_dematch_decode_crc_half(handles[cb],
                                                           d_result + cb,
                                                           d_llr + llr_offset,
                                                           cb_payload_bits[cb],
                                                           cb_filler_bits[cb],
                                                           crc_order,
                                                           stream));
        llr_offset += cb_E[cb];
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    polar_uci_decode_result_t results[codeblocks] = {};
    CUDA_CHECK(cudaMemcpy(results, d_result, sizeof(results), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int cb = 0; cb != codeblocks; ++cb) {
        ok = ok && results[cb].decoded != 0 && results[cb].status != 0 && results[cb].nof_bits == cb_payload_bits[cb];
        for (int i = 0; ok && i != cb_payload_bits[cb]; ++i) {
            if ((results[cb].payload[i] & 1U) != (expected_payload[cb][i] & 1U)) {
                std::fprintf(stderr,
                             "%s cb%d payload mismatch at %d: got=%u expected=%u\n",
                             name,
                             cb,
                             i,
                             results[cb].payload[i],
                             expected_payload[cb][i]);
                ok = false;
            }
        }
    }

    std::printf("%-18s A=%4d E=%4d split=%d/%d crc=%2d : %s\n",
                name,
                payload_bits,
                total_E,
                cb_E[0],
                cb_E[1],
                crc_order,
                ok ? "PASS" : "FAIL");

    cudaFree(d_result);
    cudaFree(d_llr);
    for (polar_handle_t handle : handles) {
        polar_destroy(handle);
    }
    cudaStreamDestroy(stream);
    return ok;
}


int main()
{
    const test_case_t cases[] = {
        {"bch_repetition", 56, 864, 9, POLAR_IBIL_NOT_PRESENT},
        {"dci_short", 40, 100, 9, POLAR_IBIL_NOT_PRESENT},
        {"uci_repetition", 20, 256, 10, POLAR_IBIL_PRESENT},
        {"uci_puncture", 18, 45, 10, POLAR_IBIL_PRESENT},
        {"uci_shorten", 18, 38, 10, POLAR_IBIL_PRESENT},
        {"uci_large", 64, 1024, 10, POLAR_IBIL_PRESENT},
    };

    bool ok = true;
    for (const auto& tc : cases) {
        ok = run_case(tc) && ok;
    }
    ok = run_uci_case("uci_crc6_pc", 12, 0, 64, 6) && ok;
    ok = run_uci_case("uci_crc11", 32, 0, 256, 11) && ok;
    ok = run_uci_case("uci_crc11_fill", 506, 1, 1024, 11) && ok;
    ok = run_uci_segmented_case("uci_2cb_even", 360, 1088, 11) && ok;
    ok = run_uci_segmented_case("uci_2cb_odd", 360, 1089, 11) && ok;
    std::printf("Polar CUDA correctness: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
