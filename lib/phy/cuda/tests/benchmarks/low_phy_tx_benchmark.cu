// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "low_phy_tx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>

static uint16_t float_to_bf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static uint32_t pack_cbf16(float re, float im)
{
    return static_cast<uint32_t>(float_to_bf16(re)) | (static_cast<uint32_t>(float_to_bf16(im)) << 16);
}

static ocudu_lowphy_tx_config_t make_20mhz_30khz_config(int nof_ports)
{
    ocudu_lowphy_tx_config_t cfg = {};
    cfg.dft_size         = 768;
    cfg.rg_size          = 612;
    cfg.nof_ports        = nof_ports;
    cfg.nof_symbols      = 14;
    cfg.ofdm_scale       = 1.0F;
    cfg.amplitude_gain   = 1.0F;
    cfg.clipping_ceiling = 1.0F;
    cfg.clipping_enabled = 0;

    int sample_offset = 0;
    for (int symbol = 0; symbol != cfg.nof_symbols; ++symbol) {
        cfg.cp_lengths[symbol]    = (symbol == 0) ? 66 : 54;
        cfg.symbol_offsets[symbol] = sample_offset;
        cfg.phase_re[symbol]       = 1.0F;
        cfg.phase_im[symbol]       = 0.0F;
        sample_offset += cfg.dft_size + cfg.cp_lengths[symbol];
    }
    cfg.nof_samples = sample_offset;
    return cfg;
}

static void fill_grid(std::vector<uint32_t>& grid, const ocudu_lowphy_tx_config_t& cfg)
{
    for (int port = 0; port != cfg.nof_ports; ++port) {
        for (int symbol = 0; symbol != cfg.nof_symbols; ++symbol) {
            for (int sc = 0; sc != cfg.rg_size; ++sc) {
                size_t index = (static_cast<size_t>(port) * cfg.nof_symbols + symbol) * cfg.rg_size + sc;
                if ((sc % 7) == ((port + symbol) % 7)) {
                    float re = static_cast<float>(((port + 3) * (symbol + 5) * (sc + 11)) % 29 - 14) * 4e-4F;
                    float im = static_cast<float>(((port + 7) * (symbol + 2) * (sc + 17)) % 31 - 15) * 4e-4F;
                    grid[index] = pack_cbf16(re, im);
                } else {
                    grid[index] = 0;
                }
            }
        }
    }
}

static double percentile_us(std::vector<double> values, double percentile)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(std::round(percentile * static_cast<double>(values.size() - 1)));
    return values[index];
}

int main(int argc, char** argv)
{
    int nof_ports = 1;
    int iterations = 1000;
    const char* mode = "pre-register";
    if (argc > 1) {
        nof_ports = std::max(1, std::atoi(argv[1]));
    }
    if (argc > 2) {
        iterations = std::max(1, std::atoi(argv[2]));
    }
    if (argc > 3) {
        mode = argv[3];
    }

    ocudu_lowphy_tx_config_t cfg = make_20mhz_30khz_config(nof_ports);

    ocudu_lowphy_tx_handle_t* handle = nullptr;
    if (!ocudu_lowphy_tx_create(&cfg, &handle)) {
        std::fprintf(stderr, "Failed to create low-PHY TX handle.\n");
        return 1;
    }

    std::vector<uint32_t> grid(static_cast<size_t>(cfg.nof_ports) * cfg.nof_symbols * cfg.rg_size, 0);
    fill_grid(grid, cfg);

    std::vector<int16_t> output(static_cast<size_t>(cfg.nof_ports) * cfg.nof_samples * 2U, 0);
    void* output_ports[OCUDU_LOWPHY_TX_MAX_PORTS] = {};
    for (int port = 0; port != cfg.nof_ports; ++port) {
        output_ports[port] = output.data() + static_cast<size_t>(port) * cfg.nof_samples * 2U;
    }
    const bool use_device_grid = std::strstr(mode, "device") != nullptr;
    const bool pre_register_output = std::strstr(mode, "pre-register") != nullptr;

    uint32_t* d_grid = nullptr;
    if (use_device_grid) {
        if (cudaMalloc(&d_grid, grid.size() * sizeof(uint32_t)) != cudaSuccess ||
            cudaMemcpy(d_grid, grid.data(), grid.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "Failed to prepare device grid.\n");
            cudaFree(d_grid);
            ocudu_lowphy_tx_destroy(handle);
            return 1;
        }
    }

    if (pre_register_output) {
        if (!ocudu_lowphy_tx_register_host_output(handle, output.data(), output.size() * sizeof(int16_t))) {
            std::fprintf(stderr, "Failed to pre-register output buffer.\n");
            cudaFree(d_grid);
            ocudu_lowphy_tx_destroy(handle);
            return 1;
        }
    }

    for (int i = 0; i != 50; ++i) {
        bool ok = use_device_grid ? ocudu_lowphy_tx_process(handle, d_grid, output_ports, nullptr)
                                  : ocudu_lowphy_tx_process_host_grid(handle, grid.data(), output_ports, nullptr);
        if (!ok ||
            !ocudu_lowphy_tx_synchronize(handle)) {
            std::fprintf(stderr, "Warmup failed.\n");
            cudaFree(d_grid);
            ocudu_lowphy_tx_destroy(handle);
            return 1;
        }
    }

    std::vector<double> elapsed_us;
    elapsed_us.reserve(iterations);
    for (int i = 0; i != iterations; ++i) {
        auto start = std::chrono::steady_clock::now();
        bool ok = use_device_grid ? ocudu_lowphy_tx_process(handle, d_grid, output_ports, nullptr)
                                  : ocudu_lowphy_tx_process_host_grid(handle, grid.data(), output_ports, nullptr);
        if (!ok ||
            !ocudu_lowphy_tx_synchronize(handle)) {
            std::fprintf(stderr, "Iteration %d failed.\n", i);
            cudaFree(d_grid);
            ocudu_lowphy_tx_destroy(handle);
            return 1;
        }
        auto stop = std::chrono::steady_clock::now();
        elapsed_us.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
    }

    double sum = std::accumulate(elapsed_us.begin(), elapsed_us.end(), 0.0);
    std::printf("20MHz 30kHz low-PHY TX host-grid, ports=%d iterations=%d samples=%d mode=%s\n",
                cfg.nof_ports,
                iterations,
                cfg.nof_samples,
                mode);
    std::printf("avg %.2f us p50 %.2f us p90 %.2f us p99 %.2f us max %.2f us\n",
                sum / static_cast<double>(elapsed_us.size()),
                percentile_us(elapsed_us, 0.50),
                percentile_us(elapsed_us, 0.90),
                percentile_us(elapsed_us, 0.99),
                *std::max_element(elapsed_us.begin(), elapsed_us.end()));

    cudaFree(d_grid);
    ocudu_lowphy_tx_destroy(handle);
    return 0;
}
