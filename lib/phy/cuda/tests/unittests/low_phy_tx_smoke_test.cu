// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "low_phy_tx.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

int main()
{
    ocudu_lowphy_tx_config_t cfg = {};
    cfg.dft_size          = 128;
    cfg.rg_size           = 72;
    cfg.nof_ports         = 4;
    cfg.nof_symbols       = 2;
    cfg.ofdm_scale        = 1.0F;
    cfg.amplitude_gain    = 1.0F;
    cfg.clipping_ceiling  = 1.0F;
    cfg.clipping_enabled  = 0;

    int sample_offset = 0;
    for (int symbol = 0; symbol != cfg.nof_symbols; ++symbol) {
        cfg.cp_lengths[symbol]    = 8;
        cfg.symbol_offsets[symbol] = sample_offset;
        cfg.phase_re[symbol]       = 1.0F;
        cfg.phase_im[symbol]       = 0.0F;
        sample_offset += cfg.dft_size + cfg.cp_lengths[symbol];
    }
    cfg.nof_samples = sample_offset;

    ocudu_lowphy_tx_handle_t* handle = nullptr;
    if (!ocudu_lowphy_tx_create(&cfg, &handle)) {
        std::fprintf(stderr, "Failed to create low-PHY TX handle.\n");
        return 1;
    }

    std::vector<uint32_t> grid(static_cast<size_t>(cfg.nof_ports) * cfg.nof_symbols * cfg.rg_size, 0);
    for (int port = 0; port != cfg.nof_ports; ++port) {
        for (int symbol = 0; symbol != cfg.nof_symbols; ++symbol) {
            size_t index =
                (static_cast<size_t>(port) * cfg.nof_symbols + symbol) * cfg.rg_size + cfg.rg_size / 2;
            grid[index] = pack_cbf16(0.25F + 0.05F * static_cast<float>(port), 0.0F);
        }
    }

    std::vector<int16_t> output(static_cast<size_t>(cfg.nof_ports) * cfg.nof_samples * 2U, 0);
    void* output_ports[OCUDU_LOWPHY_TX_MAX_PORTS] = {};
    for (int port = 0; port != cfg.nof_ports; ++port) {
        output_ports[port] = output.data() + static_cast<size_t>(port) * cfg.nof_samples * 2U;
    }

    if (!ocudu_lowphy_tx_process_host_grid(handle, grid.data(), output_ports, nullptr) ||
        !ocudu_lowphy_tx_synchronize(handle)) {
        std::fprintf(stderr, "Failed to run low-PHY TX host-grid path.\n");
        ocudu_lowphy_tx_destroy(handle);
        return 1;
    }

    ocudu_lowphy_tx_destroy(handle);

    auto is_nonzero = [](int16_t value) { return value != 0; };
    for (int port = 0; port != cfg.nof_ports; ++port) {
        auto port_begin = output.begin() + static_cast<size_t>(port) * cfg.nof_samples * 2U;
        auto port_end   = port_begin + static_cast<size_t>(cfg.nof_samples) * 2U;
        if (!std::any_of(port_begin, port_end, is_nonzero)) {
            std::fprintf(stderr, "Low-PHY TX output for port %d is all zero.\n", port);
            return 1;
        }
    }

    return 0;
}
