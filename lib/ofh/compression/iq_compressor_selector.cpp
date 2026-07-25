// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "iq_compressor_selector.h"
#include "ocudu/support/error_handling.h"

using namespace ocudu;
using namespace ofh;

iq_compressor_selector::iq_compressor_selector(
    std::array<std::unique_ptr<iq_compressor>, NOF_COMPRESSION_TYPES_SUPPORTED> compressors_) :
  compressors(std::move(compressors_))
{
  // Sanity check that all the positions in the array has a valid compressor.
  for (unsigned i = 0, e = compressors.size(); i != e; ++i) {
    report_fatal_error_if_not(compressors[i],
                              "Null compressor detected for compression type '{}'",
                              to_string(static_cast<compression_type>(i)));
  }
}

void iq_compressor_selector::compress(span<uint8_t>                buffer,
                                      span<const cbf16_t>          iq_data,
                                      const ru_compression_params& params)
{
  auto& compressor = compressors[static_cast<unsigned>(params.type)];
  compressor->compress(buffer, iq_data, params);
}

bool iq_compressor_selector::compress_device_symbol(span<uint8_t>                buffer,
                                                    const resource_grid_reader&  grid,
                                                    unsigned                     port,
                                                    unsigned                     symbol,
                                                    unsigned                     start_prb,
                                                    unsigned                     nof_prbs,
                                                    const ru_compression_params& params)
{
  auto& compressor = compressors[static_cast<unsigned>(params.type)];
  return compressor->compress_device_symbol(buffer, grid, port, symbol, start_prb, nof_prbs, params);
}

bool iq_compressor_selector::compress_device_symbols(span<uint8_t>                buffer,
                                                     unsigned                     port_stride_bytes,
                                                     const resource_grid_reader&  grid,
                                                     unsigned                     first_port,
                                                     unsigned                     nof_ports,
                                                     unsigned                     symbol,
                                                     unsigned                     start_prb,
                                                     unsigned                     nof_prbs,
                                                     const ru_compression_params& params)
{
  auto& compressor = compressors[static_cast<unsigned>(params.type)];
  return compressor->compress_device_symbols(
      buffer, port_stride_bytes, grid, first_port, nof_ports, symbol, start_prb, nof_prbs, params);
}

bool iq_compressor_selector::compress_device_symbols_to_buffers(span<span<uint8_t>>          buffers,
                                                                const resource_grid_reader&  grid,
                                                                unsigned                     first_port,
                                                                unsigned                     nof_ports,
                                                                unsigned                     symbol,
                                                                unsigned                     start_prb,
                                                                unsigned                     nof_prbs,
                                                                const ru_compression_params& params)
{
  auto& compressor = compressors[static_cast<unsigned>(params.type)];
  return compressor->compress_device_symbols_to_buffers(
      buffers, grid, first_port, nof_ports, symbol, start_prb, nof_prbs, params);
}

bool iq_compressor_selector::compress_device_symbol_batch(span<uint8_t>                buffer,
                                                          unsigned                     symbol_stride_bytes,
                                                          unsigned                     port_stride_bytes,
                                                          const resource_grid_reader&  grid,
                                                          unsigned                     first_port,
                                                          unsigned                     nof_ports,
                                                          unsigned                     first_symbol,
                                                          unsigned                     nof_symbols,
                                                          unsigned                     start_prb,
                                                          unsigned                     nof_prbs,
                                                          const ru_compression_params& params)
{
  auto& compressor = compressors[static_cast<unsigned>(params.type)];
  return compressor->compress_device_symbol_batch(buffer,
                                                 symbol_stride_bytes,
                                                 port_stride_bytes,
                                                 grid,
                                                 first_port,
                                                 nof_ports,
                                                 first_symbol,
                                                 nof_symbols,
                                                 start_prb,
                                                 nof_prbs,
                                                 params);
}

bool iq_compressor_selector::compress_device_symbol_batch_to_buffers(span<span<uint8_t>>          buffers,
                                                                     const resource_grid_reader&  grid,
                                                                     unsigned                     first_port,
                                                                     unsigned                     nof_ports,
                                                                     unsigned                     first_symbol,
                                                                     unsigned                     nof_symbols,
                                                                     unsigned                     start_prb,
                                                                     unsigned                     nof_prbs,
                                                                     const ru_compression_params& params)
{
  auto& compressor = compressors[static_cast<unsigned>(params.type)];
  return compressor->compress_device_symbol_batch_to_buffers(
      buffers, grid, first_port, nof_ports, first_symbol, nof_symbols, start_prb, nof_prbs, params);
}
