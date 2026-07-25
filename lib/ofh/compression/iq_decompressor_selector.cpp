// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "iq_decompressor_selector.h"
#include "ocudu/support/error_handling.h"
#include <algorithm>

using namespace ocudu;
using namespace ofh;

iq_decompressor_selector::iq_decompressor_selector(
    std::array<std::unique_ptr<iq_decompressor>, NOF_COMPRESSION_TYPES_SUPPORTED> decompressors_) :
  decompressors(std::move(decompressors_))
{
  // Sanity check that all the positions in the array has a decompressor.
  for (unsigned i = 0, e = decompressors.size(); i != e; ++i) {
    report_fatal_error_if_not(decompressors[i],
                              "Null decompressor detected for compression type '{}'",
                              to_string(static_cast<compression_type>(i)));
  }
}

void iq_decompressor_selector::decompress(span<cbf16_t>                iq_data,
                                          span<const uint8_t>          compressed_data,
                                          const ru_compression_params& params)
{
  return decompressors[static_cast<unsigned>(params.type)]->decompress(iq_data, compressed_data, params);
}

bool iq_decompressor_selector::supports_resource_grid_decompression() const
{
  return std::any_of(decompressors.begin(), decompressors.end(), [](const auto& decompressor) {
    return decompressor->supports_resource_grid_decompression();
  });
}

bool iq_decompressor_selector::supports_prach_buffer_decompression() const
{
  return std::any_of(decompressors.begin(), decompressors.end(), [](const auto& decompressor) {
    return decompressor->supports_prach_buffer_decompression();
  });
}

bool iq_decompressor_selector::decompress_to_resource_grid(resource_grid_writer&        grid,
                                                           unsigned                     port,
                                                           unsigned                     symbol,
                                                           unsigned                     start_prb,
                                                           unsigned                     nof_prbs,
                                                           span<const uint8_t>          compressed_data,
                                                           const ru_compression_params& params)
{
  return decompressors[static_cast<unsigned>(params.type)]->decompress_to_resource_grid(
      grid, port, symbol, start_prb, nof_prbs, compressed_data, params);
}

bool iq_decompressor_selector::decompress_to_prach_buffer(prach_buffer&                buffer,
                                                          unsigned                     port,
                                                          unsigned                     td_occasion,
                                                          unsigned                     fd_occasion,
                                                          unsigned                     symbol,
                                                          unsigned                     start_re,
                                                          unsigned                     input_start_re,
                                                          unsigned                     nof_re,
                                                          unsigned                     nof_prbs,
                                                          span<const uint8_t>          compressed_data,
                                                          const ru_compression_params& params)
{
  return decompressors[static_cast<unsigned>(params.type)]->decompress_to_prach_buffer(buffer,
                                                                                       port,
                                                                                       td_occasion,
                                                                                       fd_occasion,
                                                                                       symbol,
                                                                                       start_re,
                                                                                       input_start_re,
                                                                                       nof_re,
                                                                                       nof_prbs,
                                                                                       compressed_data,
                                                                                       params);
}
