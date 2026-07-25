// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/ofh/compression/iq_compressor.h"

namespace ocudu {
namespace ofh {

/// \brief IQ compressor selector implementation.
///
/// The selector will select the IQ compressor between the registered ones to compress IQ samples, based on the
/// given compression parameters.
class iq_compressor_selector : public iq_compressor
{
public:
  explicit iq_compressor_selector(
      std::array<std::unique_ptr<iq_compressor>, NOF_COMPRESSION_TYPES_SUPPORTED> compressors_);

  // See interface for documentation.
  void compress(span<uint8_t> buffer, span<const cbf16_t> iq_data, const ru_compression_params& params) override;

  // See interface for documentation.
  bool compress_device_symbol(span<uint8_t>                buffer,
                              const resource_grid_reader&  grid,
                              unsigned                     port,
                              unsigned                     symbol,
                              unsigned                     start_prb,
                              unsigned                     nof_prbs,
                              const ru_compression_params& params) override;

  // See interface for documentation.
  bool compress_device_symbols(span<uint8_t>                buffer,
                               unsigned                     port_stride_bytes,
                               const resource_grid_reader&  grid,
                               unsigned                     first_port,
                               unsigned                     nof_ports,
                               unsigned                     symbol,
                               unsigned                     start_prb,
                               unsigned                     nof_prbs,
                               const ru_compression_params& params) override;

  // See interface for documentation.
  bool compress_device_symbols_to_buffers(span<span<uint8_t>>          buffers,
                                          const resource_grid_reader&  grid,
                                          unsigned                     first_port,
                                          unsigned                     nof_ports,
                                          unsigned                     symbol,
                                          unsigned                     start_prb,
                                          unsigned                     nof_prbs,
                                          const ru_compression_params& params) override;

  // See interface for documentation.
  bool compress_device_symbol_batch(span<uint8_t>                buffer,
                                    unsigned                     symbol_stride_bytes,
                                    unsigned                     port_stride_bytes,
                                    const resource_grid_reader&  grid,
                                    unsigned                     first_port,
                                    unsigned                     nof_ports,
                                    unsigned                     first_symbol,
                                    unsigned                     nof_symbols,
                                    unsigned                     start_prb,
                                    unsigned                     nof_prbs,
                                    const ru_compression_params& params) override;

  // See interface for documentation.
  bool compress_device_symbol_batch_to_buffers(span<span<uint8_t>>          buffers,
                                               const resource_grid_reader&  grid,
                                               unsigned                     first_port,
                                               unsigned                     nof_ports,
                                               unsigned                     first_symbol,
                                               unsigned                     nof_symbols,
                                               unsigned                     start_prb,
                                               unsigned                     nof_prbs,
                                               const ru_compression_params& params) override;

private:
  std::array<std::unique_ptr<iq_compressor>, NOF_COMPRESSION_TYPES_SUPPORTED> compressors;
};

} // namespace ofh
} // namespace ocudu
