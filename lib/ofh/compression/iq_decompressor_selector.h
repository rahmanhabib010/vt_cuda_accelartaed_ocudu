// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/ofh/compression/iq_decompressor.h"

namespace ocudu {
namespace ofh {

/// \brief IQ decompressor selector implementation.
///
/// The selector will select the IQ decompressor between the registered ones to decompress IQ samples, based on the
/// given compression parameters.
class iq_decompressor_selector : public iq_decompressor
{
public:
  explicit iq_decompressor_selector(
      std::array<std::unique_ptr<iq_decompressor>, NOF_COMPRESSION_TYPES_SUPPORTED> decompressors_);

  // See interface for documentation.
  void
  decompress(span<cbf16_t> iq_data, span<const uint8_t> compressed_data, const ru_compression_params& params) override;

  // See interface for documentation.
  bool supports_resource_grid_decompression() const override;

  // See interface for documentation.
  bool supports_prach_buffer_decompression() const override;

  // See interface for documentation.
  bool decompress_to_resource_grid(resource_grid_writer&        grid,
                                   unsigned                     port,
                                   unsigned                     symbol,
                                   unsigned                     start_prb,
                                   unsigned                     nof_prbs,
                                   span<const uint8_t>          compressed_data,
                                   const ru_compression_params& params) override;

  // See interface for documentation.
  bool decompress_to_prach_buffer(prach_buffer&                buffer,
                                  unsigned                     port,
                                  unsigned                     td_occasion,
                                  unsigned                     fd_occasion,
                                  unsigned                     symbol,
                                  unsigned                     start_re,
                                  unsigned                     input_start_re,
                                  unsigned                     nof_re,
                                  unsigned                     nof_prbs,
                                  span<const uint8_t>          compressed_data,
                                  const ru_compression_params& params) override;

private:
  std::array<std::unique_ptr<iq_decompressor>, NOF_COMPRESSION_TYPES_SUPPORTED> decompressors;
};

} // namespace ofh
} // namespace ocudu
