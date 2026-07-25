// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/adt/complex.h"
#include "ocudu/adt/span.h"
#include "ocudu/ofh/compression/compression_params.h"

namespace ocudu {

class prach_buffer;
class resource_grid_writer;

namespace ofh {

/// \brief Describes the IQ data decompressor.
///
/// Deserializes compression parameters and decompresses received PRBs with compressed IQ data according to compression
/// methods specified in WG4.CUS.0 document.
class iq_decompressor
{
public:
  /// Default destructor.
  virtual ~iq_decompressor() = default;

  /// \brief Decompress received compressed PRBs.
  ///
  /// Decompresses compressed PRBs from the input buffer according to received compression parameters and puts the
  /// results into an array of brain floating point IQ samples.
  ///
  /// \param[out] iq_data  Resulting IQ samples after decompression.
  /// \param[in]  compressed_data A span containing received compressed IQ data and compression parameters.
  /// \param[in]  params  Compression parameters.
  virtual void
  decompress(span<cbf16_t> iq_data, span<const uint8_t> compressed_data, const ru_compression_params& params) = 0;

  /// Returns true if the decompressor can write directly into a device-capable resource grid.
  virtual bool supports_resource_grid_decompression() const { return false; }

  /// Returns true if the decompressor can write directly into a device-capable PRACH buffer.
  virtual bool supports_prach_buffer_decompression() const { return false; }

  /// \brief Decompress received compressed PRBs directly into a resource grid.
  ///
  /// Implementations return false when direct grid decompression is unavailable for the destination or compression
  /// parameters. The default implementation leaves the existing host decompression path unchanged.
  ///
  /// \param[out] grid            Destination resource grid.
  /// \param[in]  port            Resource-grid port index.
  /// \param[in]  symbol          OFDM symbol index.
  /// \param[in]  start_prb       First destination PRB.
  /// \param[in]  nof_prbs        Number of PRBs to decompress.
  /// \param[in]  compressed_data Received compressed IQ payload and compression parameters.
  /// \param[in]  params          Compression parameters.
  /// \return True on success, false when this path is unsupported or cannot be used.
  virtual bool decompress_to_resource_grid(resource_grid_writer&        grid,
                                           unsigned                     port,
                                           unsigned                     symbol,
                                           unsigned                     start_prb,
                                           unsigned                     nof_prbs,
                                           span<const uint8_t>          compressed_data,
                                           const ru_compression_params& params)
  {
    (void)grid;
    (void)port;
    (void)symbol;
    (void)start_prb;
    (void)nof_prbs;
    (void)compressed_data;
    (void)params;
    return false;
  }

  /// \brief Decompress received compressed PRBs directly into a PRACH buffer.
  ///
  /// \param[out] buffer          Destination PRACH buffer.
  /// \param[in]  port            PRACH receive port index.
  /// \param[in]  td_occasion     PRACH time-domain occasion index.
  /// \param[in]  fd_occasion     PRACH frequency-domain occasion index.
  /// \param[in]  symbol          PRACH symbol index within the occasion.
  /// \param[in]  start_re        First destination RE in the PRACH symbol.
  /// \param[in]  input_start_re  First compressed-payload RE to consume.
  /// \param[in]  nof_re          Number of RE to decompress.
  /// \param[in]  nof_prbs        Number of PRBs represented by the compressed payload.
  /// \param[in]  compressed_data Received compressed IQ payload and compression parameters.
  /// \param[in]  params          Compression parameters.
  /// \return True on success, false when this path is unsupported or cannot be used.
  virtual bool decompress_to_prach_buffer(prach_buffer&                buffer,
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
    (void)buffer;
    (void)port;
    (void)td_occasion;
    (void)fd_occasion;
    (void)symbol;
    (void)start_re;
    (void)input_start_re;
    (void)nof_re;
    (void)nof_prbs;
    (void)compressed_data;
    (void)params;
    return false;
  }
};

} // namespace ofh
} // namespace ocudu
