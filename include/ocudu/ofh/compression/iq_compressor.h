// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/adt/complex.h"
#include "ocudu/adt/span.h"
#include "ocudu/ofh/compression/compression_params.h"

namespace ocudu {
class resource_grid_reader;

namespace ofh {

/// \brief Describes the IQ data compressor.
///
/// Compresses and serializes floating point IQ samples together with compression parameters according to compression
/// methods specified in WG4.CUS.0 document.
class iq_compressor
{
public:
  /// Default destructor.
  virtual ~iq_compressor() = default;

  /// \brief Compress input IQ samples.
  ///
  /// Compresses IQ samples from the input buffer according to received compression parameters and puts the results into
  /// an array of compressed PRBs.
  ///
  /// \param[out] buffer   Buffer where the compressed IQ data and compression parameters will be stored.
  /// \param[in]  iq_data  IQ samples to be compressed.
  /// \param[in]  params   Compression parameters.
  virtual void compress(span<uint8_t> buffer, span<const cbf16_t> iq_data, const ru_compression_params& params) = 0;

  /// \brief Compress IQ samples directly from a device-visible resource-grid symbol.
  ///
  /// Implementations that cannot consume a device-visible grid return false and leave \c buffer unspecified.
  ///
  /// \param[out] buffer    Buffer where compressed IQ data and compression parameters will be stored.
  /// \param[in]  grid      Resource-grid reader exposing the source symbol.
  /// \param[in]  port      Resource-grid port index.
  /// \param[in]  symbol    OFDM symbol index.
  /// \param[in]  start_prb First PRB to compress.
  /// \param[in]  nof_prbs Number of PRBs to compress.
  /// \param[in]  params    Compression parameters.
  /// \return True on success, false when the implementation cannot use this path.
  virtual bool compress_device_symbol(span<uint8_t>                buffer,
                                      const resource_grid_reader&  grid,
                                      unsigned                     port,
                                      unsigned                     symbol,
                                      unsigned                     start_prb,
                                      unsigned                     nof_prbs,
                                      const ru_compression_params& params)
  {
    (void)buffer;
    (void)grid;
    (void)port;
    (void)symbol;
    (void)start_prb;
    (void)nof_prbs;
    (void)params;
    return false;
  }

  /// \brief Compress IQ samples for contiguous resource-grid ports directly from a device-visible grid.
  ///
  /// Implementations that cannot batch a device-visible grid return false and leave \c buffer unspecified.
  ///
  /// \param[out] buffer             Buffer containing one compressed port after another.
  /// \param[in]  port_stride_bytes  Byte distance between two adjacent compressed ports in \c buffer.
  /// \param[in]  grid               Resource-grid reader exposing the source symbols.
  /// \param[in]  first_port         First resource-grid port index.
  /// \param[in]  nof_ports          Number of contiguous ports to compress.
  /// \param[in]  symbol             OFDM symbol index.
  /// \param[in]  start_prb          First PRB to compress.
  /// \param[in]  nof_prbs           Number of PRBs to compress per port.
  /// \param[in]  params             Compression parameters.
  /// \return True on success, false when the implementation cannot use this path.
  virtual bool compress_device_symbols(span<uint8_t>                buffer,
                                       unsigned                     port_stride_bytes,
                                       const resource_grid_reader&  grid,
                                       unsigned                     first_port,
                                       unsigned                     nof_ports,
                                       unsigned                     symbol,
                                       unsigned                     start_prb,
                                       unsigned                     nof_prbs,
                                       const ru_compression_params& params)
  {
    (void)buffer;
    (void)port_stride_bytes;
    (void)grid;
    (void)first_port;
    (void)nof_ports;
    (void)symbol;
    (void)start_prb;
    (void)nof_prbs;
    (void)params;
    return false;
  }

  /// \brief Compress IQ samples for contiguous resource-grid ports directly into final payload buffers.
  ///
  /// The payload buffers are ordered by port. This optional path lets implementations avoid an intermediate host batch
  /// buffer when packet payload buffers are already available.
  ///
  /// \param[out] buffers    Payload buffers where each compressed port will be written.
  /// \param[in]  grid       Resource-grid reader exposing the source symbol.
  /// \param[in]  first_port First resource-grid port index.
  /// \param[in]  nof_ports  Number of contiguous ports to compress.
  /// \param[in]  symbol     OFDM symbol index.
  /// \param[in]  start_prb  First PRB to compress.
  /// \param[in]  nof_prbs   Number of PRBs to compress per port.
  /// \param[in]  params     Compression parameters.
  /// \return True on success, false when the implementation cannot use this path.
  virtual bool compress_device_symbols_to_buffers(span<span<uint8_t>>          buffers,
                                                  const resource_grid_reader&  grid,
                                                  unsigned                     first_port,
                                                  unsigned                     nof_ports,
                                                  unsigned                     symbol,
                                                  unsigned                     start_prb,
                                                  unsigned                     nof_prbs,
                                                  const ru_compression_params& params)
  {
    (void)buffers;
    (void)grid;
    (void)first_port;
    (void)nof_ports;
    (void)symbol;
    (void)start_prb;
    (void)nof_prbs;
    (void)params;
    return false;
  }

  /// \brief Compress IQ samples for a symbol and port batch directly from a device-visible grid.
  ///
  /// The output layout is symbol-major and then port-major.
  ///
  /// Implementations that cannot batch a device-visible grid return false and leave \c buffer unspecified.
  ///
  /// \param[out] buffer                Buffer containing compressed symbols and ports.
  /// \param[in]  symbol_stride_bytes   Byte distance between two adjacent compressed symbols in \c buffer.
  /// \param[in]  port_stride_bytes     Byte distance between two adjacent compressed ports inside one symbol.
  /// \param[in]  grid                  Resource-grid reader exposing the source symbols.
  /// \param[in]  first_port            First resource-grid port index.
  /// \param[in]  nof_ports             Number of contiguous ports to compress.
  /// \param[in]  first_symbol          First OFDM symbol index.
  /// \param[in]  nof_symbols           Number of OFDM symbols to compress.
  /// \param[in]  start_prb             First PRB to compress.
  /// \param[in]  nof_prbs              Number of PRBs to compress per symbol and port.
  /// \param[in]  params                Compression parameters.
  /// \return True on success, false when the implementation cannot use this path.
  virtual bool compress_device_symbol_batch(span<uint8_t>                buffer,
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
    (void)buffer;
    (void)symbol_stride_bytes;
    (void)port_stride_bytes;
    (void)grid;
    (void)first_port;
    (void)nof_ports;
    (void)first_symbol;
    (void)nof_symbols;
    (void)start_prb;
    (void)nof_prbs;
    (void)params;
    return false;
  }

  /// \brief Compress IQ samples for a symbol and port batch directly into final payload buffers.
  ///
  /// The payload buffers are ordered by symbol first and port second. This optional path lets implementations avoid an
  /// intermediate host batch buffer when packet payload buffers are already available.
  ///
  /// \param[out] buffers      Payload buffers ordered by symbol first and port second.
  /// \param[in]  grid         Resource-grid reader exposing the source symbols.
  /// \param[in]  first_port   First resource-grid port index.
  /// \param[in]  nof_ports    Number of contiguous ports to compress.
  /// \param[in]  first_symbol First OFDM symbol index.
  /// \param[in]  nof_symbols  Number of OFDM symbols to compress.
  /// \param[in]  start_prb    First PRB to compress.
  /// \param[in]  nof_prbs     Number of PRBs to compress per symbol and port.
  /// \param[in]  params       Compression parameters.
  /// \return True on success, false when the implementation cannot use this path.
  virtual bool compress_device_symbol_batch_to_buffers(span<span<uint8_t>>          buffers,
                                                       const resource_grid_reader&  grid,
                                                       unsigned                     first_port,
                                                       unsigned                     nof_ports,
                                                       unsigned                     first_symbol,
                                                       unsigned                     nof_symbols,
                                                       unsigned                     start_prb,
                                                       unsigned                     nof_prbs,
                                                       const ru_compression_params& params)
  {
    (void)buffers;
    (void)grid;
    (void)first_port;
    (void)nof_ports;
    (void)first_symbol;
    (void)nof_symbols;
    (void)start_prb;
    (void)nof_prbs;
    (void)params;
    return false;
  }
};

} // namespace ofh
} // namespace ocudu
