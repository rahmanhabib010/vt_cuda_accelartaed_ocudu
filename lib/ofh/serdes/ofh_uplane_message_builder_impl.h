// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/ocudulog/logger.h"
#include "ocudu/ofh/serdes/ofh_uplane_message_builder.h"

namespace ocudu {
namespace ofh {

class iq_compressor;
class network_order_binary_serializer;

/// Open Fronthaul User-Plane message builder implementation.
class uplane_message_builder_impl : public uplane_message_builder
{
public:
  /// Creates a message builder using the given logger and IQ compressor.
  uplane_message_builder_impl(ocudulog::basic_logger& logger_, iq_compressor& compressor_) :
    logger(logger_), compressor(compressor_)
  {
  }

  /// \copydoc uplane_message_builder::build_message(span<uint8_t>, span<const cbf16_t>, const uplane_message_params&)
  unsigned
  build_message(span<uint8_t> buffer, span<const cbf16_t> iq_data, const uplane_message_params& params) override;

  /// \copydoc uplane_message_builder::build_message(span<uint8_t>, const resource_grid_reader&, unsigned, const
  /// uplane_message_params&, unsigned&)
  bool build_message(span<uint8_t>                buffer,
                     const resource_grid_reader&  grid,
                     unsigned                     port,
                     const uplane_message_params& params,
                     unsigned&                    bytes_written) override;

  /// \copydoc uplane_message_builder::build_messages(span<span<uint8_t>>, const resource_grid_reader&, unsigned, const
  /// uplane_message_params&, span<unsigned>)
  bool build_messages(span<span<uint8_t>>       buffers,
                      const resource_grid_reader& grid,
                      unsigned                    first_port,
                      const uplane_message_params& params,
                      span<unsigned>              bytes_written) override;

  /// \copydoc uplane_message_builder::build_symbol_batch_messages(span<span<uint8_t>>, const resource_grid_reader&,
  /// unsigned, unsigned, unsigned, const uplane_message_params&, span<unsigned>)
  bool build_symbol_batch_messages(span<span<uint8_t>>       buffers,
                                   const resource_grid_reader& grid,
                                   unsigned                    first_port,
                                   unsigned                    first_symbol,
                                   unsigned                    nof_symbols,
                                   const uplane_message_params& params,
                                   span<unsigned>              bytes_written) override;

private:
  /// Serializes IQ data from a resource grid.
  void serialize_iq_data(network_order_binary_serializer& serializer,
                         span<const cbf16_t>              iq_data,
                         unsigned                         nof_prbs,
                         const ru_compression_params&     params);

  /// Serializes IQ data directly from a device-visible resource-grid symbol when the compressor supports it.
  bool serialize_iq_data(network_order_binary_serializer& serializer,
                         const resource_grid_reader&      grid,
                         unsigned                         port,
                         const uplane_message_params&     params);

  /// Serializes compression header. Implementation depends on whether static or non-static IQ format is configured.
  virtual void serialize_compression_header(network_order_binary_serializer& serializer,
                                            const ru_compression_params&     params) = 0;

protected:
  /// True when the builder serializes the optional udCompLen field.
  const bool              ud_comp_length_support = false;
  /// Logger used for builder diagnostics.
  ocudulog::basic_logger& logger;
  /// IQ compressor used by host and device-visible serialization paths.
  iq_compressor&          compressor;
};

} // namespace ofh
} // namespace ocudu
