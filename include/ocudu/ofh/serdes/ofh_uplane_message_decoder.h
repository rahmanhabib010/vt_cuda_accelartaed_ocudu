// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/adt/span.h"
#include "ocudu/ofh/serdes/ofh_uplane_message_decoder_properties.h"
#include "ocudu/ofh/timing/slot_symbol_point.h"
#include <optional>

namespace ocudu {

class prach_buffer;
class resource_grid_writer;

namespace ofh {
namespace uplane_peeker {

/// Peeks and returns the filter index of the given message.
///
/// \param[in] message Message to peek.
/// \return Message filter index or nullopt if the value cannot be peeked.
std::optional<filter_index_type> peek_filter_index(span<const uint8_t> message);

/// Peeks and returns the slot symbol point of the given message.
///
/// \param[in] message Message to peek.
/// \param[in] nof_symbols Number of symbols.
/// \param[in] scs Subcarrier spacing.
/// \return Expected slot symbol point or nullopt if the value cannot be peeked
std::optional<slot_symbol_point>
peek_slot_symbol_point(span<const uint8_t> message, unsigned nof_symbols, subcarrier_spacing scs);

} // namespace uplane_peeker

/// Open Fronthaul User-Plane message decoder interface.
class uplane_message_decoder
{
public:
  virtual ~uplane_message_decoder() = default;

  /// Decodes the given message into results and returns true on success, false otherwise.
  ///
  /// \param[out] results Results of decoding the message. On error, results value is undefined.
  /// \param[in] message Message to be decoded.
  /// \return True on success, false otherwise.
  virtual bool decode(uplane_message_decoder_results& results, span<const uint8_t> message) = 0;

  /// Returns true if this decoder can place decoded IQ directly into a device-capable resource grid.
  virtual bool supports_resource_grid_decompression() const { return false; }

  /// Returns true if this decoder can place decoded IQ directly into a device-capable PRACH buffer.
  virtual bool supports_prach_buffer_decompression() const { return false; }

  /// Decodes the given message and preserves compressed IQ payloads without decompressing them to host memory.
  ///
  /// The compressed IQ spans in the results are valid only while the input message buffer remains valid.
  virtual bool decode_with_compressed_iq(uplane_message_decoder_results& results, span<const uint8_t> message)
  {
    (void)results;
    (void)message;
    return false;
  }

  /// Decompresses a previously decoded section directly into a resource grid.
  ///
  /// \param[out] grid      Destination resource grid.
  /// \param[in]  port      Resource-grid port index.
  /// \param[in]  symbol    OFDM symbol index.
  /// \param[in]  start_prb First destination PRB.
  /// \param[in]  nof_prbs  Number of PRBs to decompress.
  /// \param[in]  section   Previously decoded U-Plane section.
  /// \return True on success, false when direct grid decompression is unsupported.
  virtual bool decompress_to_resource_grid(resource_grid_writer&        grid,
                                           unsigned                     port,
                                           unsigned                     symbol,
                                           unsigned                     start_prb,
                                           unsigned                     nof_prbs,
                                           const uplane_section_params& section)
  {
    (void)grid;
    (void)port;
    (void)symbol;
    (void)start_prb;
    (void)nof_prbs;
    (void)section;
    return false;
  }

  /// Decompresses a previously decoded section directly into a PRACH buffer.
  ///
  /// \param[out] buffer         Destination PRACH buffer.
  /// \param[in]  port           PRACH receive port index.
  /// \param[in]  td_occasion    PRACH time-domain occasion index.
  /// \param[in]  fd_occasion    PRACH frequency-domain occasion index.
  /// \param[in]  symbol         PRACH symbol index within the occasion.
  /// \param[in]  start_re       First destination RE in the PRACH symbol.
  /// \param[in]  input_start_re First compressed-payload RE to consume.
  /// \param[in]  nof_re         Number of RE to decompress.
  /// \param[in]  nof_prbs       Number of PRBs represented by the compressed payload.
  /// \param[in]  section        Previously decoded U-Plane section.
  /// \return True on success, false when direct PRACH-buffer decompression is unsupported.
  virtual bool decompress_to_prach_buffer(prach_buffer&                buffer,
                                          unsigned                     port,
                                          unsigned                     td_occasion,
                                          unsigned                     fd_occasion,
                                          unsigned                     symbol,
                                          unsigned                     start_re,
                                          unsigned                     input_start_re,
                                          unsigned                     nof_re,
                                          unsigned                     nof_prbs,
                                          const uplane_section_params& section)
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
    (void)section;
    return false;
  }
};

} // namespace ofh
} // namespace ocudu
