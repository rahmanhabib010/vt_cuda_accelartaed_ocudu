// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ocudu/adt/complex.h"
#include "ocudu/adt/span.h"

namespace ocudu {

/// \brief PRACH buffer interface.
///
/// Provides access to frequency-domain PRACH sequences.
///
/// A buffer for storing long PRACH sequences (\f$L_{RA}=839\f$), for a maximum of 4 OFDM symbols, can be created with
/// the factory function create_prach_buffer_long(). A buffer for storing short PRACH sequences (\f$L_{RA}=129\f$), for
/// a maximum of 12 OFDM symbols, can be created with the factory function create_prach_buffer_short().
class prach_buffer
{
public:
  /// Default destructor.
  virtual ~prach_buffer() = default;

  /// Gets the maximum number of ports.
  virtual unsigned get_max_nof_ports() const = 0;

  /// Gets the maximum number of time domain occasions.
  virtual unsigned get_max_nof_td_occasions() const = 0;

  /// Gets the maximum number of frequency domain occasions.
  virtual unsigned get_max_nof_fd_occasions() const = 0;

  /// Gets the maximum number of symbols.
  virtual unsigned get_max_nof_symbols() const = 0;

  /// Gets the sequence length.
  virtual unsigned get_sequence_length() const = 0;

  /// \brief Gets a read-write PRACH symbol for a given port, occasion and symbol.
  ///
  /// \param[in] i_port        Port identifier.
  /// \param[in] i_td_occasion Time-domain occasion.
  /// \param[in] i_fd_occasion Frequency-domain occasion.
  /// \param[in] i_symbol      Symbol index within the occasion.
  /// \return A read-write view of a PRACH OFDM symbol.
  virtual span<cbf16_t>
  get_symbol(unsigned i_port, unsigned i_td_occasion, unsigned i_fd_occasion, unsigned i_symbol) = 0;

  /// \brief Gets a read-only PRACH symbol for a given port, occasion and symbol.
  ///
  /// \param[in] i_port        Port identifier.
  /// \param[in] i_td_occasion Time-domain occasion.
  /// \param[in] i_fd_occasion Frequency-domain occasion.
  /// \param[in] i_symbol      Symbol index within the occasion.
  /// \return A read-only view of a PRACH OFDM symbol.
  virtual span<const cbf16_t>
  get_symbol(unsigned i_port, unsigned i_td_occasion, unsigned i_fd_occasion, unsigned i_symbol) const = 0;

  /// Returns true if the buffer can be consumed directly by an implementation-specific device backend.
  virtual bool supports_device_prach_buffer_reading() const { return false; }

  /// Returns true if the buffer can be produced directly by an implementation-specific device backend.
  virtual bool supports_device_prach_buffer_mapping() const { return false; }

  /// Gets an implementation-specific device pointer to the first PRACH buffer element.
  virtual const void* get_device_prach_buffer_cbf16() const { return nullptr; }

  /// Gets an implementation-specific device pointer to the first PRACH buffer element.
  virtual void* get_device_prach_buffer_cbf16() { return nullptr; }

  /// Gets the linear element offset of a PRACH symbol in the device buffer.
  virtual unsigned
  get_device_prach_symbol_offset(unsigned i_port, unsigned i_td_occasion, unsigned i_fd_occasion, unsigned i_symbol) const
  {
    (void)i_port;
    (void)i_td_occasion;
    (void)i_fd_occasion;
    (void)i_symbol;
    return 0;
  }

  /// Prepares the buffer for implementation-specific device reads on the supplied stream.
  virtual bool prepare_device_prach_buffer_reading(void* stream) const
  {
    (void)stream;
    return false;
  }

  /// Prepares the buffer for implementation-specific device writes on the supplied stream.
  virtual bool prepare_device_prach_buffer_mapping(void* stream)
  {
    (void)stream;
    return false;
  }

  /// Records that device writes have been enqueued on the supplied stream.
  virtual bool on_device_prach_buffer_mapping_enqueued(void* stream)
  {
    (void)stream;
    return false;
  }

  /// Cancels a pending device mapping operation.
  virtual bool cancel_device_prach_buffer_mapping() { return true; }

  /// Synchronizes pending device writes before host access.
  virtual bool synchronize_device_prach_buffer_mapping() const { return true; }
};

} // namespace ocudu
