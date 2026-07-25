// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief PUSCH decoder interface declaration.

#pragma once

#include "ocudu/adt/span.h"
#include "ocudu/phy/upper/acceleration/phy_acceleration.h"
#include "ocudu/phy/upper/log_likelihood_ratio.h"
#include "ocudu/ran/sch/ldpc_base_graph.h"
#include "ocudu/ran/sch/modulation_scheme.h"
#include <functional>

namespace ocudu {

class pusch_decoder_buffer;
class pusch_decoder_notifier;
class rx_buffer;
class unique_rx_buffer;
struct pusch_decoder_result;

/// \brief PUSCH decoder interface.
///
/// Recovers a UL-SCH transport block from a PUSCH codeword. Reverting the encoding operations described in TS38.212
/// Sections 6.2.1-6.2.6, the codeword is first split into rate-matched codeblocks. Then, each codeblock is restored
/// to its base rate, combined with previous retransmissions, and decoded. Finally, if all blocks pass the CRC check,
/// the data bits from all codeblocks are concatenated to form the UL-SCH transport block. If applicable, a last
/// transport-block CRC is computed and verified.
///
/// The PUSCH decoder is configured using the method \ref new_data. This method returns a \ref pusch_decoder_buffer
/// which is used to write softbits into the decoder.
///
/// The decoder could potentially start the code block processing if a codeword length is provided (see
/// \ref set_nof_softbits) prior \ref pusch_decoder_buffer::on_end_softbits.
class pusch_decoder
{
public:
  /// Collects the parameters necessary for decoding a PUSCH transport block.
  struct configuration {
    /// Code base graph.
    ldpc_base_graph_type base_graph = ldpc_base_graph_type::BG1;
    /// Redundancy version, values in {0, 1, 2, 3}.
    unsigned rv = 0;
    /// Modulation scheme.
    modulation_scheme mod = modulation_scheme::BPSK;
    /// \brief Limited buffer rate matching length in bits, as per TS38.212 Section 5.4.2.
    /// \note Set to zero for unlimited buffer length.
    unsigned Nref = 0;
    /// Number of transmission layers the transport block is mapped onto.
    unsigned nof_layers = 0;
    /// Maximum number of iterations of the LDPC decoder.
    unsigned nof_ldpc_iterations = 6;
    /// Flag for LDPC decoder early stopping: \c true to activate.
    bool use_early_stop = true;
    /// Flag to denote new data (first HARQ transmission).
    bool new_data = true;
  };

  /// Default destructor.
  virtual ~pusch_decoder() = default;

  /// \brief Decodes a PUSCH codeword.
  /// \param[out]    transport_block The decoded transport block, with packed (8 bits per entry) representation.
  /// \param[in,out] rm_buffer       A buffer for combining log-likelihood ratios from different retransmissions.
  /// \param[in]     notifier        Interface for notifying the completion of the TB processing.
  /// \param[in]     cfg             Decoder configuration parameters.
  /// \return  A \ref pusch_decoder_buffer, used to write softbits into the decoder.
  virtual pusch_decoder_buffer& new_data(span<uint8_t>           transport_block,
                                         unique_rx_buffer        rm_buffer,
                                         pusch_decoder_notifier& notifier,
                                         const configuration&    cfg) = 0;

  /// \brief Sets the number of UL-SCH codeword softbits expected by the PUSCH decoder.
  ///
  /// It allows the decoder to start decoding codeblocks before receiving the entire codeword. If it is called before
  /// \ref new_data or after decoding has started, it has no effect.
  ///
  /// \param[in] nof_softbits Number of codeword softbits, parameter \f$G^\textup{UL-SCH}\f$ in TS38.212 Section 6.2.7.
  virtual void set_nof_softbits(units::bits nof_softbits) = 0;

  /// \brief Attempts to decode a resident accelerator softbit buffer.
  virtual bool try_decode_resident(const resident_softbit_buffer& codeword)
  {
    (void)codeword;
    return false;
  }

  /// \brief Checks if resident accelerator decode is supported by this decoder.
  virtual bool supports_resident_decode() const { return false; }

  /// \brief Enables resident accelerator decode mode.
  virtual void enable_resident_decode() {}

  /// \brief Disables resident accelerator decode mode.
  virtual void disable_resident_decode() {}

  /// \brief Pass demodulator gap timing for inclusion in decode results.
  virtual void set_demod_gap_timing(float grid_staging_us, float demod_sync_us)
  {
    (void)grid_staging_us;
    (void)demod_sync_us;
  }

  /// \brief Pass processor-side timing stages for inclusion in decode results.
  virtual void set_processor_stage_timing(float ch_estimate_us, float process_data_setup_us, float demod_call_us)
  {
    (void)ch_estimate_us;
    (void)process_data_setup_us;
    (void)demod_call_us;
  }

  /// \brief Pass detailed gap timing from processor for inclusion in decode results.
  virtual void set_detailed_gap_timing(float decode_call_us, float sinr_readback_us)
  {
    (void)decode_call_us;
    (void)sinr_readback_us;
  }

  /// \brief Set a callback to be invoked after resident decode completion but before join_and_notify.
  ///
  /// Used for deferred metric readback: after the resident backend synchronizes the demodulation work, metric memory is
  /// valid. The callback reads the metrics and sets them on the notifier before join_and_notify() consumes CSI data.
  virtual void set_pre_join_callback(std::function<void()> callback) { (void)callback; }
};

} // namespace ocudu
