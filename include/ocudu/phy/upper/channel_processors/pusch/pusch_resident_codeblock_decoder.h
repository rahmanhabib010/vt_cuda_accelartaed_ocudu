// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ocudu/adt/bit_buffer.h"
#include "ocudu/adt/span.h"
#include "ocudu/phy/upper/channel_coding/crc_calculator.h"
#include "ocudu/phy/upper/codeblock_metadata.h"
#include "ocudu/phy/upper/log_likelihood_ratio.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ocudu {

/// Pipeline timing statistics for resident PUSCH decoding.
struct pusch_resident_codeblock_decoder_timing_stats {
  /// Host-to-device transfer time, when host input staging is used.
  float h2d_transfer_us = 0.0f;
  /// LDPC deinterleaving time.
  float deinterleave_us = 0.0f;
  /// LDPC rate-dematching time.
  float rate_dematch_us = 0.0f;
  /// LDPC decode time.
  float ldpc_decode_us = 0.0f;
  /// Codeblock CRC check time.
  float crc_check_us = 0.0f;
  /// Device-to-host transfer time for decoded output.
  float d2h_transfer_us = 0.0f;
  /// Host-side decoder configuration time.
  float decoder_config_us = 0.0f;
  /// Host-side rate-matcher configuration time.
  float rate_match_config_us = 0.0f;
  /// Time spent waiting for resident decode completion.
  float completion_wait_us = 0.0f;
  /// Transport-block output copy time.
  float tb_output_copy_us = 0.0f;
  /// Average iteration-statistics query time.
  float avg_iters_query_us = 0.0f;
  /// Host-side CRC time, when applicable.
  float cpu_crc_us = 0.0f;
  /// Backend execution time for the resident decode work.
  float total_backend_us = 0.0f;
  /// End-to-end decode path time.
  float total_e2e_us = 0.0f;
  /// Number of decoded codeblocks.
  unsigned nof_cbs = 0;
  /// True when timing collection was enabled for the last decode.
  bool timing_enabled = false;
  /// Host-side decoded-bit extraction time.
  float extract_bits_us = 0.0f;
  /// Resource-grid staging time reported by the demodulator.
  float grid_staging_us = 0.0f;
  /// Resident demodulation synchronization time reported by the demodulator.
  float demod_sync_us = 0.0f;
};

/// Backend-neutral interface for PUSCH codeblock decoders that can consume resident softbits.
class pusch_resident_codeblock_decoder
{
public:
  /// Result for a decoded resident PUSCH codeblock.
  struct codeblock_result {
    /// True when the decoded codeblock CRC passed.
    bool crc_ok = false;
    /// Number of LDPC iterations used for this codeblock.
    unsigned nof_iterations = 0;
  };

  /// Scrambling parameters needed when resident softbits are decoded without a host-side descrambling step.
  struct scrambling_config {
    /// RNTI used for scrambling.
    uint16_t n_rnti = 0;
    /// Scrambling identity.
    uint16_t n_id = 0;
    /// Codeword index.
    uint8_t q = 0;
    /// Slot-dependent scrambling offset.
    uint8_t n_s = 0;
    /// Bit offset within the resident softbit stream.
    unsigned bit_offset = 0;
  };

  /// Transport-block level resident decode result.
  struct tb_decode_result {
    /// True when the final transport-block CRC passed.
    bool tb_crc_ok = false;
    /// True when all codeblock CRCs passed.
    bool all_cb_crcs_ok = false;
    /// Average or representative LDPC iteration count reported by the backend.
    unsigned nof_iterations = 0;
  };

  virtual ~pusch_resident_codeblock_decoder() = default;

  /// Decode a batch of host-resident codeblocks.
  virtual std::vector<codeblock_result> decode_batch(span<const described_rx_codeblock> codeblock_llrs,
                                                     span<bit_buffer>                   cb_data_buffers,
                                                     span<span<log_likelihood_ratio>>   rm_buffers,
                                                     bool                               new_data,
                                                     crc_generator_poly                 crc_poly,
                                                     bool                               use_early_stop,
                                                     unsigned                           nof_ldpc_iterations) = 0;

  /// Rate-dematch a codeblock whose previous CRC already passed.
  virtual void rate_dematch_only(span<const log_likelihood_ratio> cb_llrs,
                                 span<log_likelihood_ratio>       rm_buffer,
                                 bool                             new_data,
                                 const codeblock_metadata&        metadata) = 0;

  /// Decode resident softbits and write the reassembled transport block.
  virtual tb_decode_result decode_resident_softbits(void*                                                resident_llrs,
                                                    size_t                                               nof_softbits,
                                                    const scrambling_config*                             scrambling_cfg,
                                                    const codeblock_metadata::tb_common_metadata&        tb_common,
                                                    span<const codeblock_metadata::cb_specific_metadata> cb_specific,
                                                    span<uint8_t>                                        tb_output,
                                                    crc_generator_poly                                   cb_crc_poly,
                                                    unsigned nof_ldpc_iterations,
                                                    void*    execution_context,
                                                    int      buffer_index = -1) = 0;

  /// Returns timing statistics for the most recent decode operation.
  virtual const pusch_resident_codeblock_decoder_timing_stats& get_last_timing_stats() const = 0;

  /// Supplies demodulator staging and synchronization timing for combined pipeline reporting.
  virtual void set_demod_timing(float grid_staging_us, float demod_sync_us) = 0;

  /// Enables or disables backend timing collection.
  virtual void set_timing_enabled(bool enable) = 0;

  /// Selects the backend LDPC decoder algorithm.
  virtual void set_ldpc_decoder_algorithm(const std::string& algorithm) = 0;
};

} // namespace ocudu
