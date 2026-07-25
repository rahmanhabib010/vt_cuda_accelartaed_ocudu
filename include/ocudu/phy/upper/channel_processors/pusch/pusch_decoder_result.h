// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ocudu/support/math/stats.h"

namespace ocudu {

/// PUSCH decoding statistics.
struct pusch_decoder_result {
  /// Denotes whether the received transport block passed the CRC.
  bool tb_crc_ok = false;
  /// Total number of codeblocks in the current codeword.
  unsigned nof_codeblocks_total = 0;
  /// \brief LDPC decoding statistics.
  ///
  /// Provides access to LDPC decoding statistics such as the number of decoded codeblocks (via
  /// <tt>ldpc_stats->get_nof_observations()</tt>) or the average number of iterations for correctly decoded
  /// codeblocks (via <tt>ldpc_stats->get_mean()</tt>).
  sample_statistics<unsigned> ldpc_decoder_stats;

  /// Resident accelerator pipeline timing breakdown (microseconds), populated when accelerator timing is enabled.
  /// Note: Always included to avoid ODR violations between TUs built with different acceleration options.
  struct acceleration_timing {
    float    deinterleave_us = 0;
    float    rate_dematch_us = 0;
    float    ldpc_decode_us  = 0;
    float    crc_check_us    = 0;
    float    d2h_transfer_us = 0;
    float    total_e2e_us    = 0;
    unsigned nof_cbs         = 0;
    bool     valid           = false;
    /// Gap profiling: host-side overhead components (microseconds).
    float extract_bits_us = 0; ///< CPU extract_decoded_bits time.
    float grid_staging_us = 0; ///< CPU grid staging memcpy time.
    float demod_sync_us   = 0; ///< Resident demodulator synchronization time.
    /// Detailed gap profiling: timestamps for critical path analysis.
    float ch_estimate_us        = 0; ///< CPU channel estimation time (estimator.estimate callback).
    float process_data_setup_us = 0; ///< process_data() config/setup overhead.
    float demod_call_us         = 0; ///< demodulator.demodulate() call time.
    float decode_call_us        = 0; ///< Resident decode call time.
    float join_notify_us        = 0; ///< join_and_notify() time (TB concat + CRC).
    float sinr_readback_us      = 0; ///< Deferred SINR readback time.
    float decoder_config_us     = 0; ///< Host-side decoder configure path inside resident decode.
    float rate_match_config_us  = 0; ///< Host-side rate dematcher configure path inside resident decode.
    float completion_wait_us    = 0; ///< Host wait on resident decode completion.
    float tb_output_copy_us     = 0; ///< Host memcpy from pinned TB staging to the caller buffer.
    float avg_iters_query_us    = 0; ///< Host query for average LDPC iterations.
  };
  acceleration_timing acceleration_pipeline_timing;
};

} // namespace ocudu
