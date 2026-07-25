// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief Batched GPU PUSCH codeblock decoder using CUDA.
///
/// This class provides efficient batch processing of multiple codeblocks
/// on GPU with single H2D transfer, single kernel launch, and single D2H transfer.

#pragma once

#include "ocudu/phy/upper/channel_coding/crc_calculator.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_decoder.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_rate_dematcher.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_resident_codeblock_decoder.h"
#include "ocudu/phy/upper/codeblock_metadata.h"
#include "ocudu/ran/sch/sch_constants.h"
#include <cuda_runtime.h>
#include <ocudu_phy_cuda.h>
#include <array>
#include <memory>
#include <mutex>
#include <rate_matching.h>
#include <scrambling.h>
#include <string>
#include <vector>

namespace ocudu {

/// \brief Batched GPU PUSCH codeblock decoder using CUDA.
///
/// This class processes all codeblocks from a transport block in a single batch:
/// - Single async H2D transfer for all codeblock LLRs
/// - Single GPU kernel call for all LDPC decodes
/// - Single async D2H transfer for all decoded bits
/// - Single synchronization at end
///
/// This approach reduces GPU overhead from N syncs to 1 sync, where N is the
/// number of codeblocks (up to 44 for large transport blocks).
class pusch_codeblock_decoder_cuda_batch : public pusch_resident_codeblock_decoder
{
public:
  /// Per-codeblock decode result type.
  using codeblock_result     = pusch_resident_codeblock_decoder::codeblock_result;
  /// Pipeline timing statistics type.
  using pipeline_timing_stats = pusch_resident_codeblock_decoder_timing_stats;
  /// Resident softbit descrambling configuration type.
  using scrambling_config    = pusch_resident_codeblock_decoder::scrambling_config;
  /// Transport-block resident decode result type.
  using tb_decode_result     = pusch_resident_codeblock_decoder::tb_decode_result;

  /// CRC calculators used in shared channels.
  struct sch_crc {
    /// For short TB checksums.
    std::unique_ptr<crc_calculator> crc16;
    /// For long TB checksums.
    std::unique_ptr<crc_calculator> crc24A;
    /// For segment-specific checksums.
    std::unique_ptr<crc_calculator> crc24B;
  };

  /// \brief Constructor - creates CUDA resources for batch processing.
  /// \param[in] rate_dematcher CPU rate dematcher for LLR combining.
  /// \param[in] crcs CRC calculators.
  pusch_codeblock_decoder_cuda_batch(std::unique_ptr<ldpc_rate_dematcher> rate_dematcher, sch_crc crcs);

  /// \brief Destructor - releases CUDA resources.
  ~pusch_codeblock_decoder_cuda_batch();

  // Disable copy/move.
  pusch_codeblock_decoder_cuda_batch(const pusch_codeblock_decoder_cuda_batch&)            = delete;
  pusch_codeblock_decoder_cuda_batch& operator=(const pusch_codeblock_decoder_cuda_batch&) = delete;
  pusch_codeblock_decoder_cuda_batch(pusch_codeblock_decoder_cuda_batch&&)                 = delete;
  pusch_codeblock_decoder_cuda_batch& operator=(pusch_codeblock_decoder_cuda_batch&&)      = delete;

  /// \brief Decode all codeblocks in batch.
  ///
  /// This method:
  /// 1. Rate-dematches each codeblock's LLRs (CPU)
  /// 2. Batches all rate-dematched LLRs to GPU
  /// 3. Decodes all codeblocks in single kernel
  /// 4. Transfers results back and checks CRCs
  ///
  /// \param[in] codeblock_llrs Rate-matched LLRs and metadata for each codeblock.
  /// \param[out] cb_data_buffers Output bit buffers for decoded data (one per CB).
  /// \param[in,out] rm_buffers Rate-matching soft buffers for LLR combining (one per CB).
  /// \param[in] new_data True if this is a new transmission (not HARQ retransmission).
  /// \param[in] crc_poly CRC polynomial for codeblock CRC check.
  /// \param[in] use_early_stop Enable early stopping on CRC match.
  /// \param[in] nof_ldpc_iterations Maximum LDPC decoder iterations.
  /// \return Vector of results for each codeblock.
  std::vector<codeblock_result> decode_batch(span<const described_rx_codeblock> codeblock_llrs,
                                             span<bit_buffer>                   cb_data_buffers,
                                             span<span<log_likelihood_ratio>>   rm_buffers,
                                             bool                               new_data,
                                             crc_generator_poly                 crc_poly,
                                             bool                               use_early_stop,
                                             unsigned                           nof_ldpc_iterations) override;

  /// \brief Only perform rate dematching (for CBs with already-OK CRC).
  void rate_dematch_only(span<const log_likelihood_ratio> cb_llrs,
                         span<log_likelihood_ratio>       rm_buffer,
                         bool                             new_data,
                         const codeblock_metadata&        metadata) override;

  /// \brief Get CUDA stream used by this decoder.
  cudaStream_t get_cuda_stream() const { return stream_; }

  /// \brief Get the timing stats from the last decode operation.
  /// \return Pipeline timing statistics.
  const pipeline_timing_stats& get_last_timing_stats() const override { return last_timing_stats_; }

  /// \brief Set demodulator gap timing (called by processor before decode).
  void set_demod_timing(float grid_staging_us, float demod_sync_us) override
  {
    last_timing_stats_.grid_staging_us = grid_staging_us;
    last_timing_stats_.demod_sync_us   = demod_sync_us;
  }

  /// \brief Enable or disable timing instrumentation.
  /// \param[in] enable True to enable timing, false to disable.
  /// Timing adds overhead due to CUDA event synchronization.
  void set_timing_enabled(bool enable) override
  {
    timing_enabled_ = enable;
    if (!timing_enabled_) {
      last_timing_stats_ = pipeline_timing_stats{};
    }
  }

  /// \brief Set the LDPC decoder algorithm ("auto", "boxplus", "min_sum").
  void set_ldpc_decoder_algorithm(const std::string& algorithm) override { ldpc_decoder_algorithm_ = algorithm; }

  /// \brief Decode with full GPU TB reassembly and CRC check.
  ///
  /// This method performs all decode + TB reassembly on GPU:
  /// 1. GPU descrambling (fp16 in-place, if scrambling_cfg provided)
  /// 2. GPU rate dematching (fp16)
  /// 3. GPU LDPC decoding
  /// 4. GPU CB CRC checking (batch)
  /// 5. GPU TB desegmentation (concatenate CBs to TB)
  /// 6. GPU TB CRC checking
  /// 7. Single D2H transfer for final TB bytes only
  ///
  /// This eliminates per-CB D2H transfer and CPU-side TB concatenation,
  /// providing optimal performance for large transport blocks.
  ///
  /// \param[in] d_llrs_half Device pointer to fp16 LLRs.
  /// \param[in] num_llrs Total number of LLRs.
  /// \param[in] scrambling_cfg Scrambling config (nullptr to skip descrambling).
  /// \param[in] tb_common Common TB metadata.
  /// \param[in] cb_specific Per-codeblock metadata.
  /// \param[out] tb_output Output span for reassembled TB bytes (A/8 bytes, no CRC).
  /// \param[in] cb_crc_poly CRC polynomial for codeblock CRC check.
  /// \param[in] nof_ldpc_iterations Maximum LDPC decoder iterations.
  /// \param[in] execution_context Opaque backend execution context, typically a CUDA stream.
  /// \param[in] buffer_index Optional resident buffer-ring index associated with \c d_llrs_half.
  /// \return TB decode result with CRC status.
  tb_decode_result
  decode_resident_softbits(void*                                                d_llrs_half,
                           size_t                                               num_llrs,
                           const scrambling_config*                             scrambling_cfg,
                           const codeblock_metadata::tb_common_metadata&        tb_common,
                           span<const codeblock_metadata::cb_specific_metadata> cb_specific,
                           span<uint8_t>                                        tb_output,
                           crc_generator_poly                                   cb_crc_poly,
                           unsigned                                             nof_ldpc_iterations,
                           void*                                                execution_context,
                           int                                                  buffer_index = -1) override;

private:
  /// Select CRC calculator from polynomial.
  crc_calculator* select_crc(crc_generator_poly poly);

  /// Configure CUDA decoder for given base graph, lifting size, filler bits, and CRC type.
  /// \param[in] base_graph LDPC base graph.
  /// \param[in] lifting_size LDPC lifting size.
  /// \param[in] nof_ldpc_iterations Maximum LDPC decoder iterations.
  /// \param[in] nof_filler_bits Number of filler bits in each codeblock.
  /// \param[in] nof_cbs Number of code blocks.
  /// \param[in] crc_poly CRC polynomial used for this transport block.
  void configure_decoder(ldpc_base_graph_type base_graph,
                         unsigned             lifting_size,
                         unsigned             nof_ldpc_iterations,
                         unsigned             nof_filler_bits,
                         unsigned             nof_cbs  = 1,
                         crc_generator_poly   crc_poly = crc_generator_poly::CRC24B);

  /// Configure CUDA rate dematcher for given metadata.
  void configure_rate_dematcher(const codeblock_metadata& cfg);

  /// Extract decoded bits for a single codeblock from GPU output.
  void extract_decoded_bits(bit_buffer& output, unsigned cb_idx, unsigned nof_bits);

  /// CPU rate dematcher for LLR combining (fallback).
  std::unique_ptr<ldpc_rate_dematcher> dematcher_;

  /// CRC calculators.
  sch_crc crc_set_;

  /// Mutex for thread-safe access when shared among multiple decoder instances.
  /// This serializes decode operations to prevent buffer corruption.
  mutable std::mutex decode_mutex_;

  /// CUDA decoder handle.
  ldpc_decoder_handle_t decoder_handle_ = nullptr;

  /// CUDA rate matcher handle for GPU dematching.
  rate_matcher_handle_t rate_matcher_handle_ = nullptr;

  /// CUDA scrambler handle for GPU descrambling.
  scrambler_handle_t scrambler_handle_ = nullptr;

  /// CUDA stream for async operations.
  cudaStream_t stream_ = nullptr;

  /// CUDA event for completion tracking.
  cudaEvent_t completion_event_ = nullptr;

  /// Pinned host memory for TB output staging (required for true async D2H).
  /// Without this, cudaMemcpyAsync to unpinned memory blocks internally.
  uint8_t* h_tb_output_pinned_ = nullptr;

  /// Pinned host memory for TB CRC result.
  int* h_tb_crc_result_pinned_ = nullptr;

  /// Pinned host memory for per-CB CRC results.
  int* h_crc_results_pinned_ = nullptr;

  /// Device memory for received (rate-matched) LLRs before dematching (fp32).
  float* d_received_llr_batch_ = nullptr;

  /// Device memory for deinterleaved LLRs (after GPU deinterleaving, before dematching).
  float* d_deinterleaved_llr_batch_ = nullptr;

  /// Device memory for dematched LLRs (full codeblock size, all CBs contiguous, fp32).
  float* d_llr_batch_ = nullptr;

  /// Device memory for dematched LLRs in fp16 (triple-buffered for GPU-resident pipeline).
  static constexpr int NUM_DECODE_BUFFERS                    = 3;
  void*                d_llr_batch_half_[NUM_DECODE_BUFFERS] = {nullptr, nullptr, nullptr};
  int                  current_decode_buf_                   = 0; ///< Rotating buffer index for triple-buffering

  /// Device memory for batched output bits (all CBs contiguous).
  uint32_t* d_output_batch_ = nullptr;

  /// Host memory for received LLRs (rate-matched, before dematching).
  std::vector<float> h_received_llr_batch_;

  /// Host memory for batched LLRs (pinned for async transfer).
  std::vector<float> h_llr_batch_;

  /// Host memory for batched output bits.
  std::vector<uint32_t> h_output_batch_;

  /// Device scratch for fail-closed CRC status fallback (kernel convention: 0=pass, non-zero=fail).
  int* d_crc_results_ = nullptr;

  /// Device memory for per-CB bit counts (for non-uniform CRC check).
  int* d_bits_per_cb_ = nullptr;

  /// Device memory for reassembled TB output.
  uint8_t* d_tb_output_ = nullptr;

  /// Device memory for TB CRC result.
  int* d_tb_crc_result_ = nullptr;

  /// Device memory for gather kernel source offsets (non-uniform CB path).
  unsigned int* d_gather_offsets_ = nullptr;

  /// Pinned host memory for gather kernel source offsets.
  unsigned int* h_gather_offsets_ = nullptr;

  /// Host memory for per-CB bit counts (staging buffer).
  std::vector<int> h_bits_per_cb_;

  /// Maximum number of codeblocks we can batch.
  static constexpr unsigned MAX_BATCH_SIZE = MAX_NOF_SEGMENTS;

  /// Maximum rate-matched LLRs per codeblock (max E).
  static constexpr unsigned MAX_RM_LLRS = 156000;

  /// Maximum LLRs per codeblock (BG1: 68*384 = 26112).
  static constexpr unsigned MAX_CB_LLRS = 68 * 384;

  /// Maximum output words per codeblock (BG1: 22*384/32 = 264).
  static constexpr unsigned MAX_CB_OUTPUT_WORDS = (22 * 384 + 31) / 32;

  /// Maximum TB bytes (max TB size is ~1.3M bits = ~165KB).
  static constexpr unsigned MAX_TB_BYTES = 200000;

  /// Cached decoder configuration.
  ldpc_base_graph_type cached_base_graph_   = ldpc_base_graph_type::BG1;
  unsigned             cached_lifting_size_ = 0;
  unsigned             cached_max_iters_    = 0;
  unsigned             cached_filler_bits_  = 0;
  unsigned             cached_nof_cbs_      = 0;
  crc_generator_poly   cached_crc_poly_     = crc_generator_poly::CRC24B;
  bool                 cached_use_boxplus_  = false;

  /// Cached rate matcher configuration (separate from decoder to avoid shared-state bugs).
  ldpc_base_graph_type cached_rm_bg_     = ldpc_base_graph_type::BG1;
  unsigned             cached_rm_Z_      = 0;
  unsigned             cached_rm_E_      = 0;
  unsigned             cached_rm_rv_     = 0;
  unsigned             cached_rm_filler_ = 0;
  unsigned             cached_rm_Qm_     = 2;

  /// Number of LLRs per codeblock for current config.
  unsigned llr_stride_ = 0;

  /// Number of output words per codeblock for current config.
  unsigned output_stride_ = 0;

  /// Current rate-matched length (E) for batch processing.
  unsigned rm_stride_ = 0;

  /// Cached direct rate-match metadata already resident on the device.
  bool                                      cached_direct_rm_metadata_valid_ = false;
  unsigned                                  cached_direct_rm_nof_cbs_        = 0;
  size_t                                    cached_direct_rm_num_llrs_       = 0;
  int                                       cached_direct_rm_n_cb_           = 0;
  int                                       cached_direct_rm_k0_             = 0;
  unsigned                                  cached_direct_rm_filler_bits_    = 0;
  std::array<int, MAX_BATCH_SIZE>           cached_direct_rm_bits_per_cb_    = {};
  std::array<unsigned int, MAX_BATCH_SIZE>  cached_direct_rm_gather_offsets_ = {};

  // ============================================================================
  // Pipeline Timing Instrumentation
  // ============================================================================

  /// Enable/disable timing instrumentation (adds overhead due to sync).
  bool timing_enabled_ = false;

  /// Last timing statistics from decode operation.
  mutable pipeline_timing_stats last_timing_stats_;

  /// Cached classification of the last caller payload pointer for opportunistic direct D2H.
  const void* cached_payload_output_ptr_        = nullptr;
  size_t      cached_payload_output_bytes_      = 0;
  bool        cached_payload_output_async_safe_ = false;

  /// CUDA events for timing (created on demand when timing is enabled).
  cudaEvent_t timing_events_[8] = {nullptr};

  /// Number of timing events.
  static constexpr unsigned NUM_TIMING_EVENTS = 8;

  /// Create timing events if not already created.
  void ensure_timing_events();

  /// Destroy timing events.
  void destroy_timing_events();

  /// Returns true when the caller payload buffer can be used directly as an async D2H target.
  bool can_copy_payload_output_direct(span<uint8_t> payload_output);

  /// LDPC decoder algorithm from YAML config: "auto", "boxplus", or "min_sum".
  std::string ldpc_decoder_algorithm_ = "auto";
};

} // namespace ocudu
