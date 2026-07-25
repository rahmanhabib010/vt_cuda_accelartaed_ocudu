// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "pusch_codeblock_decoder.h"
#include "ocudu/adt/mutexed_mpsc_queue.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_decoder.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_decoder_buffer.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_resident_codeblock_decoder.h"
#include "ocudu/phy/upper/unique_rx_buffer.h"
#include "ocudu/ran/pusch/pusch_constants.h"
#include "ocudu/support/executors/task_executor.h"
#include "ocudu/support/memory_pool/bounded_object_pool.h"
#include <atomic>
#include <functional>

namespace ocudu {

/// Implementation of the PUSCH decoder.
class pusch_decoder_impl : public pusch_decoder, private pusch_decoder_buffer
{
public:
  /// Code block decoder pool type.
  using codeblock_decoder_pool = bounded_unique_object_pool<pusch_codeblock_decoder>;

  /// CRC calculators used in shared channels.
  struct sch_crc {
    /// For short TB checksums.
    std::unique_ptr<crc_calculator> crc16;
    /// For long TB checksums.
    std::unique_ptr<crc_calculator> crc24A;
    /// For segment-specific checksums.
    std::unique_ptr<crc_calculator> crc24B;
  };

  /// \brief PUSCH decoder constructor.
  ///
  /// Sets up the internal components, namely LDPC segmenter, LDPC rate dematcher, LDPC decoder and all the CRC
  /// calculators.
  ///
  /// \param[in] segmenter_    LDPC segmenter.
  /// \param[in] decoder_pool_ Codeblock decoder.
  /// \param[in] crc_set_      Structure with pointers to three CRC calculator objects, with generator
  ///                          polynomials of type \c CRC16, \c CRC24A and \c CRC24B.
  /// \param[in] executor_     Task executor for asynchronous PUSCH code block decoding.
  /// \param[in] nof_prb       Number of PRBs.
  /// \param[in] nof_layers    Number of layers.

  pusch_decoder_impl(std::unique_ptr<ldpc_segmenter_rx>      segmenter_,
                     std::shared_ptr<codeblock_decoder_pool> decoder_pool_,
                     sch_crc                                 crc_set_,
                     task_executor*                          executor_,
                     unsigned                                nof_prb,
                     unsigned                                nof_layers) :
    logger(ocudulog::fetch_basic_logger("PHY")),
    segmenter(std::move(segmenter_)),
    decoder_pool(std::move(decoder_pool_)),
    crc_set(std::move(crc_set_)),
    executor(executor_),
    softbits_buffer(pusch_constants::get_max_codeword_size(nof_prb, nof_layers).value())
  {
    ocudu_assert(segmenter, "Invalid segmenter.");
    ocudu_assert(decoder_pool, "Invalid codeblock decoder pool.");
    ocudu_assert(crc_set.crc16, "Invalid CRC16 calculator.");
    ocudu_assert(crc_set.crc24A, "Invalid CRC24A calculator.");
    ocudu_assert(crc_set.crc24B, "Invalid CRC24B calculator.");
    ocudu_assert(crc_set.crc16->get_generator_poly() == crc_generator_poly::CRC16, "Wrong TB CRC calculator.");
    ocudu_assert(crc_set.crc24A->get_generator_poly() == crc_generator_poly::CRC24A, "Wrong TB CRC calculator.");
    ocudu_assert(crc_set.crc24B->get_generator_poly() == crc_generator_poly::CRC24B, "Wrong TB CRC calculator.");
  }

#ifdef ENABLE_CUDA
  /// Pool of lazily-created batch GPU decoders.
  class batch_gpu_decoder_pool
  {
  public:
    using decoder_ptr =
        std::unique_ptr<pusch_resident_codeblock_decoder, std::function<void(pusch_resident_codeblock_decoder*)>>;

    virtual ~batch_gpu_decoder_pool() = default;

    /// Acquires an exclusive decoder instance, blocking if the bounded pool is saturated.
    virtual decoder_ptr get() = 0;

    /// Pre-creates pooled decoders outside the slot processing path.
    virtual void warmup() {}
  };

  /// \brief PUSCH decoder constructor with optional batch GPU decoder.
  ///
  /// \param[in] segmenter_         LDPC segmenter.
  /// \param[in] decoder_pool_      Codeblock decoder pool for CPU path.
  /// \param[in] batch_gpu_decoder_ Batched GPU decoder (nullptr to disable GPU batching).
  /// \param[in] crc_set_           CRC calculators.
  /// \param[in] executor_          Task executor.
  /// \param[in] nof_prb            Number of PRBs.
  /// \param[in] nof_layers         Number of layers.
  pusch_decoder_impl(std::unique_ptr<ldpc_segmenter_rx>                     segmenter_,
                     std::shared_ptr<codeblock_decoder_pool>                decoder_pool_,
                     std::unique_ptr<pusch_resident_codeblock_decoder>       batch_gpu_decoder_,
                     sch_crc                                                crc_set_,
                     task_executor*                                         executor_,
                     unsigned                                               nof_prb,
                     unsigned                                               nof_layers) :
    logger(ocudulog::fetch_basic_logger("PHY")),
    segmenter(std::move(segmenter_)),
    decoder_pool(std::move(decoder_pool_)),
    batch_gpu_decoder(std::move(batch_gpu_decoder_)),
    crc_set(std::move(crc_set_)),
    executor(executor_),
    softbits_buffer(pusch_constants::get_max_codeword_size(nof_prb, nof_layers).value())
  {
    ocudu_assert(segmenter, "Invalid segmenter.");
    ocudu_assert(decoder_pool || batch_gpu_decoder, "Invalid codeblock decoder - need pool or batch decoder.");
    ocudu_assert(crc_set.crc16, "Invalid CRC16 calculator.");
    ocudu_assert(crc_set.crc24A, "Invalid CRC24A calculator.");
    ocudu_assert(crc_set.crc24B, "Invalid CRC24B calculator.");
    ocudu_assert(crc_set.crc16->get_generator_poly() == crc_generator_poly::CRC16, "Wrong TB CRC calculator.");
    ocudu_assert(crc_set.crc24A->get_generator_poly() == crc_generator_poly::CRC24A, "Wrong TB CRC calculator.");
    ocudu_assert(crc_set.crc24B->get_generator_poly() == crc_generator_poly::CRC24B, "Wrong TB CRC calculator.");
  }

  /// \brief PUSCH decoder constructor with a shared bounded batch GPU decoder pool.
  ///
  /// \param[in] segmenter_              LDPC segmenter.
  /// \param[in] decoder_pool_           Codeblock decoder pool for CPU path.
  /// \param[in] batch_gpu_decoder_pool_ Shared batch GPU decoder pool.
  /// \param[in] crc_set_                CRC calculators.
  /// \param[in] executor_               Task executor.
  /// \param[in] nof_prb                 Number of PRBs.
  /// \param[in] nof_layers              Number of layers.
  pusch_decoder_impl(std::unique_ptr<ldpc_segmenter_rx>      segmenter_,
                     std::shared_ptr<codeblock_decoder_pool> decoder_pool_,
                     std::shared_ptr<batch_gpu_decoder_pool> shared_batch_gpu_decoder_pool_,
                     sch_crc                                 crc_set_,
                     task_executor*                          executor_,
                     unsigned                                nof_prb,
                     unsigned                                nof_layers) :
    logger(ocudulog::fetch_basic_logger("PHY")),
    segmenter(std::move(segmenter_)),
    decoder_pool(std::move(decoder_pool_)),
    batch_gpu_decoder_pool_(std::move(shared_batch_gpu_decoder_pool_)),
    crc_set(std::move(crc_set_)),
    executor(executor_),
    softbits_buffer(pusch_constants::get_max_codeword_size(nof_prb, nof_layers).value())
  {
    ocudu_assert(segmenter, "Invalid segmenter.");
    ocudu_assert(decoder_pool || batch_gpu_decoder_pool_, "Invalid codeblock decoder - need pool or batch decoder.");
    ocudu_assert(crc_set.crc16, "Invalid CRC16 calculator.");
    ocudu_assert(crc_set.crc24A, "Invalid CRC24A calculator.");
    ocudu_assert(crc_set.crc24B, "Invalid CRC24B calculator.");
    ocudu_assert(crc_set.crc16->get_generator_poly() == crc_generator_poly::CRC16, "Wrong TB CRC calculator.");
    ocudu_assert(crc_set.crc24A->get_generator_poly() == crc_generator_poly::CRC24A, "Wrong TB CRC calculator.");
    ocudu_assert(crc_set.crc24B->get_generator_poly() == crc_generator_poly::CRC24B, "Wrong TB CRC calculator.");
  }
#endif

  // See interface for the documentation.
  pusch_decoder_buffer& new_data(span<uint8_t>           transport_block,
                                 unique_rx_buffer        rm_buffer,
                                 pusch_decoder_notifier& notifier,
                                 const configuration&    cfg) override;

  // See interface for the documentation.
  void set_nof_softbits(units::bits nof_softbits) override;

#ifdef ENABLE_CUDA
  // See interface for the documentation.
  bool try_decode_resident(const resident_softbit_buffer& codeword) override;

  // See interface for the documentation.
  bool supports_resident_decode() const override;

  // See interface for the documentation.
  void enable_resident_decode() override;

  // See interface for the documentation.
  void disable_resident_decode() override;

  /// \brief Get access to the batch GPU decoder for profiling/diagnostics.
  /// \return Pointer to the batch GPU decoder, or nullptr if not available.
  pusch_resident_codeblock_decoder* get_batch_gpu_decoder() { return batch_gpu_decoder.get(); }

  // See interface for the documentation.
  void set_demod_gap_timing(float grid_staging_us, float demod_sync_us) override
  {
    collect_acceleration_timing_    = true;
    demod_grid_staging_us_ = grid_staging_us;
    demod_sync_us_         = demod_sync_us;
  }

  // See interface for the documentation.
  void set_detailed_gap_timing(float decode_call_us, float sinr_readback_us) override
  {
    if (!collect_acceleration_timing_) {
      return;
    }
    decode_call_us_   = decode_call_us;
    sinr_readback_us_ = sinr_readback_us;
  }

  // See interface for the documentation.
  void set_processor_stage_timing(float ch_estimate_us, float process_data_setup_us, float demod_call_us) override
  {
    collect_acceleration_timing_    = true;
    ch_estimate_us_        = ch_estimate_us;
    process_data_setup_us_ = process_data_setup_us;
    demod_call_us_         = demod_call_us;
  }

  // See interface for the documentation.
  void set_pre_join_callback(std::function<void()> callback) override { pre_join_callback_ = std::move(callback); }
#endif

private:
  /// Internal states for verifying the component coherence.
  enum class internal_states : uint8_t {
    /// \brief The decoder is not configured for decoding.
    ///
    /// The decoder only accepts new transmissions. It transitions to \c collecting when a new transmission is
    /// configured.
    ///
    idle = 0,
    /// \brief The decoder is collecting soft bits.
    ///
    /// It can simultaneously decode. It transitions to \c decoded if it finishes decoding prior \c on_end_softbits is
    /// called. In this case, the decoder shall not notify the ending of processing.
    ///
    /// It transitions to \c decoding if \c on_end_softbits is called before the asynchronous decoding finishes.
    collecting,
    /// \brief The decoder does not accept soft bits and it is decoding.
    ///
    /// It transitions to \c idle when it finishes decoding. In this case, the decoder shall notify the end of the
    /// processing.
    decoding,
    /// \brief The decoder finished decoding all codeblocks asynchronously before \c on_end_softbits is called.
    ///
    /// It transitions to \c idle when \c on_end_softbits is called. In this case, the decoder notifies the end of the
    /// processing.
    decoded
  };

  /// Convert an internal state to a string.
  static const char* to_string(internal_states state)
  {
    switch (state) {
      default:
      case internal_states::idle:
        return "idle";
      case internal_states::collecting:
        return "collecting";
      case internal_states::decoding:
        return "decoding";
      case internal_states::decoded:
        return "decoded";
    }
  }

  ocudulog::basic_logger& logger;
  /// Current internal state.
  std::atomic<internal_states> current_state = {internal_states::idle};
  /// Pointer to an LDPC segmenter.
  std::unique_ptr<ldpc_segmenter_rx> segmenter;
  /// Pointer to a codeblock decoder.
  std::shared_ptr<codeblock_decoder_pool> decoder_pool;
#ifdef ENABLE_CUDA
  /// Batched GPU decoder for efficient multi-codeblock processing.
  std::unique_ptr<pusch_resident_codeblock_decoder> batch_gpu_decoder;
  /// Shared bounded batch GPU decoder pool.
  std::shared_ptr<batch_gpu_decoder_pool> batch_gpu_decoder_pool_;
  /// Flag indicating GPU-resident decode mode is enabled (LLRs stay on GPU).
  bool resident_decode_enabled = false;
  /// True when optional resident GPU timing was enabled for the current PDU.
  bool collect_acceleration_timing_ = false;
  /// Last GPU decoder timing copied before releasing a pooled decoder.
  pusch_resident_codeblock_decoder_timing_stats last_acceleration_timing_stats_;
  /// Demodulator gap timing (set before decode, applied to result after).
  float demod_grid_staging_us_ = 0;
  float demod_sync_us_         = 0;
  /// Detailed gap timing from processor.
  float ch_estimate_us_        = 0;
  float process_data_setup_us_ = 0;
  float demod_call_us_         = 0;
  float decode_call_us_        = 0;
  float sinr_readback_us_      = 0;
  /// Callback invoked after GPU sync but before join_and_notify (for deferred SINR).
  std::function<void()> pre_join_callback_;
  /// GPU TB CRC result from resident decode with TB reassembly.
  /// When set, join_and_notify() uses this instead of CPU TB concatenation + CRC check.
  std::optional<bool> gpu_tb_crc_result_;
#endif
  /// \brief Pointer to a CRC calculator for TB-wise checksum.
  ///
  /// Only the CRC calculator with generator polynomial crc_generator_poly::CRC24A, used for long transport blocks, is
  /// needed. Indeed, if a transport block is short enough not to be segmented, the CRC is verified by the decoder.
  sch_crc crc_set;
  /// Optional task executor. Used for accelerating the PUSCH decoding at code block level. Set to \c nullptr for no
  /// concurrent execution.
  task_executor* executor;
  /// Soft bit buffer.
  std::vector<log_likelihood_ratio> softbits_buffer;
  /// Counts the number of soft bits in the buffer.
  unsigned softbits_count;
  /// Current transport block.
  span<uint8_t> transport_block;
  /// Current soft bits buffer.
  unique_rx_buffer unique_rm_buffer;
  /// Current notifier.
  pusch_decoder_notifier* result_notifier = nullptr;
  /// Current PUSCH decoder configuration.
  pusch_decoder::configuration current_config;
  /// Segmentation configuration parameters.
  segmenter_config segmentation_config;
  /// Temporary buffer to store the rate-matched codeblocks (represented by LLRs) and their metadata.
  static_vector<described_rx_codeblock, MAX_NOF_SEGMENTS> codeblock_llrs;
  /// Counts the number of remaining CB decoding tasks.
  std::atomic<unsigned> cb_task_counter;
  /// Counts the number of CB available for decoding.
  unsigned available_cb_counter;
  /// Number of iterations for each of the codeblocks.
  std::array<unsigned, MAX_NOF_SEGMENTS> cb_stats;
  /// Number of UL-SCH codeword softbits. If set, the decoder will start decoding codeblocks as they become available.
  std::optional<units::bits> nof_ulsch_softbits;
  /// Number of codeblocks in the current codeword.
  unsigned nof_codeblocks;
  /// CRC calculator for inner codeblock checks.
  crc_calculator* block_crc;

  // See interface for the documentation.
  span<log_likelihood_ratio> get_next_block_view(unsigned block_size) override;

  // See interface for the documentation.
  void on_new_softbits(span<const log_likelihood_ratio> softbits) override;

  // See interface for the documentation.
  void on_end_softbits() override;

  /// \brief Creates a codeblock decoding task.
  ///
  /// \param[in] cb_id Identifier of the codeblock to decode.
  void fork_codeblock_task(unsigned cb_id);

#ifdef ENABLE_CUDA
  /// \brief Creates a batched GPU decoding task for all codeblocks.
  ///
  /// Uses the batch GPU decoder to process all codeblocks in a single GPU operation.
  void fork_batch_gpu_task();

  /// Acquires a batch GPU decoder, either the owned one or a decoder from the shared bounded pool.
  batch_gpu_decoder_pool::decoder_ptr acquire_batch_gpu_decoder();
#endif

  /// \brief Joins the multiple code block processing.
  ///
  /// Called from the last decoding code block task. It concatenates code blocks and checks the decoded transport block
  /// CRC if applicable. Also, it notifies the decoder result.
  void join_and_notify();

  /// Concatenates code blocks and returns the CRC checksum.
  unsigned concatenate_codeblocks();
};

} // namespace ocudu
