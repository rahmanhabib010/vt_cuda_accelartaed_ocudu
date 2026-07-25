// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief GPU-accelerated PUSCH demodulator implementation.
///
/// This implementation accumulates equalized symbols across all OFDM symbols
/// and performs batch soft demodulation on GPU at the end. This avoids
/// per-symbol GPU kernel launches and enables efficient end-to-end GPU processing.

#pragma once

#include "cuda/pusch_sch_llr_compactor.h"
#include "ocudu/phy/generic_functions/transform_precoding/transform_precoder.h"
#include "ocudu/phy/support/re_buffer.h"
#include "ocudu/phy/support/resource_grid_reader.h"
#include "ocudu/phy/upper/channel_modulation/demodulation_mapper.h"
#include "ocudu/phy/upper/channel_modulation/evm_calculator.h"
#include "ocudu/phy/upper/channel_processors/pusch/factories.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_demodulator.h"
#include "ocudu/phy/upper/equalization/channel_equalizer.h"
#include "ocudu/phy/upper/equalization/dynamic_ch_est_list.h"
#include "ocudu/phy/upper/equalization/modular_ch_est_list.h"
#include "ocudu/phy/upper/sequence_generators/pseudo_random_generator.h"
#include "ocudu/ran/pusch/pusch_constants.h"
#include "ocudu/ran/resource_block.h"
#include "ocudu/ran/uci/uci_constants.h"
#include <array>
#include <cuda_runtime.h>

// Forward declarations for CUDA types
struct modulator_ctx;
typedef struct modulator_ctx* modulator_handle_t;
struct scrambler_ctx;
typedef struct scrambler_ctx* scrambler_handle_t;
struct pusch_e2e_ctx;
typedef struct pusch_e2e_ctx* pusch_e2e_handle_t;

namespace ocudu {

// Forward declaration
class pusch_codeblock_decoder_cuda_batch;

/// \brief GPU-accelerated PUSCH demodulator implementation.
///
/// This implementation differs from pusch_demodulator_impl in that it:
/// 1. Accumulates all equalized symbols across OFDM symbols
/// 2. Performs batch soft demodulation on GPU at the end of the codeword
/// 3. Keeps LLRs on GPU in fp16 format for downstream GPU processing
/// 4. Integrates with the batch GPU decoder for end-to-end GPU pipeline
class pusch_demodulator_gpu_impl : public pusch_demodulator
{
public:
  /// Constructor: sets up internal components and CUDA resources.
  pusch_demodulator_gpu_impl(
      std::unique_ptr<channel_equalizer>       equalizer_,
      std::unique_ptr<transform_precoder>      precoder_,
      std::unique_ptr<demodulation_mapper>     demapper_fallback_,
      std::unique_ptr<evm_calculator>          evm_calc_,
      std::unique_ptr<pseudo_random_generator> descrambler_,
      unsigned                                 max_nof_rb,
      bool                                     compute_post_eq_sinr_,
      bool                                     compensate_cfo_     = true,
      channel_equalizer_algorithm_type         equalizer_algorithm = channel_equalizer_algorithm_type::mmse);

  /// Destructor - releases CUDA resources.
  ~pusch_demodulator_gpu_impl() override;

  // Disable copy/move due to CUDA resource management.
  pusch_demodulator_gpu_impl(const pusch_demodulator_gpu_impl&)            = delete;
  pusch_demodulator_gpu_impl& operator=(const pusch_demodulator_gpu_impl&) = delete;
  pusch_demodulator_gpu_impl(pusch_demodulator_gpu_impl&&)                 = delete;
  pusch_demodulator_gpu_impl& operator=(pusch_demodulator_gpu_impl&&)      = delete;

  // See interface for the documentation.
  void demodulate(pusch_codeword_buffer&              codeword_buffer,
                  pusch_demodulator_notifier&         notifier,
                  const resource_grid_reader&         grid,
                  const dmrs_pusch_estimator_results& est_results,
                  const configuration&                config) override;

  // See interface for documentation.
  bool supports_resident_mode() const override { return gpu_available_; }

  // See interface for documentation.
  void enable_resident_mode() override { gpu_resident_mode_ = true; }

  // See interface for documentation.
  void disable_resident_mode() override { gpu_resident_mode_ = false; }

  // See interface for documentation.
  bool is_resident_mode_enabled() const override { return gpu_resident_mode_; }

  // See interface for documentation.
  resident_softbit_buffer get_resident_softbits() const override
  {
    // Both E2E and batch paths use FP16 output for GPU-resident LDPC decode.
    // FP16 provides best performance with the optimized resident decoder kernels.
    // Return the buffer from the most recently completed processing.
    return {.data              = last_resident_llrs_,
            .nof_softbits      = last_num_llrs_,
            .execution_context = static_cast<void*>(stream_),
            .completion_token  = static_cast<void*>(kernel_complete_[last_buf_]),
            .valid             = gpu_llrs_valid_,
            .n_id              = last_n_id_,
            .rnti              = last_rnti_,
            .buffer_index      = static_cast<int>(last_buf_)};
  }

  bool used_host_codeword_fallback() const override { return host_codeword_written_; }

  // See interface for documentation.
  resident_uci_buffer get_resident_uci() const override
  {
    return {.harq_ack_llrs     = span<const log_likelihood_ratio>(h_harq_ack_llrs_),
            .csi_part1_llrs    = span<const log_likelihood_ratio>(h_csi_part1_llrs_),
            .harq_ack_payload  = span<const uint8_t>(h_harq_ack_payload_).first(h_harq_ack_payload_size_),
            .csi_part1_payload = span<const uint8_t>(h_csi_part1_payload_).first(h_csi_part1_payload_size_),
            .harq_ack_status   = h_harq_ack_status_,
            .csi_part1_status  = h_csi_part1_status_,
            .harq_ack_decoded  = gpu_harq_ack_decoded_,
            .csi_part1_decoded = gpu_csi_part1_decoded_,
            .llrs_valid        = gpu_uci_llrs_valid_,
            .decoded_valid     = gpu_uci_decoded_valid_,
            .valid             = gpu_uci_llrs_valid_ || gpu_uci_decoded_valid_};
  }

  // See interface for documentation.
  void finalize_resident_uci() override;

  /// \brief Check if GPU is available for demodulation.
  bool is_gpu_available() const { return gpu_available_; }

  /// \brief Check if GPU-resident LLRs are valid (from last demodulation).
  bool has_gpu_resident_llrs() const { return gpu_llrs_valid_; }

  /// \brief Set the batch GPU decoder for end-to-end GPU processing.
  void set_batch_decoder(pusch_codeblock_decoder_cuda_batch* decoder) { batch_decoder_ = decoder; }

  /// \brief Get device pointer to GPU-resident fp16 LLRs (from last completed processing).
  void* get_device_llrs_half() const { return d_llrs_half_[last_buf_]; }

  /// \brief Get number of LLRs from last demodulation.
  size_t get_last_num_llrs() const { return last_num_llrs_; }

  /// \brief Get CUDA stream used by this demodulator.
  cudaStream_t get_cuda_stream() const { return stream_; }

  /// \brief Get last grid staging time in microseconds.
  float get_last_grid_staging_us() const override { return last_grid_staging_us_; }

  /// \brief Get last demod sync time in microseconds.
  float get_last_demod_sync_us() const override { return last_demod_sync_us_; }

  /// \brief Report deferred SINR after decoder sync (no GPU sync needed).
  float report_deferred_sinr(pusch_demodulator_notifier& notifier) override;

private:
  /// Data type for representing an RE mask within an OFDM symbol.
  using re_symbol_mask_type = bounded_bitset<MAX_NOF_SUBCARRIERS>;

  /// Gets channel data Resource Elements from the resource grid.
  const re_buffer_reader<cbf16_t>& get_ch_data_re(const resource_grid_reader&              grid,
                                                  unsigned                                 i_symbol,
                                                  const re_symbol_mask_type&               re_mask,
                                                  const static_vector<uint8_t, MAX_PORTS>& rx_ports);

  /// Gets channel data estimates (adapted for ocudu-foss API).
  const channel_equalizer::ch_est_list& get_ch_data_estimates(const dmrs_pusch_estimator_results&      est_results,
                                                              unsigned                                 i_symbol,
                                                              unsigned                                 nof_tx_layers,
                                                              const re_symbol_mask_type&               re_mask,
                                                              std::optional<unsigned>                  dc_position,
                                                              const static_vector<uint8_t, MAX_PORTS>& rx_ports);

  /// Perform end-to-end GPU processing (soft demod + descramble).
  void process_gpu_batch(pusch_codeword_buffer&      codeword_buffer,
                         pusch_demodulator_notifier& notifier,
                         const configuration&        config);

  /// Perform end-to-end GPU processing including GPU equalization.
  void process_gpu_e2e(pusch_codeword_buffer&              codeword_buffer,
                       pusch_demodulator_notifier&         notifier,
                       const resource_grid_reader&         grid,
                       const dmrs_pusch_estimator_results& est_results,
                       const configuration&                config);

  /// Fall back to CPU equalization and demodulation after a GPU-E2E path failure.
  void process_cpu_grid_fallback(pusch_codeword_buffer&              codeword_buffer,
                                 pusch_demodulator_notifier&         notifier,
                                 const resource_grid_reader&         grid,
                                 const dmrs_pusch_estimator_results& est_results,
                                 const configuration&                config);

  /// Fall back to CPU processing (per-symbol).
  void process_cpu_fallback(pusch_codeword_buffer&      codeword_buffer,
                            pusch_demodulator_notifier& notifier,
                            const configuration&        config);

  /// Channel equalization component.
  std::unique_ptr<channel_equalizer> equalizer;
  /// Transform precoder.
  std::unique_ptr<transform_precoder> precoder;
  /// Demodulation mapper for CPU fallback.
  std::unique_ptr<demodulation_mapper> demapper_fallback;
  /// EVM calculator. Optional, set to nullptr if not available.
  std::unique_ptr<evm_calculator> evm_calc;
  /// Descrambler component (for CPU fallback).
  std::unique_ptr<pseudo_random_generator> descrambler;

  /// GPU resources
  bool               gpu_available_ = false;
  modulator_handle_t mod_handle_    = nullptr;
  scrambler_handle_t scr_handle_    = nullptr;
  pusch_e2e_handle_t e2e_handle_    = nullptr; ///< E2E handle for GPU channel estimation

  cudaStream_t stream_           = nullptr;
  cudaStream_t scr_stream_       = nullptr; ///< Separate stream for scrambler (overlaps with H2D)
  cudaEvent_t  scr_event_        = nullptr; ///< Event to sync scrambler completion
  cudaEvent_t  completion_event_ = nullptr; ///< Event for async D2H completion tracking

  /// CUDA memory pool for async allocations (reduces allocation latency).
  cudaMemPool_t mem_pool_     = nullptr;
  bool          use_mem_pool_ = false; ///< Whether memory pool is available

  /// Device memory for equalized symbols.
  void*  d_symbols_          = nullptr;
  size_t d_symbols_capacity_ = 0;

  /// Device memory for per-RE noise variances.
  void*  d_noise_vars_          = nullptr;
  size_t d_noise_vars_capacity_ = 0;

  /// Flag for end-to-end GPU pipeline (equalization on GPU).
  bool use_gpu_equalization_ = false;

  /// Flag for GPU channel estimation support.
  bool use_gpu_chest_ = false;

  /// Host buffers for accumulated equalized data (CPU equalization path).
  std::vector<cf_t>  accumulated_symbols_;
  std::vector<float> accumulated_noise_vars_;
  size_t             accumulated_count_ = 0;

  /// Host buffers for raw channel data (GPU equalization path).
  std::vector<cf_t> raw_ch_symbols_;   // [total_re * nof_rx_ports]
  std::vector<cf_t> raw_ch_estimates_; // [total_re * nof_rx_ports]
  size_t            raw_accumulated_re_  = 0;
  unsigned          cached_nof_rx_ports_ = 0;

  static constexpr unsigned MAX_GPU_UCI_CODEBLOCKS = 2;

  /// Pre-allocated host buffers for bulk GPU transfer (avoids push_back overhead).
  std::vector<cf_t>                 h_grid_symbols_;         // Full grid symbols for one slot (fp32)
  std::vector<cf_t>                 h_grid_estimates_;       // Full channel estimates for one slot (fp32)
  std::vector<int>                  h_re_indices_;           // RE indices for extraction on GPU
  std::vector<int>                  h_sch_re_indices_;       // RE indices for SCH-only device compaction
  std::vector<int>                  h_harq_ack_re_indices_;  // RE indices for device-side HARQ compaction
  std::vector<int>                  h_csi_part1_re_indices_; // RE indices for device-side CSI Part 1 compaction
  std::vector<uint16_t>             h_harq_ack_llrs_half_;   // Fallback pageable HARQ LLR D2H staging
  std::vector<uint16_t>             h_csi_part1_llrs_half_;  // Fallback pageable CSI Part 1 LLR D2H staging
  std::vector<log_likelihood_ratio> h_harq_ack_llrs_;        // Compact HARQ LLRs for host UCI decoder
  std::vector<log_likelihood_ratio> h_csi_part1_llrs_;       // Compact CSI Part 1 LLRs for host UCI decoder
  std::array<uint8_t, uci_constants::MAX_NOF_PAYLOAD_BITS>          h_harq_ack_payload_        = {};
  std::array<uint8_t, uci_constants::MAX_NOF_PAYLOAD_BITS>          h_csi_part1_payload_       = {};
  size_t                                                            h_harq_ack_payload_size_   = 0;
  size_t                                                            h_csi_part1_payload_size_  = 0;
  uci_status                                                        h_harq_ack_status_         = uci_status::unknown;
  uci_status                                                        h_csi_part1_status_        = uci_status::unknown;
  std::array<pusch_uci_short_decode_result, MAX_GPU_UCI_CODEBLOCKS> h_harq_ack_decode_result_  = {};
  std::array<pusch_uci_short_decode_result, MAX_GPU_UCI_CODEBLOCKS> h_csi_part1_decode_result_ = {};
  unsigned                                                          h_harq_ack_decode_nof_codeblocks_  = 0;
  unsigned                                                          h_csi_part1_decode_nof_codeblocks_ = 0;
  unsigned                                                          h_harq_ack_expected_payload_bits_  = 0;
  unsigned                                                          h_csi_part1_expected_payload_bits_ = 0;

  /// Host staging buffers for cbf16 direct upload.
  std::vector<uint32_t> h_grid_cbf16_;      // Full grid as cbf16 (packed uint32)
  std::vector<uint32_t> h_estimates_cbf16_; // Full estimates as cbf16 (packed uint32)

  /// Device memory for full grid upload.
  void*  d_full_grid_      = nullptr; // fp32 grid after conversion
  size_t d_full_grid_cap_  = 0;
  void*  d_full_estimates_ = nullptr; // fp32 estimates after conversion
  size_t d_full_est_cap_   = 0;
  /// Pointers into unified buffer (computed as offsets, not separately allocated).
  void* d_grid_cbf16_ = nullptr; // cbf16 grid before conversion
  int*  d_re_indices_ = nullptr;

  /// Triple-buffered unified staging for parallel H2D upload and D2H overlap.
  static constexpr int NUM_BUFFERS                     = 3;
  void*                h_unified_staging_[NUM_BUFFERS] = {nullptr, nullptr, nullptr};
  void*                d_unified_input_[NUM_BUFFERS]   = {nullptr, nullptr, nullptr};
  size_t               unified_buf_cap_                = 0; ///< Capacity of each buffer (same for all)
  int                  current_buf_                    = 0; ///< Index of buffer currently being processed by GPU
  int                  last_buf_                       = 0; ///< Index of buffer from most recent completed processing

  /// Stream dedicated to H2D transfers (overlaps with GPU compute on main stream).
  cudaStream_t h2d_stream_                = nullptr;
  cudaEvent_t  h2d_complete_[NUM_BUFFERS] = {nullptr, nullptr, nullptr};

  /// Stream dedicated to D2H transfers (overlaps with next H2D).
  cudaStream_t d2h_stream_                = nullptr;
  cudaEvent_t  d2h_complete_[NUM_BUFFERS] = {nullptr, nullptr, nullptr};

  /// Events to track kernel completion for each buffer (enables async D2H start).
  cudaEvent_t kernel_complete_[NUM_BUFFERS] = {nullptr, nullptr, nullptr};

  /// Triple-buffered FP16 LLR output buffers (one per unified input buffer).
  void*  d_llrs_half_[NUM_BUFFERS]          = {nullptr, nullptr, nullptr};
  size_t d_llrs_half_capacity_[NUM_BUFFERS] = {0, 0, 0};
  /// Triple-buffered FP16 compact SCH LLR buffers for UCI-bearing resident decode.
  void*  d_sch_llrs_half_[NUM_BUFFERS]          = {nullptr, nullptr, nullptr};
  size_t d_sch_llrs_half_capacity_[NUM_BUFFERS] = {0, 0, 0};
  /// Triple-buffered SCH RE index buffers used by the compaction kernel.
  int*   d_sch_re_indices_[NUM_BUFFERS]          = {nullptr, nullptr, nullptr};
  size_t d_sch_re_indices_capacity_[NUM_BUFFERS] = {0, 0, 0};
  /// Triple-buffered compact UCI LLR buffers and RE index buffers.
  void*          d_harq_ack_llrs_half_[NUM_BUFFERS]                            = {nullptr, nullptr, nullptr};
  size_t         d_harq_ack_llrs_half_capacity_[NUM_BUFFERS]                   = {0, 0, 0};
  int*           d_harq_ack_re_indices_[NUM_BUFFERS]                           = {nullptr, nullptr, nullptr};
  size_t         d_harq_ack_re_indices_capacity_[NUM_BUFFERS]                  = {0, 0, 0};
  void*          d_csi_part1_llrs_half_[NUM_BUFFERS]                           = {nullptr, nullptr, nullptr};
  size_t         d_csi_part1_llrs_half_capacity_[NUM_BUFFERS]                  = {0, 0, 0};
  int*           d_csi_part1_re_indices_[NUM_BUFFERS]                          = {nullptr, nullptr, nullptr};
  size_t         d_csi_part1_re_indices_capacity_[NUM_BUFFERS]                 = {0, 0, 0};
  void*          d_harq_ack_decode_result_[NUM_BUFFERS]                        = {nullptr, nullptr, nullptr};
  void*          d_csi_part1_decode_result_[NUM_BUFFERS]                       = {nullptr, nullptr, nullptr};
  polar_handle_t harq_ack_polar_handles_[NUM_BUFFERS][MAX_GPU_UCI_CODEBLOCKS]  = {};
  polar_handle_t csi_part1_polar_handles_[NUM_BUFFERS][MAX_GPU_UCI_CODEBLOCKS] = {};
  bool           gpu_uci_polar_available_                                      = false;

  /// Triple-buffered pinned host mailboxes for compact UCI D2H.
  uint16_t*                      h_harq_ack_llrs_half_pinned_[NUM_BUFFERS]      = {nullptr, nullptr, nullptr};
  size_t                         h_harq_ack_llrs_half_pinned_cap_[NUM_BUFFERS]  = {0, 0, 0};
  size_t                         h_harq_ack_llrs_half_size_[NUM_BUFFERS]        = {0, 0, 0};
  uint16_t*                      h_csi_part1_llrs_half_pinned_[NUM_BUFFERS]     = {nullptr, nullptr, nullptr};
  size_t                         h_csi_part1_llrs_half_pinned_cap_[NUM_BUFFERS] = {0, 0, 0};
  size_t                         h_csi_part1_llrs_half_size_[NUM_BUFFERS]       = {0, 0, 0};
  pusch_uci_short_decode_result* h_harq_ack_decode_result_pinned_[NUM_BUFFERS]  = {nullptr, nullptr, nullptr};
  pusch_uci_short_decode_result* h_csi_part1_decode_result_pinned_[NUM_BUFFERS] = {nullptr, nullptr, nullptr};

  /// Offsets within unified buffer for sub-arrays.
  size_t unified_grid_offset_    = 0;
  size_t unified_indices_offset_ = 0;

  /// Last demodulation info.
  size_t last_num_llrs_ = 0;
  /// Device pointer containing the last resident LLR stream.
  void* last_resident_llrs_ = nullptr;

  /// Flag indicating GPU-resident LLRs are valid.
  bool gpu_llrs_valid_ = false;
  /// Flag indicating compact UCI LLRs are valid for the last demodulation.
  bool gpu_uci_llrs_valid_ = false;
  /// Flag indicating compact UCI payloads are valid for the last demodulation.
  bool gpu_uci_decoded_valid_           = false;
  bool gpu_harq_ack_decoded_            = false;
  bool gpu_csi_part1_decoded_           = false;
  bool gpu_uci_demux_pending_           = false;
  bool gpu_uci_device_decode_requested_ = false;
  /// True when the current demodulation wrote data through the host codeword buffer.
  bool host_codeword_written_ = false;

  /// Saved config from last demodulation (for decoder).
  unsigned last_n_id_ = 0;
  uint16_t last_rnti_ = 0;

  /// Batch GPU decoder (optional, for end-to-end GPU).
  pusch_codeblock_decoder_cuda_batch* batch_decoder_ = nullptr;

  /// Flag indicating GPU-resident mode (skip D2H copy of LLRs).
  bool gpu_resident_mode_ = false;

  /// Copy buffer for channel RE.
  modular_re_buffer_reader<cbf16_t, MAX_PORTS> ch_re_view;
  /// Copy buffer for non-contiguous channel RE.
  dynamic_re_buffer<cbf16_t> ch_re_copy;
  /// Buffer for equalized RE.
  std::vector<cf_t> temp_eq_re;
  /// Buffer for noise variances.
  std::vector<float> temp_eq_noise_vars;
  /// Copy buffer for channel estimates.
  dynamic_ch_est_list ch_estimates_copy;
  /// Buffer for noise variance estimates.
  std::array<float, MAX_PORTS> noise_var_estimates;

  /// Enables post equalization SINR calculation.
  bool compute_post_eq_sinr;

  /// Enables CFO compensation in GPU channel estimation.
  bool compensate_cfo_;

  /// Equalizer algorithm requested by the factory.
  channel_equalizer_algorithm_type equalizer_algorithm_;

  /// Enable optional path tracing. Set via OCUDU_PUSCH_ACCELERATION_TRACE=1.
  bool enable_path_tracing_ = false;

  /// Time interpolation mode: 0 = average (default), 1 = linear. Set via OCUDU_TIME_INTERP=linear.
  int time_interp_mode_ = 0;

  /// Noise estimation mode: 0 = cross-validation, 1 = pilot-residual (default). Set via OCUDU_NOISE_MODE=cv.
  int noise_mode_ = 1;

  /// Maximum RB count.
  unsigned max_nof_rb_;

  /// Pre-allocated pinned host buffer for D2H LLR transfers (FP16 path).
  uint16_t* h_llrs_half_pinned_     = nullptr;
  size_t    h_llrs_half_pinned_cap_ = 0;

  /// D2H copy tracking (single event for non-buffered path)
  cudaEvent_t d2h_event_ = nullptr;

  /// Profiling events for detailed latency breakdown
  cudaEvent_t prof_h2d_start_    = nullptr;
  cudaEvent_t prof_h2d_end_      = nullptr;
  cudaEvent_t prof_kernel_start_ = nullptr;
  cudaEvent_t prof_kernel_end_   = nullptr;
  cudaEvent_t prof_d2h_start_    = nullptr;
  cudaEvent_t prof_d2h_end_      = nullptr;

  struct gpu_e2e_timing_state {
    cudaEvent_t start             = nullptr;
    cudaEvent_t h2d_end           = nullptr;
    cudaEvent_t kernel_end        = nullptr;
    cudaEvent_t final_end         = nullptr;
    bool        pending           = false;
    uint16_t    rnti              = 0;
    unsigned    nof_prb           = 0;
    unsigned    nof_re            = 0;
    unsigned    mod_order         = 0;
    size_t      h2d_bytes         = 0;
    bool        direct_grid       = false;
    bool        re_indices_cached = false;
    bool        resident          = false;
    bool        compact_sch       = false;
    bool        configured        = false;
    bool        includes_llr_d2h  = false;
  };

  /// Optional per-buffer GPU E2E timing. Created only when OCUDU_PUSCH_GPU_E2E_TIMING_WARN_US is non-zero.
  std::array<gpu_e2e_timing_state, NUM_BUFFERS> gpu_e2e_timing_;
  unsigned                                      gpu_e2e_timing_warn_us_ = 0;
  bool                                          gpu_e2e_timing_enabled_ = false;

  /// Cached E2E configuration to skip reconfiguration when parameters match.
  struct cached_e2e_config {
    uint16_t rnti                        = 0;
    uint16_t n_id                        = 0;
    int      nof_prb                     = 0;
    int      nof_symbols                 = 0;
    int      nof_rx_ports                = 0;
    int      nof_tx_layers               = 0;
    int      grid_nof_subcarriers        = 0;
    int      grid_nof_symbols            = 0;
    int      mod_order                   = 0;
    int      dmrs_type                   = 0;
    int      dmrs_symbol_mask            = 0;
    int      nof_cdm_groups_without_data = 0;
    int      scrambling_id               = 0;
    int      n_scid                      = 0;
    int      scs_khz                     = 0;
    int      use_low_papr_dmrs           = 0;
    int      n_rs_id                     = 0;
    int      compensate_cfo              = 0;
    int      time_interp_mode            = 0;
    int      noise_mode                  = 0;
    bool     valid                       = false;

    bool matches(uint16_t new_rnti,
                 uint16_t new_n_id,
                 int      new_nof_prb,
                 int      new_nof_symbols,
                 int      new_nof_rx_ports,
                 int      new_nof_tx_layers,
                 int      new_grid_nof_subcarriers,
                 int      new_grid_nof_symbols,
                 int      new_mod_order,
                 int      new_dmrs_type,
                 int      new_dmrs_symbol_mask,
                 int      new_nof_cdm_groups_without_data,
                 int      new_scrambling_id,
                 int      new_n_scid,
                 int      new_scs_khz,
                 int      new_use_low_papr_dmrs,
                 int      new_n_rs_id,
                 int      new_compensate_cfo,
                 int      new_time_interp_mode,
                 int      new_noise_mode) const
    {
      return valid && rnti == new_rnti && n_id == new_n_id && nof_prb == new_nof_prb &&
             nof_symbols == new_nof_symbols && nof_rx_ports == new_nof_rx_ports && nof_tx_layers == new_nof_tx_layers &&
             grid_nof_subcarriers == new_grid_nof_subcarriers && grid_nof_symbols == new_grid_nof_symbols &&
             mod_order == new_mod_order && dmrs_type == new_dmrs_type && dmrs_symbol_mask == new_dmrs_symbol_mask &&
             nof_cdm_groups_without_data == new_nof_cdm_groups_without_data && scrambling_id == new_scrambling_id &&
             n_scid == new_n_scid && scs_khz == new_scs_khz && use_low_papr_dmrs == new_use_low_papr_dmrs &&
             n_rs_id == new_n_rs_id && compensate_cfo == new_compensate_cfo &&
             time_interp_mode == new_time_interp_mode && noise_mode == new_noise_mode;
    }

    void update(uint16_t new_rnti,
                uint16_t new_n_id,
                int      new_nof_prb,
                int      new_nof_symbols,
                int      new_nof_rx_ports,
                int      new_nof_tx_layers,
                int      new_grid_nof_subcarriers,
                int      new_grid_nof_symbols,
                int      new_mod_order,
                int      new_dmrs_type,
                int      new_dmrs_symbol_mask,
                int      new_nof_cdm_groups_without_data,
                int      new_scrambling_id,
                int      new_n_scid,
                int      new_scs_khz,
                int      new_use_low_papr_dmrs,
                int      new_n_rs_id,
                int      new_compensate_cfo,
                int      new_time_interp_mode,
                int      new_noise_mode)
    {
      rnti                        = new_rnti;
      n_id                        = new_n_id;
      nof_prb                     = new_nof_prb;
      nof_symbols                 = new_nof_symbols;
      nof_rx_ports                = new_nof_rx_ports;
      nof_tx_layers               = new_nof_tx_layers;
      grid_nof_subcarriers        = new_grid_nof_subcarriers;
      grid_nof_symbols            = new_grid_nof_symbols;
      mod_order                   = new_mod_order;
      dmrs_type                   = new_dmrs_type;
      dmrs_symbol_mask            = new_dmrs_symbol_mask;
      nof_cdm_groups_without_data = new_nof_cdm_groups_without_data;
      scrambling_id               = new_scrambling_id;
      n_scid                      = new_n_scid;
      scs_khz                     = new_scs_khz;
      use_low_papr_dmrs           = new_use_low_papr_dmrs;
      n_rs_id                     = new_n_rs_id;
      compensate_cfo              = new_compensate_cfo;
      time_interp_mode            = new_time_interp_mode;
      noise_mode                  = new_noise_mode;
      valid                       = true;
    }
  };
  cached_e2e_config e2e_config_cache_;

  /// Cached per-slot update fields to avoid redundant CUDA slot-update calls.
  struct cached_e2e_slot_update {
    int  nof_prb          = 0;
    int  start_prb        = 0;
    int  slot_idx         = 0;
    int  dmrs_symbol_mask = 0;
    bool valid            = false;

    bool matches(int new_nof_prb, int new_start_prb, int new_slot_idx, int new_dmrs_symbol_mask) const
    {
      return valid && nof_prb == new_nof_prb && start_prb == new_start_prb && slot_idx == new_slot_idx &&
             dmrs_symbol_mask == new_dmrs_symbol_mask;
    }

    void update(int new_nof_prb, int new_start_prb, int new_slot_idx, int new_dmrs_symbol_mask)
    {
      nof_prb          = new_nof_prb;
      start_prb        = new_start_prb;
      slot_idx         = new_slot_idx;
      dmrs_symbol_mask = new_dmrs_symbol_mask;
      valid            = true;
    }
  };
  cached_e2e_slot_update e2e_slot_update_cache_;

  /// Cached RE index configuration to skip index building when PUSCH config hasn't changed.
  /// The RE indices depend only on PRB allocation, DMRS pattern, and grid geometry — not on
  /// data content. Caching them saves ~5-15μs per slot by avoiding the per-symbol for_each loop.
  struct cached_re_indices_config {
    unsigned rb_count                    = 0;
    unsigned rb_lowest                   = 0;
    unsigned rb_highest                  = 0;
    uint64_t dmrs_symb_pos               = 0;
    unsigned start_symbol                = 0;
    unsigned nof_symbols                 = 0;
    int      dmrs_config_type            = 0;
    unsigned nof_cdm_groups_without_data = 0;
    unsigned nof_subcarriers             = 0;
    size_t   cached_total_re             = 0;
    bool     valid                       = false;

    bool matches(unsigned rc,
                 unsigned rl,
                 unsigned rh,
                 uint64_t dp,
                 unsigned ss,
                 unsigned ns,
                 int      dt,
                 unsigned cdm,
                 unsigned subc) const
    {
      return valid && rb_count == rc && rb_lowest == rl && rb_highest == rh && dmrs_symb_pos == dp &&
             start_symbol == ss && nof_symbols == ns && dmrs_config_type == dt && nof_cdm_groups_without_data == cdm &&
             nof_subcarriers == subc;
    }

    void update(unsigned rc,
                unsigned rl,
                unsigned rh,
                uint64_t dp,
                unsigned ss,
                unsigned ns,
                int      dt,
                unsigned cdm,
                unsigned subc,
                size_t   total_re)
    {
      rb_count                    = rc;
      rb_lowest                   = rl;
      rb_highest                  = rh;
      dmrs_symb_pos               = dp;
      start_symbol                = ss;
      nof_symbols                 = ns;
      dmrs_config_type            = dt;
      nof_cdm_groups_without_data = cdm;
      nof_subcarriers             = subc;
      cached_total_re             = total_re;
      valid                       = true;
    }
  };
  cached_re_indices_config                          re_indices_cache_;
  std::array<cached_re_indices_config, NUM_BUFFERS> d_re_indices_cache_;

  /// Cached compact SCH/UCI RE-index configuration.
  struct cached_compaction_re_indices_config {
    bool     enabled                     = false;
    unsigned rb_count                    = 0;
    unsigned rb_lowest                   = 0;
    unsigned rb_highest                  = 0;
    uint64_t dmrs_symb_pos               = 0;
    unsigned start_symbol                = 0;
    unsigned nof_symbols                 = 0;
    int      dmrs_config_type            = 0;
    unsigned nof_cdm_groups_without_data = 0;
    unsigned modulation_order            = 0;
    unsigned nof_tx_layers               = 0;
    unsigned nof_ul_sch_bits             = 0;
    unsigned nof_harq_ack_rvd            = 0;
    unsigned nof_enc_harq_ack_bits       = 0;
    unsigned nof_harq_ack_bits           = 0;
    unsigned nof_enc_csi_part1_bits      = 0;
    unsigned nof_csi_part1_bits          = 0;
    unsigned nof_enc_csi_part2_bits      = 0;
    bool     valid                       = false;

    bool matches(bool     en,
                 unsigned rc,
                 unsigned rl,
                 unsigned rh,
                 uint64_t dp,
                 unsigned ss,
                 unsigned ns,
                 int      dt,
                 unsigned cdm,
                 unsigned mod,
                 unsigned layers,
                 unsigned sch_bits,
                 unsigned harq_rvd,
                 unsigned enc_harq_bits,
                 unsigned harq_bits,
                 unsigned enc_csi1_bits,
                 unsigned csi1_bits,
                 unsigned enc_csi2_bits) const
    {
      return valid && enabled == en && rb_count == rc && rb_lowest == rl && rb_highest == rh && dmrs_symb_pos == dp &&
             start_symbol == ss && nof_symbols == ns && dmrs_config_type == dt && nof_cdm_groups_without_data == cdm &&
             modulation_order == mod && nof_tx_layers == layers && nof_ul_sch_bits == sch_bits &&
             nof_harq_ack_rvd == harq_rvd && nof_enc_harq_ack_bits == enc_harq_bits && nof_harq_ack_bits == harq_bits &&
             nof_enc_csi_part1_bits == enc_csi1_bits && nof_csi_part1_bits == csi1_bits &&
             nof_enc_csi_part2_bits == enc_csi2_bits;
    }

    void update(bool     en,
                unsigned rc,
                unsigned rl,
                unsigned rh,
                uint64_t dp,
                unsigned ss,
                unsigned ns,
                int      dt,
                unsigned cdm,
                unsigned mod,
                unsigned layers,
                unsigned sch_bits,
                unsigned harq_rvd,
                unsigned enc_harq_bits,
                unsigned harq_bits,
                unsigned enc_csi1_bits,
                unsigned csi1_bits,
                unsigned enc_csi2_bits)
    {
      enabled                     = en;
      rb_count                    = rc;
      rb_lowest                   = rl;
      rb_highest                  = rh;
      dmrs_symb_pos               = dp;
      start_symbol                = ss;
      nof_symbols                 = ns;
      dmrs_config_type            = dt;
      nof_cdm_groups_without_data = cdm;
      modulation_order            = mod;
      nof_tx_layers               = layers;
      nof_ul_sch_bits             = sch_bits;
      nof_harq_ack_rvd            = harq_rvd;
      nof_enc_harq_ack_bits       = enc_harq_bits;
      nof_harq_ack_bits           = harq_bits;
      nof_enc_csi_part1_bits      = enc_csi1_bits;
      nof_csi_part1_bits          = csi1_bits;
      nof_enc_csi_part2_bits      = enc_csi2_bits;
      valid                       = true;
    }
  };
  cached_compaction_re_indices_config                          compaction_re_indices_cache_;
  std::array<cached_compaction_re_indices_config, NUM_BUFFERS> d_compaction_re_indices_cache_;
  std::array<bool, NUM_BUFFERS>                                d_sch_re_indices_uploaded_       = {false, false, false};
  std::array<bool, NUM_BUFFERS>                                d_harq_ack_re_indices_uploaded_  = {false, false, false};
  std::array<bool, NUM_BUFFERS>                                d_csi_part1_re_indices_uploaded_ = {false, false, false};

  /// Gap profiling: last measured grid staging and demod sync times.
  float last_grid_staging_us_ = 0;
  float last_demod_sync_us_   = 0;
};

} // namespace ocudu
