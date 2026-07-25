// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief GPU-accelerated PDSCH block processor using CUDA.
///
/// This processor performs end-to-end GPU processing of PDSCH transport blocks:
/// - CRC attachment (on GPU)
/// - Code block segmentation (on GPU)
/// - LDPC encoding (batched on GPU)
/// - Rate matching (on GPU)
/// - Scrambling (on GPU)
/// - Modulation (on GPU)
///
/// The entire TX chain is executed on GPU with only TB input and symbol output transfers.

#pragma once

#include "cuda/pdsch_tb_encoder_cuda.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_segmenter_tx.h"
#include "ocudu/phy/upper/channel_processors/pdsch/pdsch_block_processor.h"
#include <vector>

namespace ocudu {

/// GPU-accelerated PDSCH block processor implementation.
class pdsch_block_processor_gpu_impl : public pdsch_block_processor, private resource_grid_mapper::symbol_buffer
{
public:
  /// Constructs the GPU block processor with TB-level encoder.
  ///
  /// \param tb_encoder_ GPU-accelerated transport block encoder.
  explicit pdsch_block_processor_gpu_impl(std::unique_ptr<pdsch_tb_encoder_gpu> tb_encoder_);

  /// Frees GPU device-mapper scratch buffers.
  ~pdsch_block_processor_gpu_impl() override;

  // See interface for documentation.
  resource_grid_mapper::symbol_buffer& configure_new_transmission(span<const uint8_t>          data,
                                                                  unsigned                     i_cw,
                                                                  const configuration&         config,
                                                                  const ldpc_segmenter_buffer& segment_buffer,
                                                                  unsigned                     start_i_cb,
                                                                  unsigned cb_batch_len) override;

private:
  // See interface for documentation.
  span<const ci8_t> pop_symbols(unsigned block_size) override;

  // See interface for documentation.
  unsigned get_max_block_size() const override;

  // See interface for documentation.
  bool empty() const override;

  // See interface for documentation.
  bool supports_device_grid_mapping() const override;

  // See interface for documentation.
  bool map_to_device_grid(span<const uint32_t> re_offsets,
                          void*                d_grid_bf16,
                          unsigned             nof_ports,
                          unsigned             nof_layers,
                          unsigned             nof_grid_re_per_port,
                          float                weight_real) override;

  // See interface for documentation.
  void* get_device_grid_mapping_stream() const override;

  /// Processes transport block using GPU.
  void process_batch_gpu();

  /// Returns true if a cached full-TB GPU encode can serve the current CB batch.
  bool can_reuse_cached_encode(const pdsch_tb_encoder_gpu::config& cfg) const;

  /// GPU-accelerated transport block encoder.
  std::unique_ptr<pdsch_tb_encoder_gpu> tb_encoder;

  /// Pointer to the transport block data.
  span<const uint8_t> transport_block;

  /// Pointer to the LDPC segmenter buffer.
  const ldpc_segmenter_buffer* segment_buffer = nullptr;

  /// Modulation scheme for the current transmission.
  modulation_scheme modulation = modulation_scheme::QPSK;

  /// Codeword index for scrambling.
  unsigned codeword_index = 0;

  /// RNTI for scrambling.
  unsigned rnti = 0;

  /// Scrambling ID (n_ID).
  unsigned n_id = 0;

  /// Number of layers for rate matching.
  unsigned nof_layers = 1;

  /// Number of precoding output ports.
  unsigned nof_ports = 1;

  /// Index of the first CB in the batch.
  unsigned start_i_cb = 0;

  /// Index of the last CB in the batch.
  unsigned last_i_cb = 0;

  /// True if the batch has been processed on GPU.
  bool batch_processed = false;

  /// Buffer for all modulated symbols from GPU.
  std::vector<ci8_t> symbols_buffer;

  /// Full-TB host symbol cache used to serve multiple mapper slices without extra downloads.
  std::vector<ci8_t> cached_symbols_buffer;

  /// Current read position in the symbols buffer.
  unsigned read_offset = 0;

  /// Total number of symbols produced.
  unsigned total_symbols = 0;

  /// First symbol of the current CB-batch slice within the full-TB symbol cache.
  unsigned current_slice_symbol_offset = 0;

  /// True if the current CB-batch slice is backed by the full-TB symbol cache.
  bool current_slice_uses_symbol_cache = false;

  /// True when the current transmission is being mapped into a device-backed resource grid.
  bool device_grid_mapping_requested = false;

  /// True if the full-TB host symbol cache contains valid downloaded data.
  bool cached_symbols_host_valid = false;

  /// Number of symbols in the configured codeblock batch.
  unsigned batch_nof_symbols = 0;

  /// Cached full-TB encode metadata. The flexible PDSCH processor can ask the
  /// same block processor for one CB batch at a time; the GPU encoder produces
  /// the whole TB, so keep it alive until all symbol slices have been served.
  bool                         cached_encode_valid = false;
  pdsch_tb_encoder_gpu::config cached_cfg          = {};
  const uint8_t*               cached_tb_data      = nullptr;
  unsigned                     cached_tb_size      = 0;
  const ldpc_segmenter_buffer* cached_segment_buf  = nullptr;
  unsigned                     cached_nof_symbols  = 0;
  unsigned                     cached_symbols_left = 0;

  /// GPU-resident compact BF16 grid scratch buffer.
  void*  d_mapper_grid_bf16          = nullptr;
  size_t d_mapper_grid_bf16_capacity = 0;

  /// Device RE-offset list for GPU-resident real-grid mapping.
  void*                 d_mapper_re_offsets          = nullptr;
  size_t                d_mapper_re_offsets_capacity = 0;
  std::vector<uint32_t> cached_mapper_re_offsets;

  /// CUDA events for optional GPU device-mapper timing.
  void* gpu_mapper_start_event = nullptr;
  void* gpu_mapper_stop_event  = nullptr;
};

} // namespace ocudu
