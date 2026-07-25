// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "pdsch_block_processor_gpu_impl.h"
#include "cuda/cuda_rt_utils.h"
#include "cuda/pdsch_resource_grid_mapper_cuda.h"
#include "../../phy_acceleration_runtime_options.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_segmenter_buffer.h"
#include "ocudu/ran/cyclic_prefix.h"
#include "ocudu/ran/resource_block.h"
#include "ocudu/support/ocudu_assert.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

using namespace ocudu;

namespace {

bool are_configs_equal(const pdsch_tb_encoder_gpu::config& left, const pdsch_tb_encoder_gpu::config& right)
{
  return (left.tb_size_bits == right.tb_size_bits) && (left.num_layers == right.num_layers) &&
         (left.modulation_order == right.modulation_order) && (left.num_coded_bits == right.num_coded_bits) &&
         (left.rv == right.rv) && (left.n_rnti == right.n_rnti) && (left.n_id == right.n_id) &&
         (left.cw_index == right.cw_index) && (left.base_graph == right.base_graph) &&
         (left.nof_codeblocks == right.nof_codeblocks) && (left.lifting_size == right.lifting_size) &&
         (left.nof_filler_bits == right.nof_filler_bits) && (left.nof_short_segments == right.nof_short_segments) &&
         (left.E_short == right.E_short) && (left.E_long == right.E_long);
}

bool is_encode_cache_disabled()
{
  static const bool disabled = []() {
    const char* value = std::getenv("OCUDU_PDSCH_DISABLE_ENCODE_CACHE");
    return value != nullptr && std::strcmp(value, "0") != 0;
  }();
  return disabled;
}

bool is_gpu_mapper_disabled()
{
  static const bool disabled = []() {
    const char* disable_value = std::getenv("OCUDU_PDSCH_DISABLE_DEVICE_MAP");
    if (disable_value != nullptr) {
      return std::strcmp(disable_value, "0") != 0;
    }

    const char* direct_grid_value = std::getenv("OCUDU_PDSCH_DIRECT_DEVICE_GRID");
    if (direct_grid_value != nullptr) {
      return !phy_acceleration_env_flag_enabled("OCUDU_PDSCH_DIRECT_DEVICE_GRID", true);
    }

    if (phy_acceleration_cuda_visible_grid_managed_requested("OCUDU_DL_CUDA_VISIBLE_GRID")) {
      return false;
    }

    (void)cudaGetLastError();
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }

    int integrated = 0;
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device_id) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }

    // On discrete GPUs the current sidecar grid path copies and scans the full
    // resource grid back to host before TX. That is slower than the host mapper
    // unless a real CUDA-visible downstream grid is selected explicitly.
    return integrated == 0;
  }();
  return disabled;
}

bool is_symbol_d2h_deferred()
{
  static const bool deferred = []() {
    const char* value = std::getenv("OCUDU_PDSCH_DEFER_SYMBOL_D2H");
    return value != nullptr && std::strcmp(value, "0") != 0;
  }();
  return deferred;
}

bool is_gpu_mapper_timing_enabled()
{
  static const bool enabled = []() {
    const char* value = std::getenv("OCUDU_PDSCH_DEVICE_MAP_TIMING");
    return value != nullptr && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

} // namespace

pdsch_block_processor_gpu_impl::pdsch_block_processor_gpu_impl(std::unique_ptr<pdsch_tb_encoder_gpu> tb_encoder_) :
  tb_encoder(std::move(tb_encoder_))
{
  ocudu_assert(tb_encoder, "Invalid GPU TB encoder.");
  ocudu_assert(tb_encoder->is_gpu_available(), "GPU not available.");

  // Pre-allocate symbol buffer for maximum TB size.
  // Max symbols = max_G / min_mod_order = ~2M / 2 = ~1M symbols.
  symbols_buffer.reserve(1000000);
  cached_symbols_buffer.reserve(1000000);

  constexpr size_t max_mapper_offsets = static_cast<size_t>(MAX_NSYMB_PER_SLOT) * MAX_NOF_SUBCARRIERS;
  const size_t     offsets_bytes      = max_mapper_offsets * sizeof(uint32_t);
  if (cudaMalloc(&d_mapper_re_offsets, offsets_bytes) == cudaSuccess) {
    d_mapper_re_offsets_capacity = offsets_bytes;
  }
}

pdsch_block_processor_gpu_impl::~pdsch_block_processor_gpu_impl()
{
  if (d_mapper_grid_bf16 != nullptr) {
    cudaFree(d_mapper_grid_bf16);
    d_mapper_grid_bf16          = nullptr;
    d_mapper_grid_bf16_capacity = 0;
  }
  if (d_mapper_re_offsets != nullptr) {
    cudaFree(d_mapper_re_offsets);
    d_mapper_re_offsets          = nullptr;
    d_mapper_re_offsets_capacity = 0;
  }
  cached_mapper_re_offsets.clear();
  if (gpu_mapper_start_event != nullptr) {
    cudaEventDestroy(static_cast<cudaEvent_t>(gpu_mapper_start_event));
    gpu_mapper_start_event = nullptr;
  }
  if (gpu_mapper_stop_event != nullptr) {
    cudaEventDestroy(static_cast<cudaEvent_t>(gpu_mapper_stop_event));
    gpu_mapper_stop_event = nullptr;
  }
}

resource_grid_mapper::symbol_buffer&
pdsch_block_processor_gpu_impl::configure_new_transmission(span<const uint8_t>          data,
                                                           unsigned                     i_cw,
                                                           const configuration&         config,
                                                           const ldpc_segmenter_buffer& segment_buffer_,
                                                           unsigned                     start_i_cb_,
                                                           unsigned                     cb_batch_len)
{
  // Store configuration for deferred processing.
  transport_block = data;
  segment_buffer  = &segment_buffer_;
  modulation      = config.modulation;
  codeword_index  = i_cw;
  rnti            = config.rnti;
  n_id            = config.n_id;
  nof_layers      = config.nof_layers;
  nof_ports       = config.nof_ports;
  start_i_cb      = start_i_cb_;
  last_i_cb       = start_i_cb_ + cb_batch_len - 1;

  ocudu_assert(last_i_cb < segment_buffer->get_nof_codeblocks(),
               "The last codeblock index in the batch (i.e., {}) exceeds the number of codeblocks (i.e., {})",
               last_i_cb,
               segment_buffer->get_nof_codeblocks());

  // Reset state.
  batch_processed                 = false;
  read_offset                     = 0;
  total_symbols                   = 0;
  batch_nof_symbols               = 0;
  current_slice_symbol_offset     = 0;
  current_slice_uses_symbol_cache = false;
  device_grid_mapping_requested   = false;

  const unsigned mod_order = get_bits_per_symbol(modulation);
  for (unsigned cb = start_i_cb; cb <= last_i_cb; ++cb) {
    batch_nof_symbols += segment_buffer->get_rm_length(cb) / mod_order;
  }

  return *this;
}

bool pdsch_block_processor_gpu_impl::can_reuse_cached_encode(const pdsch_tb_encoder_gpu::config& cfg) const
{
  return !is_encode_cache_disabled() && cached_encode_valid && (cached_tb_data == transport_block.data()) &&
         (cached_tb_size == transport_block.size()) && (cached_segment_buf == segment_buffer) &&
         are_configs_equal(cached_cfg, cfg);
}

bool pdsch_block_processor_gpu_impl::supports_device_grid_mapping() const
{
  return !is_gpu_mapper_disabled() && (nof_layers >= 1) && (nof_layers <= 4) &&
         ((nof_layers == 1) || (nof_layers == nof_ports));
}

bool pdsch_block_processor_gpu_impl::map_to_device_grid(span<const uint32_t> re_offsets,
                                                        void*                d_grid_bf16,
                                                        unsigned             mapped_nof_ports,
                                                        unsigned             mapped_nof_layers,
                                                        unsigned             nof_grid_re_per_port,
                                                        float                weight_real)
{
  if (!supports_device_grid_mapping() || re_offsets.empty() || (mapped_nof_ports != nof_ports) ||
      (mapped_nof_layers != nof_layers)) {
    return false;
  }
  if ((nof_layers > 1) && ((start_i_cb != 0) || (last_i_cb + 1 != segment_buffer->get_nof_codeblocks()))) {
    return false;
  }

  device_grid_mapping_requested = true;

  if (!batch_processed) {
    process_batch_gpu();
  }

  if ((re_offsets.size() * nof_layers) != total_symbols) {
    return false;
  }

  size_t offsets_bytes = re_offsets.size() * sizeof(uint32_t);
  if (offsets_bytes > d_mapper_re_offsets_capacity) {
    if (d_mapper_re_offsets != nullptr) {
      cudaFree(d_mapper_re_offsets);
      d_mapper_re_offsets = nullptr;
    }
    if (cudaMalloc(&d_mapper_re_offsets, offsets_bytes) != cudaSuccess) {
      d_mapper_re_offsets_capacity = 0;
      return false;
    }
    d_mapper_re_offsets_capacity = offsets_bytes;
    cached_mapper_re_offsets.clear();
  }

  size_t grid_bytes       = static_cast<size_t>(nof_ports) * nof_grid_re_per_port * 2U * sizeof(uint16_t);
  bool   use_writer_grid  = d_grid_bf16 != nullptr;
  void*  d_grid_bf16_dest = d_grid_bf16;
  if (!use_writer_grid && (grid_bytes > d_mapper_grid_bf16_capacity)) {
    if (d_mapper_grid_bf16 != nullptr) {
      cudaFree(d_mapper_grid_bf16);
      d_mapper_grid_bf16 = nullptr;
    }
    if (cudaMalloc(&d_mapper_grid_bf16, grid_bytes) != cudaSuccess) {
      d_mapper_grid_bf16_capacity = 0;
      return false;
    }
    d_mapper_grid_bf16_capacity = grid_bytes;
  }
  if (!use_writer_grid) {
    d_grid_bf16_dest = d_mapper_grid_bf16;
  }

  auto stream = static_cast<cudaStream_t>(tb_encoder->get_execution_context());
  bool re_offsets_on_device = (cached_mapper_re_offsets.size() == re_offsets.size()) &&
                              (std::memcmp(cached_mapper_re_offsets.data(), re_offsets.data(), offsets_bytes) == 0);
  if (!re_offsets_on_device) {
    if (cudaMemcpyAsync(d_mapper_re_offsets, re_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice, stream) !=
        cudaSuccess) {
      return false;
    }
    cached_mapper_re_offsets.resize(re_offsets.size());
    std::memcpy(cached_mapper_re_offsets.data(), re_offsets.data(), offsets_bytes);
  }

  if (!use_writer_grid && (cudaMemsetAsync(d_grid_bf16_dest, 0, grid_bytes, stream) != cudaSuccess)) {
    return false;
  }

  bool timing_enabled = is_gpu_mapper_timing_enabled();
  if (timing_enabled && (gpu_mapper_start_event == nullptr)) {
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event  = nullptr;
    if ((cudaEventCreate(&start_event) == cudaSuccess) && (cudaEventCreate(&stop_event) == cudaSuccess)) {
      gpu_mapper_start_event = start_event;
      gpu_mapper_stop_event  = stop_event;
    } else {
      if (start_event != nullptr) {
        cudaEventDestroy(start_event);
      }
      if (stop_event != nullptr) {
        cudaEventDestroy(stop_event);
      }
      timing_enabled = false;
    }
  }

  const int8_t* d_symbols = tb_encoder->get_device_symbols_int8();
  if (d_symbols == nullptr) {
    return false;
  }

  if (timing_enabled) {
    cudaEventRecord(static_cast<cudaEvent_t>(gpu_mapper_start_event), stream);
  }
  bool launch_ok = pdsch_map_layers_int8_to_bf16_real_grid(d_symbols + current_slice_symbol_offset * 2U,
                                                           static_cast<const uint32_t*>(d_mapper_re_offsets),
                                                           static_cast<uint16_t*>(d_grid_bf16_dest),
                                                           re_offsets.size(),
                                                           nof_ports,
                                                           nof_layers,
                                                           nof_grid_re_per_port,
                                                           weight_real,
                                                           stream);
  if (!launch_ok) {
    return false;
  }

  if (timing_enabled) {
    cudaEventRecord(static_cast<cudaEvent_t>(gpu_mapper_stop_event), stream);
    cudaEventSynchronizeYielding(static_cast<cudaEvent_t>(gpu_mapper_stop_event));
    float elapsed_ms = 0.0F;
    cudaEventElapsedTime(
        &elapsed_ms, static_cast<cudaEvent_t>(gpu_mapper_start_event), static_cast<cudaEvent_t>(gpu_mapper_stop_event));
    std::fprintf(stderr,
                 "CUDA PDSCH GPU real-grid map: symbols=%zu ports=%u grid_bf16_bytes=%zu kernel=%.3fus\n",
                 re_offsets.size() * nof_layers,
                 nof_ports,
                 grid_bytes,
                 elapsed_ms * 1000.0F);
  }

  return true;
}

void* pdsch_block_processor_gpu_impl::get_device_grid_mapping_stream() const
{
  return tb_encoder ? tb_encoder->get_execution_context() : nullptr;
}

void pdsch_block_processor_gpu_impl::process_batch_gpu()
{
  if (batch_processed) {
    return;
  }
  // Calculate the full codeword length required by the TB encoder and the
  // requested CB-batch slice returned by this block processor instance.
  unsigned       G                   = 0;
  unsigned       first_symbol_offset = 0;
  const unsigned mod_order           = get_bits_per_symbol(modulation);
  total_symbols                      = batch_nof_symbols;
  for (unsigned cb = 0, cb_end = segment_buffer->get_nof_codeblocks(); cb != cb_end; ++cb) {
    unsigned rm_length = segment_buffer->get_rm_length(cb);
    G += rm_length;

    if (cb < start_i_cb) {
      first_symbol_offset += rm_length / mod_order;
    }
  }

  // Get RV from first CB metadata (same for all CBs).
  const codeblock_metadata& cb_meta = segment_buffer->get_cb_metadata(start_i_cb);

  // Get rate matching parameters from CPU segmenter for multi-CB consistency.
  unsigned nof_cbs   = segment_buffer->get_nof_codeblocks();
  unsigned nof_short = segment_buffer->get_nof_short_segments();
  // E_short is the rm_length of any short CB (CB 0 if nof_short > 0)
  // E_long is the rm_length of any long CB (CB nof_short if nof_short < nof_cbs)
  unsigned E_short = (nof_short > 0) ? segment_buffer->get_rm_length(0) : 0;
  unsigned E_long  = (nof_short < nof_cbs) ? segment_buffer->get_rm_length(nof_short) : E_short;
  // If all CBs are short or all are long, both values are the same
  if (E_short == 0) {
    E_short = E_long;
  }

  // Configure TB encoder.
  pdsch_tb_encoder_gpu::config cfg = {};
  cfg.tb_size_bits                 = static_cast<unsigned>(transport_block.size() * 8);
  cfg.num_layers                   = nof_layers;
  cfg.modulation_order             = get_bits_per_symbol(modulation);
  cfg.num_coded_bits               = G;
  cfg.rv                           = cb_meta.tb_common.rv;
  cfg.n_rnti                       = static_cast<uint16_t>(rnti);
  cfg.n_id                         = static_cast<uint16_t>(n_id);
  cfg.cw_index                     = static_cast<uint8_t>(codeword_index);
  cfg.base_graph                   = static_cast<unsigned>(cb_meta.tb_common.base_graph);
  cfg.nof_codeblocks               = nof_cbs;
  cfg.lifting_size                 = static_cast<unsigned>(cb_meta.tb_common.lifting_size);
  cfg.nof_filler_bits              = cb_meta.cb_specific.nof_filler_bits;
  // Pass rate matching parameters from CPU segmenter for multi-CB consistency.
  cfg.nof_short_segments = nof_short;
  cfg.E_short            = E_short;
  cfg.E_long             = E_long;

  current_slice_symbol_offset = first_symbol_offset;

  if (is_encode_cache_disabled()) {
    // Cache-disabled reference path: preserve one encode/download per configured CB batch.
    tb_encoder->set_defer_symbol_download(device_grid_mapping_requested);
    unsigned result_symbols = tb_encoder->encode(transport_block, cfg);

    ocudu_assert(first_symbol_offset + total_symbols <= result_symbols,
                 "GPU TB encoder returned {} symbols, but requested slice [{}:{})",
                 result_symbols,
                 first_symbol_offset,
                 first_symbol_offset + total_symbols);

    symbols_buffer.resize(total_symbols);
    tb_encoder->get_symbols(symbols_buffer, first_symbol_offset, total_symbols);

    cached_encode_valid = false;
    cached_symbols_left = 0;
    batch_processed     = true;
    return;
  }

  unsigned result_symbols = cached_nof_symbols;
  if (!can_reuse_cached_encode(cfg)) {
    // Encode entire TB on GPU (CRC + Segment + LDPC + RM + Scramble + Modulate).
    tb_encoder->set_defer_symbol_download(device_grid_mapping_requested);
    result_symbols = tb_encoder->encode(transport_block, cfg);

    cached_symbols_buffer.resize(result_symbols);
    cached_symbols_host_valid = false;
    if (!device_grid_mapping_requested && !is_symbol_d2h_deferred()) {
      tb_encoder->get_symbols(cached_symbols_buffer, 0, result_symbols);
      cached_symbols_host_valid = true;
    }

    cached_encode_valid = true;
    cached_cfg          = cfg;
    cached_tb_data      = transport_block.data();
    cached_tb_size      = transport_block.size();
    cached_segment_buf  = segment_buffer;
    cached_nof_symbols  = result_symbols;
    cached_symbols_left = result_symbols;
  }

  ocudu_assert(first_symbol_offset + total_symbols <= result_symbols,
               "GPU TB encoder returned {} symbols, but requested slice [{}:{})",
               result_symbols,
               first_symbol_offset,
               first_symbol_offset + total_symbols);
  current_slice_uses_symbol_cache = true;

  if (cached_symbols_left <= total_symbols) {
    cached_encode_valid = false;
    cached_symbols_left = 0;
  } else {
    cached_symbols_left -= total_symbols;
  }

  batch_processed = true;
}

span<const ci8_t> pdsch_block_processor_gpu_impl::pop_symbols(unsigned block_size)
{
  // Process batch on first call.
  if (!batch_processed) {
    process_batch_gpu();
  }

  ocudu_assert(read_offset + block_size <= total_symbols,
               "The block size (i.e., {}) exceeds the number of available symbols (i.e., {}).",
               block_size,
               total_symbols - read_offset);

  if (current_slice_uses_symbol_cache && !cached_symbols_host_valid) {
    tb_encoder->get_symbols(cached_symbols_buffer, 0, cached_nof_symbols);
    cached_symbols_host_valid = true;
  }

  // Return view of symbols.
  span<const ci8_t> result =
      current_slice_uses_symbol_cache
          ? span<const ci8_t>(cached_symbols_buffer).subspan(current_slice_symbol_offset + read_offset, block_size)
          : span<const ci8_t>(symbols_buffer).subspan(read_offset, block_size);
  read_offset += block_size;

  return result;
}

unsigned pdsch_block_processor_gpu_impl::get_max_block_size() const
{
  if (!batch_processed) {
    return batch_nof_symbols;
  }

  // Return remaining symbols.
  return total_symbols - read_offset;
}

bool pdsch_block_processor_gpu_impl::empty() const
{
  if (!batch_processed) {
    return false;
  }
  return read_offset >= total_symbols;
}
