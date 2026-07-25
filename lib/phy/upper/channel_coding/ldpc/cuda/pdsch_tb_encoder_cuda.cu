// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "cuda_rt_utils.h"
#include "pdsch_tb_encoder_cuda.h"
#include "ocudu/ocudulog/ocudulog.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <modulation.h>
#include <scrambling.h>
#include <transport_block.h>
#include <vector>

using namespace ocudu;

namespace {

static bool are_configs_equal(const pdsch_tb_encoder_gpu::config& left, const pdsch_tb_encoder_gpu::config& right)
{
  return (left.tb_size_bits == right.tb_size_bits) && (left.num_layers == right.num_layers) &&
         (left.modulation_order == right.modulation_order) && (left.num_coded_bits == right.num_coded_bits) &&
         (left.rv == right.rv) && (left.n_rnti == right.n_rnti) && (left.n_id == right.n_id) &&
         (left.cw_index == right.cw_index) && (left.base_graph == right.base_graph) &&
         (left.nof_codeblocks == right.nof_codeblocks) && (left.lifting_size == right.lifting_size) &&
         (left.nof_filler_bits == right.nof_filler_bits) && (left.nof_short_segments == right.nof_short_segments) &&
         (left.E_short == right.E_short) && (left.E_long == right.E_long);
}

bool is_symbol_d2h_deferred()
{
  static const bool deferred = []() {
    const char* value = std::getenv("OCUDU_PDSCH_DEFER_SYMBOL_D2H");
    return value != nullptr && std::strcmp(value, "0") != 0;
  }();
  return deferred;
}

/// Maximum transport block size in bytes (max TB is ~1.3M bits = ~165KB).
static constexpr unsigned MAX_TB_BYTES = 200000;

/// Maximum coded bits (G can be very large for high MCS + large BW).
static constexpr unsigned MAX_CODED_BITS = 2000000;

/// Maximum symbols (G / min_mod_order = G / 2).
static constexpr unsigned MAX_SYMBOLS = MAX_CODED_BITS / 2;

/// GPU PDSCH TB encoder implementation using CUDA.
class pdsch_tb_encoder_gpu_impl : public pdsch_tb_encoder_gpu
{
public:
  pdsch_tb_encoder_gpu_impl()
  {
    // Initialize CUDA Driver API.
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
      gpu_available_ = false;
      return;
    }

    // Get device 0.
    res = cuDeviceGet(&device_, 0);
    if (res != CUDA_SUCCESS) {
      gpu_available_ = false;
      return;
    }

    // Retain the primary context (thread-safe).
    res = cuDevicePrimaryCtxRetain(&context_, device_);
    if (res != CUDA_SUCCESS) {
      gpu_available_ = false;
      return;
    }

    // Set context ONCE here - will be inherited by calling thread.
    res = cuCtxSetCurrent(context_);
    if (res != CUDA_SUCCESS) {
      cuDevicePrimaryCtxRelease(device_);
      gpu_available_ = false;
      return;
    }

    cudaError_t err = ocudu::cudaStreamCreateUpperPhy(&stream_);
    if (err != cudaSuccess) {
      cuDevicePrimaryCtxRelease(device_);
      gpu_available_ = false;
      return;
    }

    // Create CUDA event for async completion tracking (no timing needed for speed).
    err = cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream_);
      cuDevicePrimaryCtxRelease(device_);
      gpu_available_ = false;
      return;
    }

    const char* timing_env = std::getenv("OCUDU_PDSCH_TIMING");
    timing_enabled_        = timing_env && (std::strcmp(timing_env, "0") != 0);
    if (timing_enabled_) {
      if ((cudaEventCreate(&timing_start_event_) != cudaSuccess) ||
          (cudaEventCreate(&timing_h2d_event_) != cudaSuccess) ||
          (cudaEventCreate(&timing_encode_event_) != cudaSuccess) ||
          (cudaEventCreate(&timing_d2h_event_) != cudaSuccess)) {
        cleanup();
        return;
      }
      last_timing_stats_.enabled = true;
    }

    // Allocate device memory for TB input.
    err = cudaMalloc(&d_tb_input_, MAX_TB_BYTES);
    if (err != cudaSuccess) {
      cleanup();
      return;
    }

    // Allocate device memory for coded bits (rate-matched + scrambled).
    // Need bytes for bit-packed output.
    err = cudaMalloc(&d_coded_bits_, (MAX_CODED_BITS + 7) / 8);
    if (err != cudaSuccess) {
      cleanup();
      return;
    }

    // Allocate device memory for symbols.
    err = cudaMalloc(&d_symbols_, MAX_SYMBOLS * sizeof(cuFloatComplex));
    if (err != cudaSuccess) {
      cleanup();
      return;
    }

    // Allocate device memory for int8 symbols (2 bytes per symbol: real + imag).
    err = cudaMalloc(&d_symbols_int8_, MAX_SYMBOLS * 2);
    if (err != cudaSuccess) {
      cleanup();
      return;
    }

    // Allocate pinned host memory for fast async D2H transfer.
    err = cudaMallocHost(&h_symbols_pinned_, MAX_SYMBOLS * 2);
    if (err != cudaSuccess) {
      cleanup();
      return;
    }

    err = cudaMallocHost(&h_tb_input_pinned_, MAX_TB_BYTES);
    if (err != cudaSuccess) {
      cleanup();
      return;
    }

    // Create CUDA TB encoder.
    nr_ldpc_status_t status = tb_encoder_create(&tb_encoder_);
    if (status != NR_LDPC_SUCCESS) {
      cleanup();
      return;
    }

    // Create CUDA modulator.
    if (modulator_create(&modulator_) != 0) {
      tb_encoder_destroy(tb_encoder_);
      tb_encoder_ = nullptr;
      cleanup();
      return;
    }

    // Create CUDA scrambler for fused scramble+modulate path.
    if (scrambler_create(&scrambler_) != NR_LDPC_SUCCESS) {
      modulator_destroy(modulator_);
      modulator_ = nullptr;
      tb_encoder_destroy(tb_encoder_);
      tb_encoder_ = nullptr;
      cleanup();
      return;
    }

    // Preallocate scrambling sequence buffer for max coded bits to avoid
    // runtime allocations during real-time operation.
    if (scrambler_preallocate_sequence(scrambler_, MAX_CODED_BITS) != NR_LDPC_SUCCESS) {
      // Not fatal - will allocate on demand
    }

    gpu_available_ = true;

    // Comprehensive warmup to eliminate 15-18ms cold-start penalty.
    // This forces CUDA context initialization and ALL kernel JIT compilation to happen now.
    // Warmup includes: TB encoder, scrambler sequence generation, and modulation kernels.

    ocudulog::fetch_basic_logger("PHY").info("CUDA: Warming up PDSCH GPU encoder for real-time operation...");

    // 1. Warmup TB encoder (LDPC encoding + rate matching kernels)
    nr_ldpc_status_t warmup_status = tb_encoder_warmup(tb_encoder_, stream_);
    if (warmup_status != NR_LDPC_SUCCESS) {
      ocudulog::fetch_basic_logger("PHY").warning("CUDA: TB encoder warmup failed - first call will be slower");
    }

    // 2. Warmup scrambler sequence generation for typical 100MHz PDSCH configuration
    //    Max G for 100MHz 273 PRB = 273 * 12 * 13 * 8 (256QAM) = 340704 bits
    //    Note: PDSCH uses symbols 1-13 (13 symbols), symbol 0 is reserved for PDCCH
    //    Use representative config for warmup
    constexpr int      WARMUP_N_CODED_BITS = 340704; // 273 PRB * 12 * 13 * 8
    constexpr unsigned WARMUP_N_RNTI       = 0x4601;
    constexpr unsigned WARMUP_N_ID         = 1;
    constexpr unsigned WARMUP_CW_INDEX     = 0;

    nr_scrambling_config_t scr_cfg = {};
    scr_cfg.n_RNTI                 = WARMUP_N_RNTI;
    scr_cfg.n_ID                   = WARMUP_N_ID;
    scr_cfg.q                      = WARMUP_CW_INDEX;

    if (scrambler_configure(scrambler_, &scr_cfg) == NR_LDPC_SUCCESS) {
      // Generate scrambling sequence to trigger kernel JIT compilation
      if (scrambler_generate_sequence(scrambler_, WARMUP_N_CODED_BITS, stream_) == NR_LDPC_SUCCESS) {
        // 3. Warmup modulation kernels for all common modulation orders
        //    Run each modulation kernel to force JIT compilation
        const uint32_t* d_scramble_seq = reinterpret_cast<const uint32_t*>(scrambler_get_sequence_ptr(scrambler_));

        // Warmup QPSK (Qm=2)
        modulator_scramble_and_modulate_int8(modulator_,
                                             reinterpret_cast<uint32_t*>(d_coded_bits_),
                                             d_scramble_seq,
                                             d_symbols_int8_,
                                             WARMUP_N_CODED_BITS,
                                             2, // QPSK
                                             stream_);

        // Warmup 16QAM (Qm=4)
        modulator_scramble_and_modulate_int8(modulator_,
                                             reinterpret_cast<uint32_t*>(d_coded_bits_),
                                             d_scramble_seq,
                                             d_symbols_int8_,
                                             WARMUP_N_CODED_BITS,
                                             4, // 16QAM
                                             stream_);

        // Warmup 64QAM (Qm=6)
        modulator_scramble_and_modulate_int8(modulator_,
                                             reinterpret_cast<uint32_t*>(d_coded_bits_),
                                             d_scramble_seq,
                                             d_symbols_int8_,
                                             WARMUP_N_CODED_BITS,
                                             6, // 64QAM
                                             stream_);

        // Warmup 256QAM (Qm=8) - most common for high MCS
        modulator_scramble_and_modulate_int8(modulator_,
                                             reinterpret_cast<uint32_t*>(d_coded_bits_),
                                             d_scramble_seq,
                                             d_symbols_int8_,
                                             WARMUP_N_CODED_BITS,
                                             8, // 256QAM
                                             stream_);

        // Wait for all warmup kernels to complete
        cudaStreamSynchronize(stream_);

        ocudulog::fetch_basic_logger("PHY").info("CUDA: PDSCH GPU encoder warmup complete - ready for real-time");
      } else {
        ocudulog::fetch_basic_logger("PHY").warning("CUDA: Scrambler warmup failed - first call will be slower");
      }
    } else {
      ocudulog::fetch_basic_logger("PHY").warning("CUDA: Scrambler config failed - first call will be slower");
    }
  }

  ~pdsch_tb_encoder_gpu_impl() override
  {
    if (gpu_available_) {
      gpu_available_ = false;
      cuCtxSetCurrent(context_);

      if (stream_) {
        cudaStreamSynchronize(stream_);
      }

      if (scrambler_) {
        scrambler_destroy(scrambler_);
      }
      if (modulator_) {
        modulator_destroy(modulator_);
      }
      if (tb_encoder_) {
        tb_encoder_destroy(tb_encoder_);
      }

      cleanup();
    }
  }

  unsigned encode(span<const uint8_t> tb_bytes, const config& cfg) override
  {
    if (!gpu_available_) {
      return 0;
    }

    // NOTE: Context switch removed - context is set once at construction.
    // The calling thread inherits the context, which is thread-safe for primary context.

    // Configure TB encoder - disable internal scrambling for fused path.
    tb_encoder_config_t tb_cfg = {};
    tb_cfg.tb_size_bits        = static_cast<int>(cfg.tb_size_bits);
    tb_cfg.num_layers          = static_cast<int>(cfg.num_layers);
    tb_cfg.modulation_order    = static_cast<int>(cfg.modulation_order);
    tb_cfg.num_allocated_res   = static_cast<int>(cfg.num_coded_bits);
    tb_cfg.redundancy_version  = static_cast<int>(cfg.rv);
    tb_cfg.n_RNTI              = cfg.n_rnti;
    tb_cfg.n_ID                = cfg.n_id;
    tb_cfg.q                   = cfg.cw_index;
    tb_cfg.enable_scrambling   = true; // Enable - use tb_encoder's fused kernels with correct bit ordering
    // Calculate code rate from TBS and G (num_coded_bits).
    tb_cfg.code_rate = static_cast<float>(cfg.tb_size_bits) / static_cast<float>(cfg.num_coded_bits);
    // Pass LDPC parameters from CPU segmenter for consistency.
    tb_cfg.base_graph      = static_cast<int>(cfg.base_graph);
    tb_cfg.num_code_blocks = static_cast<int>(cfg.nof_codeblocks);
    tb_cfg.lifting_size    = static_cast<int>(cfg.lifting_size);
    tb_cfg.nof_filler_bits = static_cast<int>(cfg.nof_filler_bits);
    // Pass rate matching parameters from CPU segmenter for multi-CB consistency.
    tb_cfg.nof_short_segments = static_cast<int>(cfg.nof_short_segments);
    tb_cfg.E_short            = static_cast<int>(cfg.E_short);
    tb_cfg.E_long             = static_cast<int>(cfg.E_long);

    // Save modulation order for later use in get_symbols() scaling.
    last_modulation_order_ = cfg.modulation_order;

#ifdef OCUDU_PDSCH_DIAGNOSTICS
    fprintf(stderr,
            "[GPU PDSCH] TBS=%d bits, G=%d, rate=%.4f, Qm=%d, rv=%d, n_RNTI=0x%04X, n_ID=%d, q=%d\n",
            tb_cfg.tb_size_bits,
            tb_cfg.num_allocated_res,
            tb_cfg.code_rate,
            tb_cfg.modulation_order,
            tb_cfg.redundancy_version,
            tb_cfg.n_RNTI,
            tb_cfg.n_ID,
            tb_cfg.q);
#endif

    nr_ldpc_status_t status = NR_LDPC_SUCCESS;
    if (!last_config_valid_ || !are_configs_equal(cfg, last_config_)) {
      status = tb_encoder_configure(tb_encoder_, &tb_cfg);
      if (status != NR_LDPC_SUCCESS) {
        return 0;
      }
      last_config_       = cfg;
      last_config_valid_ = true;
    }

    // Upload TB bytes to GPU (single H2D transfer).
    unsigned tb_bytes_count = tb_bytes.size();
    if (tb_bytes_count > MAX_TB_BYTES) {
      return 0;
    }
    if (timing_enabled_) {
      cudaEventRecord(timing_start_event_, stream_);
    }
    std::memcpy(h_tb_input_pinned_, tb_bytes.data(), tb_bytes_count);
    cudaMemcpyAsync(d_tb_input_, h_tb_input_pinned_, tb_bytes_count, cudaMemcpyHostToDevice, stream_);
    if (timing_enabled_) {
      cudaEventRecord(timing_h2d_event_, stream_);
    }

    // Use tb_encoder_encode_to_symbols_int8 for complete pipeline:
    // CRC + Segment + LDPC + RM + Interleave + Scramble + Modulate → INT8
    // This uses the fused kernels with correct bit ordering.
    unsigned num_symbols = cfg.num_coded_bits / cfg.modulation_order;
    status               = tb_encoder_encode_to_symbols_int8(tb_encoder_, d_tb_input_, d_symbols_int8_, stream_);
    if (status != NR_LDPC_SUCCESS) {
      return 0;
    }
    if (timing_enabled_) {
      cudaEventRecord(timing_encode_event_, stream_);
    }

    symbols_pinned_valid_ = false;
    if (!defer_symbol_download_ && !is_symbol_d2h_deferred()) {
      // Start D2H transfer NOW so it overlaps with CPU work.
      // Copy all symbols to pinned memory in one transfer.
      cudaMemcpyAsync(h_symbols_pinned_,
                      d_symbols_int8_,
                      num_symbols * 2, // 2 bytes per symbol (I + Q)
                      cudaMemcpyDeviceToHost,
                      stream_);
      symbols_pinned_valid_ = true;
    }
    if (timing_enabled_) {
      cudaEventRecord(timing_d2h_event_, stream_);
    }

    // Record event for completion tracking - get_symbols() can poll this.
    cudaEventRecord(completion_event_, stream_);

    num_symbols_            = num_symbols;
    pending_timing_nof_cbs_ = cfg.nof_codeblocks;
    timing_report_pending_  = timing_enabled_;
    return num_symbols;
  }

  void get_symbols(span<ci8_t> symbols, unsigned offset, unsigned count) override
  {
    if (!gpu_available_ || count == 0) {
      return;
    }

    // NOTE: Context switch removed - using inherited context.

    // Wait for GPU completion using event with yielding (RT-friendly).
    // Event was recorded at end of encode() after the optional full D2H copy.
    cudaEventSynchronizeYielding(completion_event_);
    collect_timing_stats();

    if (symbols_pinned_valid_) {
      // Data is already in pinned memory from the async D2H in encode().
      // Just copy the requested slice to output buffer.
      std::memcpy(symbols.data(),
                  h_symbols_pinned_ + offset * 2, // 2 bytes per symbol
                  count * sizeof(ci8_t));
      return;
    }

    cudaMemcpyAsync(h_symbols_pinned_ + offset * 2,
                    d_symbols_int8_ + offset * 2,
                    count * sizeof(ci8_t),
                    cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronizeYielding(stream_);
    std::memcpy(symbols.data(), h_symbols_pinned_ + offset * 2, count * sizeof(ci8_t));
    if (count == num_symbols_) {
      symbols_pinned_valid_ = true;
    }
  }

  void get_symbols_float(span<cf_t> symbols, unsigned offset, unsigned count) override
  {
    if (!gpu_available_ || count == 0) {
      return;
    }

    // NOTE: Context switch removed - using inherited context.

    // Copy directly - cuFloatComplex is layout-compatible with cf_t.
    static_assert(sizeof(cf_t) == sizeof(cuFloatComplex), "cf_t and cuFloatComplex must have same size");

    // Wait for completion and copy (RT-friendly yielding sync).
    cudaEventSynchronize(completion_event_);
    collect_timing_stats();
    cudaMemcpyAsync(symbols.data(),
                    reinterpret_cast<cuFloatComplex*>(d_symbols_) + offset,
                    count * sizeof(cuFloatComplex),
                    cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronizeYielding(stream_);
  }

  void* get_device_symbols() const override { return d_symbols_; }

  const int8_t* get_device_symbols_int8() const override { return d_symbols_int8_; }

  void* get_execution_context() const override { return stream_; }

  void synchronize() override
  {
    if (!gpu_available_) {
      return;
    }
    cudaEventSynchronizeYielding(completion_event_);
    collect_timing_stats();
  }

  unsigned get_num_symbols() const override { return num_symbols_; }

  const timing_stats& get_last_timing_stats() const override { return last_timing_stats_; }

  bool is_gpu_available() const override { return gpu_available_; }

  void set_defer_symbol_download(bool defer) override { defer_symbol_download_ = defer; }

private:
  void collect_timing_stats()
  {
    if (!timing_enabled_) {
      return;
    }
    if (!timing_report_pending_) {
      return;
    }
    timing_report_pending_ = false;

    float h2d_ms    = 0.0F;
    float encode_ms = 0.0F;
    float d2h_ms    = 0.0F;
    float total_ms  = 0.0F;
    cudaEventElapsedTime(&h2d_ms, timing_start_event_, timing_h2d_event_);
    cudaEventElapsedTime(&encode_ms, timing_h2d_event_, timing_encode_event_);
    cudaEventElapsedTime(&d2h_ms, timing_encode_event_, timing_d2h_event_);
    cudaEventElapsedTime(&total_ms, timing_start_event_, timing_d2h_event_);

    last_timing_stats_.valid       = true;
    last_timing_stats_.enabled     = true;
    last_timing_stats_.h2d_us      = h2d_ms * 1000.0F;
    last_timing_stats_.encode_us   = encode_ms * 1000.0F;
    last_timing_stats_.d2h_us      = d2h_ms * 1000.0F;
    last_timing_stats_.total_us    = total_ms * 1000.0F;
    last_timing_stats_.nof_cbs     = pending_timing_nof_cbs_;
    last_timing_stats_.nof_symbols = num_symbols_;

    std::fprintf(stderr,
                 "CUDA PDSCH timing: cbs=%u symbols=%u h2d=%.1fus encode=%.1fus d2h=%.1fus total=%.1fus\n",
                 last_timing_stats_.nof_cbs,
                 last_timing_stats_.nof_symbols,
                 last_timing_stats_.h2d_us,
                 last_timing_stats_.encode_us,
                 last_timing_stats_.d2h_us,
                 last_timing_stats_.total_us);
  }

  void cleanup()
  {
    if (h_symbols_pinned_) {
      cudaFreeHost(h_symbols_pinned_);
      h_symbols_pinned_ = nullptr;
    }
    if (h_tb_input_pinned_) {
      cudaFreeHost(h_tb_input_pinned_);
      h_tb_input_pinned_ = nullptr;
    }
    if (d_symbols_int8_) {
      cudaFree(d_symbols_int8_);
      d_symbols_int8_ = nullptr;
    }
    if (d_symbols_) {
      cudaFree(d_symbols_);
      d_symbols_ = nullptr;
    }
    if (d_coded_bits_) {
      cudaFree(d_coded_bits_);
      d_coded_bits_ = nullptr;
    }
    if (d_tb_input_) {
      cudaFree(d_tb_input_);
      d_tb_input_ = nullptr;
    }
    if (completion_event_) {
      cudaEventDestroy(completion_event_);
      completion_event_ = nullptr;
    }
    if (timing_start_event_) {
      cudaEventDestroy(timing_start_event_);
      timing_start_event_ = nullptr;
    }
    if (timing_h2d_event_) {
      cudaEventDestroy(timing_h2d_event_);
      timing_h2d_event_ = nullptr;
    }
    if (timing_encode_event_) {
      cudaEventDestroy(timing_encode_event_);
      timing_encode_event_ = nullptr;
    }
    if (timing_d2h_event_) {
      cudaEventDestroy(timing_d2h_event_);
      timing_d2h_event_ = nullptr;
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    if (context_) {
      cuDevicePrimaryCtxRelease(device_);
      context_ = nullptr;
    }
    gpu_available_ = false;
  }

  bool gpu_available_ = false;

  // CUDA context and stream.
  CUdevice     device_              = 0;
  CUcontext    context_             = nullptr;
  cudaStream_t stream_              = nullptr;
  cudaEvent_t  completion_event_    = nullptr; // Event for async completion tracking
  cudaEvent_t  timing_start_event_  = nullptr;
  cudaEvent_t  timing_h2d_event_    = nullptr;
  cudaEvent_t  timing_encode_event_ = nullptr;
  cudaEvent_t  timing_d2h_event_    = nullptr;

  // Device memory.
  uint8_t* d_tb_input_     = nullptr;
  uint8_t* d_coded_bits_   = nullptr;
  void*    d_symbols_      = nullptr;
  int8_t*  d_symbols_int8_ = nullptr; // GPU buffer for int8 symbols

  // Pinned host memory for fast D2H transfer.
  int8_t*  h_symbols_pinned_  = nullptr;
  uint8_t* h_tb_input_pinned_ = nullptr;

  // CUDA handles.
  tb_encoder_handle_t tb_encoder_ = nullptr;
  modulator_handle_t  modulator_  = nullptr;
  scrambler_handle_t  scrambler_  = nullptr;

  // Output info.
  unsigned     num_symbols_            = 0;
  unsigned     last_modulation_order_  = 2; // Default to QPSK
  bool         symbols_pinned_valid_   = false;
  bool         defer_symbol_download_  = false;
  bool         timing_enabled_         = false;
  bool         timing_report_pending_  = false;
  unsigned     pending_timing_nof_cbs_ = 0;
  timing_stats last_timing_stats_;
  config       last_config_       = {};
  bool         last_config_valid_ = false;

  // Cached scrambling config for sequence reuse.
};

} // namespace

std::unique_ptr<pdsch_tb_encoder_gpu> ocudu::create_pdsch_tb_encoder_cuda()
{
  auto encoder = std::make_unique<pdsch_tb_encoder_gpu_impl>();
  if (!encoder->is_gpu_available()) {
    return nullptr;
  }
  return encoder;
}

bool ocudu::is_pdsch_tb_encoder_gpu_available()
{
  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    return false;
  }

  int device_count = 0;
  res              = cuDeviceGetCount(&device_count);
  return (res == CUDA_SUCCESS) && (device_count > 0);
}
