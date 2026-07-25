// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ldpc_encoder_cuda.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_encoder_buffer.h"
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ldpc_encoder.h>
#include <mutex>
#include <nr_ldpc_defs.h>
#include <unordered_map>
#include <vector>

using namespace ocudu;

namespace {

/// Convert logical bit index to the physical bit position within a uint32_t word,
/// using MSB-first-per-byte format expected by CUDA GPU kernels.
inline unsigned cuda_bit_pos(unsigned bit_idx)
{
  unsigned bit_in_word  = bit_idx % 32;
  unsigned byte_in_word = bit_in_word / 8;
  unsigned bit_in_byte  = 7 - (bit_in_word % 8);
  return byte_in_word * 8 + bit_in_byte;
}

/// Minimum number of bits to use GPU (avoid GPU overhead for small blocks).
static constexpr unsigned MIN_BITS_FOR_GPU = 1000;

/// Maximum output size for CUDA encoder (68 * max_lifting_size for BG1).
/// This is larger than ldpc::MAX_CODEBLOCK_SIZE which is 66 * max_lifting_size.
static constexpr unsigned MAX_LDPC_OUTPUT_SIZE = 68 * 384; // 26112 bits

/// Internal buffer for GPU encoder output implementing ldpc_encoder_buffer interface.
class ldpc_encoder_buffer_cuda : public ldpc_encoder_buffer
{
public:
  ldpc_encoder_buffer_cuda() : buffer_(MAX_LDPC_OUTPUT_SIZE) {}

  unsigned get_codeblock_length() const override { return codeblock_length_; }

  void write_codeblock(span<uint8_t> data, unsigned offset) const override
  {
    unsigned bytes_to_write = std::min(static_cast<unsigned>(data.size()), codeblock_length_ - offset);
    std::memcpy(data.data(), buffer_.data() + offset, bytes_to_write);
  }

  void set_codeblock_length(unsigned len) { codeblock_length_ = len; }

  span<uint8_t> data() { return span<uint8_t>(buffer_.data(), codeblock_length_); }

private:
  std::vector<uint8_t> buffer_;
  unsigned             codeblock_length_ = 0;
};

/// Key for encoder cache
struct encoder_config_key {
  int  bg;
  int  z;
  bool operator==(const encoder_config_key& other) const { return bg == other.bg && z == other.z; }
};

struct encoder_config_key_hash {
  size_t operator()(const encoder_config_key& k) const { return std::hash<int>()(k.bg) ^ (std::hash<int>()(k.z) << 1); }
};

/// Thread-local CUDA context setup helper.
/// Uses cuCtxSetCurrent to ensure the primary context is active on this thread.
class cuda_thread_context
{
public:
  /// Ensure the given context is current on this thread.
  static bool ensure_current(CUcontext ctx)
  {
    if (!ctx)
      return false;

    // Check if already current
    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == ctx) {
      return true;
    }

    // Set as current (not push - just set)
    return cuCtxSetCurrent(ctx) == CUDA_SUCCESS;
  }
};

/// LDPC encoder implementation using CUDA with thread-safe CUDA context management.
class ldpc_encoder_cuda_impl : public ldpc_encoder
{
public:
  ldpc_encoder_cuda_impl(std::unique_ptr<ldpc_encoder> fallback) : fallback_(std::move(fallback))
  {
    // Initialize CUDA Driver API
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
      gpu_available_ = false;
      return;
    }

    // Get device 0
    res = cuDeviceGet(&device_, 0);
    if (res != CUDA_SUCCESS) {
      gpu_available_ = false;
      return;
    }

    // Retain the primary context (shared across all threads)
    res = cuDevicePrimaryCtxRetain(&context_, device_);
    if (res != CUDA_SUCCESS) {
      gpu_available_ = false;
      return;
    }

    // Set context to initialize CUDA runtime on this thread
    res = cuCtxSetCurrent(context_);
    if (res != CUDA_SUCCESS) {
      cuDevicePrimaryCtxRelease(device_);
      gpu_available_ = false;
      return;
    }

    // Create CUDA stream
    cudaError_t err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      cuDevicePrimaryCtxRelease(device_);
      gpu_available_ = false;
      return;
    }

    // Allocate device memory
    constexpr size_t max_input_words  = 264; // 22 * 384 bits / 32
    constexpr size_t max_output_words = 816; // 68 * 384 bits / 32

    err = cudaMalloc(&d_input_, max_input_words * sizeof(uint32_t));
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream_);
      cuDevicePrimaryCtxRelease(device_);
      gpu_available_ = false;
      return;
    }

    err = cudaMalloc(&d_output_, max_output_words * sizeof(uint32_t));
    if (err != cudaSuccess) {
      cudaFree(d_input_);
      cudaStreamDestroy(stream_);
      cuDevicePrimaryCtxRelease(device_);
      gpu_available_ = false;
      return;
    }

    // Context stays set - no need to pop

    // Allocate host memory
    h_input_words_.resize(max_input_words);
    h_output_words_.resize(max_output_words);

    gpu_available_ = true;
  }

  ~ldpc_encoder_cuda_impl()
  {
    if (gpu_available_) {
      gpu_available_ = false; // Prevent re-entry

      // Set context for cleanup
      cuCtxSetCurrent(context_);

      // Synchronize all pending operations
      if (stream_) {
        cudaStreamSynchronize(stream_);
      }

      // Destroy cached encoders
      {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (auto& pair : encoder_cache_) {
          if (pair.second) {
            ldpc_encoder_destroy(pair.second);
            pair.second = nullptr;
          }
        }
        encoder_cache_.clear();
      }

      // Free resources in reverse order of allocation
      if (d_output_) {
        cudaFree(d_output_);
        d_output_ = nullptr;
      }
      if (d_input_) {
        cudaFree(d_input_);
        d_input_ = nullptr;
      }
      if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
      }

      // Release primary context
      if (context_) {
        cuDevicePrimaryCtxRelease(device_);
        context_ = nullptr;
      }
    }
  }

  const ldpc_encoder_buffer& encode(const bit_buffer& input, const configuration& cfg) override
  {
    int bg = (cfg.base_graph == ldpc_base_graph_type::BG1) ? 1 : 2;
    int Z  = static_cast<int>(cfg.lifting_size);

    // Check if we should use GPU
    if (!gpu_available_ || input.size() < MIN_BITS_FOR_GPU) {
      return fallback_->encode(input, cfg);
    }

    // Ensure CUDA context is current on this thread
    if (!cuda_thread_context::ensure_current(context_)) {
      return fallback_->encode(input, cfg);
    }

    // Get or create encoder for this configuration (with mutex for thread safety)
    ldpc_encoder_handle_t encoder = nullptr;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      encoder = get_or_create_encoder_locked(bg, Z, input.size());
    }

    if (!encoder) {
      return fallback_->encode(input, cfg);
    }

    // Get sizes from the configured encoder
    int K = ldpc_encoder_get_input_bits(encoder);
    int N = ldpc_encoder_get_output_bits(encoder);

    // Convert input bits to packed words
    unsigned input_words = (static_cast<unsigned>(K) + 31) / 32;
    std::memset(h_input_words_.data(), 0, input_words * sizeof(uint32_t));

    for (unsigned bit_idx = 0; bit_idx < input.size(); ++bit_idx) {
      if (input.extract(bit_idx, 1)) {
        unsigned word_idx = bit_idx / 32;
        unsigned bit_pos  = cuda_bit_pos(bit_idx);
        h_input_words_[word_idx] |= (1u << bit_pos);
      }
    }

    // Upload to GPU
    cudaMemcpyAsync(d_input_, h_input_words_.data(), input_words * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_);

    // Encode on GPU
    if (ldpc_encoder_encode(encoder, d_input_, d_output_, stream_) != NR_LDPC_SUCCESS) {
      cudaStreamSynchronize(stream_);
      return fallback_->encode(input, cfg);
    }

    // Download result
    unsigned output_words = (static_cast<unsigned>(N) + 31) / 32;
    cudaMemcpyAsync(
        h_output_words_.data(), d_output_, output_words * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Convert packed words to output buffer
    output_buffer_.set_codeblock_length(static_cast<unsigned>(N));
    span<uint8_t> out = output_buffer_.data();

    for (unsigned bit_idx = 0; bit_idx < static_cast<unsigned>(N); ++bit_idx) {
      unsigned word_idx = bit_idx / 32;
      unsigned bit_pos  = cuda_bit_pos(bit_idx);
      out[bit_idx]      = (h_output_words_[word_idx] >> bit_pos) & 1;
    }

    return output_buffer_;
  }

  bool is_gpu_available() const { return gpu_available_; }

private:
  /// Get or create encoder - must be called with cache_mutex_ held
  ldpc_encoder_handle_t get_or_create_encoder_locked(int bg, int z, size_t input_size)
  {
    encoder_config_key key{bg, z};

    auto it = encoder_cache_.find(key);
    if (it != encoder_cache_.end()) {
      return it->second;
    }

    // Create new encoder
    ldpc_encoder_handle_t encoder = nullptr;
    if (ldpc_encoder_create(&encoder) != NR_LDPC_SUCCESS) {
      return nullptr;
    }

    // Configure encoder
    nr_ldpc_config_t ldpc_cfg  = {};
    ldpc_cfg.base_graph        = bg;
    ldpc_cfg.lifting_size      = z;
    ldpc_cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(z);

    if (bg == 1) {
      ldpc_cfg.num_info_bits    = 22 * z;
      ldpc_cfg.num_parity_bits  = 46 * z;
      ldpc_cfg.max_parity_nodes = 46;
    } else {
      ldpc_cfg.num_info_bits    = 10 * z;
      ldpc_cfg.num_parity_bits  = 42 * z;
      ldpc_cfg.max_parity_nodes = 42;
    }
    ldpc_cfg.num_codeword_bits  = ldpc_cfg.num_info_bits + ldpc_cfg.num_parity_bits;
    ldpc_cfg.num_filler_bits    = ldpc_cfg.num_info_bits - static_cast<int>(input_size);
    ldpc_cfg.puncture           = true;
    ldpc_cfg.redundancy_version = 0;

    if (ldpc_encoder_configure(encoder, &ldpc_cfg) != NR_LDPC_SUCCESS) {
      ldpc_encoder_destroy(encoder);
      return nullptr;
    }

    encoder_cache_[key] = encoder;
    return encoder;
  }

  std::unique_ptr<ldpc_encoder> fallback_;
  bool                          gpu_available_ = false;

  // CUDA Driver API handles for thread-safe context management
  CUdevice  device_  = 0;
  CUcontext context_ = nullptr;

  // CUDA Runtime resources
  cudaStream_t stream_ = nullptr;

  // Device memory
  uint32_t* d_input_  = nullptr;
  uint32_t* d_output_ = nullptr;

  // Host memory for bit packing
  std::vector<uint32_t> h_input_words_;
  std::vector<uint32_t> h_output_words_;

  // Output buffer
  ldpc_encoder_buffer_cuda output_buffer_;

  // Cache of pre-configured encoders (one per BG/Z combination)
  std::unordered_map<encoder_config_key, ldpc_encoder_handle_t, encoder_config_key_hash> encoder_cache_;
  std::mutex                                                                             cache_mutex_;
};

} // namespace

std::unique_ptr<ldpc_encoder> ocudu::create_ldpc_encoder_cuda(std::unique_ptr<ldpc_encoder> fallback)
{
  return std::make_unique<ldpc_encoder_cuda_impl>(std::move(fallback));
}

bool ocudu::is_ldpc_encoder_gpu_available()
{
  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    return false;
  }

  int device_count = 0;
  res              = cuDeviceGetCount(&device_count);
  return (res == CUDA_SUCCESS) && (device_count > 0);
}
