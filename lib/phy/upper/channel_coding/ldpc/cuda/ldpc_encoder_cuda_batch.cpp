// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ldpc_encoder_cuda_batch.h"
#include "ocudu/phy/upper/channel_coding/channel_coding_factories.h"
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

/// Minimum number of codeblocks to use GPU batching.
static constexpr unsigned MIN_CBS_FOR_GPU_BATCH = 2;

/// Internal buffer for batch encoder output.
class ldpc_encoder_buffer_batch : public ldpc_encoder_buffer
{
public:
  ldpc_encoder_buffer_batch() : buffer_(MAX_CB_OUTPUT_WORDS * 32) {}

  unsigned get_codeblock_length() const override { return codeblock_length_; }

  void write_codeblock(span<uint8_t> data, unsigned offset) const override
  {
    unsigned bytes_to_write = std::min(static_cast<unsigned>(data.size()), codeblock_length_ - offset);
    std::memcpy(data.data(), buffer_.data() + offset, bytes_to_write);
  }

  void          set_codeblock_length(unsigned len) { codeblock_length_ = len; }
  span<uint8_t> data() { return span<uint8_t>(buffer_.data(), codeblock_length_); }

private:
  std::vector<uint8_t> buffer_;
  unsigned             codeblock_length_ = 0;
};

/// Key for encoder cache.
struct encoder_config_key {
  int  bg;
  int  z;
  bool operator==(const encoder_config_key& other) const { return bg == other.bg && z == other.z; }
};

struct encoder_config_key_hash {
  size_t operator()(const encoder_config_key& k) const { return std::hash<int>()(k.bg) ^ (std::hash<int>()(k.z) << 1); }
};

/// Batched LDPC encoder implementation using CUDA.
class ldpc_encoder_cuda_batch_impl : public ldpc_encoder_batch
{
public:
  ldpc_encoder_cuda_batch_impl(std::shared_ptr<ldpc_encoder_factory> cpu_factory) :
    cpu_factory_(std::move(cpu_factory))
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

    // Retain the primary context.
    res = cuDevicePrimaryCtxRetain(&context_, device_);
    if (res != CUDA_SUCCESS) {
      gpu_available_ = false;
      return;
    }

    // Set context.
    res = cuCtxSetCurrent(context_);
    if (res != CUDA_SUCCESS) {
      cuDevicePrimaryCtxRelease(device_);
      gpu_available_ = false;
      return;
    }

    // Create CUDA stream.
    cudaError_t err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      cuDevicePrimaryCtxRelease(device_);
      gpu_available_ = false;
      return;
    }

    // Allocate device memory for batched input.
    size_t input_batch_size = MAX_LDPC_BATCH_SIZE * MAX_CB_INPUT_WORDS * sizeof(uint32_t);
    err                     = cudaMalloc(&d_input_batch_, input_batch_size);
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream_);
      cuDevicePrimaryCtxRelease(device_);
      gpu_available_ = false;
      return;
    }

    // Allocate device memory for batched output.
    size_t output_batch_size = MAX_LDPC_BATCH_SIZE * MAX_CB_OUTPUT_WORDS * sizeof(uint32_t);
    err                      = cudaMalloc(&d_output_batch_, output_batch_size);
    if (err != cudaSuccess) {
      cudaFree(d_input_batch_);
      cudaStreamDestroy(stream_);
      cuDevicePrimaryCtxRelease(device_);
      gpu_available_ = false;
      return;
    }

    // Allocate host memory for bit packing.
    h_input_batch_.resize(MAX_LDPC_BATCH_SIZE * MAX_CB_INPUT_WORDS);
    h_output_batch_.resize(MAX_LDPC_BATCH_SIZE * MAX_CB_OUTPUT_WORDS);

    // Pre-allocate output buffers.
    output_buffers_.resize(MAX_LDPC_BATCH_SIZE);

    gpu_available_ = true;
  }

  ~ldpc_encoder_cuda_batch_impl() override
  {
    if (gpu_available_) {
      gpu_available_ = false;

      cuCtxSetCurrent(context_);

      if (stream_) {
        cudaStreamSynchronize(stream_);
      }

      // Destroy cached encoders.
      {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (auto& pair : encoder_cache_) {
          if (pair.second) {
            ldpc_encoder_destroy(pair.second);
          }
        }
        encoder_cache_.clear();
      }

      if (d_output_batch_) {
        cudaFree(d_output_batch_);
      }
      if (d_input_batch_) {
        cudaFree(d_input_batch_);
      }
      if (stream_) {
        cudaStreamDestroy(stream_);
      }
      if (context_) {
        cuDevicePrimaryCtxRelease(device_);
      }
    }
  }

  void encode_batch(span<const bit_buffer>             inputs,
                    span<ldpc_encoder_buffer*>         outputs,
                    const ldpc_encoder::configuration& cfg) override
  {
    unsigned num_cbs = inputs.size();

    // Fall back to CPU for small batches.
    if (!gpu_available_ || num_cbs < MIN_CBS_FOR_GPU_BATCH || num_cbs > MAX_LDPC_BATCH_SIZE) {
      encode_batch_cpu(inputs, outputs, cfg);
      return;
    }

    // Ensure CUDA context is current.
    cuCtxSetCurrent(context_);

    int bg = (cfg.base_graph == ldpc_base_graph_type::BG1) ? 1 : 2;
    int Z  = static_cast<int>(cfg.lifting_size);

    // Get or create encoder.
    ldpc_encoder_handle_t encoder = nullptr;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      encoder = get_or_create_encoder_locked(bg, Z, inputs[0].size());
    }

    if (!encoder) {
      encode_batch_cpu(inputs, outputs, cfg);
      return;
    }

    int N            = ldpc_encoder_get_output_bits(encoder);
    int input_words  = ldpc_encoder_get_input_words(encoder);
    int output_words = ldpc_encoder_get_output_words(encoder);

    // Pack all CB inputs into contiguous host buffer.
    std::memset(h_input_batch_.data(), 0, num_cbs * input_words * sizeof(uint32_t));

    for (unsigned cb = 0; cb < num_cbs; ++cb) {
      const bit_buffer& input     = inputs[cb];
      uint32_t*         cb_buffer = h_input_batch_.data() + cb * input_words;

      for (unsigned bit_idx = 0; bit_idx < input.size(); ++bit_idx) {
        if (input.extract(bit_idx, 1)) {
          unsigned word_idx = bit_idx / 32;
          unsigned bit_pos  = cuda_bit_pos(bit_idx);
          cb_buffer[word_idx] |= (1u << bit_pos);
        }
      }
    }

    // Upload all inputs to GPU in single transfer.
    cudaMemcpyAsync(d_input_batch_,
                    h_input_batch_.data(),
                    num_cbs * input_words * sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    stream_);

    // Batch encode on GPU.
    nr_ldpc_status_t status = ldpc_encoder_encode_batch(encoder, d_input_batch_, d_output_batch_, num_cbs, stream_);

    if (status != NR_LDPC_SUCCESS) {
      cudaStreamSynchronize(stream_);
      encode_batch_cpu(inputs, outputs, cfg);
      return;
    }

    // Download all outputs from GPU in single transfer.
    cudaMemcpyAsync(h_output_batch_.data(),
                    d_output_batch_,
                    num_cbs * output_words * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronize(stream_);

    // Unpack outputs to individual buffers.
    for (unsigned cb = 0; cb < num_cbs; ++cb) {
      const uint32_t* cb_output = h_output_batch_.data() + cb * output_words;

      output_buffers_[cb].set_codeblock_length(static_cast<unsigned>(N));
      span<uint8_t> out = output_buffers_[cb].data();

      for (unsigned bit_idx = 0; bit_idx < static_cast<unsigned>(N); ++bit_idx) {
        unsigned word_idx = bit_idx / 32;
        unsigned bit_pos  = cuda_bit_pos(bit_idx);
        out[bit_idx]      = (cb_output[word_idx] >> bit_pos) & 1;
      }

      outputs[cb] = &output_buffers_[cb];
    }

    // Save info for GPU-resident mode.
    last_gpu_info_.d_encoded_bits = d_output_batch_;
    last_gpu_info_.num_cbs        = num_cbs;
    last_gpu_info_.output_stride  = output_words;
    last_gpu_info_.stream         = stream_;
    last_gpu_info_.valid          = true;
  }

  void encode_batch_gpu_resident(const uint32_t*                    d_inputs,
                                 uint32_t*                          d_outputs,
                                 unsigned                           num_cbs,
                                 unsigned                           input_stride,
                                 unsigned                           output_stride,
                                 const ldpc_encoder::configuration& cfg,
                                 cudaStream_t                       stream) override
  {
    if (!gpu_available_ || num_cbs == 0) {
      return;
    }

    cuCtxSetCurrent(context_);

    int bg = (cfg.base_graph == ldpc_base_graph_type::BG1) ? 1 : 2;
    int Z  = static_cast<int>(cfg.lifting_size);

    ldpc_encoder_handle_t encoder = nullptr;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      // For GPU-resident mode, use full K bits as input size.
      unsigned K = (bg == 1) ? 22 * Z : 10 * Z;
      encoder    = get_or_create_encoder_locked(bg, Z, K);
    }

    if (!encoder) {
      return;
    }

    // Batch encode directly on device.
    ldpc_encoder_encode_batch(encoder, d_inputs, d_outputs, num_cbs, stream);

    // Save GPU info.
    last_gpu_info_.d_encoded_bits = d_outputs;
    last_gpu_info_.num_cbs        = num_cbs;
    last_gpu_info_.output_stride  = output_stride;
    last_gpu_info_.stream         = stream;
    last_gpu_info_.valid          = true;
  }

  gpu_ldpc_buffer_info get_gpu_buffer_info() const override { return last_gpu_info_; }

  bool is_gpu_available() const override { return gpu_available_; }

  unsigned get_max_batch_size() const override { return MAX_LDPC_BATCH_SIZE; }

private:
  /// CPU fallback for small batches.
  void encode_batch_cpu(span<const bit_buffer>             inputs,
                        span<ldpc_encoder_buffer*>         outputs,
                        const ldpc_encoder::configuration& cfg)
  {
    // Create CPU encoder if needed.
    if (!cpu_encoder_) {
      cpu_encoder_ = cpu_factory_->create();
    }

    for (unsigned cb = 0; cb < inputs.size(); ++cb) {
      const ldpc_encoder_buffer& result = cpu_encoder_->encode(inputs[cb], cfg);
      // Copy to our internal buffer.
      output_buffers_[cb].set_codeblock_length(result.get_codeblock_length());
      result.write_codeblock(output_buffers_[cb].data(), 0);
      outputs[cb] = &output_buffers_[cb];
    }

    last_gpu_info_.valid = false;
  }

  /// Get or create encoder - must be called with cache_mutex_ held.
  ldpc_encoder_handle_t get_or_create_encoder_locked(int bg, int z, size_t input_size)
  {
    encoder_config_key key{bg, z};

    auto it = encoder_cache_.find(key);
    if (it != encoder_cache_.end()) {
      return it->second;
    }

    ldpc_encoder_handle_t encoder = nullptr;
    if (ldpc_encoder_create(&encoder) != NR_LDPC_SUCCESS) {
      return nullptr;
    }

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

  std::shared_ptr<ldpc_encoder_factory> cpu_factory_;
  std::unique_ptr<ldpc_encoder>         cpu_encoder_;
  bool                                  gpu_available_ = false;

  // CUDA context.
  CUdevice     device_  = 0;
  CUcontext    context_ = nullptr;
  cudaStream_t stream_  = nullptr;

  // Device memory.
  uint32_t* d_input_batch_  = nullptr;
  uint32_t* d_output_batch_ = nullptr;

  // Host memory.
  std::vector<uint32_t> h_input_batch_;
  std::vector<uint32_t> h_output_batch_;

  // Output buffers.
  std::vector<ldpc_encoder_buffer_batch> output_buffers_;

  // Encoder cache.
  std::unordered_map<encoder_config_key, ldpc_encoder_handle_t, encoder_config_key_hash> encoder_cache_;
  std::mutex                                                                             cache_mutex_;

  // Last GPU operation info.
  gpu_ldpc_buffer_info last_gpu_info_ = {};
};

} // namespace

std::unique_ptr<ldpc_encoder_batch>
ocudu::create_ldpc_encoder_batch_cuda(std::shared_ptr<ldpc_encoder_factory> cpu_encoder_factory)
{
  return std::make_unique<ldpc_encoder_cuda_batch_impl>(std::move(cpu_encoder_factory));
}

bool ocudu::is_ldpc_encoder_batch_gpu_available()
{
  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    return false;
  }

  int device_count = 0;
  res              = cuDeviceGetCount(&device_count);
  return (res == CUDA_SUCCESS) && (device_count > 0);
}
