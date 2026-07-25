// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ofh_compression.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdint.h>
#include <thread>

struct ocudu_ofh_compression_handle {
    static constexpr unsigned HOST_INPUT_STAGING_SLOTS = 4U;

    cudaStream_t stream = nullptr;
    uint8_t*     d_in   = nullptr;
    uint8_t*     d_out  = nullptr;
    uint8_t*     h_out  = nullptr;
    uint8_t*     h_in[HOST_INPUT_STAGING_SLOTS] = {};
    cudaEvent_t  h_in_ready_events[HOST_INPUT_STAGING_SLOTS] = {};
    bool         h_in_event_valid[HOST_INPUT_STAGING_SLOTS] = {};
    size_t       in_cap                                    = 0;
    size_t       out_cap                                   = 0;
    size_t       h_out_cap                                 = 0;
    size_t       h_in_cap[HOST_INPUT_STAGING_SLOTS] = {};
    unsigned     h_in_slot                         = 0;
};

namespace {

constexpr unsigned NOF_SUBCARRIERS_PER_RB = 12U;
constexpr unsigned NOF_IQ_SAMPLES_PER_PRB = 2U * NOF_SUBCARRIERS_PER_RB;
constexpr unsigned COMPRESS_9B_WARPS_PER_BLOCK = 4U;
constexpr unsigned COMPRESS_9B_NONE_GROUP_LOW_PRB_THRESHOLD = 1024U;
constexpr unsigned COMPRESS_9B_NONE_GROUP_HIGH_PRB_THRESHOLD = 8192U;
constexpr unsigned COMPRESS_12B_WARPS_PER_BLOCK = 8U;
constexpr unsigned DECOMPRESS_9B_RE_PARALLEL_PRB_THRESHOLD = 4096U;
constexpr unsigned DECOMPRESS_12B_RE_PARALLEL_PRB_THRESHOLD = 4096U;

static bool use_9b_none_group_compressor(unsigned total_prbs)
{
    return (total_prbs < COMPRESS_9B_NONE_GROUP_LOW_PRB_THRESHOLD) ||
           (total_prbs >= COMPRESS_9B_NONE_GROUP_HIGH_PRB_THRESHOLD);
}

static unsigned compressed_prb_size(int compression_type, unsigned data_width)
{
    unsigned bits = NOF_IQ_SAMPLES_PER_PRB * data_width;
    if (compression_type == OCUDU_OFH_COMPRESSION_TYPE_BFP) {
        bits += 8U;
    }
    return (bits + 7U) / 8U;
}

static bool valid_request(int compression_type, unsigned nof_prbs, unsigned data_width)
{
    if ((compression_type != OCUDU_OFH_COMPRESSION_TYPE_NONE) &&
        (compression_type != OCUDU_OFH_COMPRESSION_TYPE_BFP)) {
        return false;
    }
    return (data_width > 0U) && (data_width <= 16U) && (nof_prbs > 0U);
}

static bool valid_grid_request(int      compression_type,
                               unsigned nof_symbols,
                               unsigned nof_subc,
                               unsigned symbol,
                               unsigned start_prb,
                               unsigned nof_prbs,
                               unsigned data_width)
{
    if (!valid_request(compression_type, nof_prbs, data_width)) {
        return false;
    }
    if ((nof_symbols == 0U) || (nof_subc == 0U) || (symbol >= nof_symbols)) {
        return false;
    }
    if ((nof_subc % NOF_SUBCARRIERS_PER_RB) != 0U) {
        return false;
    }
    return (start_prb <= (nof_subc / NOF_SUBCARRIERS_PER_RB)) &&
           (nof_prbs <= (nof_subc / NOF_SUBCARRIERS_PER_RB - start_prb));
}

static bool valid_grid_symbol_batch_request(int      compression_type,
                                            unsigned nof_grid_symbols,
                                            unsigned nof_subc,
                                            unsigned first_symbol,
                                            unsigned nof_symbols,
                                            unsigned start_prb,
                                            unsigned nof_prbs,
                                            unsigned data_width)
{
    if ((nof_symbols == 0U) ||
        !valid_grid_request(compression_type, nof_grid_symbols, nof_subc, first_symbol, start_prb, nof_prbs, data_width)) {
        return false;
    }
    return nof_symbols <= (nof_grid_symbols - first_symbol);
}

static bool ensure_capacity(uint8_t** ptr, size_t* capacity, size_t required)
{
    if (*capacity >= required) {
        return true;
    }
    if (*ptr != nullptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
        *capacity = 0;
    }
    if (required == 0) {
        return true;
    }
    if (cudaMalloc(reinterpret_cast<void**>(ptr), required) != cudaSuccess) {
        return false;
    }
    *capacity = required;
    return true;
}

static bool ensure_host_capacity(uint8_t** ptr, size_t* capacity, size_t required)
{
    if (*capacity >= required) {
        return true;
    }
    if (*ptr != nullptr) {
        cudaFreeHost(*ptr);
        *ptr = nullptr;
        *capacity = 0;
    }
    if (required == 0) {
        return true;
    }
    if (cudaMallocHost(reinterpret_cast<void**>(ptr), required) != cudaSuccess) {
        return false;
    }
    *capacity = required;
    return true;
}

static bool use_pinned_host_output_transfer(int compression_type, unsigned data_width)
{
    return (compression_type == OCUDU_OFH_COMPRESSION_TYPE_BFP) &&
           ((data_width == 9U) || (data_width == 12U) || (data_width == 14U));
}

static bool use_pinned_host_input_transfer(int compression_type, unsigned data_width)
{
    return (compression_type == OCUDU_OFH_COMPRESSION_TYPE_BFP) &&
           ((data_width == 9U) || (data_width == 12U) || (data_width == 14U));
}

static cudaError_t wait_event_yielding(cudaEvent_t event)
{
    cudaError_t status;
    while ((status = cudaEventQuery(event)) == cudaErrorNotReady) {
        std::this_thread::yield();
    }
    return status;
}

static bool copy_device_to_host(ocudu_ofh_compression_handle_t* handle,
                                void*                           dst,
                                const uint8_t*                   src,
                                size_t                           bytes,
                                bool                             use_pinned_staging)
{
    if (use_pinned_staging) {
        if (!ensure_host_capacity(&handle->h_out, &handle->h_out_cap, bytes)) {
            return false;
        }
        if (cudaMemcpyAsync(handle->h_out, src, bytes, cudaMemcpyDeviceToHost, handle->stream) != cudaSuccess) {
            return false;
        }
        if (cudaStreamSynchronize(handle->stream) != cudaSuccess) {
            return false;
        }
        std::memcpy(dst, handle->h_out, bytes);
        return true;
    }

    if (cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, handle->stream) != cudaSuccess) {
        return false;
    }
    return cudaStreamSynchronize(handle->stream) == cudaSuccess;
}

static bool copy_device_to_host_buffers(ocudu_ofh_compression_handle_t* handle,
                                        void* const*                    dst_buffers,
                                        unsigned                        nof_dst_buffers,
                                        unsigned                        dst_buffer_bytes,
                                        const uint8_t*                  src)
{
    if ((nof_dst_buffers == 0U) || (dst_buffer_bytes == 0U)) {
        return true;
    }

    size_t total_bytes = static_cast<size_t>(nof_dst_buffers) * dst_buffer_bytes;
    if (!ensure_host_capacity(&handle->h_out, &handle->h_out_cap, total_bytes)) {
        return false;
    }
    if (cudaMemcpyAsync(handle->h_out, src, total_bytes, cudaMemcpyDeviceToHost, handle->stream) != cudaSuccess) {
        return false;
    }
    if (cudaStreamSynchronize(handle->stream) != cudaSuccess) {
        return false;
    }

    for (unsigned i = 0; i != nof_dst_buffers; ++i) {
        std::memcpy(dst_buffers[i], handle->h_out + static_cast<size_t>(i) * dst_buffer_bytes, dst_buffer_bytes);
    }
    return true;
}

static bool copy_host_to_device(ocudu_ofh_compression_handle_t* handle,
                                uint8_t*                        dst,
                                const void*                     src,
                                size_t                          bytes,
                                bool                            use_pinned_staging)
{
    if (!use_pinned_staging) {
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, handle->stream) == cudaSuccess;
    }

    const unsigned slot = handle->h_in_slot++ % ocudu_ofh_compression_handle::HOST_INPUT_STAGING_SLOTS;
    if (handle->h_in_event_valid[slot] &&
        (wait_event_yielding(handle->h_in_ready_events[slot]) != cudaSuccess)) {
        return false;
    }

    if (!ensure_host_capacity(&handle->h_in[slot], &handle->h_in_cap[slot], bytes)) {
        return false;
    }
    if (handle->h_in_ready_events[slot] == nullptr) {
        if (cudaEventCreateWithFlags(&handle->h_in_ready_events[slot], cudaEventDisableTiming) != cudaSuccess) {
            return false;
        }
    }

    std::memcpy(handle->h_in[slot], src, bytes);
    if (cudaMemcpyAsync(dst, handle->h_in[slot], bytes, cudaMemcpyHostToDevice, handle->stream) != cudaSuccess) {
        return false;
    }
    if (cudaEventRecord(handle->h_in_ready_events[slot], handle->stream) != cudaSuccess) {
        return false;
    }
    handle->h_in_event_valid[slot] = true;
    return true;
}

__device__ __forceinline__ float bf16_to_float(uint16_t value)
{
    return __uint_as_float(static_cast<uint32_t>(value) << 16U);
}

__device__ __forceinline__ uint16_t float_to_bf16(float value)
{
    uint32_t bits = __float_as_uint(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

__device__ __forceinline__ int16_t quantize_bf16(uint16_t value, float scale)
{
    return static_cast<int16_t>(roundf(bf16_to_float(value) * scale));
}

__device__ __forceinline__ unsigned determine_bfp_exponent(uint16_t x, unsigned data_width)
{
    unsigned max_shift = 16U - data_width;
    unsigned lz_without_sign = max_shift;
    if ((x > 0U) && (max_shift > 0U)) {
        lz_without_sign = __clz(static_cast<unsigned>(x)) - 16U - 1U;
    }
    unsigned raw_exp = min(max_shift, lz_without_sign);
    return (16U - data_width) - raw_exp;
}

__device__ __forceinline__ int bfp_abs_for_exponent(int sample)
{
    return (sample >= 0) ? sample : (-sample - 1);
}

template <unsigned DATA_WIDTH, bool BFP>
__device__ __forceinline__ uint16_t prepare_quantized_sample(int16_t sample, unsigned exponent)
{
    if constexpr (BFP) {
        sample = static_cast<int16_t>(sample >> exponent);
    }
    const uint16_t mask = (DATA_WIDTH == 16U) ? 0xffffU : static_cast<uint16_t>((1U << DATA_WIDTH) - 1U);
    return static_cast<uint16_t>(sample) & mask;
}

template <unsigned DATA_WIDTH, bool BFP>
__device__ __forceinline__ void pack_prb_samples_msb(uint8_t* output, const int16_t* quantized, unsigned exponent)
{
    if constexpr (DATA_WIDTH == 8U) {
#pragma unroll
        for (unsigned i = 0; i != NOF_IQ_SAMPLES_PER_PRB; ++i) {
            output[i] = static_cast<uint8_t>(prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i], exponent));
        }
        return;
    }

    if constexpr (DATA_WIDTH == 9U) {
#pragma unroll
        for (unsigned i = 0, out_idx = 0; i != NOF_IQ_SAMPLES_PER_PRB; i += 8U, out_idx += 9U) {
            uint16_t s0 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i], exponent);
            uint16_t s1 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 1U], exponent);
            uint16_t s2 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 2U], exponent);
            uint16_t s3 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 3U], exponent);
            uint16_t s4 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 4U], exponent);
            uint16_t s5 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 5U], exponent);
            uint16_t s6 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 6U], exponent);
            uint16_t s7 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 7U], exponent);
            output[out_idx]      = static_cast<uint8_t>(s0 >> 1U);
            output[out_idx + 1U] = static_cast<uint8_t>((s0 << 7U) | (s1 >> 2U));
            output[out_idx + 2U] = static_cast<uint8_t>((s1 << 6U) | (s2 >> 3U));
            output[out_idx + 3U] = static_cast<uint8_t>((s2 << 5U) | (s3 >> 4U));
            output[out_idx + 4U] = static_cast<uint8_t>((s3 << 4U) | (s4 >> 5U));
            output[out_idx + 5U] = static_cast<uint8_t>((s4 << 3U) | (s5 >> 6U));
            output[out_idx + 6U] = static_cast<uint8_t>((s5 << 2U) | (s6 >> 7U));
            output[out_idx + 7U] = static_cast<uint8_t>((s6 << 1U) | (s7 >> 8U));
            output[out_idx + 8U] = static_cast<uint8_t>(s7);
        }
        return;
    }

    if constexpr (DATA_WIDTH == 10U) {
#pragma unroll
        for (unsigned i = 0, out_idx = 0; i != NOF_IQ_SAMPLES_PER_PRB; i += 4U, out_idx += 5U) {
            uint16_t s0 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i], exponent);
            uint16_t s1 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 1U], exponent);
            uint16_t s2 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 2U], exponent);
            uint16_t s3 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 3U], exponent);
            output[out_idx]      = static_cast<uint8_t>(s0 >> 2U);
            output[out_idx + 1U] = static_cast<uint8_t>((s0 << 6U) | (s1 >> 4U));
            output[out_idx + 2U] = static_cast<uint8_t>((s1 << 4U) | (s2 >> 6U));
            output[out_idx + 3U] = static_cast<uint8_t>((s2 << 2U) | (s3 >> 8U));
            output[out_idx + 4U] = static_cast<uint8_t>(s3);
        }
        return;
    }

    if constexpr (DATA_WIDTH == 12U) {
#pragma unroll
        for (unsigned i = 0, out_idx = 0; i != NOF_IQ_SAMPLES_PER_PRB; i += 2U, out_idx += 3U) {
            uint16_t first  = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i], exponent);
            uint16_t second = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 1U], exponent);
            output[out_idx]      = static_cast<uint8_t>(first >> 4U);
            output[out_idx + 1U] = static_cast<uint8_t>((first << 4U) | (second >> 8U));
            output[out_idx + 2U] = static_cast<uint8_t>(second);
        }
        return;
    }

    if constexpr (DATA_WIDTH == 14U) {
#pragma unroll
        for (unsigned i = 0, out_idx = 0; i != NOF_IQ_SAMPLES_PER_PRB; i += 4U, out_idx += 7U) {
            uint16_t s0 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i], exponent);
            uint16_t s1 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 1U], exponent);
            uint16_t s2 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 2U], exponent);
            uint16_t s3 = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i + 3U], exponent);
            output[out_idx]      = static_cast<uint8_t>(s0 >> 6U);
            output[out_idx + 1U] = static_cast<uint8_t>((s0 << 2U) | (s1 >> 12U));
            output[out_idx + 2U] = static_cast<uint8_t>(s1 >> 4U);
            output[out_idx + 3U] = static_cast<uint8_t>((s1 << 4U) | (s2 >> 10U));
            output[out_idx + 4U] = static_cast<uint8_t>(s2 >> 2U);
            output[out_idx + 5U] = static_cast<uint8_t>((s2 << 6U) | (s3 >> 8U));
            output[out_idx + 6U] = static_cast<uint8_t>(s3);
        }
        return;
    }

    if constexpr (DATA_WIDTH == 16U) {
#pragma unroll
        for (unsigned i = 0, out_idx = 0; i != NOF_IQ_SAMPLES_PER_PRB; ++i, out_idx += 2U) {
            uint16_t sample      = prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i], exponent);
            output[out_idx]      = static_cast<uint8_t>(sample >> 8U);
            output[out_idx + 1U] = static_cast<uint8_t>(sample);
        }
        return;
    }

    uint32_t       acc      = 0;
    unsigned       acc_bits = 0;
    unsigned       out_idx  = 0;

#pragma unroll
    for (unsigned i = 0; i != NOF_IQ_SAMPLES_PER_PRB; ++i) {
        acc = (acc << DATA_WIDTH) | prepare_quantized_sample<DATA_WIDTH, BFP>(quantized[i], exponent);
        acc_bits += DATA_WIDTH;
        while (acc_bits >= 8U) {
            acc_bits -= 8U;
            output[out_idx++] = static_cast<uint8_t>((acc >> acc_bits) & 0xffU);
            acc = (acc_bits == 0U) ? 0U : (acc & ((1U << acc_bits) - 1U));
        }
    }
}

template <bool BFP>
__device__ __forceinline__ void pack_prb_samples_msb_runtime(uint8_t*       output,
                                                             const int16_t* quantized,
                                                             unsigned       exponent,
                                                             unsigned       data_width)
{
    uint32_t       acc      = 0;
    unsigned       acc_bits = 0;
    unsigned       out_idx  = 0;
    const uint32_t mask     = (data_width == 16U) ? 0xffffU : ((1U << data_width) - 1U);

#pragma unroll
    for (unsigned i = 0; i != NOF_IQ_SAMPLES_PER_PRB; ++i) {
        int16_t sample = quantized[i];
        if constexpr (BFP) {
            sample = static_cast<int16_t>(sample >> exponent);
        }

        acc = (acc << data_width) | (static_cast<uint16_t>(sample) & mask);
        acc_bits += data_width;
        while (acc_bits >= 8U) {
            acc_bits -= 8U;
            output[out_idx++] = static_cast<uint8_t>((acc >> acc_bits) & 0xffU);
            acc = (acc_bits == 0U) ? 0U : (acc & ((1U << acc_bits) - 1U));
        }
    }
}

template <unsigned DATA_WIDTH>
__device__ __forceinline__ uint16_t unpack_prb_sample_msb(const uint8_t* input, unsigned& in_idx, uint32_t& acc, unsigned& acc_bits)
{
    while (acc_bits < DATA_WIDTH) {
        acc = (acc << 8U) | input[in_idx++];
        acc_bits += 8U;
    }
    acc_bits -= DATA_WIDTH;
    uint16_t value = static_cast<uint16_t>((acc >> acc_bits) & ((DATA_WIDTH == 16U) ? 0xffffU : ((1U << DATA_WIDTH) - 1U)));
    acc = (acc_bits == 0U) ? 0U : (acc & ((1U << acc_bits) - 1U));
    return value;
}

__device__ __forceinline__ uint16_t unpack_prb_sample_msb_runtime(const uint8_t* input,
                                                                  unsigned&      in_idx,
                                                                  uint32_t&      acc,
                                                                  unsigned&      acc_bits,
                                                                  unsigned       data_width)
{
    while (acc_bits < data_width) {
        acc = (acc << 8U) | input[in_idx++];
        acc_bits += 8U;
    }
    acc_bits -= data_width;
    uint16_t value = static_cast<uint16_t>((acc >> acc_bits) &
                                           ((data_width == 16U) ? 0xffffU : ((1U << data_width) - 1U)));
    acc = (acc_bits == 0U) ? 0U : (acc & ((1U << acc_bits) - 1U));
    return value;
}

__device__ __forceinline__ int16_t sign_extend(uint16_t value, unsigned width)
{
    unsigned shift = 16U - width;
    return static_cast<int16_t>(static_cast<int16_t>(value << shift) >> shift);
}

__device__ __forceinline__ int16_t quantize_prb_sample_signed(const uint32_t* in_prb, unsigned iq_index, float gain)
{
    uint32_t packed = in_prb[iq_index >> 1U];
    uint16_t value  = (iq_index & 1U) ? static_cast<uint16_t>(packed >> 16U) : static_cast<uint16_t>(packed);
    return quantize_bf16(value, gain);
}

template <unsigned DATA_WIDTH>
__device__ __forceinline__ uint16_t read_packed_prb_sample(const uint8_t* input, unsigned iq_index)
{
    if constexpr (DATA_WIDTH == 8U) {
        return input[iq_index];
    }

    if constexpr (DATA_WIDTH == 9U) {
        const uint8_t* group = input + (iq_index >> 3U) * 9U;
        switch (iq_index & 7U) {
            case 0:
                return static_cast<uint16_t>((static_cast<uint16_t>(group[0]) << 1U) | (group[1] >> 7U));
            case 1:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[1]) & 0x7fU) << 2U) | (group[2] >> 6U));
            case 2:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[2]) & 0x3fU) << 3U) | (group[3] >> 5U));
            case 3:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[3]) & 0x1fU) << 4U) | (group[4] >> 4U));
            case 4:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[4]) & 0x0fU) << 5U) | (group[5] >> 3U));
            case 5:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[5]) & 0x07U) << 6U) | (group[6] >> 2U));
            case 6:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[6]) & 0x03U) << 7U) | (group[7] >> 1U));
            default:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[7]) & 0x01U) << 8U) | group[8]);
        }
    }

    if constexpr (DATA_WIDTH == 10U) {
        const uint8_t* group = input + (iq_index >> 2U) * 5U;
        switch (iq_index & 3U) {
            case 0:
                return static_cast<uint16_t>((static_cast<uint16_t>(group[0]) << 2U) | (group[1] >> 6U));
            case 1:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[1]) & 0x3fU) << 4U) | (group[2] >> 4U));
            case 2:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[2]) & 0x0fU) << 6U) | (group[3] >> 2U));
            default:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[3]) & 0x03U) << 8U) | group[4]);
        }
    }

    if constexpr (DATA_WIDTH == 12U) {
        const uint8_t* group = input + (iq_index >> 1U) * 3U;
        if ((iq_index & 1U) == 0U) {
            return static_cast<uint16_t>((static_cast<uint16_t>(group[0]) << 4U) | (group[1] >> 4U));
        }
        return static_cast<uint16_t>(((static_cast<uint16_t>(group[1]) & 0x0fU) << 8U) | group[2]);
    }

    if constexpr (DATA_WIDTH == 14U) {
        const uint8_t* group = input + (iq_index >> 2U) * 7U;
        switch (iq_index & 3U) {
            case 0:
                return static_cast<uint16_t>((static_cast<uint16_t>(group[0]) << 6U) | (group[1] >> 2U));
            case 1:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[1]) & 0x03U) << 12U) |
                                             (static_cast<uint16_t>(group[2]) << 4U) | (group[3] >> 4U));
            case 2:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[3]) & 0x0fU) << 10U) |
                                             (static_cast<uint16_t>(group[4]) << 2U) | (group[5] >> 6U));
            default:
                return static_cast<uint16_t>(((static_cast<uint16_t>(group[5]) & 0x3fU) << 8U) | group[6]);
        }
    }

    if constexpr (DATA_WIDTH == 16U) {
        const uint8_t* sample = input + iq_index * 2U;
        return static_cast<uint16_t>((static_cast<uint16_t>(sample[0]) << 8U) | sample[1]);
    }
    return 0;
}

template <unsigned DATA_WIDTH>
__device__ __forceinline__ uint16_t quantize_prb_sample_bits(const uint32_t* in_prb, unsigned iq_index, float gain)
{
    uint32_t packed = in_prb[iq_index >> 1U];
    uint16_t value  = (iq_index & 1U) ? static_cast<uint16_t>(packed >> 16U) : static_cast<uint16_t>(packed);
    uint16_t sample = static_cast<uint16_t>(quantize_bf16(value, gain));
    if constexpr (DATA_WIDTH == 16U) {
        return sample;
    }
    return sample & static_cast<uint16_t>((1U << DATA_WIDTH) - 1U);
}

__device__ __forceinline__ void pack_9b_prb_warp(uint8_t* output, uint16_t sample_bits)
{
    const unsigned lane = threadIdx.x & 31U;
    if (lane >= NOF_IQ_SAMPLES_PER_PRB) {
        return;
    }

    const unsigned group      = lane >> 3U;
    const unsigned group_lane = lane & 7U;
    const unsigned group_base = group << 3U;
    const unsigned group_mask = 0xffU << group_base;

    uint16_t s0 = __shfl_sync(group_mask, sample_bits, group_base);
    uint16_t s1 = __shfl_sync(group_mask, sample_bits, group_base + 1U);
    uint16_t s2 = __shfl_sync(group_mask, sample_bits, group_base + 2U);
    uint16_t s3 = __shfl_sync(group_mask, sample_bits, group_base + 3U);
    uint16_t s4 = __shfl_sync(group_mask, sample_bits, group_base + 4U);
    uint16_t s5 = __shfl_sync(group_mask, sample_bits, group_base + 5U);
    uint16_t s6 = __shfl_sync(group_mask, sample_bits, group_base + 6U);
    uint16_t s7 = __shfl_sync(group_mask, sample_bits, group_base + 7U);

    if (group_lane == 0U) {
        uint8_t* group_output = output + group * 9U;
        group_output[0]       = static_cast<uint8_t>(s0 >> 1U);
        group_output[1]       = static_cast<uint8_t>((s0 << 7U) | (s1 >> 2U));
        group_output[2]       = static_cast<uint8_t>((s1 << 6U) | (s2 >> 3U));
        group_output[3]       = static_cast<uint8_t>((s2 << 5U) | (s3 >> 4U));
        group_output[4]       = static_cast<uint8_t>((s3 << 4U) | (s4 >> 5U));
        group_output[5]       = static_cast<uint8_t>((s4 << 3U) | (s5 >> 6U));
        group_output[6]       = static_cast<uint8_t>((s5 << 2U) | (s6 >> 7U));
        group_output[7]       = static_cast<uint8_t>((s6 << 1U) | (s7 >> 8U));
        group_output[8]       = static_cast<uint8_t>(s7);
    }
}

template <bool BFP>
__device__ __forceinline__ void compress_9b_prb_warp(const uint32_t* in_prb, uint8_t* out_prb, float iq_scaling)
{
    const unsigned lane = threadIdx.x & 31U;
    constexpr unsigned full_warp_mask = 0xffffffffU;

    const float gain = static_cast<float>((1U << (BFP ? 15U : 8U)) - 1U) * iq_scaling;
    int         sample  = 0;
    int         max_abs = 0;
    if (lane < NOF_IQ_SAMPLES_PER_PRB) {
        sample = quantize_prb_sample_signed(in_prb, lane, gain);
        if constexpr (BFP) {
            max_abs = bfp_abs_for_exponent(sample);
        }
    }

    unsigned exponent = 0;
    if constexpr (BFP) {
#pragma unroll
        for (unsigned offset = 16U; offset != 0U; offset >>= 1U) {
            max_abs = max(max_abs, __shfl_down_sync(full_warp_mask, max_abs, offset));
        }
        if (lane == 0U) {
            exponent = determine_bfp_exponent(static_cast<uint16_t>(static_cast<unsigned>(max_abs)), 9U);
            out_prb[0] = static_cast<uint8_t>(exponent);
        }
        exponent = __shfl_sync(full_warp_mask, exponent, 0);
    }

    uint16_t sample_bits = 0;
    if (lane < NOF_IQ_SAMPLES_PER_PRB) {
        int16_t prepared = static_cast<int16_t>(sample);
        if constexpr (BFP) {
            prepared = static_cast<int16_t>(prepared >> exponent);
        }
        sample_bits = static_cast<uint16_t>(prepared) & 0x01ffU;
    }

    pack_9b_prb_warp(BFP ? out_prb + 1U : out_prb, sample_bits);
}

__device__ __forceinline__ void compress_9b_none_group(const uint32_t* in_prb,
                                                       unsigned        group,
                                                       uint8_t*        output,
                                                       float           iq_scaling)
{
    const float gain = 255.0F * iq_scaling;
    const unsigned iq = group * 8U;
    uint16_t       s0 = quantize_prb_sample_bits<9>(in_prb, iq, gain);
    uint16_t       s1 = quantize_prb_sample_bits<9>(in_prb, iq + 1U, gain);
    uint16_t       s2 = quantize_prb_sample_bits<9>(in_prb, iq + 2U, gain);
    uint16_t       s3 = quantize_prb_sample_bits<9>(in_prb, iq + 3U, gain);
    uint16_t       s4 = quantize_prb_sample_bits<9>(in_prb, iq + 4U, gain);
    uint16_t       s5 = quantize_prb_sample_bits<9>(in_prb, iq + 5U, gain);
    uint16_t       s6 = quantize_prb_sample_bits<9>(in_prb, iq + 6U, gain);
    uint16_t       s7 = quantize_prb_sample_bits<9>(in_prb, iq + 7U, gain);

    output[0] = static_cast<uint8_t>(s0 >> 1U);
    output[1] = static_cast<uint8_t>((s0 << 7U) | (s1 >> 2U));
    output[2] = static_cast<uint8_t>((s1 << 6U) | (s2 >> 3U));
    output[3] = static_cast<uint8_t>((s2 << 5U) | (s3 >> 4U));
    output[4] = static_cast<uint8_t>((s3 << 4U) | (s4 >> 5U));
    output[5] = static_cast<uint8_t>((s4 << 3U) | (s5 >> 6U));
    output[6] = static_cast<uint8_t>((s5 << 2U) | (s6 >> 7U));
    output[7] = static_cast<uint8_t>((s6 << 1U) | (s7 >> 8U));
    output[8] = static_cast<uint8_t>(s7);
}

__device__ __forceinline__ void pack_12b_prb_warp(uint8_t* output, uint16_t sample_bits)
{
    const unsigned lane = threadIdx.x & 31U;
    uint16_t       second = __shfl_down_sync(0xffffffffU, sample_bits, 1U);
    if ((lane >= NOF_IQ_SAMPLES_PER_PRB) || ((lane & 1U) != 0U)) {
        return;
    }

    uint16_t first  = sample_bits;
    uint8_t* out    = output + (lane >> 1U) * 3U;
    out[0]          = static_cast<uint8_t>(first >> 4U);
    out[1]          = static_cast<uint8_t>((first << 4U) | (second >> 8U));
    out[2]          = static_cast<uint8_t>(second);
}

template <bool BFP>
__device__ __forceinline__ void compress_12b_prb_warp(const uint32_t* in_prb, uint8_t* out_prb, float iq_scaling)
{
    const unsigned lane = threadIdx.x & 31U;
    constexpr unsigned full_warp_mask = 0xffffffffU;

    const float gain = static_cast<float>((1U << (BFP ? 15U : 11U)) - 1U) * iq_scaling;
    int         sample  = 0;
    int         max_abs = 0;
    if (lane < NOF_IQ_SAMPLES_PER_PRB) {
        sample = quantize_prb_sample_signed(in_prb, lane, gain);
        if constexpr (BFP) {
            max_abs = bfp_abs_for_exponent(sample);
        }
    }

    unsigned exponent = 0;
    if constexpr (BFP) {
#pragma unroll
        for (unsigned offset = 16U; offset != 0U; offset >>= 1U) {
            max_abs = max(max_abs, __shfl_down_sync(full_warp_mask, max_abs, offset));
        }
        if (lane == 0U) {
            exponent = determine_bfp_exponent(static_cast<uint16_t>(static_cast<unsigned>(max_abs)), 12U);
            out_prb[0] = static_cast<uint8_t>(exponent);
        }
        exponent = __shfl_sync(full_warp_mask, exponent, 0);
    }

    uint16_t sample_bits = 0;
    if (lane < NOF_IQ_SAMPLES_PER_PRB) {
        int16_t prepared = static_cast<int16_t>(sample);
        if constexpr (BFP) {
            prepared = static_cast<int16_t>(prepared >> exponent);
        }
        sample_bits = static_cast<uint16_t>(prepared) & 0x0fffU;
    }

    pack_12b_prb_warp(BFP ? out_prb + 1U : out_prb, sample_bits);
}

template <unsigned DATA_WIDTH>
__device__ __forceinline__ void compress_prb_none_t(const uint32_t* in_prb, uint8_t* out_prb, float iq_scaling)
{
    float gain = static_cast<float>((1U << (DATA_WIDTH - 1U)) - 1U) * iq_scaling;

    if constexpr (DATA_WIDTH == 8U) {
#pragma unroll
        for (unsigned i = 0; i != NOF_IQ_SAMPLES_PER_PRB; ++i) {
            out_prb[i] = static_cast<uint8_t>(quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i, gain));
        }
        return;
    }

    if constexpr (DATA_WIDTH == 9U) {
#pragma unroll
        for (unsigned i = 0, out_idx = 0; i != NOF_IQ_SAMPLES_PER_PRB; i += 8U, out_idx += 9U) {
            uint16_t s0 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i, gain);
            uint16_t s1 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 1U, gain);
            uint16_t s2 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 2U, gain);
            uint16_t s3 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 3U, gain);
            uint16_t s4 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 4U, gain);
            uint16_t s5 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 5U, gain);
            uint16_t s6 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 6U, gain);
            uint16_t s7 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 7U, gain);
            out_prb[out_idx]      = static_cast<uint8_t>(s0 >> 1U);
            out_prb[out_idx + 1U] = static_cast<uint8_t>((s0 << 7U) | (s1 >> 2U));
            out_prb[out_idx + 2U] = static_cast<uint8_t>((s1 << 6U) | (s2 >> 3U));
            out_prb[out_idx + 3U] = static_cast<uint8_t>((s2 << 5U) | (s3 >> 4U));
            out_prb[out_idx + 4U] = static_cast<uint8_t>((s3 << 4U) | (s4 >> 5U));
            out_prb[out_idx + 5U] = static_cast<uint8_t>((s4 << 3U) | (s5 >> 6U));
            out_prb[out_idx + 6U] = static_cast<uint8_t>((s5 << 2U) | (s6 >> 7U));
            out_prb[out_idx + 7U] = static_cast<uint8_t>((s6 << 1U) | (s7 >> 8U));
            out_prb[out_idx + 8U] = static_cast<uint8_t>(s7);
        }
        return;
    }

    if constexpr (DATA_WIDTH == 10U) {
#pragma unroll
        for (unsigned i = 0, out_idx = 0; i != NOF_IQ_SAMPLES_PER_PRB; i += 4U, out_idx += 5U) {
            uint16_t s0 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i, gain);
            uint16_t s1 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 1U, gain);
            uint16_t s2 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 2U, gain);
            uint16_t s3 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 3U, gain);
            out_prb[out_idx]      = static_cast<uint8_t>(s0 >> 2U);
            out_prb[out_idx + 1U] = static_cast<uint8_t>((s0 << 6U) | (s1 >> 4U));
            out_prb[out_idx + 2U] = static_cast<uint8_t>((s1 << 4U) | (s2 >> 6U));
            out_prb[out_idx + 3U] = static_cast<uint8_t>((s2 << 2U) | (s3 >> 8U));
            out_prb[out_idx + 4U] = static_cast<uint8_t>(s3);
        }
        return;
    }

    if constexpr (DATA_WIDTH == 12U) {
#pragma unroll
        for (unsigned i = 0, out_idx = 0; i != NOF_IQ_SAMPLES_PER_PRB; i += 2U, out_idx += 3U) {
            uint16_t first  = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i, gain);
            uint16_t second = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 1U, gain);
            out_prb[out_idx]      = static_cast<uint8_t>(first >> 4U);
            out_prb[out_idx + 1U] = static_cast<uint8_t>((first << 4U) | (second >> 8U));
            out_prb[out_idx + 2U] = static_cast<uint8_t>(second);
        }
        return;
    }

    if constexpr (DATA_WIDTH == 14U) {
#pragma unroll
        for (unsigned i = 0, out_idx = 0; i != NOF_IQ_SAMPLES_PER_PRB; i += 4U, out_idx += 7U) {
            uint16_t s0 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i, gain);
            uint16_t s1 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 1U, gain);
            uint16_t s2 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 2U, gain);
            uint16_t s3 = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i + 3U, gain);
            out_prb[out_idx]      = static_cast<uint8_t>(s0 >> 6U);
            out_prb[out_idx + 1U] = static_cast<uint8_t>((s0 << 2U) | (s1 >> 12U));
            out_prb[out_idx + 2U] = static_cast<uint8_t>(s1 >> 4U);
            out_prb[out_idx + 3U] = static_cast<uint8_t>((s1 << 4U) | (s2 >> 10U));
            out_prb[out_idx + 4U] = static_cast<uint8_t>(s2 >> 2U);
            out_prb[out_idx + 5U] = static_cast<uint8_t>((s2 << 6U) | (s3 >> 8U));
            out_prb[out_idx + 6U] = static_cast<uint8_t>(s3);
        }
        return;
    }

    if constexpr (DATA_WIDTH == 16U) {
#pragma unroll
        for (unsigned i = 0, out_idx = 0; i != NOF_IQ_SAMPLES_PER_PRB; ++i, out_idx += 2U) {
            uint16_t sample      = quantize_prb_sample_bits<DATA_WIDTH>(in_prb, i, gain);
            out_prb[out_idx]     = static_cast<uint8_t>(sample >> 8U);
            out_prb[out_idx + 1U] = static_cast<uint8_t>(sample);
        }
        return;
    }
}

template <unsigned DATA_WIDTH, bool BFP>
__device__ __forceinline__ void compress_prb_t(const uint32_t* in_prb, uint8_t* out_prb, float iq_scaling)
{
    if constexpr (!BFP) {
        compress_prb_none_t<DATA_WIDTH>(in_prb, out_prb, iq_scaling);
        return;
    }
    if constexpr (DATA_WIDTH == 16U) {
        out_prb[0] = 0;
        compress_prb_none_t<DATA_WIDTH>(in_prb, out_prb + 1, iq_scaling);
        return;
    }

    int16_t quantized[NOF_IQ_SAMPLES_PER_PRB];
    float   gain = static_cast<float>((1U << (BFP ? 15U : (DATA_WIDTH - 1U))) - 1U) * iq_scaling;

    int16_t max_value = quantize_bf16(static_cast<uint16_t>(in_prb[0] & 0xffffU), gain);
    int16_t min_value = max_value;
    for (unsigned sc = 0; sc != NOF_SUBCARRIERS_PER_RB; ++sc) {
        uint32_t packed = in_prb[sc];
        int16_t  re     = quantize_bf16(static_cast<uint16_t>(packed & 0xffffU), gain);
        int16_t  im     = quantize_bf16(static_cast<uint16_t>(packed >> 16U), gain);
        quantized[2U * sc]      = re;
        quantized[2U * sc + 1U] = im;
        max_value = max(max_value, max(re, im));
        min_value = min(min_value, min(re, im));
    }

    uint8_t* payload = out_prb;
    unsigned exponent = 0;
    if constexpr (BFP) {
        int max_abs_signed = max(abs(static_cast<int>(max_value)), abs(static_cast<int>(min_value)) - 1);
        unsigned max_abs = static_cast<unsigned>(max_abs_signed);
        exponent = determine_bfp_exponent(static_cast<uint16_t>(max_abs), DATA_WIDTH);
        out_prb[0] = static_cast<uint8_t>(exponent);
        payload = out_prb + 1;
    }

    pack_prb_samples_msb<DATA_WIDTH, BFP>(payload, quantized, exponent);
}

template <unsigned DATA_WIDTH, bool BFP>
__global__ void ofh_compress_kernel_t(const uint32_t* input,
                                      uint8_t*        output,
                                      unsigned        nof_prbs,
                                      unsigned        prb_size,
                                      float           iq_scaling)
{
    unsigned prb = blockIdx.x * blockDim.x + threadIdx.x;
    if (prb >= nof_prbs) {
        return;
    }

    compress_prb_t<DATA_WIDTH, BFP>(
        input + prb * NOF_SUBCARRIERS_PER_RB, output + prb * prb_size, iq_scaling);
}

template <unsigned DATA_WIDTH, bool BFP>
__global__ void ofh_compress_grid_ports_kernel_t(const uint32_t* input_grid,
                                                 uint8_t*        output,
                                                 unsigned        output_port_stride_bytes,
                                                 unsigned        nof_symbols,
                                                 unsigned        nof_subc,
                                                 unsigned        first_port,
                                                 unsigned        nof_ports,
                                                 unsigned        symbol,
                                                 unsigned        start_prb,
                                                 unsigned        nof_prbs,
                                                 unsigned        prb_size,
                                                 float           iq_scaling)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_prbs = nof_ports * nof_prbs;
    if (index >= total_prbs) {
        return;
    }

    unsigned port_index = index / nof_prbs;
    unsigned prb        = index - port_index * nof_prbs;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    uint8_t* out_prb = output + static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;

    compress_prb_t<DATA_WIDTH, BFP>(input_grid + input_re, out_prb, iq_scaling);
}

__device__ __forceinline__ void store_i16_msb(uint8_t* output, int16_t value)
{
    uint16_t sample = static_cast<uint16_t>(value);
    output[0]       = static_cast<uint8_t>(sample >> 8U);
    output[1]       = static_cast<uint8_t>(sample);
}

__global__ void ofh_compress_bfp16_kernel_t(const uint32_t* input,
                                            uint8_t*        output,
                                            unsigned        nof_prbs,
                                            unsigned        prb_size,
                                            float           iq_scaling)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_re = nof_prbs * NOF_SUBCARRIERS_PER_RB;
    if (index >= total_re) {
        return;
    }

    unsigned prb = index / NOF_SUBCARRIERS_PER_RB;
    unsigned sc  = index - prb * NOF_SUBCARRIERS_PER_RB;
    uint8_t* out_prb = output + prb * prb_size;
    if (sc == 0U) {
        out_prb[0] = 0;
    }

    float    gain   = 32767.0F * iq_scaling;
    uint32_t packed = input[index];
    int16_t  re     = quantize_bf16(static_cast<uint16_t>(packed), gain);
    int16_t  im     = quantize_bf16(static_cast<uint16_t>(packed >> 16U), gain);
    uint8_t* out_re = out_prb + 1U + sc * 4U;
    store_i16_msb(out_re, re);
    store_i16_msb(out_re + 2U, im);
}

__global__ void ofh_compress_bfp16_grid_ports_kernel_t(const uint32_t* input_grid,
                                                       uint8_t*        output,
                                                       unsigned        output_port_stride_bytes,
                                                       unsigned        nof_symbols,
                                                       unsigned        nof_subc,
                                                       unsigned        first_port,
                                                       unsigned        nof_ports,
                                                       unsigned        symbol,
                                                       unsigned        start_prb,
                                                       unsigned        nof_prbs,
                                                       unsigned        prb_size,
                                                       float           iq_scaling)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_re = nof_ports * nof_prbs * NOF_SUBCARRIERS_PER_RB;
    if (index >= total_re) {
        return;
    }

    unsigned re_in_prb  = index % NOF_SUBCARRIERS_PER_RB;
    unsigned prb_index  = index / NOF_SUBCARRIERS_PER_RB;
    unsigned port_index = prb_index / nof_prbs;
    unsigned prb        = prb_index - port_index * nof_prbs;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB + re_in_prb;
    uint8_t* out_prb = output + static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;
    if (re_in_prb == 0U) {
        out_prb[0] = 0;
    }

    float    gain   = 32767.0F * iq_scaling;
    uint32_t packed = input_grid[input_re];
    int16_t  re     = quantize_bf16(static_cast<uint16_t>(packed), gain);
    int16_t  im     = quantize_bf16(static_cast<uint16_t>(packed >> 16U), gain);
    uint8_t* out_re = out_prb + 1U + re_in_prb * 4U;
    store_i16_msb(out_re, re);
    store_i16_msb(out_re + 2U, im);
}

template <unsigned DATA_WIDTH, bool BFP>
__global__ void ofh_compress_grid_symbol_batch_kernel_t(const uint32_t* input_grid,
                                                        uint8_t*        output,
                                                        unsigned        output_symbol_stride_bytes,
                                                        unsigned        output_port_stride_bytes,
                                                        unsigned        nof_grid_symbols,
                                                        unsigned        nof_subc,
                                                        unsigned        first_port,
                                                        unsigned        nof_ports,
                                                        unsigned        first_symbol,
                                                        unsigned        nof_symbols,
                                                        unsigned        start_prb,
                                                        unsigned        nof_prbs,
                                                        unsigned        prb_size,
                                                        float           iq_scaling)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_prbs = nof_symbols * nof_ports * nof_prbs;
    if (index >= total_prbs) {
        return;
    }

    unsigned prb        = index % nof_prbs;
    unsigned port_index = (index / nof_prbs) % nof_ports;
    unsigned symbol_idx = index / (nof_prbs * nof_ports);
    unsigned symbol     = first_symbol + symbol_idx;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_grid_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    uint8_t* out_prb = output + static_cast<size_t>(symbol_idx) * output_symbol_stride_bytes +
                       static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;

    compress_prb_t<DATA_WIDTH, BFP>(input_grid + input_re, out_prb, iq_scaling);
}

__global__ void ofh_compress_bfp16_grid_symbol_batch_kernel_t(const uint32_t* input_grid,
                                                              uint8_t*        output,
                                                              unsigned        output_symbol_stride_bytes,
                                                              unsigned        output_port_stride_bytes,
                                                              unsigned        nof_grid_symbols,
                                                              unsigned        nof_subc,
                                                              unsigned        first_port,
                                                              unsigned        nof_ports,
                                                              unsigned        first_symbol,
                                                              unsigned        nof_symbols,
                                                              unsigned        start_prb,
                                                              unsigned        nof_prbs,
                                                              unsigned        prb_size,
                                                              float           iq_scaling)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_re = nof_symbols * nof_ports * nof_prbs * NOF_SUBCARRIERS_PER_RB;
    if (index >= total_re) {
        return;
    }

    unsigned re_in_prb  = index % NOF_SUBCARRIERS_PER_RB;
    unsigned prb_linear = index / NOF_SUBCARRIERS_PER_RB;
    unsigned prb        = prb_linear % nof_prbs;
    unsigned port_index = (prb_linear / nof_prbs) % nof_ports;
    unsigned symbol_idx = prb_linear / (nof_prbs * nof_ports);
    unsigned symbol     = first_symbol + symbol_idx;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_grid_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB + re_in_prb;
    uint8_t* out_prb = output + static_cast<size_t>(symbol_idx) * output_symbol_stride_bytes +
                       static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;
    if (re_in_prb == 0U) {
        out_prb[0] = 0;
    }

    float    gain   = 32767.0F * iq_scaling;
    uint32_t packed = input_grid[input_re];
    int16_t  re     = quantize_bf16(static_cast<uint16_t>(packed), gain);
    int16_t  im     = quantize_bf16(static_cast<uint16_t>(packed >> 16U), gain);
    uint8_t* out_re = out_prb + 1U + re_in_prb * 4U;
    store_i16_msb(out_re, re);
    store_i16_msb(out_re + 2U, im);
}

template <bool BFP>
__global__ void ofh_compress_9b_warp_kernel_t(const uint32_t* input,
                                              uint8_t*        output,
                                              unsigned        nof_prbs,
                                              unsigned        prb_size,
                                              float           iq_scaling)
{
    unsigned prb = blockIdx.x * COMPRESS_9B_WARPS_PER_BLOCK + (threadIdx.x >> 5U);
    if (prb >= nof_prbs) {
        return;
    }

    compress_9b_prb_warp<BFP>(input + prb * NOF_SUBCARRIERS_PER_RB, output + prb * prb_size, iq_scaling);
}

template <bool BFP>
__global__ void ofh_compress_9b_warp_grid_ports_kernel_t(const uint32_t* input_grid,
                                                         uint8_t*        output,
                                                         unsigned        output_port_stride_bytes,
                                                         unsigned        nof_symbols,
                                                         unsigned        nof_subc,
                                                         unsigned        first_port,
                                                         unsigned        nof_ports,
                                                         unsigned        symbol,
                                                         unsigned        start_prb,
                                                         unsigned        nof_prbs,
                                                         unsigned        prb_size,
                                                         float           iq_scaling)
{
    unsigned prb_index = blockIdx.x * COMPRESS_9B_WARPS_PER_BLOCK + (threadIdx.x >> 5U);
    unsigned total_prbs = nof_ports * nof_prbs;
    if (prb_index >= total_prbs) {
        return;
    }

    unsigned port_index = prb_index / nof_prbs;
    unsigned prb        = prb_index - port_index * nof_prbs;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    uint8_t* out_prb = output + static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;

    compress_9b_prb_warp<BFP>(input_grid + input_re, out_prb, iq_scaling);
}

template <bool BFP>
__global__ void ofh_compress_9b_warp_grid_symbol_batch_kernel_t(const uint32_t* input_grid,
                                                                uint8_t*        output,
                                                                unsigned        output_symbol_stride_bytes,
                                                                unsigned        output_port_stride_bytes,
                                                                unsigned        nof_grid_symbols,
                                                                unsigned        nof_subc,
                                                                unsigned        first_port,
                                                                unsigned        nof_ports,
                                                                unsigned        first_symbol,
                                                                unsigned        nof_symbols,
                                                                unsigned        start_prb,
                                                                unsigned        nof_prbs,
                                                                unsigned        prb_size,
                                                                float           iq_scaling)
{
    unsigned prb_index = blockIdx.x * COMPRESS_9B_WARPS_PER_BLOCK + (threadIdx.x >> 5U);
    unsigned total_prbs = nof_symbols * nof_ports * nof_prbs;
    if (prb_index >= total_prbs) {
        return;
    }

    unsigned prb        = prb_index % nof_prbs;
    unsigned port_index = (prb_index / nof_prbs) % nof_ports;
    unsigned symbol_idx = prb_index / (nof_prbs * nof_ports);
    unsigned symbol     = first_symbol + symbol_idx;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_grid_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    uint8_t* out_prb = output + static_cast<size_t>(symbol_idx) * output_symbol_stride_bytes +
                       static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;

    compress_9b_prb_warp<BFP>(input_grid + input_re, out_prb, iq_scaling);
}

__global__ void ofh_compress_9b_none_group_kernel_t(const uint32_t* input,
                                                    uint8_t*        output,
                                                    unsigned        nof_prbs,
                                                    unsigned        prb_size,
                                                    float           iq_scaling)
{
    unsigned group_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_groups = nof_prbs * 3U;
    if (group_index >= total_groups) {
        return;
    }

    unsigned prb   = group_index / 3U;
    unsigned group = group_index - prb * 3U;
    compress_9b_none_group(
        input + prb * NOF_SUBCARRIERS_PER_RB, group, output + prb * prb_size + group * 9U, iq_scaling);
}

__global__ void ofh_compress_9b_none_group_grid_ports_kernel_t(const uint32_t* input_grid,
                                                               uint8_t*        output,
                                                               unsigned        output_port_stride_bytes,
                                                               unsigned        nof_symbols,
                                                               unsigned        nof_subc,
                                                               unsigned        first_port,
                                                               unsigned        nof_ports,
                                                               unsigned        symbol,
                                                               unsigned        start_prb,
                                                               unsigned        nof_prbs,
                                                               unsigned        prb_size,
                                                               float           iq_scaling)
{
    unsigned group_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_groups = nof_ports * nof_prbs * 3U;
    if (group_index >= total_groups) {
        return;
    }

    unsigned prb_group  = group_index / 3U;
    unsigned group      = group_index - prb_group * 3U;
    unsigned port_index = prb_group / nof_prbs;
    unsigned prb        = prb_group - port_index * nof_prbs;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    uint8_t* out_group = output + static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size +
                         group * 9U;

    compress_9b_none_group(input_grid + input_re, group, out_group, iq_scaling);
}

__global__ void ofh_compress_9b_none_group_grid_symbol_batch_kernel_t(const uint32_t* input_grid,
                                                                      uint8_t*        output,
                                                                      unsigned        output_symbol_stride_bytes,
                                                                      unsigned        output_port_stride_bytes,
                                                                      unsigned        nof_grid_symbols,
                                                                      unsigned        nof_subc,
                                                                      unsigned        first_port,
                                                                      unsigned        nof_ports,
                                                                      unsigned        first_symbol,
                                                                      unsigned        nof_symbols,
                                                                      unsigned        start_prb,
                                                                      unsigned        nof_prbs,
                                                                      unsigned        prb_size,
                                                                      float           iq_scaling)
{
    unsigned group_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_groups = nof_symbols * nof_ports * nof_prbs * 3U;
    if (group_index >= total_groups) {
        return;
    }

    unsigned prb_group  = group_index / 3U;
    unsigned group      = group_index - prb_group * 3U;
    unsigned prb        = prb_group % nof_prbs;
    unsigned port_index = (prb_group / nof_prbs) % nof_ports;
    unsigned symbol_idx = prb_group / (nof_prbs * nof_ports);
    unsigned symbol     = first_symbol + symbol_idx;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_grid_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    uint8_t* out_group = output + static_cast<size_t>(symbol_idx) * output_symbol_stride_bytes +
                         static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size + group * 9U;

    compress_9b_none_group(input_grid + input_re, group, out_group, iq_scaling);
}

template <bool BFP>
__global__ void ofh_compress_12b_warp_kernel_t(const uint32_t* input,
                                               uint8_t*        output,
                                               unsigned        nof_prbs,
                                               unsigned        prb_size,
                                               float           iq_scaling)
{
    unsigned prb = blockIdx.x * COMPRESS_12B_WARPS_PER_BLOCK + (threadIdx.x >> 5U);
    if (prb >= nof_prbs) {
        return;
    }

    compress_12b_prb_warp<BFP>(input + prb * NOF_SUBCARRIERS_PER_RB, output + prb * prb_size, iq_scaling);
}

template <bool BFP>
__global__ void ofh_compress_12b_warp_grid_ports_kernel_t(const uint32_t* input_grid,
                                                          uint8_t*        output,
                                                          unsigned        output_port_stride_bytes,
                                                          unsigned        nof_symbols,
                                                          unsigned        nof_subc,
                                                          unsigned        first_port,
                                                          unsigned        nof_ports,
                                                          unsigned        symbol,
                                                          unsigned        start_prb,
                                                          unsigned        nof_prbs,
                                                          unsigned        prb_size,
                                                          float           iq_scaling)
{
    unsigned prb_index = blockIdx.x * COMPRESS_12B_WARPS_PER_BLOCK + (threadIdx.x >> 5U);
    unsigned total_prbs = nof_ports * nof_prbs;
    if (prb_index >= total_prbs) {
        return;
    }

    unsigned port_index = prb_index / nof_prbs;
    unsigned prb        = prb_index - port_index * nof_prbs;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    uint8_t* out_prb = output + static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;

    compress_12b_prb_warp<BFP>(input_grid + input_re, out_prb, iq_scaling);
}

template <bool BFP>
__global__ void ofh_compress_12b_warp_grid_symbol_batch_kernel_t(const uint32_t* input_grid,
                                                                 uint8_t*        output,
                                                                 unsigned        output_symbol_stride_bytes,
                                                                 unsigned        output_port_stride_bytes,
                                                                 unsigned        nof_grid_symbols,
                                                                 unsigned        nof_subc,
                                                                 unsigned        first_port,
                                                                 unsigned        nof_ports,
                                                                 unsigned        first_symbol,
                                                                 unsigned        nof_symbols,
                                                                 unsigned        start_prb,
                                                                 unsigned        nof_prbs,
                                                                 unsigned        prb_size,
                                                                 float           iq_scaling)
{
    unsigned prb_index = blockIdx.x * COMPRESS_12B_WARPS_PER_BLOCK + (threadIdx.x >> 5U);
    unsigned total_prbs = nof_symbols * nof_ports * nof_prbs;
    if (prb_index >= total_prbs) {
        return;
    }

    unsigned prb        = prb_index % nof_prbs;
    unsigned port_index = (prb_index / nof_prbs) % nof_ports;
    unsigned symbol_idx = prb_index / (nof_prbs * nof_ports);
    unsigned symbol     = first_symbol + symbol_idx;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_grid_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    uint8_t* out_prb = output + static_cast<size_t>(symbol_idx) * output_symbol_stride_bytes +
                       static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;

    compress_12b_prb_warp<BFP>(input_grid + input_re, out_prb, iq_scaling);
}

template <unsigned DATA_WIDTH>
__global__ void ofh_compress_bfp_warp_kernel_t(const uint32_t* input,
                                               uint8_t*        output,
                                               unsigned        nof_prbs,
                                               unsigned        prb_size,
                                               float           iq_scaling)
{
    unsigned prb = blockIdx.x;
    if (prb >= nof_prbs) {
        return;
    }

    const unsigned lane = threadIdx.x;
    __shared__ int16_t  quantized[NOF_IQ_SAMPLES_PER_PRB];
    __shared__ unsigned exponent;

    const uint32_t* in_prb = input + prb * NOF_SUBCARRIERS_PER_RB;
    float           gain   = 32767.0F * iq_scaling;
    int             max_value = -32768;
    int             min_value = 32767;
    if (lane < NOF_IQ_SAMPLES_PER_PRB) {
        int16_t sample = quantize_prb_sample_signed(in_prb, lane, gain);
        quantized[lane] = sample;
        max_value = sample;
        min_value = sample;
    }

    constexpr unsigned full_warp_mask = 0xffffffffU;
#pragma unroll
    for (unsigned offset = 16U; offset != 0U; offset >>= 1U) {
        max_value = max(max_value, __shfl_down_sync(full_warp_mask, max_value, offset));
        min_value = min(min_value, __shfl_down_sync(full_warp_mask, min_value, offset));
    }

    uint8_t* out_prb = output + prb * prb_size;
    if (lane == 0U) {
        int max_abs_signed = max(abs(max_value), abs(min_value) - 1);
        exponent = determine_bfp_exponent(static_cast<uint16_t>(static_cast<unsigned>(max_abs_signed)), DATA_WIDTH);
        out_prb[0] = static_cast<uint8_t>(exponent);
    }
    __syncthreads();

    if (lane == 0U) {
        pack_prb_samples_msb<DATA_WIDTH, true>(out_prb + 1U, quantized, exponent);
    }
}

template <unsigned DATA_WIDTH>
__global__ void ofh_compress_bfp_warp_grid_ports_kernel_t(const uint32_t* input_grid,
                                                          uint8_t*        output,
                                                          unsigned        output_port_stride_bytes,
                                                          unsigned        nof_symbols,
                                                          unsigned        nof_subc,
                                                          unsigned        first_port,
                                                          unsigned        nof_ports,
                                                          unsigned        symbol,
                                                          unsigned        start_prb,
                                                          unsigned        nof_prbs,
                                                          unsigned        prb_size,
                                                          float           iq_scaling)
{
    unsigned prb_index = blockIdx.x;
    unsigned total_prbs = nof_ports * nof_prbs;
    if (prb_index >= total_prbs) {
        return;
    }

    const unsigned lane = threadIdx.x;
    __shared__ int16_t  quantized[NOF_IQ_SAMPLES_PER_PRB];
    __shared__ unsigned exponent;

    unsigned port_index = prb_index / nof_prbs;
    unsigned prb        = prb_index - port_index * nof_prbs;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    const uint32_t* in_prb = input_grid + input_re;

    float gain      = 32767.0F * iq_scaling;
    int   max_value = -32768;
    int   min_value = 32767;
    if (lane < NOF_IQ_SAMPLES_PER_PRB) {
        int16_t sample = quantize_prb_sample_signed(in_prb, lane, gain);
        quantized[lane] = sample;
        max_value = sample;
        min_value = sample;
    }

    constexpr unsigned full_warp_mask = 0xffffffffU;
#pragma unroll
    for (unsigned offset = 16U; offset != 0U; offset >>= 1U) {
        max_value = max(max_value, __shfl_down_sync(full_warp_mask, max_value, offset));
        min_value = min(min_value, __shfl_down_sync(full_warp_mask, min_value, offset));
    }

    uint8_t* out_prb = output + static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;
    if (lane == 0U) {
        int max_abs_signed = max(abs(max_value), abs(min_value) - 1);
        exponent = determine_bfp_exponent(static_cast<uint16_t>(static_cast<unsigned>(max_abs_signed)), DATA_WIDTH);
        out_prb[0] = static_cast<uint8_t>(exponent);
    }
    __syncthreads();

    if (lane == 0U) {
        pack_prb_samples_msb<DATA_WIDTH, true>(out_prb + 1U, quantized, exponent);
    }
}

template <unsigned DATA_WIDTH>
__global__ void ofh_compress_bfp_warp_grid_symbol_batch_kernel_t(const uint32_t* input_grid,
                                                                 uint8_t*        output,
                                                                 unsigned        output_symbol_stride_bytes,
                                                                 unsigned        output_port_stride_bytes,
                                                                 unsigned        nof_grid_symbols,
                                                                 unsigned        nof_subc,
                                                                 unsigned        first_port,
                                                                 unsigned        nof_ports,
                                                                 unsigned        first_symbol,
                                                                 unsigned        nof_symbols,
                                                                 unsigned        start_prb,
                                                                 unsigned        nof_prbs,
                                                                 unsigned        prb_size,
                                                                 float           iq_scaling)
{
    unsigned prb_index = blockIdx.x;
    unsigned total_prbs = nof_symbols * nof_ports * nof_prbs;
    if (prb_index >= total_prbs) {
        return;
    }

    const unsigned lane = threadIdx.x;
    __shared__ int16_t  quantized[NOF_IQ_SAMPLES_PER_PRB];
    __shared__ unsigned exponent;

    unsigned prb        = prb_index % nof_prbs;
    unsigned port_index = (prb_index / nof_prbs) % nof_ports;
    unsigned symbol_idx = prb_index / (nof_prbs * nof_ports);
    unsigned symbol     = first_symbol + symbol_idx;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_grid_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    const uint32_t* in_prb = input_grid + input_re;

    float gain      = 32767.0F * iq_scaling;
    int   max_value = -32768;
    int   min_value = 32767;
    if (lane < NOF_IQ_SAMPLES_PER_PRB) {
        int16_t sample = quantize_prb_sample_signed(in_prb, lane, gain);
        quantized[lane] = sample;
        max_value = sample;
        min_value = sample;
    }

    constexpr unsigned full_warp_mask = 0xffffffffU;
#pragma unroll
    for (unsigned offset = 16U; offset != 0U; offset >>= 1U) {
        max_value = max(max_value, __shfl_down_sync(full_warp_mask, max_value, offset));
        min_value = min(min_value, __shfl_down_sync(full_warp_mask, min_value, offset));
    }

    uint8_t* out_prb = output + static_cast<size_t>(symbol_idx) * output_symbol_stride_bytes +
                       static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;
    if (lane == 0U) {
        int max_abs_signed = max(abs(max_value), abs(min_value) - 1);
        exponent = determine_bfp_exponent(static_cast<uint16_t>(static_cast<unsigned>(max_abs_signed)), DATA_WIDTH);
        out_prb[0] = static_cast<uint8_t>(exponent);
    }
    __syncthreads();

    if (lane == 0U) {
        pack_prb_samples_msb<DATA_WIDTH, true>(out_prb + 1U, quantized, exponent);
    }
}

template <unsigned DATA_WIDTH>
__global__ void ofh_compress_none_warp_kernel_t(const uint32_t* input,
                                                uint8_t*        output,
                                                unsigned        nof_prbs,
                                                unsigned        prb_size,
                                                float           iq_scaling)
{
    unsigned prb = blockIdx.x;
    if (prb >= nof_prbs) {
        return;
    }

    const unsigned lane = threadIdx.x;
    __shared__ int16_t quantized[NOF_IQ_SAMPLES_PER_PRB];

    const uint32_t* in_prb = input + prb * NOF_SUBCARRIERS_PER_RB;
    float gain = static_cast<float>((1U << (DATA_WIDTH - 1U)) - 1U) * iq_scaling;
    if (lane < NOF_IQ_SAMPLES_PER_PRB) {
        quantized[lane] = quantize_prb_sample_signed(in_prb, lane, gain);
    }
    __syncthreads();

    if (lane == 0U) {
        pack_prb_samples_msb<DATA_WIDTH, false>(output + prb * prb_size, quantized, 0);
    }
}

template <unsigned DATA_WIDTH>
__global__ void ofh_compress_none_warp_grid_ports_kernel_t(const uint32_t* input_grid,
                                                           uint8_t*        output,
                                                           unsigned        output_port_stride_bytes,
                                                           unsigned        nof_symbols,
                                                           unsigned        nof_subc,
                                                           unsigned        first_port,
                                                           unsigned        nof_ports,
                                                           unsigned        symbol,
                                                           unsigned        start_prb,
                                                           unsigned        nof_prbs,
                                                           unsigned        prb_size,
                                                           float           iq_scaling)
{
    unsigned prb_index = blockIdx.x;
    unsigned total_prbs = nof_ports * nof_prbs;
    if (prb_index >= total_prbs) {
        return;
    }

    const unsigned lane = threadIdx.x;
    __shared__ int16_t quantized[NOF_IQ_SAMPLES_PER_PRB];

    unsigned port_index = prb_index / nof_prbs;
    unsigned prb        = prb_index - port_index * nof_prbs;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    const uint32_t* in_prb = input_grid + input_re;

    float gain = static_cast<float>((1U << (DATA_WIDTH - 1U)) - 1U) * iq_scaling;
    if (lane < NOF_IQ_SAMPLES_PER_PRB) {
        quantized[lane] = quantize_prb_sample_signed(in_prb, lane, gain);
    }
    __syncthreads();

    if (lane == 0U) {
        uint8_t* out_prb = output + static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;
        pack_prb_samples_msb<DATA_WIDTH, false>(out_prb, quantized, 0);
    }
}

template <unsigned DATA_WIDTH>
__global__ void ofh_compress_none_warp_grid_symbol_batch_kernel_t(const uint32_t* input_grid,
                                                                  uint8_t*        output,
                                                                  unsigned        output_symbol_stride_bytes,
                                                                  unsigned        output_port_stride_bytes,
                                                                  unsigned        nof_grid_symbols,
                                                                  unsigned        nof_subc,
                                                                  unsigned        first_port,
                                                                  unsigned        nof_ports,
                                                                  unsigned        first_symbol,
                                                                  unsigned        nof_symbols,
                                                                  unsigned        start_prb,
                                                                  unsigned        nof_prbs,
                                                                  unsigned        prb_size,
                                                                  float           iq_scaling)
{
    unsigned prb_index = blockIdx.x;
    unsigned total_prbs = nof_symbols * nof_ports * nof_prbs;
    if (prb_index >= total_prbs) {
        return;
    }

    const unsigned lane = threadIdx.x;
    __shared__ int16_t quantized[NOF_IQ_SAMPLES_PER_PRB];

    unsigned prb        = prb_index % nof_prbs;
    unsigned port_index = (prb_index / nof_prbs) % nof_ports;
    unsigned symbol_idx = prb_index / (nof_prbs * nof_ports);
    unsigned symbol     = first_symbol + symbol_idx;
    size_t   input_re   = (static_cast<size_t>(first_port + port_index) * nof_grid_symbols + symbol) * nof_subc +
                        (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    const uint32_t* in_prb = input_grid + input_re;

    float gain = static_cast<float>((1U << (DATA_WIDTH - 1U)) - 1U) * iq_scaling;
    if (lane < NOF_IQ_SAMPLES_PER_PRB) {
        quantized[lane] = quantize_prb_sample_signed(in_prb, lane, gain);
    }
    __syncthreads();

    if (lane == 0U) {
        uint8_t* out_prb = output + static_cast<size_t>(symbol_idx) * output_symbol_stride_bytes +
                           static_cast<size_t>(port_index) * output_port_stride_bytes + prb * prb_size;
        pack_prb_samples_msb<DATA_WIDTH, false>(out_prb, quantized, 0);
    }
}

__global__ void ofh_compress_kernel(const uint32_t* input,
                                    uint8_t*        output,
                                    int             compression_type,
                                    unsigned        nof_prbs,
                                    unsigned        data_width,
                                    unsigned        prb_size,
                                    float           iq_scaling)
{
    unsigned prb = blockIdx.x * blockDim.x + threadIdx.x;
    if (prb >= nof_prbs) {
        return;
    }

    const uint32_t* in_prb  = input + prb * NOF_SUBCARRIERS_PER_RB;
    uint8_t*        out_prb = output + prb * prb_size;

    int16_t quantized[NOF_IQ_SAMPLES_PER_PRB];
    bool    is_bfp = compression_type == OCUDU_OFH_COMPRESSION_TYPE_BFP;
    float   gain = static_cast<float>((1U << (is_bfp ? 15U : (data_width - 1U))) - 1U) * iq_scaling;

    int16_t max_value = quantize_bf16(static_cast<uint16_t>(in_prb[0] & 0xffffU), gain);
    int16_t min_value = max_value;
    for (unsigned sc = 0; sc != NOF_SUBCARRIERS_PER_RB; ++sc) {
        uint32_t packed = in_prb[sc];
        int16_t  re     = quantize_bf16(static_cast<uint16_t>(packed & 0xffffU), gain);
        int16_t  im     = quantize_bf16(static_cast<uint16_t>(packed >> 16U), gain);
        quantized[2U * sc]      = re;
        quantized[2U * sc + 1U] = im;
        max_value = max(max_value, max(re, im));
        min_value = min(min_value, min(re, im));
    }

    uint8_t* payload = out_prb;
    unsigned exponent = 0;
    if (is_bfp) {
        int max_abs_signed = max(abs(static_cast<int>(max_value)), abs(static_cast<int>(min_value)) - 1);
        unsigned max_abs = static_cast<unsigned>(max_abs_signed);
        exponent = determine_bfp_exponent(static_cast<uint16_t>(max_abs), data_width);
        out_prb[0] = static_cast<uint8_t>(exponent);
        payload = out_prb + 1;
    }

    if (is_bfp) {
        pack_prb_samples_msb_runtime<true>(payload, quantized, exponent, data_width);
    } else {
        pack_prb_samples_msb_runtime<false>(payload, quantized, exponent, data_width);
    }
}

template <bool BFP>
__device__ __forceinline__ uint32_t decompress_9b_re(const uint8_t* in_prb, unsigned sc)
{
    const uint8_t* payload  = BFP ? in_prb + 1U : in_prb;
    unsigned       exponent = BFP ? static_cast<unsigned>(in_prb[0]) : 0U;
    float          gain     = static_cast<float>((1U << (BFP ? 15U : 8U)) - 1U);
    int            scaler   = 1 << exponent;

    const uint8_t* group = payload + (sc >> 2U) * 9U;
    uint16_t       re_bits;
    uint16_t       im_bits;
    switch (sc & 3U) {
        case 0:
            re_bits = static_cast<uint16_t>((static_cast<uint16_t>(group[0]) << 1U) | (group[1] >> 7U));
            im_bits = static_cast<uint16_t>(((static_cast<uint16_t>(group[1]) & 0x7fU) << 2U) | (group[2] >> 6U));
            break;
        case 1:
            re_bits = static_cast<uint16_t>(((static_cast<uint16_t>(group[2]) & 0x3fU) << 3U) | (group[3] >> 5U));
            im_bits = static_cast<uint16_t>(((static_cast<uint16_t>(group[3]) & 0x1fU) << 4U) | (group[4] >> 4U));
            break;
        case 2:
            re_bits = static_cast<uint16_t>(((static_cast<uint16_t>(group[4]) & 0x0fU) << 5U) | (group[5] >> 3U));
            im_bits = static_cast<uint16_t>(((static_cast<uint16_t>(group[5]) & 0x07U) << 6U) | (group[6] >> 2U));
            break;
        default:
            re_bits = static_cast<uint16_t>(((static_cast<uint16_t>(group[6]) & 0x03U) << 7U) | (group[7] >> 1U));
            im_bits = static_cast<uint16_t>(((static_cast<uint16_t>(group[7]) & 0x01U) << 8U) | group[8]);
            break;
    }

    int16_t re = sign_extend(re_bits, 9U);
    int16_t im = sign_extend(im_bits, 9U);

    float re_float = static_cast<float>(static_cast<int>(re) * scaler) / gain;
    float im_float = static_cast<float>(static_cast<int>(im) * scaler) / gain;
    return static_cast<uint32_t>(float_to_bf16(re_float)) |
           (static_cast<uint32_t>(float_to_bf16(im_float)) << 16U);
}

template <bool BFP>
__global__ void ofh_decompress_9b_re_kernel_t(uint32_t*      output,
                                              const uint8_t* input,
                                              unsigned       nof_prbs,
                                              unsigned       prb_size)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_re = nof_prbs * NOF_SUBCARRIERS_PER_RB;
    if (index >= total_re) {
        return;
    }

    unsigned prb = index / NOF_SUBCARRIERS_PER_RB;
    unsigned sc  = index - prb * NOF_SUBCARRIERS_PER_RB;
    output[index] = decompress_9b_re<BFP>(input + prb * prb_size, sc);
}

template <bool BFP>
__global__ void ofh_decompress_9b_re_grid_ports_kernel_t(uint32_t*      output_grid,
                                                         const uint8_t* input,
                                                         unsigned       input_port_stride_bytes,
                                                         unsigned       nof_symbols,
                                                         unsigned       nof_subc,
                                                         unsigned       first_port,
                                                         unsigned       nof_ports,
                                                         unsigned       symbol,
                                                         unsigned       start_prb,
                                                         unsigned       nof_prbs,
                                                         unsigned       prb_size)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_re = nof_ports * nof_prbs * NOF_SUBCARRIERS_PER_RB;
    if (index >= total_re) {
        return;
    }

    unsigned sc         = index % NOF_SUBCARRIERS_PER_RB;
    unsigned prb_index  = index / NOF_SUBCARRIERS_PER_RB;
    unsigned port_index = prb_index / nof_prbs;
    unsigned prb        = prb_index - port_index * nof_prbs;
    size_t   output_re  = (static_cast<size_t>(first_port + port_index) * nof_symbols + symbol) * nof_subc +
                         (start_prb + prb) * NOF_SUBCARRIERS_PER_RB + sc;
    const uint8_t* in_prb = input + static_cast<size_t>(port_index) * input_port_stride_bytes + prb * prb_size;

    output_grid[output_re] = decompress_9b_re<BFP>(in_prb, sc);
}

template <bool BFP>
__global__ void ofh_decompress_9b_re_grid_symbol_batch_kernel_t(uint32_t*      output_grid,
                                                                const uint8_t* input,
                                                                unsigned       input_symbol_stride_bytes,
                                                                unsigned       input_port_stride_bytes,
                                                                unsigned       nof_grid_symbols,
                                                                unsigned       nof_subc,
                                                                unsigned       first_port,
                                                                unsigned       nof_ports,
                                                                unsigned       first_symbol,
                                                                unsigned       nof_symbols,
                                                                unsigned       start_prb,
                                                                unsigned       nof_prbs,
                                                                unsigned       prb_size)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_re = nof_symbols * nof_ports * nof_prbs * NOF_SUBCARRIERS_PER_RB;
    if (index >= total_re) {
        return;
    }

    unsigned sc         = index % NOF_SUBCARRIERS_PER_RB;
    unsigned prb_linear = index / NOF_SUBCARRIERS_PER_RB;
    unsigned prb        = prb_linear % nof_prbs;
    unsigned port_index = (prb_linear / nof_prbs) % nof_ports;
    unsigned symbol_idx = prb_linear / (nof_prbs * nof_ports);
    unsigned symbol     = first_symbol + symbol_idx;
    size_t   output_re  = (static_cast<size_t>(first_port + port_index) * nof_grid_symbols + symbol) * nof_subc +
                         (start_prb + prb) * NOF_SUBCARRIERS_PER_RB + sc;
    const uint8_t* in_prb = input + static_cast<size_t>(symbol_idx) * input_symbol_stride_bytes +
                            static_cast<size_t>(port_index) * input_port_stride_bytes + prb * prb_size;

    output_grid[output_re] = decompress_9b_re<BFP>(in_prb, sc);
}

template <bool BFP>
__device__ __forceinline__ uint32_t decompress_12b_re(const uint8_t* in_prb, unsigned sc)
{
    const uint8_t* payload  = BFP ? in_prb + 1U : in_prb;
    unsigned       exponent = BFP ? static_cast<unsigned>(in_prb[0]) : 0U;
    float          gain     = static_cast<float>((1U << (BFP ? 15U : 11U)) - 1U);
    int            scaler   = 1 << exponent;

    const uint8_t* group = payload + sc * 3U;
    int16_t        re = sign_extend(static_cast<uint16_t>((static_cast<uint16_t>(group[0]) << 4U) | (group[1] >> 4U)), 12U);
    int16_t        im = sign_extend(static_cast<uint16_t>(((static_cast<uint16_t>(group[1]) & 0x0fU) << 8U) | group[2]), 12U);

    float re_float = static_cast<float>(static_cast<int>(re) * scaler) / gain;
    float im_float = static_cast<float>(static_cast<int>(im) * scaler) / gain;
    return static_cast<uint32_t>(float_to_bf16(re_float)) |
           (static_cast<uint32_t>(float_to_bf16(im_float)) << 16U);
}

template <bool BFP>
__global__ void ofh_decompress_12b_re_kernel_t(uint32_t*      output,
                                               const uint8_t* input,
                                               unsigned       nof_prbs,
                                               unsigned       prb_size)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_re = nof_prbs * NOF_SUBCARRIERS_PER_RB;
    if (index >= total_re) {
        return;
    }

    unsigned prb = index / NOF_SUBCARRIERS_PER_RB;
    unsigned sc  = index - prb * NOF_SUBCARRIERS_PER_RB;
    output[index] = decompress_12b_re<BFP>(input + prb * prb_size, sc);
}

template <bool BFP>
__global__ void ofh_decompress_12b_re_grid_ports_kernel_t(uint32_t*      output_grid,
                                                          const uint8_t* input,
                                                          unsigned       input_port_stride_bytes,
                                                          unsigned       nof_symbols,
                                                          unsigned       nof_subc,
                                                          unsigned       first_port,
                                                          unsigned       nof_ports,
                                                          unsigned       symbol,
                                                          unsigned       start_prb,
                                                          unsigned       nof_prbs,
                                                          unsigned       prb_size)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_re = nof_ports * nof_prbs * NOF_SUBCARRIERS_PER_RB;
    if (index >= total_re) {
        return;
    }

    unsigned sc         = index % NOF_SUBCARRIERS_PER_RB;
    unsigned prb_index  = index / NOF_SUBCARRIERS_PER_RB;
    unsigned port_index = prb_index / nof_prbs;
    unsigned prb        = prb_index - port_index * nof_prbs;
    size_t   output_re  = (static_cast<size_t>(first_port + port_index) * nof_symbols + symbol) * nof_subc +
                         (start_prb + prb) * NOF_SUBCARRIERS_PER_RB + sc;
    const uint8_t* in_prb = input + static_cast<size_t>(port_index) * input_port_stride_bytes + prb * prb_size;

    output_grid[output_re] = decompress_12b_re<BFP>(in_prb, sc);
}

template <bool BFP>
__global__ void ofh_decompress_12b_re_grid_symbol_batch_kernel_t(uint32_t*      output_grid,
                                                                 const uint8_t* input,
                                                                 unsigned       input_symbol_stride_bytes,
                                                                 unsigned       input_port_stride_bytes,
                                                                 unsigned       nof_grid_symbols,
                                                                 unsigned       nof_subc,
                                                                 unsigned       first_port,
                                                                 unsigned       nof_ports,
                                                                 unsigned       first_symbol,
                                                                 unsigned       nof_symbols,
                                                                 unsigned       start_prb,
                                                                 unsigned       nof_prbs,
                                                                 unsigned       prb_size)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_re = nof_symbols * nof_ports * nof_prbs * NOF_SUBCARRIERS_PER_RB;
    if (index >= total_re) {
        return;
    }

    unsigned sc         = index % NOF_SUBCARRIERS_PER_RB;
    unsigned prb_linear = index / NOF_SUBCARRIERS_PER_RB;
    unsigned prb        = prb_linear % nof_prbs;
    unsigned port_index = (prb_linear / nof_prbs) % nof_ports;
    unsigned symbol_idx = prb_linear / (nof_prbs * nof_ports);
    unsigned symbol     = first_symbol + symbol_idx;
    size_t   output_re  = (static_cast<size_t>(first_port + port_index) * nof_grid_symbols + symbol) * nof_subc +
                         (start_prb + prb) * NOF_SUBCARRIERS_PER_RB + sc;
    const uint8_t* in_prb = input + static_cast<size_t>(symbol_idx) * input_symbol_stride_bytes +
                            static_cast<size_t>(port_index) * input_port_stride_bytes + prb * prb_size;

    output_grid[output_re] = decompress_12b_re<BFP>(in_prb, sc);
}

template <unsigned DATA_WIDTH, bool BFP>
__device__ __forceinline__ void decompress_prb_t(uint32_t* out_prb, const uint8_t* in_prb)
{
    const uint8_t*  payload = in_prb;
    unsigned        exponent = 0;
    if constexpr (BFP) {
        exponent = in_prb[0];
        payload  = in_prb + 1;
    }

    float gain = static_cast<float>((1U << (BFP ? 15U : (DATA_WIDTH - 1U))) - 1U);
    int scaler = 1 << exponent;

    if constexpr (DATA_WIDTH == 16U) {
        for (unsigned sc = 0; sc != NOF_SUBCARRIERS_PER_RB; ++sc) {
            int16_t re = sign_extend(read_packed_prb_sample<DATA_WIDTH>(payload, 2U * sc), DATA_WIDTH);
            int16_t im = sign_extend(read_packed_prb_sample<DATA_WIDTH>(payload, 2U * sc + 1U), DATA_WIDTH);

            float re_float = static_cast<float>(static_cast<int>(re) * scaler) / gain;
            float im_float = static_cast<float>(static_cast<int>(im) * scaler) / gain;
            uint32_t packed = static_cast<uint32_t>(float_to_bf16(re_float)) |
                              (static_cast<uint32_t>(float_to_bf16(im_float)) << 16U);
            out_prb[sc] = packed;
        }
        return;
    }

    uint32_t acc      = 0;
    unsigned acc_bits = 0;
    unsigned in_idx   = 0;
    for (unsigned sc = 0; sc != NOF_SUBCARRIERS_PER_RB; ++sc) {
        int16_t re = sign_extend(unpack_prb_sample_msb<DATA_WIDTH>(payload, in_idx, acc, acc_bits), DATA_WIDTH);
        int16_t im = sign_extend(unpack_prb_sample_msb<DATA_WIDTH>(payload, in_idx, acc, acc_bits), DATA_WIDTH);

        float re_float = static_cast<float>(static_cast<int>(re) * scaler) / gain;
        float im_float = static_cast<float>(static_cast<int>(im) * scaler) / gain;
        uint32_t packed = static_cast<uint32_t>(float_to_bf16(re_float)) |
                          (static_cast<uint32_t>(float_to_bf16(im_float)) << 16U);
        out_prb[sc] = packed;
    }
}

template <unsigned DATA_WIDTH, bool BFP>
__global__ void ofh_decompress_kernel_t(uint32_t*      output,
                                        const uint8_t* input,
                                        unsigned       nof_prbs,
                                        unsigned       prb_size)
{
    unsigned prb = blockIdx.x * blockDim.x + threadIdx.x;
    if (prb >= nof_prbs) {
        return;
    }

    decompress_prb_t<DATA_WIDTH, BFP>(output + prb * NOF_SUBCARRIERS_PER_RB, input + prb * prb_size);
}

template <unsigned DATA_WIDTH, bool BFP>
__global__ void ofh_decompress_grid_ports_kernel_t(uint32_t*      output_grid,
                                                   const uint8_t* input,
                                                   unsigned       input_port_stride_bytes,
                                                   unsigned       nof_symbols,
                                                   unsigned       nof_subc,
                                                   unsigned       first_port,
                                                   unsigned       nof_ports,
                                                   unsigned       symbol,
                                                   unsigned       start_prb,
                                                   unsigned       nof_prbs,
                                                   unsigned       prb_size)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_prbs = nof_ports * nof_prbs;
    if (index >= total_prbs) {
        return;
    }

    unsigned port_index = index / nof_prbs;
    unsigned prb        = index - port_index * nof_prbs;
    size_t   output_re  = (static_cast<size_t>(first_port + port_index) * nof_symbols + symbol) * nof_subc +
                         (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    const uint8_t* in_prb = input + static_cast<size_t>(port_index) * input_port_stride_bytes + prb * prb_size;

    decompress_prb_t<DATA_WIDTH, BFP>(output_grid + output_re, in_prb);
}

template <unsigned DATA_WIDTH, bool BFP>
__global__ void ofh_decompress_grid_symbol_batch_kernel_t(uint32_t*      output_grid,
                                                          const uint8_t* input,
                                                          unsigned       input_symbol_stride_bytes,
                                                          unsigned       input_port_stride_bytes,
                                                          unsigned       nof_grid_symbols,
                                                          unsigned       nof_subc,
                                                          unsigned       first_port,
                                                          unsigned       nof_ports,
                                                          unsigned       first_symbol,
                                                          unsigned       nof_symbols,
                                                          unsigned       start_prb,
                                                          unsigned       nof_prbs,
                                                          unsigned       prb_size)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_prbs = nof_symbols * nof_ports * nof_prbs;
    if (index >= total_prbs) {
        return;
    }

    unsigned prb        = index % nof_prbs;
    unsigned port_index = (index / nof_prbs) % nof_ports;
    unsigned symbol_idx = index / (nof_prbs * nof_ports);
    unsigned symbol     = first_symbol + symbol_idx;
    size_t   output_re  = (static_cast<size_t>(first_port + port_index) * nof_grid_symbols + symbol) * nof_subc +
                         (start_prb + prb) * NOF_SUBCARRIERS_PER_RB;
    const uint8_t* in_prb = input + static_cast<size_t>(symbol_idx) * input_symbol_stride_bytes +
                            static_cast<size_t>(port_index) * input_port_stride_bytes + prb * prb_size;

    decompress_prb_t<DATA_WIDTH, BFP>(output_grid + output_re, in_prb);
}

__global__ void ofh_decompress_kernel(uint32_t*      output,
                                      const uint8_t* input,
                                      int            compression_type,
                                      unsigned       nof_prbs,
                                      unsigned       data_width,
                                      unsigned       prb_size)
{
    unsigned prb = blockIdx.x * blockDim.x + threadIdx.x;
    if (prb >= nof_prbs) {
        return;
    }

    uint32_t*       out_prb = output + prb * NOF_SUBCARRIERS_PER_RB;
    const uint8_t*  in_prb  = input + prb * prb_size;
    const uint8_t*  payload = in_prb;
    unsigned        exponent = 0;
    bool            is_bfp = compression_type == OCUDU_OFH_COMPRESSION_TYPE_BFP;
    if (is_bfp) {
        exponent = in_prb[0];
        payload  = in_prb + 1;
    }

    float gain = static_cast<float>((1U << (is_bfp ? 15U : (data_width - 1U))) - 1U);
    int scaler = 1 << exponent;

    uint32_t acc      = 0;
    unsigned acc_bits = 0;
    unsigned in_idx   = 0;
    for (unsigned sc = 0; sc != NOF_SUBCARRIERS_PER_RB; ++sc) {
        int16_t re = sign_extend(unpack_prb_sample_msb_runtime(payload, in_idx, acc, acc_bits, data_width), data_width);
        int16_t im = sign_extend(unpack_prb_sample_msb_runtime(payload, in_idx, acc, acc_bits, data_width), data_width);

        float re_float = static_cast<float>(static_cast<int>(re) * scaler) / gain;
        float im_float = static_cast<float>(static_cast<int>(im) * scaler) / gain;
        uint32_t packed = static_cast<uint32_t>(float_to_bf16(re_float)) |
                          (static_cast<uint32_t>(float_to_bf16(im_float)) << 16U);
        out_prb[sc] = packed;
    }
}

template <bool BFP>
static void launch_compress_width(const uint32_t* input,
                                  uint8_t*        output,
                                  unsigned        nof_prbs,
                                  unsigned        data_width,
                                  unsigned        prb_size,
                                  float           iq_scaling,
                                  cudaStream_t    stream)
{
    dim3 block(128);
    dim3 grid((nof_prbs + block.x - 1U) / block.x);
    switch (data_width) {
        case 8:
            ofh_compress_kernel_t<8, BFP><<<grid, block, 0, stream>>>(input, output, nof_prbs, prb_size, iq_scaling);
            return;
        case 9:
        {
            if constexpr (BFP) {
                dim3 warp_block(COMPRESS_9B_WARPS_PER_BLOCK * 32U);
                dim3 warp_grid((nof_prbs + COMPRESS_9B_WARPS_PER_BLOCK - 1U) / COMPRESS_9B_WARPS_PER_BLOCK);
                ofh_compress_9b_warp_kernel_t<BFP><<<warp_grid, warp_block, 0, stream>>>(
                    input, output, nof_prbs, prb_size, iq_scaling);
            } else {
                if (use_9b_none_group_compressor(nof_prbs)) {
                    unsigned total_groups = nof_prbs * 3U;
                    dim3     group_block(256);
                    dim3     group_grid((total_groups + group_block.x - 1U) / group_block.x);
                    ofh_compress_9b_none_group_kernel_t<<<group_grid, group_block, 0, stream>>>(
                        input, output, nof_prbs, prb_size, iq_scaling);
                } else {
                    dim3 warp_block(COMPRESS_9B_WARPS_PER_BLOCK * 32U);
                    dim3 warp_grid((nof_prbs + COMPRESS_9B_WARPS_PER_BLOCK - 1U) / COMPRESS_9B_WARPS_PER_BLOCK);
                    ofh_compress_9b_warp_kernel_t<BFP><<<warp_grid, warp_block, 0, stream>>>(
                        input, output, nof_prbs, prb_size, iq_scaling);
                }
            }
            return;
        }
        case 10:
            ofh_compress_kernel_t<10, BFP><<<grid, block, 0, stream>>>(input, output, nof_prbs, prb_size, iq_scaling);
            return;
        case 12:
        {
            if constexpr (BFP) {
                dim3 warp_block(COMPRESS_12B_WARPS_PER_BLOCK * 32U);
                dim3 warp_grid((nof_prbs + COMPRESS_12B_WARPS_PER_BLOCK - 1U) / COMPRESS_12B_WARPS_PER_BLOCK);
                ofh_compress_12b_warp_kernel_t<BFP><<<warp_grid, warp_block, 0, stream>>>(
                    input, output, nof_prbs, prb_size, iq_scaling);
            } else {
                ofh_compress_kernel_t<12, BFP><<<grid, block, 0, stream>>>(
                    input, output, nof_prbs, prb_size, iq_scaling);
            }
            return;
        }
        case 14:
            if constexpr (BFP) {
                ofh_compress_bfp_warp_kernel_t<14><<<nof_prbs, 32, 0, stream>>>(
                    input, output, nof_prbs, prb_size, iq_scaling);
                return;
            } else {
                ofh_compress_none_warp_kernel_t<14><<<nof_prbs, 32, 0, stream>>>(
                    input, output, nof_prbs, prb_size, iq_scaling);
                return;
            }
            ofh_compress_kernel_t<14, BFP><<<grid, block, 0, stream>>>(input, output, nof_prbs, prb_size, iq_scaling);
            return;
        case 16:
            if constexpr (BFP) {
                dim3 bfp16_block(256);
                dim3 bfp16_grid((nof_prbs * NOF_SUBCARRIERS_PER_RB + bfp16_block.x - 1U) / bfp16_block.x);
                ofh_compress_bfp16_kernel_t<<<bfp16_grid, bfp16_block, 0, stream>>>(
                    input, output, nof_prbs, prb_size, iq_scaling);
                return;
            }
            ofh_compress_kernel_t<16, BFP><<<grid, block, 0, stream>>>(input, output, nof_prbs, prb_size, iq_scaling);
            return;
        default:
            ofh_compress_kernel<<<grid, block, 0, stream>>>(
                input, output, BFP ? OCUDU_OFH_COMPRESSION_TYPE_BFP : OCUDU_OFH_COMPRESSION_TYPE_NONE, nof_prbs, data_width, prb_size, iq_scaling);
            return;
    }
}

template <bool BFP>
static void launch_decompress_width(uint32_t*      output,
                                    const uint8_t* input,
                                    unsigned       nof_prbs,
                                    unsigned       data_width,
                                    unsigned       prb_size,
                                    cudaStream_t   stream)
{
    dim3 block(128);
    dim3 grid((nof_prbs + block.x - 1U) / block.x);
    switch (data_width) {
        case 8:
            ofh_decompress_kernel_t<8, BFP><<<grid, block, 0, stream>>>(output, input, nof_prbs, prb_size);
            return;
        case 9:
        {
            if (nof_prbs >= DECOMPRESS_9B_RE_PARALLEL_PRB_THRESHOLD) {
                dim3 re_block(256);
                dim3 re_grid((nof_prbs * NOF_SUBCARRIERS_PER_RB + re_block.x - 1U) / re_block.x);
                ofh_decompress_9b_re_kernel_t<BFP><<<re_grid, re_block, 0, stream>>>(
                    output, input, nof_prbs, prb_size);
            } else {
                ofh_decompress_kernel_t<9, BFP><<<grid, block, 0, stream>>>(output, input, nof_prbs, prb_size);
            }
            return;
        }
        case 10:
            ofh_decompress_kernel_t<10, BFP><<<grid, block, 0, stream>>>(output, input, nof_prbs, prb_size);
            return;
        case 12:
        {
            if (nof_prbs >= DECOMPRESS_12B_RE_PARALLEL_PRB_THRESHOLD) {
                dim3 re_block(256);
                dim3 re_grid((nof_prbs * NOF_SUBCARRIERS_PER_RB + re_block.x - 1U) / re_block.x);
                ofh_decompress_12b_re_kernel_t<BFP><<<re_grid, re_block, 0, stream>>>(
                    output, input, nof_prbs, prb_size);
            } else {
                ofh_decompress_kernel_t<12, BFP><<<grid, block, 0, stream>>>(output, input, nof_prbs, prb_size);
            }
            return;
        }
        case 14:
            ofh_decompress_kernel_t<14, BFP><<<grid, block, 0, stream>>>(output, input, nof_prbs, prb_size);
            return;
        case 16:
            ofh_decompress_kernel_t<16, BFP><<<grid, block, 0, stream>>>(output, input, nof_prbs, prb_size);
            return;
        default:
            ofh_decompress_kernel<<<grid, block, 0, stream>>>(
                output, input, BFP ? OCUDU_OFH_COMPRESSION_TYPE_BFP : OCUDU_OFH_COMPRESSION_TYPE_NONE, nof_prbs, data_width, prb_size);
            return;
    }
}

static void launch_compress(const uint32_t* input,
                            uint8_t*        output,
                            int             compression_type,
                            unsigned        nof_prbs,
                            unsigned        data_width,
                            unsigned        prb_size,
                            float           iq_scaling,
                            cudaStream_t    stream)
{
    if (compression_type == OCUDU_OFH_COMPRESSION_TYPE_BFP) {
        launch_compress_width<true>(input, output, nof_prbs, data_width, prb_size, iq_scaling, stream);
    } else {
        launch_compress_width<false>(input, output, nof_prbs, data_width, prb_size, iq_scaling, stream);
    }
}

static void launch_decompress(uint32_t*      output,
                              const uint8_t* input,
                              int            compression_type,
                              unsigned       nof_prbs,
                              unsigned       data_width,
                              unsigned       prb_size,
                              cudaStream_t   stream)
{
    if (compression_type == OCUDU_OFH_COMPRESSION_TYPE_BFP) {
        launch_decompress_width<true>(output, input, nof_prbs, data_width, prb_size, stream);
    } else {
        launch_decompress_width<false>(output, input, nof_prbs, data_width, prb_size, stream);
    }
}

__global__ void copy_decompressed_re_kernel(uint32_t*       output,
                                            const uint32_t* input,
                                            unsigned        input_start_re,
                                            unsigned        nof_re)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nof_re) {
        return;
    }
    output[idx] = input[input_start_re + idx];
}

static void launch_copy_decompressed_re(uint32_t*       output,
                                        const uint32_t* input,
                                        unsigned        input_start_re,
                                        unsigned        nof_re,
                                        cudaStream_t    stream)
{
    if (nof_re == 0U) {
        return;
    }
    constexpr unsigned threads = 256;
    unsigned           blocks  = (nof_re + threads - 1U) / threads;
    copy_decompressed_re_kernel<<<blocks, threads, 0, stream>>>(output, input, input_start_re, nof_re);
}

template <bool BFP>
static void launch_compress_ports_width(const uint32_t* input_grid,
                                        uint8_t*        output,
                                        unsigned        output_port_stride_bytes,
                                        unsigned        nof_symbols,
                                        unsigned        nof_subc,
                                        unsigned        first_port,
                                        unsigned        nof_ports,
                                        unsigned        symbol,
                                        unsigned        start_prb,
                                        unsigned        nof_prbs,
                                        unsigned        data_width,
                                        unsigned        prb_size,
                                        float           iq_scaling,
                                        cudaStream_t    stream)
{
    unsigned total_prbs = nof_ports * nof_prbs;
    dim3 block(128);
    dim3 grid((total_prbs + block.x - 1U) / block.x);
    switch (data_width) {
        case 8:
            ofh_compress_grid_ports_kernel_t<8, BFP><<<grid, block, 0, stream>>>(
                input_grid, output, output_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                start_prb, nof_prbs, prb_size, iq_scaling);
            return;
        case 9:
        {
            if constexpr (BFP) {
                dim3 warp_block(COMPRESS_9B_WARPS_PER_BLOCK * 32U);
                dim3 warp_grid((total_prbs + COMPRESS_9B_WARPS_PER_BLOCK - 1U) / COMPRESS_9B_WARPS_PER_BLOCK);
                ofh_compress_9b_warp_grid_ports_kernel_t<BFP><<<warp_grid, warp_block, 0, stream>>>(
                    input_grid, output, output_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                    start_prb, nof_prbs, prb_size, iq_scaling);
            } else if (use_9b_none_group_compressor(total_prbs)) {
                unsigned total_groups = total_prbs * 3U;
                dim3     group_block(256);
                dim3     group_grid((total_groups + group_block.x - 1U) / group_block.x);
                ofh_compress_9b_none_group_grid_ports_kernel_t<<<group_grid, group_block, 0, stream>>>(
                    input_grid, output, output_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                    start_prb, nof_prbs, prb_size, iq_scaling);
            } else {
                dim3 warp_block(COMPRESS_9B_WARPS_PER_BLOCK * 32U);
                dim3 warp_grid((total_prbs + COMPRESS_9B_WARPS_PER_BLOCK - 1U) / COMPRESS_9B_WARPS_PER_BLOCK);
                ofh_compress_9b_warp_grid_ports_kernel_t<BFP><<<warp_grid, warp_block, 0, stream>>>(
                    input_grid, output, output_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                    start_prb, nof_prbs, prb_size, iq_scaling);
            }
            return;
        }
        case 10:
            ofh_compress_grid_ports_kernel_t<10, BFP><<<grid, block, 0, stream>>>(
                input_grid, output, output_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                start_prb, nof_prbs, prb_size, iq_scaling);
            return;
        case 12:
        {
            if constexpr (BFP) {
                dim3 warp_block(COMPRESS_12B_WARPS_PER_BLOCK * 32U);
                dim3 warp_grid((total_prbs + COMPRESS_12B_WARPS_PER_BLOCK - 1U) / COMPRESS_12B_WARPS_PER_BLOCK);
                ofh_compress_12b_warp_grid_ports_kernel_t<BFP><<<warp_grid, warp_block, 0, stream>>>(
                    input_grid, output, output_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                    start_prb, nof_prbs, prb_size, iq_scaling);
            } else {
                ofh_compress_grid_ports_kernel_t<12, BFP><<<grid, block, 0, stream>>>(
                    input_grid, output, output_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                    start_prb, nof_prbs, prb_size, iq_scaling);
            }
            return;
        }
        case 14:
            if constexpr (BFP) {
                ofh_compress_bfp_warp_grid_ports_kernel_t<14><<<total_prbs, 32, 0, stream>>>(input_grid,
                                                                                            output,
                                                                                            output_port_stride_bytes,
                                                                                            nof_symbols,
                                                                                            nof_subc,
                                                                                            first_port,
                                                                                            nof_ports,
                                                                                            symbol,
                                                                                            start_prb,
                                                                                            nof_prbs,
                                                                                            prb_size,
                                                                                            iq_scaling);
                return;
            } else {
                ofh_compress_none_warp_grid_ports_kernel_t<14><<<total_prbs, 32, 0, stream>>>(input_grid,
                                                                                              output,
                                                                                              output_port_stride_bytes,
                                                                                              nof_symbols,
                                                                                              nof_subc,
                                                                                              first_port,
                                                                                              nof_ports,
                                                                                              symbol,
                                                                                              start_prb,
                                                                                              nof_prbs,
                                                                                              prb_size,
                                                                                              iq_scaling);
                return;
            }
            ofh_compress_grid_ports_kernel_t<14, BFP><<<grid, block, 0, stream>>>(
                input_grid, output, output_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                start_prb, nof_prbs, prb_size, iq_scaling);
            return;
        case 16:
            if constexpr (BFP) {
                dim3 bfp16_block(256);
                dim3 bfp16_grid((total_prbs * NOF_SUBCARRIERS_PER_RB + bfp16_block.x - 1U) / bfp16_block.x);
                ofh_compress_bfp16_grid_ports_kernel_t<<<bfp16_grid, bfp16_block, 0, stream>>>(input_grid,
                                                                                              output,
                                                                                              output_port_stride_bytes,
                                                                                              nof_symbols,
                                                                                              nof_subc,
                                                                                              first_port,
                                                                                              nof_ports,
                                                                                              symbol,
                                                                                              start_prb,
                                                                                              nof_prbs,
                                                                                              prb_size,
                                                                                              iq_scaling);
                return;
            }
            ofh_compress_grid_ports_kernel_t<16, BFP><<<grid, block, 0, stream>>>(
                input_grid, output, output_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                start_prb, nof_prbs, prb_size, iq_scaling);
            return;
        default:
            return;
    }
}

static void launch_compress_ports(const uint32_t* input_grid,
                                  uint8_t*        output,
                                  unsigned        output_port_stride_bytes,
                                  int             compression_type,
                                  unsigned        nof_symbols,
                                  unsigned        nof_subc,
                                  unsigned        first_port,
                                  unsigned        nof_ports,
                                  unsigned        symbol,
                                  unsigned        start_prb,
                                  unsigned        nof_prbs,
                                  unsigned        data_width,
                                  unsigned        prb_size,
                                  float           iq_scaling,
                                  cudaStream_t    stream)
{
    if (compression_type == OCUDU_OFH_COMPRESSION_TYPE_BFP) {
        launch_compress_ports_width<true>(input_grid,
                                          output,
                                          output_port_stride_bytes,
                                          nof_symbols,
                                          nof_subc,
                                          first_port,
                                          nof_ports,
                                          symbol,
                                          start_prb,
                                          nof_prbs,
                                          data_width,
                                          prb_size,
                                          iq_scaling,
                                          stream);
    } else {
        launch_compress_ports_width<false>(input_grid,
                                           output,
                                           output_port_stride_bytes,
                                           nof_symbols,
                                           nof_subc,
                                           first_port,
                                           nof_ports,
                                           symbol,
                                           start_prb,
                                           nof_prbs,
                                           data_width,
                                           prb_size,
                                           iq_scaling,
                                           stream);
    }
}

template <bool BFP>
static void launch_compress_symbol_batch_width(const uint32_t* input_grid,
                                               uint8_t*        output,
                                               unsigned        output_symbol_stride_bytes,
                                               unsigned        output_port_stride_bytes,
                                               unsigned        nof_grid_symbols,
                                               unsigned        nof_subc,
                                               unsigned        first_port,
                                               unsigned        nof_ports,
                                               unsigned        first_symbol,
                                               unsigned        nof_symbols,
                                               unsigned        start_prb,
                                               unsigned        nof_prbs,
                                               unsigned        data_width,
                                               unsigned        prb_size,
                                               float           iq_scaling,
                                               cudaStream_t    stream)
{
    unsigned total_prbs = nof_symbols * nof_ports * nof_prbs;
    dim3 block(128);
    dim3 grid((total_prbs + block.x - 1U) / block.x);
    switch (data_width) {
        case 8:
            ofh_compress_grid_symbol_batch_kernel_t<8, BFP><<<grid, block, 0, stream>>>(input_grid,
                                                                                        output,
                                                                                        output_symbol_stride_bytes,
                                                                                        output_port_stride_bytes,
                                                                                        nof_grid_symbols,
                                                                                        nof_subc,
                                                                                        first_port,
                                                                                        nof_ports,
                                                                                        first_symbol,
                                                                                        nof_symbols,
                                                                                        start_prb,
                                                                                        nof_prbs,
                                                                                        prb_size,
                                                                                        iq_scaling);
            return;
        case 9:
        {
            if constexpr (BFP) {
                dim3 warp_block(COMPRESS_9B_WARPS_PER_BLOCK * 32U);
                dim3 warp_grid((total_prbs + COMPRESS_9B_WARPS_PER_BLOCK - 1U) / COMPRESS_9B_WARPS_PER_BLOCK);
                ofh_compress_9b_warp_grid_symbol_batch_kernel_t<BFP><<<warp_grid, warp_block, 0, stream>>>(
                    input_grid,
                    output,
                    output_symbol_stride_bytes,
                    output_port_stride_bytes,
                    nof_grid_symbols,
                    nof_subc,
                    first_port,
                    nof_ports,
                    first_symbol,
                    nof_symbols,
                    start_prb,
                    nof_prbs,
                    prb_size,
                    iq_scaling);
            } else if (use_9b_none_group_compressor(total_prbs)) {
                unsigned total_groups = total_prbs * 3U;
                dim3     group_block(256);
                dim3     group_grid((total_groups + group_block.x - 1U) / group_block.x);
                ofh_compress_9b_none_group_grid_symbol_batch_kernel_t<<<group_grid, group_block, 0, stream>>>(
                    input_grid,
                    output,
                    output_symbol_stride_bytes,
                    output_port_stride_bytes,
                    nof_grid_symbols,
                    nof_subc,
                    first_port,
                    nof_ports,
                    first_symbol,
                    nof_symbols,
                    start_prb,
                    nof_prbs,
                    prb_size,
                    iq_scaling);
            } else {
                dim3 warp_block(COMPRESS_9B_WARPS_PER_BLOCK * 32U);
                dim3 warp_grid((total_prbs + COMPRESS_9B_WARPS_PER_BLOCK - 1U) / COMPRESS_9B_WARPS_PER_BLOCK);
                ofh_compress_9b_warp_grid_symbol_batch_kernel_t<BFP><<<warp_grid, warp_block, 0, stream>>>(
                    input_grid,
                    output,
                    output_symbol_stride_bytes,
                    output_port_stride_bytes,
                    nof_grid_symbols,
                    nof_subc,
                    first_port,
                    nof_ports,
                    first_symbol,
                    nof_symbols,
                    start_prb,
                    nof_prbs,
                    prb_size,
                    iq_scaling);
            }
            return;
        }
        case 10:
            ofh_compress_grid_symbol_batch_kernel_t<10, BFP><<<grid, block, 0, stream>>>(input_grid,
                                                                                         output,
                                                                                         output_symbol_stride_bytes,
                                                                                         output_port_stride_bytes,
                                                                                         nof_grid_symbols,
                                                                                         nof_subc,
                                                                                         first_port,
                                                                                         nof_ports,
                                                                                         first_symbol,
                                                                                         nof_symbols,
                                                                                         start_prb,
                                                                                         nof_prbs,
                                                                                         prb_size,
                                                                                         iq_scaling);
            return;
        case 12:
        {
            if constexpr (BFP) {
                dim3 warp_block(COMPRESS_12B_WARPS_PER_BLOCK * 32U);
                dim3 warp_grid((total_prbs + COMPRESS_12B_WARPS_PER_BLOCK - 1U) / COMPRESS_12B_WARPS_PER_BLOCK);
                ofh_compress_12b_warp_grid_symbol_batch_kernel_t<BFP><<<warp_grid, warp_block, 0, stream>>>(
                    input_grid,
                    output,
                    output_symbol_stride_bytes,
                    output_port_stride_bytes,
                    nof_grid_symbols,
                    nof_subc,
                    first_port,
                    nof_ports,
                    first_symbol,
                    nof_symbols,
                    start_prb,
                    nof_prbs,
                    prb_size,
                    iq_scaling);
            } else {
                ofh_compress_grid_symbol_batch_kernel_t<12, BFP><<<grid, block, 0, stream>>>(input_grid,
                                                                                             output,
                                                                                             output_symbol_stride_bytes,
                                                                                             output_port_stride_bytes,
                                                                                             nof_grid_symbols,
                                                                                             nof_subc,
                                                                                             first_port,
                                                                                             nof_ports,
                                                                                             first_symbol,
                                                                                             nof_symbols,
                                                                                             start_prb,
                                                                                             nof_prbs,
                                                                                             prb_size,
                                                                                             iq_scaling);
            }
            return;
        }
        case 14:
            if constexpr (BFP) {
                ofh_compress_bfp_warp_grid_symbol_batch_kernel_t<14><<<total_prbs, 32, 0, stream>>>(input_grid,
                                                                                                    output,
                                                                                                    output_symbol_stride_bytes,
                                                                                                    output_port_stride_bytes,
                                                                                                    nof_grid_symbols,
                                                                                                    nof_subc,
                                                                                                    first_port,
                                                                                                    nof_ports,
                                                                                                    first_symbol,
                                                                                                    nof_symbols,
                                                                                                    start_prb,
                                                                                                    nof_prbs,
                                                                                                    prb_size,
                                                                                                    iq_scaling);
                return;
            } else {
                ofh_compress_none_warp_grid_symbol_batch_kernel_t<14><<<total_prbs, 32, 0, stream>>>(input_grid,
                                                                                                     output,
                                                                                                     output_symbol_stride_bytes,
                                                                                                     output_port_stride_bytes,
                                                                                                     nof_grid_symbols,
                                                                                                     nof_subc,
                                                                                                     first_port,
                                                                                                     nof_ports,
                                                                                                     first_symbol,
                                                                                                     nof_symbols,
                                                                                                     start_prb,
                                                                                                     nof_prbs,
                                                                                                     prb_size,
                                                                                                     iq_scaling);
                return;
            }
            return;
        case 16:
            if constexpr (BFP) {
                dim3 bfp16_block(256);
                dim3 bfp16_grid((total_prbs * NOF_SUBCARRIERS_PER_RB + bfp16_block.x - 1U) / bfp16_block.x);
                ofh_compress_bfp16_grid_symbol_batch_kernel_t<<<bfp16_grid, bfp16_block, 0, stream>>>(input_grid,
                                                                                                      output,
                                                                                                      output_symbol_stride_bytes,
                                                                                                      output_port_stride_bytes,
                                                                                                      nof_grid_symbols,
                                                                                                      nof_subc,
                                                                                                      first_port,
                                                                                                      nof_ports,
                                                                                                      first_symbol,
                                                                                                      nof_symbols,
                                                                                                      start_prb,
                                                                                                      nof_prbs,
                                                                                                      prb_size,
                                                                                                      iq_scaling);
                return;
            }
            ofh_compress_grid_symbol_batch_kernel_t<16, BFP><<<grid, block, 0, stream>>>(input_grid,
                                                                                         output,
                                                                                         output_symbol_stride_bytes,
                                                                                         output_port_stride_bytes,
                                                                                         nof_grid_symbols,
                                                                                         nof_subc,
                                                                                         first_port,
                                                                                         nof_ports,
                                                                                         first_symbol,
                                                                                         nof_symbols,
                                                                                         start_prb,
                                                                                         nof_prbs,
                                                                                         prb_size,
                                                                                         iq_scaling);
            return;
        default:
            return;
    }
}

static void launch_compress_symbol_batch(const uint32_t* input_grid,
                                         uint8_t*        output,
                                         unsigned        output_symbol_stride_bytes,
                                         unsigned        output_port_stride_bytes,
                                         int             compression_type,
                                         unsigned        nof_grid_symbols,
                                         unsigned        nof_subc,
                                         unsigned        first_port,
                                         unsigned        nof_ports,
                                         unsigned        first_symbol,
                                         unsigned        nof_symbols,
                                         unsigned        start_prb,
                                         unsigned        nof_prbs,
                                         unsigned        data_width,
                                         unsigned        prb_size,
                                         float           iq_scaling,
                                         cudaStream_t    stream)
{
    if (compression_type == OCUDU_OFH_COMPRESSION_TYPE_BFP) {
        launch_compress_symbol_batch_width<true>(input_grid,
                                                 output,
                                                 output_symbol_stride_bytes,
                                                 output_port_stride_bytes,
                                                 nof_grid_symbols,
                                                 nof_subc,
                                                 first_port,
                                                 nof_ports,
                                                 first_symbol,
                                                 nof_symbols,
                                                 start_prb,
                                                 nof_prbs,
                                                 data_width,
                                                 prb_size,
                                                 iq_scaling,
                                                 stream);
    } else {
        launch_compress_symbol_batch_width<false>(input_grid,
                                                  output,
                                                  output_symbol_stride_bytes,
                                                  output_port_stride_bytes,
                                                  nof_grid_symbols,
                                                  nof_subc,
                                                  first_port,
                                                  nof_ports,
                                                  first_symbol,
                                                  nof_symbols,
                                                  start_prb,
                                                  nof_prbs,
                                                  data_width,
                                                  prb_size,
                                                  iq_scaling,
                                                  stream);
    }
}

template <bool BFP>
static void launch_decompress_ports_width(uint32_t*      output_grid,
                                          const uint8_t* input,
                                          unsigned       input_port_stride_bytes,
                                          unsigned       nof_symbols,
                                          unsigned       nof_subc,
                                          unsigned       first_port,
                                          unsigned       nof_ports,
                                          unsigned       symbol,
                                          unsigned       start_prb,
                                          unsigned       nof_prbs,
                                          unsigned       data_width,
                                          unsigned       prb_size,
                                          cudaStream_t   stream)
{
    unsigned total_prbs = nof_ports * nof_prbs;
    dim3 block(128);
    dim3 grid((total_prbs + block.x - 1U) / block.x);
    switch (data_width) {
        case 8:
            ofh_decompress_grid_ports_kernel_t<8, BFP><<<grid, block, 0, stream>>>(
                output_grid, input, input_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                start_prb, nof_prbs, prb_size);
            return;
        case 9:
        {
            if (total_prbs >= DECOMPRESS_9B_RE_PARALLEL_PRB_THRESHOLD) {
                dim3 re_block(256);
                dim3 re_grid((total_prbs * NOF_SUBCARRIERS_PER_RB + re_block.x - 1U) / re_block.x);
                ofh_decompress_9b_re_grid_ports_kernel_t<BFP><<<re_grid, re_block, 0, stream>>>(
                    output_grid, input, input_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                    start_prb, nof_prbs, prb_size);
            } else {
                ofh_decompress_grid_ports_kernel_t<9, BFP><<<grid, block, 0, stream>>>(
                    output_grid, input, input_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                    start_prb, nof_prbs, prb_size);
            }
            return;
        }
        case 10:
            ofh_decompress_grid_ports_kernel_t<10, BFP><<<grid, block, 0, stream>>>(
                output_grid, input, input_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                start_prb, nof_prbs, prb_size);
            return;
        case 12:
        {
            if (total_prbs >= DECOMPRESS_12B_RE_PARALLEL_PRB_THRESHOLD) {
                dim3 re_block(256);
                dim3 re_grid((total_prbs * NOF_SUBCARRIERS_PER_RB + re_block.x - 1U) / re_block.x);
                ofh_decompress_12b_re_grid_ports_kernel_t<BFP><<<re_grid, re_block, 0, stream>>>(
                    output_grid, input, input_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                    start_prb, nof_prbs, prb_size);
            } else {
                ofh_decompress_grid_ports_kernel_t<12, BFP><<<grid, block, 0, stream>>>(
                    output_grid, input, input_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                    start_prb, nof_prbs, prb_size);
            }
            return;
        }
        case 14:
            ofh_decompress_grid_ports_kernel_t<14, BFP><<<grid, block, 0, stream>>>(
                output_grid, input, input_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                start_prb, nof_prbs, prb_size);
            return;
        case 16:
            ofh_decompress_grid_ports_kernel_t<16, BFP><<<grid, block, 0, stream>>>(
                output_grid, input, input_port_stride_bytes, nof_symbols, nof_subc, first_port, nof_ports, symbol,
                start_prb, nof_prbs, prb_size);
            return;
        default:
            return;
    }
}

static void launch_decompress_ports(uint32_t*      output_grid,
                                    const uint8_t* input,
                                    unsigned       input_port_stride_bytes,
                                    int            compression_type,
                                    unsigned       nof_symbols,
                                    unsigned       nof_subc,
                                    unsigned       first_port,
                                    unsigned       nof_ports,
                                    unsigned       symbol,
                                    unsigned       start_prb,
                                    unsigned       nof_prbs,
                                    unsigned       data_width,
                                    unsigned       prb_size,
                                    cudaStream_t   stream)
{
    if (compression_type == OCUDU_OFH_COMPRESSION_TYPE_BFP) {
        launch_decompress_ports_width<true>(output_grid,
                                            input,
                                            input_port_stride_bytes,
                                            nof_symbols,
                                            nof_subc,
                                            first_port,
                                            nof_ports,
                                            symbol,
                                            start_prb,
                                            nof_prbs,
                                            data_width,
                                            prb_size,
                                            stream);
    } else {
        launch_decompress_ports_width<false>(output_grid,
                                             input,
                                             input_port_stride_bytes,
                                             nof_symbols,
                                             nof_subc,
                                             first_port,
                                             nof_ports,
                                             symbol,
                                             start_prb,
                                             nof_prbs,
                                             data_width,
                                             prb_size,
                                             stream);
    }
}

template <bool BFP>
static void launch_decompress_symbol_batch_width(uint32_t*      output_grid,
                                                 const uint8_t* input,
                                                 unsigned       input_symbol_stride_bytes,
                                                 unsigned       input_port_stride_bytes,
                                                 unsigned       nof_grid_symbols,
                                                 unsigned       nof_subc,
                                                 unsigned       first_port,
                                                 unsigned       nof_ports,
                                                 unsigned       first_symbol,
                                                 unsigned       nof_symbols,
                                                 unsigned       start_prb,
                                                 unsigned       nof_prbs,
                                                 unsigned       data_width,
                                                 unsigned       prb_size,
                                                 cudaStream_t   stream)
{
    unsigned total_prbs = nof_symbols * nof_ports * nof_prbs;
    dim3 block(128);
    dim3 grid((total_prbs + block.x - 1U) / block.x);
    switch (data_width) {
        case 8:
            ofh_decompress_grid_symbol_batch_kernel_t<8, BFP><<<grid, block, 0, stream>>>(output_grid,
                                                                                          input,
                                                                                          input_symbol_stride_bytes,
                                                                                          input_port_stride_bytes,
                                                                                          nof_grid_symbols,
                                                                                          nof_subc,
                                                                                          first_port,
                                                                                          nof_ports,
                                                                                          first_symbol,
                                                                                          nof_symbols,
                                                                                          start_prb,
                                                                                          nof_prbs,
                                                                                          prb_size);
            return;
        case 9:
        {
            if (total_prbs >= DECOMPRESS_9B_RE_PARALLEL_PRB_THRESHOLD) {
                dim3 re_block(256);
                dim3 re_grid((total_prbs * NOF_SUBCARRIERS_PER_RB + re_block.x - 1U) / re_block.x);
                ofh_decompress_9b_re_grid_symbol_batch_kernel_t<BFP><<<re_grid, re_block, 0, stream>>>(
                    output_grid,
                    input,
                    input_symbol_stride_bytes,
                    input_port_stride_bytes,
                    nof_grid_symbols,
                    nof_subc,
                    first_port,
                    nof_ports,
                    first_symbol,
                    nof_symbols,
                    start_prb,
                    nof_prbs,
                    prb_size);
            } else {
                ofh_decompress_grid_symbol_batch_kernel_t<9, BFP><<<grid, block, 0, stream>>>(output_grid,
                                                                                              input,
                                                                                              input_symbol_stride_bytes,
                                                                                              input_port_stride_bytes,
                                                                                              nof_grid_symbols,
                                                                                              nof_subc,
                                                                                              first_port,
                                                                                              nof_ports,
                                                                                              first_symbol,
                                                                                              nof_symbols,
                                                                                              start_prb,
                                                                                              nof_prbs,
                                                                                              prb_size);
            }
            return;
        }
        case 10:
            ofh_decompress_grid_symbol_batch_kernel_t<10, BFP><<<grid, block, 0, stream>>>(output_grid,
                                                                                           input,
                                                                                           input_symbol_stride_bytes,
                                                                                           input_port_stride_bytes,
                                                                                           nof_grid_symbols,
                                                                                           nof_subc,
                                                                                           first_port,
                                                                                           nof_ports,
                                                                                           first_symbol,
                                                                                           nof_symbols,
                                                                                           start_prb,
                                                                                           nof_prbs,
                                                                                           prb_size);
            return;
        case 12:
        {
            if (total_prbs >= DECOMPRESS_12B_RE_PARALLEL_PRB_THRESHOLD) {
                dim3 re_block(256);
                dim3 re_grid((total_prbs * NOF_SUBCARRIERS_PER_RB + re_block.x - 1U) / re_block.x);
                ofh_decompress_12b_re_grid_symbol_batch_kernel_t<BFP><<<re_grid, re_block, 0, stream>>>(
                    output_grid,
                    input,
                    input_symbol_stride_bytes,
                    input_port_stride_bytes,
                    nof_grid_symbols,
                    nof_subc,
                    first_port,
                    nof_ports,
                    first_symbol,
                    nof_symbols,
                    start_prb,
                    nof_prbs,
                    prb_size);
            } else {
                ofh_decompress_grid_symbol_batch_kernel_t<12, BFP><<<grid, block, 0, stream>>>(output_grid,
                                                                                               input,
                                                                                               input_symbol_stride_bytes,
                                                                                               input_port_stride_bytes,
                                                                                               nof_grid_symbols,
                                                                                               nof_subc,
                                                                                               first_port,
                                                                                               nof_ports,
                                                                                               first_symbol,
                                                                                               nof_symbols,
                                                                                               start_prb,
                                                                                               nof_prbs,
                                                                                               prb_size);
            }
            return;
        }
        case 14:
            ofh_decompress_grid_symbol_batch_kernel_t<14, BFP><<<grid, block, 0, stream>>>(output_grid,
                                                                                           input,
                                                                                           input_symbol_stride_bytes,
                                                                                           input_port_stride_bytes,
                                                                                           nof_grid_symbols,
                                                                                           nof_subc,
                                                                                           first_port,
                                                                                           nof_ports,
                                                                                           first_symbol,
                                                                                           nof_symbols,
                                                                                           start_prb,
                                                                                           nof_prbs,
                                                                                           prb_size);
            return;
        case 16:
            ofh_decompress_grid_symbol_batch_kernel_t<16, BFP><<<grid, block, 0, stream>>>(output_grid,
                                                                                           input,
                                                                                           input_symbol_stride_bytes,
                                                                                           input_port_stride_bytes,
                                                                                           nof_grid_symbols,
                                                                                           nof_subc,
                                                                                           first_port,
                                                                                           nof_ports,
                                                                                           first_symbol,
                                                                                           nof_symbols,
                                                                                           start_prb,
                                                                                           nof_prbs,
                                                                                           prb_size);
            return;
        default:
            return;
    }
}

static void launch_decompress_symbol_batch(uint32_t*      output_grid,
                                           const uint8_t* input,
                                           unsigned       input_symbol_stride_bytes,
                                           unsigned       input_port_stride_bytes,
                                           int            compression_type,
                                           unsigned       nof_grid_symbols,
                                           unsigned       nof_subc,
                                           unsigned       first_port,
                                           unsigned       nof_ports,
                                           unsigned       first_symbol,
                                           unsigned       nof_symbols,
                                           unsigned       start_prb,
                                           unsigned       nof_prbs,
                                           unsigned       data_width,
                                           unsigned       prb_size,
                                           cudaStream_t   stream)
{
    if (compression_type == OCUDU_OFH_COMPRESSION_TYPE_BFP) {
        launch_decompress_symbol_batch_width<true>(output_grid,
                                                   input,
                                                   input_symbol_stride_bytes,
                                                   input_port_stride_bytes,
                                                   nof_grid_symbols,
                                                   nof_subc,
                                                   first_port,
                                                   nof_ports,
                                                   first_symbol,
                                                   nof_symbols,
                                                   start_prb,
                                                   nof_prbs,
                                                   data_width,
                                                   prb_size,
                                                   stream);
    } else {
        launch_decompress_symbol_batch_width<false>(output_grid,
                                                    input,
                                                    input_symbol_stride_bytes,
                                                    input_port_stride_bytes,
                                                    nof_grid_symbols,
                                                    nof_subc,
                                                    first_port,
                                                    nof_ports,
                                                    first_symbol,
                                                    nof_symbols,
                                                    start_prb,
                                                    nof_prbs,
                                                    data_width,
                                                    prb_size,
                                                    stream);
    }
}

} // namespace

extern "C" int ocudu_ofh_compression_available(void)
{
    int device = 0;
    return cudaGetDevice(&device) == cudaSuccess;
}

extern "C" int ocudu_ofh_compression_create(ocudu_ofh_compression_handle_t** out)
{
    if (out == nullptr) {
        return 0;
    }
    *out = nullptr;
    if (!ocudu_ofh_compression_available()) {
        return 0;
    }
    auto* handle = new ocudu_ofh_compression_handle();
    if (cudaStreamCreateWithFlags(&handle->stream, cudaStreamNonBlocking) != cudaSuccess) {
        delete handle;
        return 0;
    }
    *out = handle;
    return 1;
}

extern "C" void ocudu_ofh_compression_destroy(ocudu_ofh_compression_handle_t* handle)
{
    if (handle == nullptr) {
        return;
    }
    if (handle->stream != nullptr) {
        cudaStreamSynchronize(handle->stream);
    }
    if (handle->d_in != nullptr) {
        cudaFree(handle->d_in);
    }
    if (handle->d_out != nullptr) {
        cudaFree(handle->d_out);
    }
    if (handle->h_out != nullptr) {
        cudaFreeHost(handle->h_out);
    }
    for (unsigned slot = 0; slot != ocudu_ofh_compression_handle::HOST_INPUT_STAGING_SLOTS; ++slot) {
        if (handle->h_in_ready_events[slot] != nullptr) {
            cudaEventDestroy(handle->h_in_ready_events[slot]);
        }
        if (handle->h_in[slot] != nullptr) {
            cudaFreeHost(handle->h_in[slot]);
        }
    }
    if (handle->stream != nullptr) {
        cudaStreamDestroy(handle->stream);
    }
    delete handle;
}

extern "C" void* ocudu_ofh_compression_get_stream(ocudu_ofh_compression_handle_t* handle)
{
    return (handle == nullptr) ? nullptr : static_cast<void*>(handle->stream);
}

extern "C" int ocudu_ofh_compression_synchronize(ocudu_ofh_compression_handle_t* handle)
{
    if (handle == nullptr) {
        return 0;
    }
    return cudaStreamSynchronize(handle->stream) == cudaSuccess;
}

extern "C" int ocudu_ofh_compress(ocudu_ofh_compression_handle_t* handle,
                                  int                             compression_type,
                                  void*                           output_bytes,
                                  const void*                     input_cbf16,
                                  unsigned                        nof_prbs,
                                  unsigned                        data_width,
                                  float                           iq_scaling)
{
    if ((handle == nullptr) || (output_bytes == nullptr) || (input_cbf16 == nullptr)) {
        return 0;
    }
    if (!valid_request(compression_type, nof_prbs, data_width)) {
        return nof_prbs == 0U;
    }

    unsigned prb_size = compressed_prb_size(compression_type, data_width);
    size_t in_bytes   = static_cast<size_t>(nof_prbs) * NOF_SUBCARRIERS_PER_RB * sizeof(uint32_t);
    size_t out_bytes  = static_cast<size_t>(nof_prbs) * prb_size;
    if (!ensure_capacity(&handle->d_in, &handle->in_cap, in_bytes) ||
        !ensure_capacity(&handle->d_out, &handle->out_cap, out_bytes)) {
        return 0;
    }

    if (!copy_host_to_device(
            handle, handle->d_in, input_cbf16, in_bytes, use_pinned_host_input_transfer(compression_type, data_width))) {
        return 0;
    }
    launch_compress(reinterpret_cast<const uint32_t*>(handle->d_in),
                    handle->d_out,
                    compression_type,
                    nof_prbs,
                    data_width,
                    prb_size,
                    iq_scaling,
                    handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return copy_device_to_host(
        handle, output_bytes, handle->d_out, out_bytes, use_pinned_host_output_transfer(compression_type, data_width));
}

extern "C" int ocudu_ofh_decompress(ocudu_ofh_compression_handle_t* handle,
                                    int                             compression_type,
                                    void*                           output_cbf16,
                                    const void*                     input_bytes,
                                    unsigned                        nof_prbs,
                                    unsigned                        data_width)
{
    if ((handle == nullptr) || (output_cbf16 == nullptr) || (input_bytes == nullptr)) {
        return 0;
    }
    if (!valid_request(compression_type, nof_prbs, data_width)) {
        return nof_prbs == 0U;
    }

    unsigned prb_size = compressed_prb_size(compression_type, data_width);
    size_t in_bytes   = static_cast<size_t>(nof_prbs) * prb_size;
    size_t out_bytes  = static_cast<size_t>(nof_prbs) * NOF_SUBCARRIERS_PER_RB * sizeof(uint32_t);
    if (!ensure_capacity(&handle->d_in, &handle->in_cap, in_bytes) ||
        !ensure_capacity(&handle->d_out, &handle->out_cap, out_bytes)) {
        return 0;
    }

    if (!copy_host_to_device(
            handle, handle->d_in, input_bytes, in_bytes, use_pinned_host_input_transfer(compression_type, data_width))) {
        return 0;
    }
    launch_decompress(reinterpret_cast<uint32_t*>(handle->d_out),
                      handle->d_in,
                      compression_type,
                      nof_prbs,
                      data_width,
                      prb_size,
                      handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return copy_device_to_host(handle, output_cbf16, handle->d_out, out_bytes, false);
}

extern "C" int ocudu_ofh_compress_device_grid(ocudu_ofh_compression_handle_t* handle,
                                              int                             compression_type,
                                              void*                           output_bytes,
                                              const void*                     input_grid_cbf16,
                                              unsigned                        nof_symbols,
                                              unsigned                        nof_subc,
                                              unsigned                        port,
                                              unsigned                        symbol,
                                              unsigned                        start_prb,
                                              unsigned                        nof_prbs,
                                              unsigned                        data_width,
                                              float                           iq_scaling)
{
    if ((handle == nullptr) || (output_bytes == nullptr) || (input_grid_cbf16 == nullptr)) {
        return 0;
    }
    if (!valid_grid_request(compression_type, nof_symbols, nof_subc, symbol, start_prb, nof_prbs, data_width)) {
        return nof_prbs == 0U;
    }

    unsigned prb_size = compressed_prb_size(compression_type, data_width);
    size_t out_bytes  = static_cast<size_t>(nof_prbs) * prb_size;
    if (!ensure_capacity(&handle->d_out, &handle->out_cap, out_bytes)) {
        return 0;
    }

    size_t grid_offset_re =
        (static_cast<size_t>(port) * nof_symbols + symbol) * nof_subc + start_prb * NOF_SUBCARRIERS_PER_RB;
    const uint32_t* input_prbs = reinterpret_cast<const uint32_t*>(input_grid_cbf16) + grid_offset_re;

    launch_compress(input_prbs,
                    handle->d_out,
                    compression_type,
                    nof_prbs,
                    data_width,
                    prb_size,
                    iq_scaling,
                    handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return copy_device_to_host(
        handle, output_bytes, handle->d_out, out_bytes, use_pinned_host_output_transfer(compression_type, data_width));
}

extern "C" int ocudu_ofh_decompress_to_device_grid(ocudu_ofh_compression_handle_t* handle,
                                                   int                             compression_type,
                                                   void*                           output_grid_cbf16,
                                                   const void*                     input_bytes,
                                                   unsigned                        nof_symbols,
                                                   unsigned                        nof_subc,
                                                   unsigned                        port,
                                                   unsigned                        symbol,
                                                   unsigned                        start_prb,
                                                   unsigned                        nof_prbs,
                                                   unsigned                        data_width)
{
    if (!ocudu_ofh_decompress_to_device_grid_async(handle,
                                                   compression_type,
                                                   output_grid_cbf16,
                                                   input_bytes,
                                                   nof_symbols,
                                                   nof_subc,
                                                   port,
                                                   symbol,
                                                   start_prb,
                                                   nof_prbs,
                                                   data_width)) {
        return 0;
    }
    return ocudu_ofh_compression_synchronize(handle);
}

extern "C" int ocudu_ofh_decompress_to_device_grid_async(ocudu_ofh_compression_handle_t* handle,
                                                         int                             compression_type,
                                                         void*                           output_grid_cbf16,
                                                         const void*                     input_bytes,
                                                         unsigned                        nof_symbols,
                                                         unsigned                        nof_subc,
                                                         unsigned                        port,
                                                         unsigned                        symbol,
                                                         unsigned                        start_prb,
                                                         unsigned                        nof_prbs,
                                                         unsigned                        data_width)
{
    if ((handle == nullptr) || (output_grid_cbf16 == nullptr) || (input_bytes == nullptr)) {
        return 0;
    }
    if (!valid_grid_request(compression_type, nof_symbols, nof_subc, symbol, start_prb, nof_prbs, data_width)) {
        return nof_prbs == 0U;
    }

    unsigned prb_size = compressed_prb_size(compression_type, data_width);
    size_t in_bytes   = static_cast<size_t>(nof_prbs) * prb_size;
    if (!ensure_capacity(&handle->d_in, &handle->in_cap, in_bytes)) {
        return 0;
    }

    if (!copy_host_to_device(
            handle, handle->d_in, input_bytes, in_bytes, use_pinned_host_input_transfer(compression_type, data_width))) {
        return 0;
    }

    size_t grid_offset_re =
        (static_cast<size_t>(port) * nof_symbols + symbol) * nof_subc + start_prb * NOF_SUBCARRIERS_PER_RB;
    uint32_t* output_prbs = reinterpret_cast<uint32_t*>(output_grid_cbf16) + grid_offset_re;

    launch_decompress(output_prbs,
                      handle->d_in,
                      compression_type,
                      nof_prbs,
                      data_width,
                      prb_size,
                      handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return 1;
}

extern "C" int ocudu_ofh_decompress_to_device_prach_buffer_async(ocudu_ofh_compression_handle_t* handle,
                                                                 int                             compression_type,
                                                                 void*                           output_prach_cbf16,
                                                                 unsigned                        output_offset_re,
                                                                 const void*                     input_bytes,
                                                                 unsigned                        input_start_re,
                                                                 unsigned                        nof_re,
                                                                 unsigned                        nof_prbs,
                                                                 unsigned                        data_width)
{
    if ((handle == nullptr) || (output_prach_cbf16 == nullptr) || (input_bytes == nullptr)) {
        return 0;
    }
    if (!valid_request(compression_type, nof_prbs, data_width)) {
        return nof_prbs == 0U;
    }
    if (nof_re == 0U) {
        return 1;
    }
    if (input_start_re + nof_re > nof_prbs * NOF_SUBCARRIERS_PER_RB) {
        return 0;
    }

    unsigned prb_size = compressed_prb_size(compression_type, data_width);
    size_t in_bytes   = static_cast<size_t>(nof_prbs) * prb_size;
    size_t out_bytes  = static_cast<size_t>(nof_prbs) * NOF_SUBCARRIERS_PER_RB * sizeof(uint32_t);
    if (!ensure_capacity(&handle->d_in, &handle->in_cap, in_bytes) ||
        !ensure_capacity(&handle->d_out, &handle->out_cap, out_bytes)) {
        return 0;
    }

    if (!copy_host_to_device(
            handle, handle->d_in, input_bytes, in_bytes, use_pinned_host_input_transfer(compression_type, data_width))) {
        return 0;
    }
    launch_decompress(reinterpret_cast<uint32_t*>(handle->d_out),
                      handle->d_in,
                      compression_type,
                      nof_prbs,
                      data_width,
                      prb_size,
                      handle->stream);
    launch_copy_decompressed_re(reinterpret_cast<uint32_t*>(output_prach_cbf16) + output_offset_re,
                                reinterpret_cast<const uint32_t*>(handle->d_out),
                                input_start_re,
                                nof_re,
                                handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return 1;
}

extern "C" int ocudu_ofh_compress_device_grid_ports(ocudu_ofh_compression_handle_t* handle,
                                                    int                             compression_type,
                                                    void*                           output_bytes,
                                                    unsigned                        output_port_stride_bytes,
                                                    const void*                     input_grid_cbf16,
                                                    unsigned                        nof_symbols,
                                                    unsigned                        nof_subc,
                                                    unsigned                        first_port,
                                                    unsigned                        nof_ports,
                                                    unsigned                        symbol,
                                                    unsigned                        start_prb,
                                                    unsigned                        nof_prbs,
                                                    unsigned                        data_width,
                                                    float                           iq_scaling)
{
    if ((handle == nullptr) || (output_bytes == nullptr) || (input_grid_cbf16 == nullptr) || (nof_ports == 0U)) {
        return 0;
    }
    if (!valid_grid_request(compression_type, nof_symbols, nof_subc, symbol, start_prb, nof_prbs, data_width)) {
        return nof_prbs == 0U;
    }

    unsigned prb_size = compressed_prb_size(compression_type, data_width);
    size_t port_bytes = static_cast<size_t>(nof_prbs) * prb_size;
    if (output_port_stride_bytes < port_bytes) {
        return 0;
    }
    size_t out_bytes = static_cast<size_t>(nof_ports - 1U) * output_port_stride_bytes + port_bytes;
    if (!ensure_capacity(&handle->d_out, &handle->out_cap, out_bytes)) {
        return 0;
    }

    launch_compress_ports(reinterpret_cast<const uint32_t*>(input_grid_cbf16),
                          handle->d_out,
                          output_port_stride_bytes,
                          compression_type,
                          nof_symbols,
                          nof_subc,
                          first_port,
                          nof_ports,
                          symbol,
                          start_prb,
                          nof_prbs,
                          data_width,
                          prb_size,
                          iq_scaling,
                          handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return copy_device_to_host(
        handle, output_bytes, handle->d_out, out_bytes, use_pinned_host_output_transfer(compression_type, data_width));
}

extern "C" int ocudu_ofh_compress_device_grid_ports_to_host_buffers(ocudu_ofh_compression_handle_t* handle,
                                                                    int                             compression_type,
                                                                    void* const*                    output_host_buffers,
                                                                    unsigned nof_output_buffers,
                                                                    unsigned output_buffer_size_bytes,
                                                                    const void* input_grid_cbf16,
                                                                    unsigned    nof_symbols,
                                                                    unsigned    nof_subc,
                                                                    unsigned    first_port,
                                                                    unsigned    nof_ports,
                                                                    unsigned    symbol,
                                                                    unsigned    start_prb,
                                                                    unsigned    nof_prbs,
                                                                    unsigned    data_width,
                                                                    float       iq_scaling)
{
    if ((handle == nullptr) || (output_host_buffers == nullptr) || (input_grid_cbf16 == nullptr) ||
        (nof_ports == 0U) || (nof_output_buffers < nof_ports)) {
        return 0;
    }
    for (unsigned i = 0; i != nof_ports; ++i) {
        if (output_host_buffers[i] == nullptr) {
            return 0;
        }
    }
    if (!valid_grid_request(compression_type, nof_symbols, nof_subc, symbol, start_prb, nof_prbs, data_width)) {
        return nof_prbs == 0U;
    }

    unsigned prb_size   = compressed_prb_size(compression_type, data_width);
    unsigned port_bytes = nof_prbs * prb_size;
    if (output_buffer_size_bytes < port_bytes) {
        return 0;
    }

    size_t out_bytes = static_cast<size_t>(nof_ports) * port_bytes;
    if (!ensure_capacity(&handle->d_out, &handle->out_cap, out_bytes)) {
        return 0;
    }

    launch_compress_ports(reinterpret_cast<const uint32_t*>(input_grid_cbf16),
                          handle->d_out,
                          port_bytes,
                          compression_type,
                          nof_symbols,
                          nof_subc,
                          first_port,
                          nof_ports,
                          symbol,
                          start_prb,
                          nof_prbs,
                          data_width,
                          prb_size,
                          iq_scaling,
                          handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }

    return copy_device_to_host_buffers(handle, output_host_buffers, nof_ports, port_bytes, handle->d_out);
}

extern "C" int ocudu_ofh_compress_device_grid_ports_to_device_async(ocudu_ofh_compression_handle_t* handle,
                                                                    int                             compression_type,
                                                                    void*                           output_device_bytes,
                                                                    unsigned                        output_port_stride_bytes,
                                                                    const void*                     input_grid_cbf16,
                                                                    unsigned                        nof_symbols,
                                                                    unsigned                        nof_subc,
                                                                    unsigned                        first_port,
                                                                    unsigned                        nof_ports,
                                                                    unsigned                        symbol,
                                                                    unsigned                        start_prb,
                                                                    unsigned                        nof_prbs,
                                                                    unsigned                        data_width,
                                                                    float                           iq_scaling)
{
    if ((handle == nullptr) || (output_device_bytes == nullptr) || (input_grid_cbf16 == nullptr) ||
        (nof_ports == 0U)) {
        return 0;
    }
    if (!valid_grid_request(compression_type, nof_symbols, nof_subc, symbol, start_prb, nof_prbs, data_width)) {
        return nof_prbs == 0U;
    }

    unsigned prb_size = compressed_prb_size(compression_type, data_width);
    size_t port_bytes = static_cast<size_t>(nof_prbs) * prb_size;
    if (output_port_stride_bytes < port_bytes) {
        return 0;
    }

    launch_compress_ports(reinterpret_cast<const uint32_t*>(input_grid_cbf16),
                          reinterpret_cast<uint8_t*>(output_device_bytes),
                          output_port_stride_bytes,
                          compression_type,
                          nof_symbols,
                          nof_subc,
                          first_port,
                          nof_ports,
                          symbol,
                          start_prb,
                          nof_prbs,
                          data_width,
                          prb_size,
                          iq_scaling,
                          handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return 1;
}

extern "C" int ocudu_ofh_compress_device_grid_ports_to_device(ocudu_ofh_compression_handle_t* handle,
                                                              int                             compression_type,
                                                              void*                           output_device_bytes,
                                                              unsigned                        output_port_stride_bytes,
                                                              const void*                     input_grid_cbf16,
                                                              unsigned                        nof_symbols,
                                                              unsigned                        nof_subc,
                                                              unsigned                        first_port,
                                                              unsigned                        nof_ports,
                                                              unsigned                        symbol,
                                                              unsigned                        start_prb,
                                                              unsigned                        nof_prbs,
                                                              unsigned                        data_width,
                                                              float                           iq_scaling)
{
    if (!ocudu_ofh_compress_device_grid_ports_to_device_async(handle,
                                                              compression_type,
                                                              output_device_bytes,
                                                              output_port_stride_bytes,
                                                              input_grid_cbf16,
                                                              nof_symbols,
                                                              nof_subc,
                                                              first_port,
                                                              nof_ports,
                                                              symbol,
                                                              start_prb,
                                                              nof_prbs,
                                                              data_width,
                                                              iq_scaling)) {
        return 0;
    }
    return ocudu_ofh_compression_synchronize(handle);
}

extern "C" int ocudu_ofh_compress_device_grid_symbol_batch(ocudu_ofh_compression_handle_t* handle,
                                                           int                             compression_type,
                                                           void*                           output_bytes,
                                                           unsigned                        output_symbol_stride_bytes,
                                                           unsigned                        output_port_stride_bytes,
                                                           const void*                     input_grid_cbf16,
                                                           unsigned                        nof_grid_symbols,
                                                           unsigned                        nof_subc,
                                                           unsigned                        first_port,
                                                           unsigned                        nof_ports,
                                                           unsigned                        first_symbol,
                                                           unsigned                        nof_symbols,
                                                           unsigned                        start_prb,
                                                           unsigned                        nof_prbs,
                                                           unsigned                        data_width,
                                                           float                           iq_scaling)
{
    if ((handle == nullptr) || (output_bytes == nullptr) || (input_grid_cbf16 == nullptr) || (nof_ports == 0U)) {
        return 0;
    }
    if (!valid_grid_symbol_batch_request(
            compression_type, nof_grid_symbols, nof_subc, first_symbol, nof_symbols, start_prb, nof_prbs, data_width)) {
        return (nof_prbs == 0U) || (nof_symbols == 0U);
    }

    unsigned prb_size   = compressed_prb_size(compression_type, data_width);
    size_t   port_bytes = static_cast<size_t>(nof_prbs) * prb_size;
    if ((output_port_stride_bytes < port_bytes) ||
        (output_symbol_stride_bytes < static_cast<size_t>(nof_ports - 1U) * output_port_stride_bytes + port_bytes)) {
        return 0;
    }
    size_t out_bytes = static_cast<size_t>(nof_symbols - 1U) * output_symbol_stride_bytes +
                       static_cast<size_t>(nof_ports - 1U) * output_port_stride_bytes + port_bytes;
    if (!ensure_capacity(&handle->d_out, &handle->out_cap, out_bytes)) {
        return 0;
    }

    launch_compress_symbol_batch(reinterpret_cast<const uint32_t*>(input_grid_cbf16),
                                 handle->d_out,
                                 output_symbol_stride_bytes,
                                 output_port_stride_bytes,
                                 compression_type,
                                 nof_grid_symbols,
                                 nof_subc,
                                 first_port,
                                 nof_ports,
                                 first_symbol,
                                 nof_symbols,
                                 start_prb,
                                 nof_prbs,
                                 data_width,
                                 prb_size,
                                 iq_scaling,
                                 handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return copy_device_to_host(
        handle, output_bytes, handle->d_out, out_bytes, use_pinned_host_output_transfer(compression_type, data_width));
}

extern "C" int ocudu_ofh_compress_device_grid_symbol_batch_to_host_buffers(
    ocudu_ofh_compression_handle_t* handle,
    int                             compression_type,
    void* const*                    output_host_buffers,
    unsigned                        nof_output_buffers,
    unsigned                        output_buffer_size_bytes,
    const void*                     input_grid_cbf16,
    unsigned                        nof_grid_symbols,
    unsigned                        nof_subc,
    unsigned                        first_port,
    unsigned                        nof_ports,
    unsigned                        first_symbol,
    unsigned                        nof_symbols,
    unsigned                        start_prb,
    unsigned                        nof_prbs,
    unsigned                        data_width,
    float                           iq_scaling)
{
    if ((handle == nullptr) || (output_host_buffers == nullptr) || (input_grid_cbf16 == nullptr) ||
        (nof_ports == 0U)) {
        return 0;
    }
    if (!valid_grid_symbol_batch_request(
            compression_type, nof_grid_symbols, nof_subc, first_symbol, nof_symbols, start_prb, nof_prbs, data_width)) {
        return (nof_prbs == 0U) || (nof_symbols == 0U);
    }

    unsigned nof_buffers = nof_symbols * nof_ports;
    if (nof_output_buffers < nof_buffers) {
        return 0;
    }
    for (unsigned i = 0; i != nof_buffers; ++i) {
        if (output_host_buffers[i] == nullptr) {
            return 0;
        }
    }

    unsigned prb_size      = compressed_prb_size(compression_type, data_width);
    unsigned port_bytes    = nof_prbs * prb_size;
    unsigned port_stride   = port_bytes;
    unsigned symbol_stride = nof_ports * port_stride;
    if (output_buffer_size_bytes < port_bytes) {
        return 0;
    }

    size_t out_bytes = static_cast<size_t>(nof_symbols) * symbol_stride;
    if (!ensure_capacity(&handle->d_out, &handle->out_cap, out_bytes)) {
        return 0;
    }

    launch_compress_symbol_batch(reinterpret_cast<const uint32_t*>(input_grid_cbf16),
                                 handle->d_out,
                                 symbol_stride,
                                 port_stride,
                                 compression_type,
                                 nof_grid_symbols,
                                 nof_subc,
                                 first_port,
                                 nof_ports,
                                 first_symbol,
                                 nof_symbols,
                                 start_prb,
                                 nof_prbs,
                                 data_width,
                                 prb_size,
                                 iq_scaling,
                                 handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }

    return copy_device_to_host_buffers(handle, output_host_buffers, nof_buffers, port_bytes, handle->d_out);
}

extern "C" int ocudu_ofh_compress_device_grid_symbol_batch_to_device_async(ocudu_ofh_compression_handle_t* handle,
                                                                          int                             compression_type,
                                                                          void*                           output_device_bytes,
                                                                          unsigned output_symbol_stride_bytes,
                                                                          unsigned output_port_stride_bytes,
                                                                          const void* input_grid_cbf16,
                                                                          unsigned    nof_grid_symbols,
                                                                          unsigned    nof_subc,
                                                                          unsigned    first_port,
                                                                          unsigned    nof_ports,
                                                                          unsigned    first_symbol,
                                                                          unsigned    nof_symbols,
                                                                          unsigned    start_prb,
                                                                          unsigned    nof_prbs,
                                                                          unsigned    data_width,
                                                                          float       iq_scaling)
{
    if ((handle == nullptr) || (output_device_bytes == nullptr) || (input_grid_cbf16 == nullptr) ||
        (nof_ports == 0U)) {
        return 0;
    }
    if (!valid_grid_symbol_batch_request(
            compression_type, nof_grid_symbols, nof_subc, first_symbol, nof_symbols, start_prb, nof_prbs, data_width)) {
        return (nof_prbs == 0U) || (nof_symbols == 0U);
    }

    unsigned prb_size   = compressed_prb_size(compression_type, data_width);
    size_t   port_bytes = static_cast<size_t>(nof_prbs) * prb_size;
    if ((output_port_stride_bytes < port_bytes) ||
        (output_symbol_stride_bytes < static_cast<size_t>(nof_ports - 1U) * output_port_stride_bytes + port_bytes)) {
        return 0;
    }

    launch_compress_symbol_batch(reinterpret_cast<const uint32_t*>(input_grid_cbf16),
                                 reinterpret_cast<uint8_t*>(output_device_bytes),
                                 output_symbol_stride_bytes,
                                 output_port_stride_bytes,
                                 compression_type,
                                 nof_grid_symbols,
                                 nof_subc,
                                 first_port,
                                 nof_ports,
                                 first_symbol,
                                 nof_symbols,
                                 start_prb,
                                 nof_prbs,
                                 data_width,
                                 prb_size,
                                 iq_scaling,
                                 handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return 1;
}

extern "C" int ocudu_ofh_compress_device_grid_symbol_batch_to_device(ocudu_ofh_compression_handle_t* handle,
                                                                     int                             compression_type,
                                                                     void*                           output_device_bytes,
                                                                     unsigned output_symbol_stride_bytes,
                                                                     unsigned output_port_stride_bytes,
                                                                     const void* input_grid_cbf16,
                                                                     unsigned    nof_grid_symbols,
                                                                     unsigned    nof_subc,
                                                                     unsigned    first_port,
                                                                     unsigned    nof_ports,
                                                                     unsigned    first_symbol,
                                                                     unsigned    nof_symbols,
                                                                     unsigned    start_prb,
                                                                     unsigned    nof_prbs,
                                                                     unsigned    data_width,
                                                                     float       iq_scaling)
{
    if (!ocudu_ofh_compress_device_grid_symbol_batch_to_device_async(handle,
                                                                     compression_type,
                                                                     output_device_bytes,
                                                                     output_symbol_stride_bytes,
                                                                     output_port_stride_bytes,
                                                                     input_grid_cbf16,
                                                                     nof_grid_symbols,
                                                                     nof_subc,
                                                                     first_port,
                                                                     nof_ports,
                                                                     first_symbol,
                                                                     nof_symbols,
                                                                     start_prb,
                                                                     nof_prbs,
                                                                     data_width,
                                                                     iq_scaling)) {
        return 0;
    }
    return ocudu_ofh_compression_synchronize(handle);
}

extern "C" int ocudu_ofh_decompress_to_device_grid_ports(ocudu_ofh_compression_handle_t* handle,
                                                         int                             compression_type,
                                                         void*                           output_grid_cbf16,
                                                         const void*                     input_bytes,
                                                         unsigned                        input_port_stride_bytes,
                                                         unsigned                        nof_symbols,
                                                         unsigned                        nof_subc,
                                                         unsigned                        first_port,
                                                         unsigned                        nof_ports,
                                                         unsigned                        symbol,
                                                         unsigned                        start_prb,
                                                         unsigned                        nof_prbs,
                                                         unsigned                        data_width)
{
    if ((handle == nullptr) || (output_grid_cbf16 == nullptr) || (input_bytes == nullptr) || (nof_ports == 0U)) {
        return 0;
    }
    if (!valid_grid_request(compression_type, nof_symbols, nof_subc, symbol, start_prb, nof_prbs, data_width)) {
        return nof_prbs == 0U;
    }

    unsigned prb_size = compressed_prb_size(compression_type, data_width);
    size_t port_bytes = static_cast<size_t>(nof_prbs) * prb_size;
    if (input_port_stride_bytes < port_bytes) {
        return 0;
    }
    size_t in_bytes = static_cast<size_t>(nof_ports - 1U) * input_port_stride_bytes + port_bytes;
    if (!ensure_capacity(&handle->d_in, &handle->in_cap, in_bytes)) {
        return 0;
    }

    if (!copy_host_to_device(
            handle, handle->d_in, input_bytes, in_bytes, use_pinned_host_input_transfer(compression_type, data_width))) {
        return 0;
    }
    launch_decompress_ports(reinterpret_cast<uint32_t*>(output_grid_cbf16),
                            handle->d_in,
                            input_port_stride_bytes,
                            compression_type,
                            nof_symbols,
                            nof_subc,
                            first_port,
                            nof_ports,
                            symbol,
                            start_prb,
                            nof_prbs,
                            data_width,
                            prb_size,
                            handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return cudaStreamSynchronize(handle->stream) == cudaSuccess;
}

extern "C" int ocudu_ofh_decompress_device_bytes_to_device_grid_ports(ocudu_ofh_compression_handle_t* handle,
                                                                      int                             compression_type,
                                                                      void*                           output_grid_cbf16,
                                                                      const void*                     input_device_bytes,
                                                                      unsigned                        input_port_stride_bytes,
                                                                      unsigned                        nof_symbols,
                                                                      unsigned                        nof_subc,
                                                                      unsigned                        first_port,
                                                                      unsigned                        nof_ports,
                                                                      unsigned                        symbol,
                                                                      unsigned                        start_prb,
                                                                      unsigned                        nof_prbs,
                                                                      unsigned                        data_width)
{
    if ((handle == nullptr) || (output_grid_cbf16 == nullptr) || (input_device_bytes == nullptr) ||
        (nof_ports == 0U)) {
        return 0;
    }
    if (!valid_grid_request(compression_type, nof_symbols, nof_subc, symbol, start_prb, nof_prbs, data_width)) {
        return nof_prbs == 0U;
    }

    unsigned prb_size = compressed_prb_size(compression_type, data_width);
    size_t port_bytes = static_cast<size_t>(nof_prbs) * prb_size;
    if (input_port_stride_bytes < port_bytes) {
        return 0;
    }

    launch_decompress_ports(reinterpret_cast<uint32_t*>(output_grid_cbf16),
                            reinterpret_cast<const uint8_t*>(input_device_bytes),
                            input_port_stride_bytes,
                            compression_type,
                            nof_symbols,
                            nof_subc,
                            first_port,
                            nof_ports,
                            symbol,
                            start_prb,
                            nof_prbs,
                            data_width,
                            prb_size,
                            handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return cudaStreamSynchronize(handle->stream) == cudaSuccess;
}

extern "C" int ocudu_ofh_decompress_to_device_grid_symbol_batch(ocudu_ofh_compression_handle_t* handle,
                                                                int                             compression_type,
                                                                void*                           output_grid_cbf16,
                                                                const void*                     input_bytes,
                                                                unsigned                        input_symbol_stride_bytes,
                                                                unsigned                        input_port_stride_bytes,
                                                                unsigned                        nof_grid_symbols,
                                                                unsigned                        nof_subc,
                                                                unsigned                        first_port,
                                                                unsigned                        nof_ports,
                                                                unsigned                        first_symbol,
                                                                unsigned                        nof_symbols,
                                                                unsigned                        start_prb,
                                                                unsigned                        nof_prbs,
                                                                unsigned                        data_width)
{
    if ((handle == nullptr) || (output_grid_cbf16 == nullptr) || (input_bytes == nullptr) || (nof_ports == 0U)) {
        return 0;
    }
    if (!valid_grid_symbol_batch_request(
            compression_type, nof_grid_symbols, nof_subc, first_symbol, nof_symbols, start_prb, nof_prbs, data_width)) {
        return (nof_prbs == 0U) || (nof_symbols == 0U);
    }

    unsigned prb_size   = compressed_prb_size(compression_type, data_width);
    size_t   port_bytes = static_cast<size_t>(nof_prbs) * prb_size;
    if ((input_port_stride_bytes < port_bytes) ||
        (input_symbol_stride_bytes < static_cast<size_t>(nof_ports - 1U) * input_port_stride_bytes + port_bytes)) {
        return 0;
    }
    size_t in_bytes = static_cast<size_t>(nof_symbols - 1U) * input_symbol_stride_bytes +
                      static_cast<size_t>(nof_ports - 1U) * input_port_stride_bytes + port_bytes;
    if (!ensure_capacity(&handle->d_in, &handle->in_cap, in_bytes)) {
        return 0;
    }

    if (!copy_host_to_device(
            handle, handle->d_in, input_bytes, in_bytes, use_pinned_host_input_transfer(compression_type, data_width))) {
        return 0;
    }
    launch_decompress_symbol_batch(reinterpret_cast<uint32_t*>(output_grid_cbf16),
                                   handle->d_in,
                                   input_symbol_stride_bytes,
                                   input_port_stride_bytes,
                                   compression_type,
                                   nof_grid_symbols,
                                   nof_subc,
                                   first_port,
                                   nof_ports,
                                   first_symbol,
                                   nof_symbols,
                                   start_prb,
                                   nof_prbs,
                                   data_width,
                                   prb_size,
                                   handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return cudaStreamSynchronize(handle->stream) == cudaSuccess;
}

extern "C" int ocudu_ofh_decompress_device_bytes_to_device_grid_symbol_batch(
    ocudu_ofh_compression_handle_t* handle,
    int                             compression_type,
    void*                           output_grid_cbf16,
    const void*                     input_device_bytes,
    unsigned                        input_symbol_stride_bytes,
    unsigned                        input_port_stride_bytes,
    unsigned                        nof_grid_symbols,
    unsigned                        nof_subc,
    unsigned                        first_port,
    unsigned                        nof_ports,
    unsigned                        first_symbol,
    unsigned                        nof_symbols,
    unsigned                        start_prb,
    unsigned                        nof_prbs,
    unsigned                        data_width)
{
    if ((handle == nullptr) || (output_grid_cbf16 == nullptr) || (input_device_bytes == nullptr) ||
        (nof_ports == 0U)) {
        return 0;
    }
    if (!valid_grid_symbol_batch_request(
            compression_type, nof_grid_symbols, nof_subc, first_symbol, nof_symbols, start_prb, nof_prbs, data_width)) {
        return (nof_prbs == 0U) || (nof_symbols == 0U);
    }

    unsigned prb_size   = compressed_prb_size(compression_type, data_width);
    size_t   port_bytes = static_cast<size_t>(nof_prbs) * prb_size;
    if ((input_port_stride_bytes < port_bytes) ||
        (input_symbol_stride_bytes < static_cast<size_t>(nof_ports - 1U) * input_port_stride_bytes + port_bytes)) {
        return 0;
    }

    launch_decompress_symbol_batch(reinterpret_cast<uint32_t*>(output_grid_cbf16),
                                   reinterpret_cast<const uint8_t*>(input_device_bytes),
                                   input_symbol_stride_bytes,
                                   input_port_stride_bytes,
                                   compression_type,
                                   nof_grid_symbols,
                                   nof_subc,
                                   first_port,
                                   nof_ports,
                                   first_symbol,
                                   nof_symbols,
                                   start_prb,
                                   nof_prbs,
                                   data_width,
                                   prb_size,
                                   handle->stream);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return cudaStreamSynchronize(handle->stream) == cudaSuccess;
}
