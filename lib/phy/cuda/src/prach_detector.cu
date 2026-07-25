// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "prach_detector.h"
#include <algorithm>
#include <cmath>
#include <cuComplex.h>
#include <cuda.h>
#include <cstring>
#include <new>
#include "vkFFT.h"

#ifndef OCUDU_PHY_CUDA_ENABLE_VKFFT
#error "OCUDU PHY CUDA PRACH acceleration requires the bundled VkFFT backend."
#endif

struct prach_detector_ctx {
    uint32_t* d_input = nullptr;
    size_t d_input_capacity = 0;
    cuFloatComplex* d_roots = nullptr;
    size_t d_roots_capacity = 0;
    cuFloatComplex* d_idft = nullptr;
    size_t d_idft_capacity = 0;
    prach_detector_candidate_t* d_candidates = nullptr;
    size_t d_candidates_capacity = 0;
    float* d_rssi = nullptr;
    prach_detector_candidate_t* h_candidates = nullptr;
    float* h_rssi = nullptr;
    bool h_result_pinned = false;
    VkFFTApplication vkfft_app{};
    CUdevice vkfft_device = 0;
    cudaStream_t vkfft_stream = nullptr;
    void* vkfft_buffer = nullptr;
    pfUINT vkfft_buffer_size = 0;
    int vkfft_dft_size = 0;
    int vkfft_batch = 0;
    bool vkfft_valid = false;
    uint64_t root_cache_key = 0;
    int cached_nof_sequences = 0;
    int cached_sequence_length = 0;
};

namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ float bf16_to_float(uint16_t v)
{
    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = static_cast<uint32_t>(v) << 16;
    return conv.f;
}

__device__ __forceinline__ cuFloatComplex cbf16_to_cfloat(uint32_t packed)
{
    return make_cuFloatComplex(bf16_to_float(static_cast<uint16_t>(packed & 0xffffu)),
                               bf16_to_float(static_cast<uint16_t>(packed >> 16)));
}

__device__ __forceinline__ cuFloatComplex cmul_conj(cuFloatComplex x, cuFloatComplex y)
{
    return make_cuFloatComplex(x.x * y.x + x.y * y.y, x.y * y.x - x.x * y.y);
}

__device__ __forceinline__ float cabs2(cuFloatComplex x)
{
    return x.x * x.x + x.y * x.y;
}

__global__ void kernel_prepare_idft(const uint32_t* __restrict__ input,
                                    const cuFloatComplex* __restrict__ roots,
                                    cuFloatComplex* __restrict__ idft,
                                    float* __restrict__ rssi,
                                    prach_detector_config_t cfg,
                                    int batches_per_sequence)
{
    __shared__ float rssi_partial[THREADS];

    int local_sequence = blockIdx.y;
    int batch_in_sequence = blockIdx.x;
    int freq_bin = threadIdx.x + blockIdx.z * blockDim.x;
    if (local_sequence >= cfg.nof_sequences || batch_in_sequence >= batches_per_sequence) {
        return;
    }

    int port = 0;
    int symbol = 0;
    if (cfg.combine_symbols) {
        port = batch_in_sequence;
    } else {
        port = batch_in_sequence / cfg.nof_symbols;
        symbol = batch_in_sequence - port * cfg.nof_symbols;
    }

    const int half = cfg.sequence_length / 2;
    int root_re = -1;
    if (freq_bin < cfg.sequence_length - half) {
        root_re = freq_bin + half;
    } else if (freq_bin >= cfg.dft_size - half && freq_bin < cfg.dft_size) {
        root_re = freq_bin - (cfg.dft_size - half);
    }

    cuFloatComplex sample = make_cuFloatComplex(0.0f, 0.0f);
    float rssi_sum = 0.0f;
    if (root_re >= 0) {
        if (cfg.combine_symbols) {
            for (int i_symbol = 0; i_symbol != cfg.nof_symbols; ++i_symbol) {
                const uint32_t packed =
                    input[port * cfg.input_port_stride + i_symbol * cfg.input_symbol_stride + root_re];
                cuFloatComplex v = cbf16_to_cfloat(packed);
                sample.x += v.x;
                sample.y += v.y;
                rssi_sum += cabs2(v);
            }
        } else {
            sample = cbf16_to_cfloat(input[port * cfg.input_port_stride + symbol * cfg.input_symbol_stride + root_re]);
            rssi_sum = cabs2(sample);
        }
    }

    if (local_sequence == 0) {
        rssi_partial[threadIdx.x] = rssi_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                rssi_partial[threadIdx.x] += rssi_partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            atomicAdd(rssi, rssi_partial[0]);
        }
    }

    int batch = local_sequence * batches_per_sequence + batch_in_sequence;
    if (root_re < 0) {
        if (freq_bin < cfg.dft_size) {
            idft[batch * cfg.dft_size + freq_bin] = make_cuFloatComplex(0.0f, 0.0f);
        }
        return;
    }

    cuFloatComplex no_root = cmul_conj(sample, roots[local_sequence * cfg.sequence_length + root_re]);
    idft[batch * cfg.dft_size + freq_bin] = no_root;
}

__device__ float compute_metric_for_delay(const cuFloatComplex* __restrict__ idft,
                                          prach_detector_config_t cfg,
                                          int sequence,
                                          int window,
                                          int delay,
                                          int batches_per_sequence,
                                          float* numerator_out)
{
    const int window_start = (cfg.n_cs == 0) ? 0 : (cfg.n_cs * window * cfg.dft_size) / cfg.sequence_length;
    const float mod_scale = 1.0f / static_cast<float>(cfg.dft_size * cfg.sequence_length);
    const float window_scale = static_cast<float>(cfg.dft_size) / static_cast<float>(cfg.sequence_length);

    float num = 0.0f;
    float den = 0.0f;
    for (int b = 0; b != batches_per_sequence; ++b) {
        const cuFloatComplex* corr = idft + (sequence * batches_per_sequence + b) * cfg.dft_size;
        float reference = 0.0f;
        int ref_start = window_start + cfg.dft_size - cfg.win_margin;
        int ref_len = 2 * cfg.win_margin + cfg.win_width;
        for (int i = 0; i != ref_len; ++i) {
            int idx = (ref_start + i) % cfg.dft_size;
            reference += cabs2(corr[idx]) * mod_scale;
        }

        int sample_idx = window_start;
        // The VkFFT correlation layout mirrors unrestricted (Ncs=0) timing offsets. Keep non-zero cyclic-shift
        // windows on the established forward scan to avoid changing their preamble-selection behavior.
        if (cfg.n_cs == 0) {
            sample_idx -= delay;
            if (sample_idx < 0) {
                sample_idx += cfg.dft_size;
            }
        } else {
            sample_idx += delay;
            if (sample_idx >= cfg.dft_size) {
                sample_idx -= cfg.dft_size;
            }
        }
        float window_power = cabs2(corr[sample_idx]) * mod_scale * window_scale;
        float diff = reference - window_power;
        if (!isfinite(diff) || fabsf(diff) < 1e-20f) {
            diff = 1e-9f;
        }
        num += window_power;
        den += diff;
    }

    den = fabsf(den);
    if (!isfinite(den) || den <= 0.0f) {
        den = 1e-9f;
    }
    *numerator_out = num;
    return num / den;
}

__global__ void kernel_find_candidates(const cuFloatComplex* __restrict__ idft,
                                       prach_detector_candidate_t* __restrict__ candidates,
                                       prach_detector_config_t cfg,
                                       int batches_per_sequence)
{
    __shared__ float best_metric[THREADS];
    __shared__ float best_power[THREADS];
    __shared__ int best_delay[THREADS];

    int local_sequence = blockIdx.y;
    int sequence = cfg.sequence_start + local_sequence;
    int window = blockIdx.x;
    int preamble_index = sequence * cfg.nof_shifts + window;
    int out_index = preamble_index;

    float local_metric = -1.0f;
    float local_power = 0.0f;
    int local_delay = 0;

    if (preamble_index < 64 && preamble_index >= cfg.start_preamble_index &&
        preamble_index < cfg.start_preamble_index + cfg.nof_preamble_indices) {
        for (int delay = threadIdx.x; delay < cfg.win_width; delay += blockDim.x) {
            float numerator = 0.0f;
            float metric =
                compute_metric_for_delay(idft, cfg, local_sequence, window, delay, batches_per_sequence, &numerator);
            if (metric > local_metric) {
                local_metric = metric;
                local_power = numerator;
                local_delay = delay;
            }
        }
    }

    best_metric[threadIdx.x] = local_metric;
    best_power[threadIdx.x] = local_power;
    best_delay[threadIdx.x] = local_delay;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && best_metric[threadIdx.x + stride] > best_metric[threadIdx.x]) {
            best_metric[threadIdx.x] = best_metric[threadIdx.x + stride];
            best_power[threadIdx.x] = best_power[threadIdx.x + stride];
            best_delay[threadIdx.x] = best_delay[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && out_index < 64) {
        prach_detector_candidate_t out{};
        float max_delay_gate = static_cast<float>(cfg.max_delay_samples) * 0.8f;
        if (preamble_index < 64 && best_metric[0] > cfg.threshold &&
            static_cast<float>(best_delay[0]) < max_delay_gate) {
            int power_norm = cfg.nof_rx_ports * cfg.sequence_length * cfg.nof_symbols;
            if (cfg.combine_symbols) {
                power_norm *= cfg.nof_symbols;
            }
            out.valid = 1;
            out.preamble_index = preamble_index;
            out.delay_samples = best_delay[0];
            out.detection_metric = best_metric[0] / cfg.threshold;
            out.preamble_power = best_power[0] / static_cast<float>(power_norm);
        }
        candidates[out_index] = out;
    }
}

bool check_cuda(cudaError_t status)
{
    return status == cudaSuccess;
}

bool ensure_capacity(void** ptr, size_t* capacity, size_t bytes)
{
    if (*capacity >= bytes) {
        return true;
    }
    if (*ptr != nullptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
        *capacity = 0;
    }
    if (bytes == 0) {
        return true;
    }
    if (cudaMalloc(ptr, bytes) != cudaSuccess) {
        return false;
    }
    *capacity = bytes;
    return true;
}

bool allocate_host_result_buffers(prach_detector_ctx* ctx)
{
    constexpr size_t candidates_bytes = 64 * sizeof(prach_detector_candidate_t);
    cudaError_t rssi_status = cudaMallocHost(reinterpret_cast<void**>(&ctx->h_rssi), sizeof(float));
    cudaError_t candidates_status = cudaMallocHost(reinterpret_cast<void**>(&ctx->h_candidates), candidates_bytes);
    if (rssi_status == cudaSuccess && candidates_status == cudaSuccess) {
        ctx->h_result_pinned = true;
        return true;
    }

    if (ctx->h_rssi != nullptr) {
        cudaFreeHost(ctx->h_rssi);
        ctx->h_rssi = nullptr;
    }
    if (ctx->h_candidates != nullptr) {
        cudaFreeHost(ctx->h_candidates);
        ctx->h_candidates = nullptr;
    }
    (void)cudaGetLastError();

    ctx->h_rssi = new (std::nothrow) float;
    ctx->h_candidates = new (std::nothrow) prach_detector_candidate_t[64];
    if ((ctx->h_rssi == nullptr) || (ctx->h_candidates == nullptr)) {
        delete ctx->h_rssi;
        delete[] ctx->h_candidates;
        ctx->h_rssi = nullptr;
        ctx->h_candidates = nullptr;
        return false;
    }
    ctx->h_result_pinned = false;
    return true;
}

bool ensure_plan(prach_detector_ctx* ctx, int dft_size, int batch, cudaStream_t stream)
{
    if (ctx->vkfft_valid && ctx->vkfft_dft_size == dft_size && ctx->vkfft_batch == batch &&
        ctx->vkfft_stream == stream && ctx->vkfft_buffer == ctx->d_idft) {
        return true;
    }
    if (ctx->vkfft_valid) {
        deleteVkFFT(&ctx->vkfft_app);
        std::memset(&ctx->vkfft_app, 0, sizeof(ctx->vkfft_app));
        ctx->vkfft_valid = false;
    }

    int cuda_device = 0;
    if (!check_cuda(cudaGetDevice(&cuda_device)) || (cuDeviceGet(&ctx->vkfft_device, cuda_device) != CUDA_SUCCESS)) {
        return false;
    }

    ctx->vkfft_stream = stream;
    ctx->vkfft_buffer = ctx->d_idft;
    ctx->vkfft_buffer_size = static_cast<pfUINT>(static_cast<size_t>(batch) * dft_size * sizeof(cuFloatComplex));

    VkFFTConfiguration configuration = {};
    configuration.FFTdim = 1;
    configuration.size[0] = static_cast<pfUINT>(dft_size);
    configuration.device = &ctx->vkfft_device;
    configuration.stream = &ctx->vkfft_stream;
    configuration.num_streams = 1;
    configuration.numberBatches = batch;
    configuration.bufferSize = &ctx->vkfft_buffer_size;
    configuration.buffer = &ctx->vkfft_buffer;
    configuration.makeForwardPlanOnly = 1;
    configuration.normalize = 0;
    configuration.disableReorderFourStep = 1;
    configuration.useLUT = 1;
    configuration.aimThreads = 128;

    if (initializeVkFFT(&ctx->vkfft_app, configuration) != VKFFT_SUCCESS) {
        std::memset(&ctx->vkfft_app, 0, sizeof(ctx->vkfft_app));
        return false;
    }
    ctx->vkfft_dft_size = dft_size;
    ctx->vkfft_batch = batch;
    ctx->vkfft_valid = true;
    return true;
}

bool validate_config(const prach_detector_config_t* cfg)
{
    return cfg != nullptr && cfg->sequence_length > 0 && cfg->dft_size >= cfg->sequence_length &&
           cfg->nof_rx_ports > 0 && cfg->nof_symbols > 0 && cfg->nof_sequences > 0 && cfg->nof_shifts > 0 &&
           cfg->sequence_start >= 0 && cfg->sequence_start + cfg->nof_sequences <= 64 && cfg->win_width > 0 &&
           cfg->threshold > 0.0f &&
           cfg->input_symbol_stride >= cfg->sequence_length &&
           cfg->input_port_stride >= cfg->input_symbol_stride * cfg->nof_symbols;
}

nr_ldpc_status_t enqueue_detector_work(prach_detector_ctx* ctx,
                                       const uint32_t* d_input,
                                       const prach_detector_config_t& cfg,
                                       int batches_per_sequence,
                                       int total_batches,
                                       int nof_candidate_slots,
                                       cudaStream_t stream)
{
    if (!check_cuda(cudaMemsetAsync(ctx->d_rssi,
                                    0,
                                    sizeof(float),
                                    stream))) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    dim3 prep_grid(batches_per_sequence, cfg.nof_sequences, (cfg.dft_size + THREADS - 1) / THREADS);
    kernel_prepare_idft<<<prep_grid, THREADS, 0, stream>>>(
        d_input, ctx->d_roots, ctx->d_idft, ctx->d_rssi, cfg, batches_per_sequence);
    if (!check_cuda(cudaGetLastError())) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    VkFFTLaunchParams launch_params = {};
    launch_params.buffer = &ctx->vkfft_buffer;
    if (VkFFTAppend(&ctx->vkfft_app, -1, &launch_params) != VKFFT_SUCCESS) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    dim3 candidate_grid(cfg.nof_shifts, cfg.nof_sequences, 1);
    kernel_find_candidates<<<candidate_grid, THREADS, 0, stream>>>(
        ctx->d_idft, ctx->d_candidates, cfg, batches_per_sequence);
    if (!check_cuda(cudaGetLastError())) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    return NR_LDPC_SUCCESS;
}

} // namespace

extern "C" nr_ldpc_status_t prach_detector_create(prach_detector_handle_t* handle)
{
    if (handle == nullptr) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    auto* ctx = new (std::nothrow) prach_detector_ctx();
    if (ctx == nullptr) {
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }
    if ((cudaMalloc(&ctx->d_rssi, sizeof(float)) != cudaSuccess) || !allocate_host_result_buffers(ctx)) {
        prach_detector_destroy(ctx);
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }
    *handle = ctx;
    return NR_LDPC_SUCCESS;
}

extern "C" void prach_detector_destroy(prach_detector_handle_t handle)
{
    if (handle == nullptr) {
        return;
    }
    if (handle->vkfft_valid) {
        deleteVkFFT(&handle->vkfft_app);
        handle->vkfft_valid = false;
    }
    cudaFree(handle->d_input);
    cudaFree(handle->d_roots);
    cudaFree(handle->d_idft);
    cudaFree(handle->d_candidates);
    cudaFree(handle->d_rssi);
    if (handle->h_result_pinned) {
        cudaFreeHost(handle->h_rssi);
        cudaFreeHost(handle->h_candidates);
    } else {
        delete handle->h_rssi;
        delete[] handle->h_candidates;
    }
    delete handle;
}

extern "C" nr_ldpc_status_t prach_detector_detect(prach_detector_handle_t         handle,
                                                   const void*                    prach_cbf16,
                                                   const void*                    roots_cf32,
                                                   const prach_detector_config_t* config,
                                                   prach_detector_result_t*       result,
                                                   void*                          external_stream)
{
    if (handle == nullptr || prach_cbf16 == nullptr || roots_cf32 == nullptr || result == nullptr ||
        !validate_config(config)) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(external_stream);
    prach_detector_config_t cfg = *config;
    const int batches_per_sequence = cfg.nof_rx_ports * (cfg.combine_symbols ? 1 : cfg.nof_symbols);
    const int total_batches = batches_per_sequence * cfg.nof_sequences;
    const int nof_candidate_slots = 64;

    if (!ensure_capacity(reinterpret_cast<void**>(&handle->d_idft),
                         &handle->d_idft_capacity,
                         static_cast<size_t>(total_batches) * cfg.dft_size * sizeof(cuFloatComplex)) ||
        !ensure_capacity(reinterpret_cast<void**>(&handle->d_candidates),
                         &handle->d_candidates_capacity,
                         static_cast<size_t>(nof_candidate_slots) * sizeof(prach_detector_candidate_t)) ||
        !ensure_capacity(reinterpret_cast<void**>(&handle->d_roots),
                         &handle->d_roots_capacity,
                         static_cast<size_t>(cfg.nof_sequences) * cfg.sequence_length * sizeof(cuFloatComplex))) {
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }

    const uint32_t* d_input = static_cast<const uint32_t*>(prach_cbf16);
    if (!cfg.input_is_device) {
        size_t input_bytes = static_cast<size_t>(cfg.nof_rx_ports) * cfg.nof_symbols * cfg.sequence_length * sizeof(uint32_t);
        if (!ensure_capacity(reinterpret_cast<void**>(&handle->d_input), &handle->d_input_capacity, input_bytes)) {
            return NR_LDPC_ERROR_ALLOC_FAILED;
        }
        if (!check_cuda(cudaMemcpyAsync(handle->d_input, prach_cbf16, input_bytes, cudaMemcpyHostToDevice, stream))) {
            return NR_LDPC_ERROR_CUDA_FAILED;
        }
        d_input = handle->d_input;
        cfg.input_symbol_stride = cfg.sequence_length;
        cfg.input_port_stride = cfg.nof_symbols * cfg.sequence_length;
    }

    if (handle->root_cache_key != cfg.root_cache_key || handle->cached_nof_sequences != cfg.nof_sequences ||
        handle->cached_sequence_length != cfg.sequence_length) {
        size_t roots_bytes = static_cast<size_t>(cfg.nof_sequences) * cfg.sequence_length * sizeof(cuFloatComplex);
        if (!check_cuda(cudaMemcpyAsync(handle->d_roots, roots_cf32, roots_bytes, cudaMemcpyHostToDevice, stream))) {
            return NR_LDPC_ERROR_CUDA_FAILED;
        }
        handle->root_cache_key = cfg.root_cache_key;
        handle->cached_nof_sequences = cfg.nof_sequences;
        handle->cached_sequence_length = cfg.sequence_length;
    }

    if (!ensure_plan(handle, cfg.dft_size, total_batches, stream)) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    nr_ldpc_status_t enqueue_status =
        enqueue_detector_work(handle, d_input, cfg, batches_per_sequence, total_batches, nof_candidate_slots, stream);
    if (enqueue_status != NR_LDPC_SUCCESS) {
        return enqueue_status;
    }

    const int nof_candidate_results = std::min(nof_candidate_slots, 64);
    if (!check_cuda(cudaMemcpyAsync(handle->h_rssi, handle->d_rssi, sizeof(float), cudaMemcpyDeviceToHost, stream)) ||
        !check_cuda(cudaMemcpyAsync(handle->h_candidates,
                                    handle->d_candidates,
                                    static_cast<size_t>(nof_candidate_results) * sizeof(prach_detector_candidate_t),
                                    cudaMemcpyDeviceToHost,
                                    stream)) ||
        !check_cuda(cudaStreamSynchronize(stream))) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    std::memset(result, 0, sizeof(*result));
    result->rssi = *handle->h_rssi / static_cast<float>(cfg.nof_rx_ports * cfg.nof_symbols * cfg.sequence_length);
    const int last_requested_preamble = cfg.start_preamble_index + cfg.nof_preamble_indices;
    for (int i = 0; i != nof_candidate_results; ++i) {
        const prach_detector_candidate_t& candidate = handle->h_candidates[i];
        if (!candidate.valid) {
            continue;
        }
        if (candidate.preamble_index < cfg.start_preamble_index ||
            candidate.preamble_index >= last_requested_preamble) {
            continue;
        }
        if (result->nof_candidates == 64) {
            break;
        }
        result->candidates[result->nof_candidates++] = candidate;
    }

    return NR_LDPC_SUCCESS;
}
