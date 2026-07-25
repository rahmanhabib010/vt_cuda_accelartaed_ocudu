// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "low_phy_prach_rx.h"

#include <algorithm>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sched.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "vkFFT.h"

#ifndef OCUDU_PHY_CUDA_ENABLE_VKFFT
#error "OCUDU PHY CUDA PRACH acceleration requires the bundled VkFFT backend."
#endif

namespace {

static constexpr int THREADS = 256;
static constexpr int HOST_STAGING_SLOTS = 8;

static bool check_cuda(cudaError_t err)
{
    return err == cudaSuccess;
}

static cudaError_t create_realtime_stream(cudaStream_t* stream)
{
    int least_priority = 0;
    int greatest_priority = 0;
    cudaError_t status = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    if (status == cudaSuccess) {
        status = cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking, greatest_priority);
        if (status == cudaSuccess) {
            return cudaSuccess;
        }
        (void)cudaGetLastError();
    }

    return cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
}

static bool env_flag_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    return (std::strcmp(value, "1") == 0) || (std::strcmp(value, "true") == 0) ||
           (std::strcmp(value, "on") == 0) || (std::strcmp(value, "yes") == 0) ||
           (std::strcmp(value, "enabled") == 0);
}

static bool runtime_host_registration_enabled()
{
    static const bool enabled = env_flag_enabled("OCUDU_LOWPHY_PRACH_RUNTIME_HOST_REGISTRATION");
    return enabled;
}

__device__ __forceinline__ uint16_t float_to_bf16(float value)
{
    uint32_t bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ uint32_t cfloat_to_cbf16(cuFloatComplex value)
{
    return static_cast<uint32_t>(float_to_bf16(value.x)) |
           (static_cast<uint32_t>(float_to_bf16(value.y)) << 16);
}

__global__ void load_prach_fft_input_kernel(cuFloatComplex* __restrict__ d_freq,
                                            const cuFloatComplex* __restrict__ d_input,
                                            int dft_size,
                                            int nof_symbols,
                                            int cyclic_prefix_length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dft_size * nof_symbols;
    if (idx >= total) {
        return;
    }

    int re = idx % dft_size;
    int symbol = idx / dft_size;
    d_freq[idx] = d_input[cyclic_prefix_length + symbol * dft_size + re];
}

__global__ void load_prach_fft_input_ci16_kernel(cuFloatComplex* __restrict__ d_freq,
                                                 const int16_t* __restrict__ d_input,
                                                 int dft_size,
                                                 int nof_symbols,
                                                 int cyclic_prefix_length,
                                                 float input_gain)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dft_size * nof_symbols;
    if (idx >= total) {
        return;
    }

    int re = idx % dft_size;
    int symbol = idx / dft_size;
    int src = cyclic_prefix_length + symbol * dft_size + re;
    d_freq[idx] = make_cuFloatComplex(static_cast<float>(d_input[2 * src]) * input_gain,
                                      static_cast<float>(d_input[2 * src + 1]) * input_gain);
}

__global__ void extract_prach_grid_kernel(uint32_t* __restrict__ output,
                                          const cuFloatComplex* __restrict__ d_freq,
                                          ocudu_lowphy_prach_rx_config_t cfg)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = cfg.nof_fd_occasions * cfg.nof_symbols * cfg.sequence_length;
    if (idx >= total) {
        return;
    }

    int re = idx % cfg.sequence_length;
    int tmp = idx / cfg.sequence_length;
    int symbol = tmp % cfg.nof_symbols;
    int fd = tmp / cfg.nof_symbols;

    int half_grid = cfg.prach_grid_size / 2;
    int k = cfg.k_start[fd] + re;
    int src = (k < half_grid) ? (cfg.dft_size - half_grid + k) : (k - half_grid);
    cuFloatComplex x = d_freq[symbol * cfg.dft_size + src];
    x.x *= cfg.dft_scale;
    x.y *= cfg.dft_scale;

    output[fd * cfg.output_fd_stride + symbol * cfg.output_symbol_stride + re] = cfloat_to_cbf16(x);
}

} // namespace

struct ocudu_lowphy_prach_rx_handle {
    ocudu_lowphy_prach_rx_config_t cfg{};
    cudaStream_t owned_stream = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t done_event = nullptr;

    cuFloatComplex* d_input = nullptr;
    int16_t* d_input_ci16 = nullptr;
    cuFloatComplex* d_freq = nullptr;
    size_t input_count = 0;
    size_t input_ci16_count = 0;
    size_t freq_count = 0;

    cuFloatComplex* h_input_staging[HOST_STAGING_SLOTS] = {};
    cudaEvent_t h_input_ready_events[HOST_STAGING_SLOTS] = {};
    bool h_input_event_recorded[HOST_STAGING_SLOTS] = {};
    size_t h_input_staging_count = 0;
    int h_input_next_slot = 0;

    int16_t* h_input_ci16_staging[HOST_STAGING_SLOTS] = {};
    cudaEvent_t h_input_ci16_ready_events[HOST_STAGING_SLOTS] = {};
    bool h_input_ci16_event_recorded[HOST_STAGING_SLOTS] = {};
    size_t h_input_ci16_staging_count = 0;
    int h_input_ci16_next_slot = 0;

    VkFFTApplication vkfft_app{};
    CUdevice vkfft_device = 0;
    cudaStream_t vkfft_stream = nullptr;
    void* vkfft_buffer = nullptr;
    pfUINT vkfft_buffer_size = 0;
    bool vkfft_valid = false;

    uint32_t* d_warmup_output = nullptr;
    size_t warmup_output_count = 0;

    struct host_registration {
        const void* ptr = nullptr;
        size_t bytes = 0;
        bool unregister_on_destroy = false;
    };
    std::vector<host_registration> registered_hosts;
    std::vector<host_registration> unregistered_hosts;
};

namespace {

static void destroy_fft_plan(ocudu_lowphy_prach_rx_handle* h)
{
    if (h->vkfft_valid) {
        deleteVkFFT(&h->vkfft_app);
        std::memset(&h->vkfft_app, 0, sizeof(h->vkfft_app));
        h->vkfft_valid = false;
    }
}

static void release_buffers(ocudu_lowphy_prach_rx_handle* h)
{
    if (h->stream != nullptr) {
        (void)cudaStreamSynchronize(h->stream);
    }
    for (int slot = 0; slot != HOST_STAGING_SLOTS; ++slot) {
        if (h->h_input_ready_events[slot] != nullptr) {
            cudaEventDestroy(h->h_input_ready_events[slot]);
            h->h_input_ready_events[slot] = nullptr;
        }
        if (h->h_input_staging[slot] != nullptr) {
            cudaFreeHost(h->h_input_staging[slot]);
            h->h_input_staging[slot] = nullptr;
        }
        h->h_input_event_recorded[slot] = false;

        if (h->h_input_ci16_ready_events[slot] != nullptr) {
            cudaEventDestroy(h->h_input_ci16_ready_events[slot]);
            h->h_input_ci16_ready_events[slot] = nullptr;
        }
        if (h->h_input_ci16_staging[slot] != nullptr) {
            cudaFreeHost(h->h_input_ci16_staging[slot]);
            h->h_input_ci16_staging[slot] = nullptr;
        }
        h->h_input_ci16_event_recorded[slot] = false;
    }
    if (h->d_input != nullptr) {
        cudaFree(h->d_input);
        h->d_input = nullptr;
    }
    if (h->d_input_ci16 != nullptr) {
        cudaFree(h->d_input_ci16);
        h->d_input_ci16 = nullptr;
    }
    if (h->d_freq != nullptr) {
        cudaFree(h->d_freq);
        h->d_freq = nullptr;
    }
    if (h->d_warmup_output != nullptr) {
        cudaFree(h->d_warmup_output);
        h->d_warmup_output = nullptr;
    }
    h->input_count = 0;
    h->input_ci16_count = 0;
    h->freq_count = 0;
    h->warmup_output_count = 0;
    h->h_input_staging_count = 0;
    h->h_input_next_slot = 0;
    h->h_input_ci16_staging_count = 0;
    h->h_input_ci16_next_slot = 0;
}

static bool contains_registered_range(const std::vector<ocudu_lowphy_prach_rx_handle::host_registration>& registrations,
                                      const void* ptr,
                                      size_t bytes)
{
    if ((ptr == nullptr) || (bytes == 0)) {
        return true;
    }

    const auto begin = reinterpret_cast<uintptr_t>(ptr);
    const auto end = begin + bytes;
    for (const auto& registration : registrations) {
        const auto registered_begin = reinterpret_cast<uintptr_t>(registration.ptr);
        const auto registered_end = registered_begin + registration.bytes;
        if ((begin >= registered_begin) && (end <= registered_end)) {
            return true;
        }
    }
    return false;
}

static bool register_host_range(ocudu_lowphy_prach_rx_handle* h, const void* ptr, size_t bytes)
{
    if ((ptr == nullptr) || (bytes == 0)) {
        return true;
    }
    if (contains_registered_range(h->registered_hosts, ptr, bytes)) {
        return true;
    }
    if (contains_registered_range(h->unregistered_hosts, ptr, bytes)) {
        return false;
    }

    cudaError_t err = cudaHostRegister(const_cast<void*>(ptr), bytes, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        h->registered_hosts.push_back({ptr, bytes, true});
        return true;
    }
    if (err == cudaErrorHostMemoryAlreadyRegistered) {
        (void)cudaGetLastError();
        h->registered_hosts.push_back({ptr, bytes, false});
        return true;
    }

    (void)cudaGetLastError();
    h->unregistered_hosts.push_back({ptr, bytes, false});
    return false;
}

static bool validate_config(const ocudu_lowphy_prach_rx_config_t& cfg)
{
    if ((cfg.dft_size <= 0) || (cfg.sequence_length <= 0) || (cfg.nof_symbols <= 0) ||
        (cfg.nof_fd_occasions <= 0) || (cfg.input_nof_samples <= 0) || (cfg.cyclic_prefix_length < 0) ||
        (cfg.prach_grid_size <= 0) || (cfg.output_symbol_stride < cfg.sequence_length) ||
        (cfg.output_fd_stride < cfg.output_symbol_stride * cfg.nof_symbols) || (cfg.dft_scale <= 0.0f)) {
        return false;
    }
    if ((cfg.sequence_length > cfg.prach_grid_size) || (cfg.prach_grid_size > cfg.dft_size) ||
        ((cfg.prach_grid_size % 2) != 0) ||
        (cfg.nof_fd_occasions > OCUDU_LOWPHY_PRACH_RX_MAX_FD_OCCASIONS) ||
        (cfg.cyclic_prefix_length + cfg.nof_symbols * cfg.dft_size > cfg.input_nof_samples)) {
        return false;
    }
    for (int fd = 0; fd != cfg.nof_fd_occasions; ++fd) {
        if ((cfg.k_start[fd] < 0) || (cfg.k_start[fd] + cfg.sequence_length > cfg.prach_grid_size)) {
            return false;
        }
    }
    return true;
}

static bool ensure_buffers(ocudu_lowphy_prach_rx_handle* h)
{
    size_t input_count = static_cast<size_t>(h->cfg.input_nof_samples);
    size_t freq_count = static_cast<size_t>(h->cfg.nof_symbols) * h->cfg.dft_size;
    size_t output_count = static_cast<size_t>(h->cfg.nof_fd_occasions) * h->cfg.output_fd_stride;
    if ((h->input_count == input_count) && (h->freq_count == freq_count) && (h->d_input != nullptr) &&
        (h->d_freq != nullptr) && (h->h_input_staging_count == freq_count) && (h->h_input_staging[0] != nullptr) &&
        (h->d_warmup_output != nullptr) && (h->warmup_output_count >= output_count)) {
        return true;
    }

    destroy_fft_plan(h);
    release_buffers(h);
    if (!check_cuda(cudaMalloc(&h->d_input, input_count * sizeof(cuFloatComplex))) ||
        !check_cuda(cudaMalloc(&h->d_freq, freq_count * sizeof(cuFloatComplex))) ||
        !check_cuda(cudaMalloc(&h->d_warmup_output, output_count * sizeof(uint32_t)))) {
        release_buffers(h);
        return false;
    }
    for (int slot = 0; slot != HOST_STAGING_SLOTS; ++slot) {
        if (!check_cuda(cudaHostAlloc(&h->h_input_staging[slot],
                                      freq_count * sizeof(cuFloatComplex),
                                      cudaHostAllocDefault)) ||
            !check_cuda(cudaEventCreateWithFlags(&h->h_input_ready_events[slot], cudaEventDisableTiming))) {
            release_buffers(h);
            return false;
        }
    }
    h->input_count = input_count;
    h->freq_count = freq_count;
    h->warmup_output_count = output_count;
    h->h_input_staging_count = freq_count;
    return true;
}

static bool ensure_ci16_input_buffer(ocudu_lowphy_prach_rx_handle* h)
{
    size_t input_ci16_count = static_cast<size_t>(h->cfg.input_nof_samples) * 2;
    size_t payload_ci16_count = h->freq_count * 2;
    if ((h->input_ci16_count == input_ci16_count) && (h->d_input_ci16 != nullptr)) {
        return true;
    }
    if (h->stream != nullptr) {
        (void)cudaStreamSynchronize(h->stream);
    }
    if (h->d_input_ci16 != nullptr) {
        cudaFree(h->d_input_ci16);
        h->d_input_ci16 = nullptr;
    }
    if (!check_cuda(cudaMalloc(&h->d_input_ci16, input_ci16_count * sizeof(int16_t)))) {
        h->input_ci16_count = 0;
        return false;
    }
    for (int slot = 0; slot != HOST_STAGING_SLOTS; ++slot) {
        if (h->h_input_ci16_staging[slot] != nullptr) {
            cudaFreeHost(h->h_input_ci16_staging[slot]);
            h->h_input_ci16_staging[slot] = nullptr;
        }
        if (h->h_input_ci16_ready_events[slot] != nullptr) {
            cudaEventDestroy(h->h_input_ci16_ready_events[slot]);
            h->h_input_ci16_ready_events[slot] = nullptr;
        }
        h->h_input_ci16_event_recorded[slot] = false;
        if (!check_cuda(cudaHostAlloc(&h->h_input_ci16_staging[slot],
                                      payload_ci16_count * sizeof(int16_t),
                                      cudaHostAllocDefault)) ||
            !check_cuda(cudaEventCreateWithFlags(&h->h_input_ci16_ready_events[slot], cudaEventDisableTiming))) {
            release_buffers(h);
            return false;
        }
    }
    h->input_ci16_count = input_ci16_count;
    h->h_input_ci16_staging_count = payload_ci16_count;
    h->h_input_ci16_next_slot = 0;
    return true;
}

static bool ensure_plan(ocudu_lowphy_prach_rx_handle* h)
{
    if (h->vkfft_valid && (h->vkfft_stream == h->stream) && (h->vkfft_buffer == h->d_freq)) {
        return true;
    }
    if (h->vkfft_valid) {
        deleteVkFFT(&h->vkfft_app);
        std::memset(&h->vkfft_app, 0, sizeof(h->vkfft_app));
        h->vkfft_valid = false;
    }
    int cuda_device = 0;
    if (!check_cuda(cudaGetDevice(&cuda_device)) || (cuDeviceGet(&h->vkfft_device, cuda_device) != CUDA_SUCCESS)) {
        return false;
    }

    h->vkfft_stream = h->stream;
    h->vkfft_buffer = h->d_freq;
    h->vkfft_buffer_size = static_cast<pfUINT>(h->freq_count * sizeof(cuFloatComplex));

    VkFFTConfiguration configuration = {};
    configuration.FFTdim = 1;
    configuration.size[0] = static_cast<pfUINT>(h->cfg.dft_size);
    configuration.device = &h->vkfft_device;
    configuration.stream = &h->vkfft_stream;
    configuration.num_streams = 1;
    configuration.numberBatches = h->cfg.nof_symbols;
    configuration.bufferSize = &h->vkfft_buffer_size;
    configuration.buffer = &h->vkfft_buffer;
    configuration.makeForwardPlanOnly = 1;
    configuration.normalize = 0;
    configuration.useLUT = 1;
    configuration.aimThreads = 128;

    if (initializeVkFFT(&h->vkfft_app, configuration) != VKFFT_SUCCESS) {
        std::memset(&h->vkfft_app, 0, sizeof(h->vkfft_app));
        return false;
    }
    h->vkfft_valid = true;
    return true;
}

static cudaError_t wait_event_yielding(cudaEvent_t event)
{
    cudaError_t status;
    while ((status = cudaEventQuery(event)) == cudaErrorNotReady) {
        sched_yield();
    }
    return status;
}

static cuFloatComplex* acquire_cf32_staging_slot(ocudu_lowphy_prach_rx_handle* h, int& slot)
{
    slot = h->h_input_next_slot;
    h->h_input_next_slot = (h->h_input_next_slot + 1) % HOST_STAGING_SLOTS;

    if ((h->h_input_staging[slot] == nullptr) || (h->h_input_ready_events[slot] == nullptr)) {
        return nullptr;
    }
    if (h->h_input_event_recorded[slot]) {
        if (wait_event_yielding(h->h_input_ready_events[slot]) != cudaSuccess) {
            h->h_input_event_recorded[slot] = false;
            return nullptr;
        }
        h->h_input_event_recorded[slot] = false;
    }
    return h->h_input_staging[slot];
}

static int16_t* acquire_ci16_staging_slot(ocudu_lowphy_prach_rx_handle* h, int& slot)
{
    slot = h->h_input_ci16_next_slot;
    h->h_input_ci16_next_slot = (h->h_input_ci16_next_slot + 1) % HOST_STAGING_SLOTS;

    if ((h->h_input_ci16_staging[slot] == nullptr) || (h->h_input_ci16_ready_events[slot] == nullptr)) {
        return nullptr;
    }
    if (h->h_input_ci16_event_recorded[slot]) {
        if (wait_event_yielding(h->h_input_ci16_ready_events[slot]) != cudaSuccess) {
            h->h_input_ci16_event_recorded[slot] = false;
            return nullptr;
        }
        h->h_input_ci16_event_recorded[slot] = false;
    }
    return h->h_input_ci16_staging[slot];
}

static bool enqueue_prach_demod(ocudu_lowphy_prach_rx_handle* h,
                                const cuFloatComplex* d_input,
                                void* output_cbf16,
                                bool input_is_fft_payload)
{
    int load_blocks = (static_cast<int>(h->freq_count) + THREADS - 1) / THREADS;
    int cyclic_prefix_length = input_is_fft_payload ? 0 : h->cfg.cyclic_prefix_length;
    load_prach_fft_input_kernel<<<load_blocks, THREADS, 0, h->stream>>>(
        h->d_freq, d_input, h->cfg.dft_size, h->cfg.nof_symbols, cyclic_prefix_length);
    if (!check_cuda(cudaGetLastError())) {
        return false;
    }

    VkFFTLaunchParams launch_params = {};
    launch_params.buffer = &h->vkfft_buffer;
    if (VkFFTAppend(&h->vkfft_app, -1, &launch_params) != VKFFT_SUCCESS) {
        return false;
    }

    int extract_total = h->cfg.nof_fd_occasions * h->cfg.nof_symbols * h->cfg.sequence_length;
    int extract_blocks = (extract_total + THREADS - 1) / THREADS;
    extract_prach_grid_kernel<<<extract_blocks, THREADS, 0, h->stream>>>(
        static_cast<uint32_t*>(output_cbf16), h->d_freq, h->cfg);
    if (!check_cuda(cudaGetLastError())) {
        return false;
    }

    return check_cuda(cudaEventRecord(h->done_event, h->stream));
}

static bool
enqueue_prach_demod_ci16(ocudu_lowphy_prach_rx_handle* h,
                         const int16_t* d_input,
                         float input_scale,
                         void* output_cbf16,
                         bool input_is_fft_payload)
{
    if (!std::isfinite(input_scale) || (input_scale <= 0.0f)) {
        return false;
    }

    int load_blocks = (static_cast<int>(h->freq_count) + THREADS - 1) / THREADS;
    int cyclic_prefix_length = input_is_fft_payload ? 0 : h->cfg.cyclic_prefix_length;
    load_prach_fft_input_ci16_kernel<<<load_blocks, THREADS, 0, h->stream>>>(
        h->d_freq,
        d_input,
        h->cfg.dft_size,
        h->cfg.nof_symbols,
        cyclic_prefix_length,
        1.0f / input_scale);
    if (!check_cuda(cudaGetLastError())) {
        return false;
    }

    VkFFTLaunchParams launch_params = {};
    launch_params.buffer = &h->vkfft_buffer;
    if (VkFFTAppend(&h->vkfft_app, -1, &launch_params) != VKFFT_SUCCESS) {
        return false;
    }

    int extract_total = h->cfg.nof_fd_occasions * h->cfg.nof_symbols * h->cfg.sequence_length;
    int extract_blocks = (extract_total + THREADS - 1) / THREADS;
    extract_prach_grid_kernel<<<extract_blocks, THREADS, 0, h->stream>>>(
        static_cast<uint32_t*>(output_cbf16), h->d_freq, h->cfg);
    if (!check_cuda(cudaGetLastError())) {
        return false;
    }

    return check_cuda(cudaEventRecord(h->done_event, h->stream));
}

static bool warmup_handle(ocudu_lowphy_prach_rx_handle* h)
{
    if ((h == nullptr) || (h->d_warmup_output == nullptr)) {
        return false;
    }

    cudaStream_t saved_stream = h->stream;
    h->stream = (h->owned_stream != nullptr) ? h->owned_stream : h->stream;
    if ((h->stream == nullptr) || !ensure_buffers(h) || !ensure_ci16_input_buffer(h) || !ensure_plan(h)) {
        h->stream = saved_stream;
        return false;
    }

    if (!check_cuda(cudaMemsetAsync(h->d_input, 0, h->input_count * sizeof(cuFloatComplex), h->stream))) {
        h->stream = saved_stream;
        return false;
    }
    if (!check_cuda(cudaMemsetAsync(h->d_input_ci16, 0, h->input_ci16_count * sizeof(int16_t), h->stream))) {
        h->stream = saved_stream;
        return false;
    }

    bool ok = enqueue_prach_demod(h, h->d_input, h->d_warmup_output, false) &&
              enqueue_prach_demod_ci16(h, h->d_input_ci16, 1.0f, h->d_warmup_output, false) &&
              check_cuda(cudaStreamSynchronize(h->stream));
    h->stream = saved_stream;
    return ok;
}

} // namespace

int ocudu_lowphy_prach_rx_create(const ocudu_lowphy_prach_rx_config_t* config,
                                 ocudu_lowphy_prach_rx_handle_t** handle)
{
    if ((config == nullptr) || (handle == nullptr) || !validate_config(*config)) {
        return 0;
    }

    auto* h = new ocudu_lowphy_prach_rx_handle();
    h->cfg = *config;
    if (!check_cuda(create_realtime_stream(&h->owned_stream)) ||
        !check_cuda(cudaEventCreateWithFlags(&h->done_event, cudaEventDisableTiming))) {
        ocudu_lowphy_prach_rx_destroy(h);
        return 0;
    }
    h->stream = h->owned_stream;

    if (!ensure_buffers(h) || !ensure_plan(h) || !warmup_handle(h)) {
        ocudu_lowphy_prach_rx_destroy(h);
        return 0;
    }

    *handle = h;
    return 1;
}

void ocudu_lowphy_prach_rx_destroy(ocudu_lowphy_prach_rx_handle_t* handle)
{
    if (handle == nullptr) {
        return;
    }
    destroy_fft_plan(handle);
    release_buffers(handle);
    for (const auto& registration : handle->registered_hosts) {
        if (registration.unregister_on_destroy) {
            cudaHostUnregister(const_cast<void*>(registration.ptr));
        }
    }
    handle->registered_hosts.clear();
    handle->unregistered_hosts.clear();
    if (handle->done_event != nullptr) {
        cudaEventDestroy(handle->done_event);
        handle->done_event = nullptr;
    }
    if (handle->owned_stream != nullptr) {
        cudaStreamDestroy(handle->owned_stream);
        handle->owned_stream = nullptr;
    }
    delete handle;
}

int ocudu_lowphy_prach_rx_update_config(ocudu_lowphy_prach_rx_handle_t* handle,
                                        const ocudu_lowphy_prach_rx_config_t* config)
{
    if ((handle == nullptr) || (config == nullptr) || !validate_config(*config)) {
        return 0;
    }

    bool plan_changed = (handle->cfg.dft_size != config->dft_size) ||
                        (handle->cfg.nof_symbols != config->nof_symbols);
    handle->cfg = *config;
    if (plan_changed) {
        destroy_fft_plan(handle);
    }
    if (!ensure_buffers(handle) || !ensure_ci16_input_buffer(handle) || !ensure_plan(handle)) {
        return 0;
    }
    return !plan_changed || warmup_handle(handle);
}

int ocudu_lowphy_prach_rx_process(ocudu_lowphy_prach_rx_handle_t* handle,
                                  const void* input_cf32,
                                  void* output_cbf16,
                                  void* external_stream)
{
    if ((handle == nullptr) || (input_cf32 == nullptr) || (output_cbf16 == nullptr)) {
        return 0;
    }
    handle->stream = (external_stream != nullptr) ? static_cast<cudaStream_t>(external_stream) : handle->owned_stream;
    if (!ensure_buffers(handle) || !ensure_plan(handle)) {
        return 0;
    }

    const cuFloatComplex* d_input = static_cast<const cuFloatComplex*>(input_cf32);
    bool input_is_fft_payload = false;
    if (!handle->cfg.input_is_device) {
        size_t input_bytes = handle->input_count * sizeof(cuFloatComplex);
        bool direct_copy_enqueued = false;
        if (runtime_host_registration_enabled()) {
            direct_copy_enqueued =
                register_host_range(handle, input_cf32, input_bytes) &&
                check_cuda(cudaMemcpyAsync(handle->d_input, input_cf32, input_bytes, cudaMemcpyHostToDevice, handle->stream));
            if (!direct_copy_enqueued) {
                (void)cudaGetLastError();
            }
        }
        if (!direct_copy_enqueued) {
            int slot = 0;
            cuFloatComplex* staged = acquire_cf32_staging_slot(handle, slot);
            if (staged == nullptr) {
                return 0;
            }
            const auto* payload = static_cast<const cuFloatComplex*>(input_cf32) + handle->cfg.cyclic_prefix_length;
            size_t payload_bytes = handle->freq_count * sizeof(cuFloatComplex);
            std::memcpy(staged, payload, payload_bytes);
            if (!check_cuda(cudaMemcpyAsync(handle->d_input, staged, payload_bytes, cudaMemcpyHostToDevice, handle->stream)) ||
                !check_cuda(cudaEventRecord(handle->h_input_ready_events[slot], handle->stream))) {
                handle->h_input_event_recorded[slot] = false;
                return 0;
            }
            handle->h_input_event_recorded[slot] = true;
            input_is_fft_payload = true;
        }
        d_input = handle->d_input;
    }

    return enqueue_prach_demod(handle, d_input, output_cbf16, input_is_fft_payload) ? 1 : 0;
}

int ocudu_lowphy_prach_rx_process_ci16(ocudu_lowphy_prach_rx_handle_t* handle,
                                       const void* input_ci16,
                                       float input_scale,
                                       void* output_cbf16,
                                       void* external_stream)
{
    if ((handle == nullptr) || (input_ci16 == nullptr) || (output_cbf16 == nullptr)) {
        return 0;
    }
    handle->stream = (external_stream != nullptr) ? static_cast<cudaStream_t>(external_stream) : handle->owned_stream;
    if (!ensure_buffers(handle) || !ensure_plan(handle) || !ensure_ci16_input_buffer(handle)) {
        return 0;
    }

    const int16_t* d_input = static_cast<const int16_t*>(input_ci16);
    bool input_is_fft_payload = false;
    if (!handle->cfg.input_is_device) {
        size_t input_bytes = handle->input_ci16_count * sizeof(int16_t);
        bool direct_copy_enqueued = false;
        if (runtime_host_registration_enabled()) {
            direct_copy_enqueued =
                register_host_range(handle, input_ci16, input_bytes) &&
                check_cuda(cudaMemcpyAsync(handle->d_input_ci16, input_ci16, input_bytes, cudaMemcpyHostToDevice, handle->stream));
            if (!direct_copy_enqueued) {
                (void)cudaGetLastError();
            }
        }
        if (!direct_copy_enqueued) {
            int slot = 0;
            int16_t* staged = acquire_ci16_staging_slot(handle, slot);
            if (staged == nullptr) {
                return 0;
            }
            const auto* payload = static_cast<const int16_t*>(input_ci16) + static_cast<size_t>(handle->cfg.cyclic_prefix_length) * 2U;
            size_t payload_bytes = handle->freq_count * 2U * sizeof(int16_t);
            std::memcpy(staged, payload, payload_bytes);
            if (!check_cuda(cudaMemcpyAsync(handle->d_input_ci16, staged, payload_bytes, cudaMemcpyHostToDevice, handle->stream)) ||
                !check_cuda(cudaEventRecord(handle->h_input_ci16_ready_events[slot], handle->stream))) {
                handle->h_input_ci16_event_recorded[slot] = false;
                return 0;
            }
            handle->h_input_ci16_event_recorded[slot] = true;
            input_is_fft_payload = true;
        }
        d_input = handle->d_input_ci16;
    }

    return enqueue_prach_demod_ci16(handle, d_input, input_scale, output_cbf16, input_is_fft_payload) ? 1 : 0;
}

int ocudu_lowphy_prach_rx_synchronize(ocudu_lowphy_prach_rx_handle_t* handle)
{
    if ((handle == nullptr) || (handle->done_event == nullptr)) {
        return 0;
    }
    cudaError_t status;
    while ((status = cudaEventQuery(handle->done_event)) == cudaErrorNotReady) {
        sched_yield();
    }
    return status == cudaSuccess;
}

void* ocudu_lowphy_prach_rx_get_completion_event(ocudu_lowphy_prach_rx_handle_t* handle)
{
    return (handle == nullptr) ? nullptr : static_cast<void*>(handle->done_event);
}

void* ocudu_lowphy_prach_rx_get_stream(ocudu_lowphy_prach_rx_handle_t* handle)
{
    return (handle == nullptr) ? nullptr : static_cast<void*>(handle->stream ? handle->stream : handle->owned_stream);
}
