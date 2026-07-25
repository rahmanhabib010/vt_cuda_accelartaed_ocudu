// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "low_phy_puxch_rx.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <mutex>
#include <sched.h>
#include <vector>
#include "vkFFT.h"

#ifndef OCUDU_PHY_CUDA_ENABLE_VKFFT
#error "OCUDU PHY CUDA low-PHY PUxCH RX requires the bundled VkFFT backend."
#endif

namespace {

static constexpr int THREADS = 256;
static constexpr int HOST_STAGING_SLOTS = 32;
static constexpr size_t DEVICE_GRAPH_CACHE_LIMIT = 256;

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
    static const bool enabled = env_flag_enabled("OCUDU_LOWPHY_RX_RUNTIME_HOST_REGISTRATION");
    return enabled;
}

static bool cuda_graphs_enabled()
{
    static const bool enabled = env_flag_enabled("OCUDU_LOWPHY_RX_CUDA_GRAPHS");
    return enabled;
}

static size_t warmup_output_words(const ocudu_lowphy_puxch_rx_config_t& cfg)
{
    return std::max(static_cast<size_t>(cfg.nof_ports) * cfg.rg_size,
                    static_cast<size_t>(cfg.nof_ports) * cfg.grid_nof_symbols * cfg.grid_nof_subc);
}

static std::mutex& graph_capture_mutex()
{
    static std::mutex mutex;
    return mutex;
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

__device__ __forceinline__ cuFloatComplex cmul(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__global__ void load_puxch_fft_input_ci16_kernel(cuFloatComplex* __restrict__ d_freq,
                                                 const int16_t* __restrict__ d_input,
                                                 int dft_size,
                                                 int input_nof_samples,
                                                 int input_offset,
                                                 int nof_ports,
                                                 float input_gain)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dft_size * nof_ports;
    if (idx >= total) {
        return;
    }

    int re = idx % dft_size;
    int port = idx / dft_size;
    int src = port * input_nof_samples + input_offset + re;
    d_freq[idx] = make_cuFloatComplex(static_cast<float>(d_input[2 * src]) * input_gain,
                                      static_cast<float>(d_input[2 * src + 1]) * input_gain);
}

__global__ void init_window_phase_kernel(cuFloatComplex* __restrict__ d_window_phase, int dft_size, int window_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dft_size) {
        return;
    }

    float angle = static_cast<float>(window_offset) * 6.28318530717958647692f * static_cast<float>(idx) /
                  static_cast<float>(dft_size);
    float s;
    float c;
    sincosf(angle, &s, &c);
    d_window_phase[idx] = make_cuFloatComplex(c, s);
}

__global__ void extract_puxch_grid_kernel(uint32_t* __restrict__ output,
                                          const cuFloatComplex* __restrict__ d_freq,
                                          const cuFloatComplex* __restrict__ d_window_phase,
                                          ocudu_lowphy_puxch_rx_config_t cfg)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = cfg.nof_ports * cfg.rg_size;
    if (idx >= total) {
        return;
    }

    int k = idx % cfg.rg_size;
    int port_batch = idx / cfg.rg_size;
    int half_grid = cfg.rg_size / 2;
    int src = (k < half_grid) ? (cfg.dft_size - half_grid + k) : (k - half_grid);
    cuFloatComplex x = d_freq[port_batch * cfg.dft_size + src];

    cuFloatComplex phase = make_cuFloatComplex(cfg.phase_re * cfg.dft_scale, cfg.phase_im * cfg.dft_scale);
    x = cmul(x, phase);
    if (d_window_phase != nullptr) {
        x = cmul(x, d_window_phase[src]);
    }

    int output_port = cfg.port_indices[port_batch];
    size_t dst = (static_cast<size_t>(output_port) * cfg.grid_nof_symbols + cfg.symbol_index) *
                     cfg.grid_nof_subc +
                 k;
    output[dst] = cfloat_to_cbf16(x);
}

} // namespace

struct ocudu_lowphy_puxch_rx_handle {
    ocudu_lowphy_puxch_rx_config_t cfg{};
    cudaStream_t owned_stream = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t done_event = nullptr;

    int16_t* d_input_ci16 = nullptr;
    cuFloatComplex* d_freq = nullptr;
    cuFloatComplex* d_window_phase = nullptr;
    size_t input_ci16_count = 0;
    size_t input_ci16_capacity = 0;
    size_t window_ci16_count = 0;
    size_t freq_count = 0;
    size_t window_phase_count = 0;
    int window_phase_offset = -1;

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

    struct device_graph_entry {
        const int16_t* d_input = nullptr;
        void* output_cbf16 = nullptr;
        cudaStream_t stream = nullptr;
        float input_scale = 0.0f;
        bool input_is_fft_window = false;
        ocudu_lowphy_puxch_rx_config_t cfg{};
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaGraphNode_t extract_node = nullptr;
    };
    std::vector<device_graph_entry> device_graphs;
    size_t next_device_graph_replacement = 0;
    bool device_graph_capture_failed = false;

    struct host_registration {
        const void* ptr = nullptr;
        size_t bytes = 0;
        bool unregister_on_destroy = false;
    };
    std::vector<host_registration> registered_hosts;
    std::vector<host_registration> unregistered_hosts;
    bool done_event_recorded = false;
};

namespace {

static void destroy_device_graph_entry(ocudu_lowphy_puxch_rx_handle::device_graph_entry& entry)
{
    if (entry.exec != nullptr) {
        cudaGraphExecDestroy(entry.exec);
        entry.exec = nullptr;
    }
    if (entry.graph != nullptr) {
        cudaGraphDestroy(entry.graph);
        entry.graph = nullptr;
    }
    entry.d_input = nullptr;
    entry.output_cbf16 = nullptr;
    entry.stream = nullptr;
    entry.extract_node = nullptr;
}

static void destroy_device_graphs(ocudu_lowphy_puxch_rx_handle* h)
{
    if (h == nullptr) {
        return;
    }
    for (auto& entry : h->device_graphs) {
        destroy_device_graph_entry(entry);
    }
    h->device_graphs.clear();
    h->next_device_graph_replacement = 0;
    h->device_graph_capture_failed = false;
}

static void destroy_fft_plan(ocudu_lowphy_puxch_rx_handle* h)
{
    destroy_device_graphs(h);
    if (h->vkfft_valid) {
        deleteVkFFT(&h->vkfft_app);
        std::memset(&h->vkfft_app, 0, sizeof(h->vkfft_app));
        h->vkfft_valid = false;
    }
}

static void release_buffers(ocudu_lowphy_puxch_rx_handle* h)
{
    destroy_device_graphs(h);
    if (h->stream != nullptr) {
        (void)cudaStreamSynchronize(h->stream);
    }
    for (int slot = 0; slot != HOST_STAGING_SLOTS; ++slot) {
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
    if (h->d_input_ci16 != nullptr) {
        cudaFree(h->d_input_ci16);
        h->d_input_ci16 = nullptr;
    }
    if (h->d_freq != nullptr) {
        cudaFree(h->d_freq);
        h->d_freq = nullptr;
    }
    if (h->d_window_phase != nullptr) {
        cudaFree(h->d_window_phase);
        h->d_window_phase = nullptr;
    }
    if (h->d_warmup_output != nullptr) {
        cudaFree(h->d_warmup_output);
        h->d_warmup_output = nullptr;
    }
    h->input_ci16_count = 0;
    h->input_ci16_capacity = 0;
    h->window_ci16_count = 0;
    h->freq_count = 0;
    h->window_phase_count = 0;
    h->window_phase_offset = -1;
    h->warmup_output_count = 0;
    h->h_input_ci16_staging_count = 0;
    h->h_input_ci16_next_slot = 0;
}

static bool contains_registered_range(const std::vector<ocudu_lowphy_puxch_rx_handle::host_registration>& registrations,
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

static bool register_host_range(ocudu_lowphy_puxch_rx_handle* h, const void* ptr, size_t bytes)
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

static bool validate_config(const ocudu_lowphy_puxch_rx_config_t& cfg)
{
    if ((cfg.dft_size <= 0) || (cfg.rg_size <= 0) || (cfg.grid_nof_subc <= 0) || (cfg.nof_ports <= 0) ||
        (cfg.grid_nof_symbols <= 0) || (cfg.input_nof_samples <= 0) || (cfg.cyclic_prefix_length < 0) ||
        (cfg.window_offset < 0) || (cfg.symbol_index < 0) ||
        !std::isfinite(cfg.dft_scale) || (cfg.dft_scale <= 0.0f) || !std::isfinite(cfg.phase_re) ||
        !std::isfinite(cfg.phase_im)) {
        return false;
    }
    if ((cfg.rg_size > cfg.dft_size) || ((cfg.rg_size % 2) != 0) || (cfg.rg_size > cfg.grid_nof_subc) ||
        (cfg.symbol_index >= cfg.grid_nof_symbols) || (cfg.window_offset > cfg.cyclic_prefix_length) ||
        (cfg.nof_ports > OCUDU_LOWPHY_PUXCH_RX_MAX_PORTS)) {
        return false;
    }
    if (cfg.cyclic_prefix_length - cfg.window_offset + cfg.dft_size > cfg.input_nof_samples) {
        return false;
    }
    for (int port = 0; port != cfg.nof_ports; ++port) {
        if (cfg.port_indices[port] < 0) {
            return false;
        }
    }
    return true;
}

static bool ensure_buffers(ocudu_lowphy_puxch_rx_handle* h)
{
    size_t input_ci16_count = static_cast<size_t>(h->cfg.nof_ports) * h->cfg.input_nof_samples * 2U;
    size_t window_ci16_count = static_cast<size_t>(h->cfg.nof_ports) * h->cfg.dft_size * 2U;
    size_t input_samples_capacity =
        static_cast<size_t>(h->cfg.dft_size) +
        std::max(static_cast<size_t>(h->cfg.cyclic_prefix_length),
                 static_cast<size_t>(std::max(1, h->cfg.dft_size / 4)));
    size_t input_ci16_capacity = static_cast<size_t>(h->cfg.nof_ports) * input_samples_capacity * 2U;
    input_ci16_capacity = std::max(input_ci16_capacity, window_ci16_count);
    size_t freq_count = static_cast<size_t>(h->cfg.nof_ports) * h->cfg.dft_size;
    const size_t warmup_words = warmup_output_words(h->cfg);

    if ((h->input_ci16_capacity >= window_ci16_count) && (h->freq_count == freq_count) &&
        (h->d_input_ci16 != nullptr) && (h->d_freq != nullptr) &&
        (h->h_input_ci16_staging_count >= window_ci16_count) && (h->h_input_ci16_staging[0] != nullptr) &&
        (h->d_warmup_output != nullptr) && (h->warmup_output_count >= warmup_words)) {
        h->input_ci16_count = input_ci16_count;
        h->window_ci16_count = window_ci16_count;
        return true;
    }

    destroy_fft_plan(h);
    release_buffers(h);
    if (!check_cuda(cudaMalloc(&h->d_input_ci16, input_ci16_capacity * sizeof(int16_t))) ||
        !check_cuda(cudaMalloc(&h->d_freq, freq_count * sizeof(cuFloatComplex))) ||
        !check_cuda(cudaMalloc(&h->d_warmup_output, warmup_words * sizeof(uint32_t)))) {
        release_buffers(h);
        return false;
    }
    for (int slot = 0; slot != HOST_STAGING_SLOTS; ++slot) {
        if (!check_cuda(cudaHostAlloc(&h->h_input_ci16_staging[slot],
                                      input_ci16_capacity * sizeof(int16_t),
                                      cudaHostAllocDefault)) ||
            !check_cuda(cudaEventCreateWithFlags(&h->h_input_ci16_ready_events[slot], cudaEventDisableTiming))) {
            release_buffers(h);
            return false;
        }
    }
    h->input_ci16_count = input_ci16_count;
    h->input_ci16_capacity = input_ci16_capacity;
    h->window_ci16_count = window_ci16_count;
    h->freq_count = freq_count;
    h->h_input_ci16_staging_count = input_ci16_capacity;
    h->warmup_output_count = warmup_words;
    return true;
}

static bool ensure_window_phase(ocudu_lowphy_puxch_rx_handle* h)
{
    if (h->cfg.window_offset == 0) {
        if (h->d_window_phase != nullptr) {
            cudaFree(h->d_window_phase);
            h->d_window_phase = nullptr;
        }
        h->window_phase_count = 0;
        h->window_phase_offset = 0;
        return true;
    }

    size_t phase_count = static_cast<size_t>(h->cfg.dft_size);
    if ((h->d_window_phase == nullptr) || (h->window_phase_count != phase_count)) {
        if (h->d_window_phase != nullptr) {
            cudaFree(h->d_window_phase);
            h->d_window_phase = nullptr;
        }
        if (!check_cuda(cudaMalloc(&h->d_window_phase, phase_count * sizeof(cuFloatComplex)))) {
            h->window_phase_count = 0;
            h->window_phase_offset = -1;
            return false;
        }
        h->window_phase_count = phase_count;
        h->window_phase_offset = -1;
    }

    if (h->window_phase_offset != h->cfg.window_offset) {
        int blocks = (h->cfg.dft_size + THREADS - 1) / THREADS;
        init_window_phase_kernel<<<blocks, THREADS, 0, h->stream>>>(
            h->d_window_phase, h->cfg.dft_size, h->cfg.window_offset);
        if (!check_cuda(cudaGetLastError())) {
            h->window_phase_offset = -1;
            return false;
        }
        h->window_phase_offset = h->cfg.window_offset;
    }
    return true;
}

static bool ensure_plan(ocudu_lowphy_puxch_rx_handle* h)
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
    configuration.numberBatches = h->cfg.nof_ports;
    configuration.bufferSize = &h->vkfft_buffer_size;
    configuration.buffer = &h->vkfft_buffer;
    configuration.makeForwardPlanOnly = 1;
    configuration.normalize = 0;
    configuration.disableReorderFourStep = 1;
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

static int16_t* acquire_ci16_staging_slot(ocudu_lowphy_puxch_rx_handle* h, int* acquired_slot)
{
    int slot = h->h_input_ci16_next_slot;
    h->h_input_ci16_next_slot = (h->h_input_ci16_next_slot + 1) % HOST_STAGING_SLOTS;
    if (acquired_slot != nullptr) {
        *acquired_slot = slot;
    }

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

static bool record_staging_slot_ready_event(ocudu_lowphy_puxch_rx_handle* h, int event_slot)
{
    if ((event_slot < 0) || (event_slot >= HOST_STAGING_SLOTS) ||
        !check_cuda(cudaEventRecord(h->h_input_ci16_ready_events[event_slot], h->stream))) {
        h->h_input_ci16_event_recorded[event_slot] = false;
        return false;
    }
    h->h_input_ci16_event_recorded[event_slot] = true;
    return true;
}

static bool
copy_staged_ci16_to_device(ocudu_lowphy_puxch_rx_handle* h, const int16_t* staged, size_t samples_i16, int event_slot)
{
    if (!check_cuda(cudaMemcpyAsync(
            h->d_input_ci16, staged, samples_i16 * sizeof(int16_t), cudaMemcpyHostToDevice, h->stream))) {
        return false;
    }
    return record_staging_slot_ready_event(h, event_slot);
}

static bool copy_registered_ci16_windows_to_device(ocudu_lowphy_puxch_rx_handle* h, const void* const* input_ports_ci16)
{
    const unsigned input_offset = h->cfg.cyclic_prefix_length - h->cfg.window_offset;
    const size_t window_i16_per_port = static_cast<size_t>(h->cfg.dft_size) * 2U;
    const size_t window_bytes_per_port = window_i16_per_port * sizeof(int16_t);

    for (int port = 0; port != h->cfg.nof_ports; ++port) {
        const auto* src = static_cast<const int16_t*>(input_ports_ci16[port]) + static_cast<size_t>(input_offset) * 2U;
        if (!register_host_range(h, src, window_bytes_per_port)) {
            return false;
        }
    }

    for (int port = 0; port != h->cfg.nof_ports; ++port) {
        const auto* src = static_cast<const int16_t*>(input_ports_ci16[port]) + static_cast<size_t>(input_offset) * 2U;
        int16_t* dst = h->d_input_ci16 + static_cast<size_t>(port) * window_i16_per_port;
        if (!check_cuda(cudaMemcpyAsync(dst, src, window_bytes_per_port, cudaMemcpyHostToDevice, h->stream))) {
            return false;
        }
    }
    return true;
}

static void copy_puxch_fft_window_to_staging(ocudu_lowphy_puxch_rx_handle* h,
                                             int16_t* staged,
                                             const void* const* input_ports_ci16)
{
    const unsigned input_offset = h->cfg.cyclic_prefix_length - h->cfg.window_offset;
    const size_t window_i16_per_port = static_cast<size_t>(h->cfg.dft_size) * 2U;
    const size_t window_bytes_per_port = window_i16_per_port * sizeof(int16_t);
    for (int port = 0; port != h->cfg.nof_ports; ++port) {
        const auto* src = static_cast<const int16_t*>(input_ports_ci16[port]) + static_cast<size_t>(input_offset) * 2U;
        std::memcpy(staged + static_cast<size_t>(port) * window_i16_per_port, src, window_bytes_per_port);
    }
}

static bool
enqueue_puxch_demod_ci16_uncaptured(ocudu_lowphy_puxch_rx_handle* h,
                                    const int16_t* d_input,
                                    float input_scale,
                                    void* output_cbf16,
                                    bool input_is_fft_window)
{
    if (!std::isfinite(input_scale) || (input_scale <= 0.0f)) {
        return false;
    }

    int load_blocks = (h->cfg.nof_ports * h->cfg.dft_size + THREADS - 1) / THREADS;
    int input_offset = input_is_fft_window ? 0 : (h->cfg.cyclic_prefix_length - h->cfg.window_offset);
    int input_stride = input_is_fft_window ? h->cfg.dft_size : h->cfg.input_nof_samples;
    load_puxch_fft_input_ci16_kernel<<<load_blocks, THREADS, 0, h->stream>>>(
        h->d_freq,
        d_input,
        h->cfg.dft_size,
        input_stride,
        input_offset,
        h->cfg.nof_ports,
        1.0f / input_scale);
    if (!check_cuda(cudaGetLastError())) {
        return false;
    }

    VkFFTLaunchParams launch_params = {};
    launch_params.buffer = &h->vkfft_buffer;
    if (VkFFTAppend(&h->vkfft_app, -1, &launch_params) != VKFFT_SUCCESS) {
        return false;
    }

    int extract_blocks = (h->cfg.nof_ports * h->cfg.rg_size + THREADS - 1) / THREADS;
    extract_puxch_grid_kernel<<<extract_blocks, THREADS, 0, h->stream>>>(
        static_cast<uint32_t*>(output_cbf16), h->d_freq, h->d_window_phase, h->cfg);
    if (!check_cuda(cudaGetLastError())) {
        return false;
    }

    return true;
}

static bool graph_configs_match(const ocudu_lowphy_puxch_rx_config_t& lhs,
                                const ocudu_lowphy_puxch_rx_config_t& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(lhs)) == 0;
}

static ocudu_lowphy_puxch_rx_handle::device_graph_entry*
find_device_graph(ocudu_lowphy_puxch_rx_handle* h,
                  const int16_t* d_input,
                  float input_scale,
                  bool input_is_fft_window)
{
    for (auto& entry : h->device_graphs) {
        // Resource grids rotate in the live receiver. Keep the graph keyed on the invariant demodulation parameters
        // and patch the extract kernel output pointer before each launch.
        if ((entry.d_input == d_input) && (entry.stream == h->stream) && (entry.input_scale == input_scale) &&
            (entry.input_is_fft_window == input_is_fft_window) && graph_configs_match(entry.cfg, h->cfg) &&
            (entry.exec != nullptr) && (entry.extract_node != nullptr)) {
            return &entry;
        }
    }
    return nullptr;
}

static cudaGraphNode_t find_extract_kernel_node(cudaGraph_t graph)
{
    size_t nof_nodes = 0;
    if ((graph == nullptr) || (cudaGraphGetNodes(graph, nullptr, &nof_nodes) != cudaSuccess) || (nof_nodes == 0)) {
        (void)cudaGetLastError();
        return nullptr;
    }

    std::vector<cudaGraphNode_t> nodes(nof_nodes);
    if (cudaGraphGetNodes(graph, nodes.data(), &nof_nodes) != cudaSuccess) {
        (void)cudaGetLastError();
        return nullptr;
    }

    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type;
        if ((cudaGraphNodeGetType(node, &type) != cudaSuccess) || (type != cudaGraphNodeTypeKernel)) {
            continue;
        }

        cudaKernelNodeParams params = {};
        if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) {
            continue;
        }
        if (params.func == reinterpret_cast<void*>(extract_puxch_grid_kernel)) {
            return node;
        }
    }

    return nullptr;
}

static bool update_graph_extract_output(ocudu_lowphy_puxch_rx_handle::device_graph_entry& entry,
                                        ocudu_lowphy_puxch_rx_handle*                     h,
                                        void*                                             output_cbf16)
{
    if ((entry.exec == nullptr) || (entry.extract_node == nullptr) || (output_cbf16 == nullptr)) {
        return false;
    }

    uint32_t* output = static_cast<uint32_t*>(output_cbf16);
    const cuFloatComplex* d_freq = h->d_freq;
    const cuFloatComplex* d_window_phase = h->d_window_phase;
    ocudu_lowphy_puxch_rx_config_t cfg = h->cfg;
    void* args[] = {&output, &d_freq, &d_window_phase, &cfg};

    cudaKernelNodeParams params = {};
    params.func = reinterpret_cast<void*>(extract_puxch_grid_kernel);
    params.gridDim = dim3((h->cfg.nof_ports * h->cfg.rg_size + THREADS - 1) / THREADS);
    params.blockDim = dim3(THREADS);
    params.sharedMemBytes = 0;
    params.kernelParams = args;
    params.extra = nullptr;

    if (cudaGraphExecKernelNodeSetParams(entry.exec, entry.extract_node, &params) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }

    entry.output_cbf16 = output_cbf16;
    return true;
}

static bool launch_device_graph(ocudu_lowphy_puxch_rx_handle::device_graph_entry& entry,
                                ocudu_lowphy_puxch_rx_handle*                     h,
                                void*                                             output_cbf16)
{
    if (!update_graph_extract_output(entry, h, output_cbf16)) {
        return false;
    }
    return cudaGraphLaunch(entry.exec, h->stream) == cudaSuccess;
}

static ocudu_lowphy_puxch_rx_handle::device_graph_entry*
reserve_device_graph_slot(ocudu_lowphy_puxch_rx_handle* h)
{
    if (h->device_graphs.size() < DEVICE_GRAPH_CACHE_LIMIT) {
        h->device_graphs.push_back({});
        return &h->device_graphs.back();
    }

    auto& entry = h->device_graphs[h->next_device_graph_replacement % h->device_graphs.size()];
    h->next_device_graph_replacement = (h->next_device_graph_replacement + 1) % h->device_graphs.size();
    destroy_device_graph_entry(entry);
    return &entry;
}

static bool capture_device_graph(ocudu_lowphy_puxch_rx_handle* h,
                                 const int16_t* d_input,
                                 float input_scale,
                                 void* output_cbf16,
                                 bool input_is_fft_window)
{
    cudaGraph_t captured_graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    if (cudaStreamBeginCapture(h->stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        (void)cudaGetLastError();
        h->device_graph_capture_failed = true;
        return false;
    }

    bool enqueue_ok = enqueue_puxch_demod_ci16_uncaptured(h, d_input, input_scale, output_cbf16, input_is_fft_window);
    cudaError_t end_status = cudaStreamEndCapture(h->stream, &captured_graph);
    if (!enqueue_ok || (end_status != cudaSuccess) || (captured_graph == nullptr)) {
        if (captured_graph != nullptr) {
            cudaGraphDestroy(captured_graph);
        }
        (void)cudaGetLastError();
        h->device_graph_capture_failed = true;
        return false;
    }

    if (cudaGraphInstantiate(&graph_exec, captured_graph, nullptr, nullptr, 0) != cudaSuccess) {
        cudaGraphDestroy(captured_graph);
        (void)cudaGetLastError();
        h->device_graph_capture_failed = true;
        return false;
    }

    cudaGraphNode_t extract_node = find_extract_kernel_node(captured_graph);
    if (extract_node == nullptr) {
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(captured_graph);
        h->device_graph_capture_failed = true;
        return false;
    }

    auto* entry = reserve_device_graph_slot(h);
    entry->d_input = d_input;
    entry->output_cbf16 = output_cbf16;
    entry->stream = h->stream;
    entry->input_scale = input_scale;
    entry->input_is_fft_window = input_is_fft_window;
    entry->cfg = h->cfg;
    entry->graph = captured_graph;
    entry->exec = graph_exec;
    entry->extract_node = extract_node;
    return true;
}

static bool
enqueue_puxch_demod_ci16(ocudu_lowphy_puxch_rx_handle* h,
                         const int16_t* d_input,
                         float input_scale,
                         void* output_cbf16,
                         bool input_is_fft_window,
                         bool record_completion_event)
{
    bool enqueued = false;
    h->done_event_recorded = false;

    if (!cuda_graphs_enabled() || h->device_graph_capture_failed) {
        enqueued = enqueue_puxch_demod_ci16_uncaptured(h, d_input, input_scale, output_cbf16, input_is_fft_window);
    } else if (auto* entry = find_device_graph(h, d_input, input_scale, input_is_fft_window); entry != nullptr) {
        enqueued = launch_device_graph(*entry, h, output_cbf16);
        if (!enqueued) {
            destroy_device_graph_entry(*entry);
            (void)cudaGetLastError();
        }
    }

    if (!enqueued && cuda_graphs_enabled() && !h->device_graph_capture_failed) {
        std::lock_guard<std::mutex> lock(graph_capture_mutex());
        if (auto* entry = find_device_graph(h, d_input, input_scale, input_is_fft_window); entry != nullptr) {
            enqueued = launch_device_graph(*entry, h, output_cbf16);
            if (!enqueued) {
                destroy_device_graph_entry(*entry);
                (void)cudaGetLastError();
            }
        }
        if (!enqueued && capture_device_graph(h, d_input, input_scale, output_cbf16, input_is_fft_window)) {
            if (auto* entry = find_device_graph(h, d_input, input_scale, input_is_fft_window); entry != nullptr) {
                enqueued = launch_device_graph(*entry, h, output_cbf16);
                if (!enqueued) {
                    destroy_device_graph_entry(*entry);
                    (void)cudaGetLastError();
                }
            }
        }
    }

    if (!enqueued) {
        enqueued = enqueue_puxch_demod_ci16_uncaptured(h, d_input, input_scale, output_cbf16, input_is_fft_window);
    }

    if (!enqueued) {
        return false;
    }
    if (!record_completion_event) {
        return true;
    }
    if (!check_cuda(cudaEventRecord(h->done_event, h->stream))) {
        h->done_event_recorded = false;
        return false;
    }
    h->done_event_recorded = true;
    return true;
}

static bool warmup_handle(ocudu_lowphy_puxch_rx_handle* h)
{
    if ((h == nullptr) || (h->d_input_ci16 == nullptr) || (h->d_warmup_output == nullptr)) {
        return false;
    }

    cudaStream_t saved_stream = h->stream;
    h->stream = (h->owned_stream != nullptr) ? h->owned_stream : h->stream;
    if ((h->stream == nullptr) || !ensure_buffers(h) || !ensure_plan(h) || !ensure_window_phase(h)) {
        h->stream = saved_stream;
        return false;
    }

    if (!check_cuda(cudaMemsetAsync(h->d_input_ci16, 0, h->window_ci16_count * sizeof(int16_t), h->stream))) {
        h->stream = saved_stream;
        return false;
    }

    ocudu_lowphy_puxch_rx_config_t saved_cfg = h->cfg;
    h->cfg.grid_nof_subc = h->cfg.rg_size;
    h->cfg.grid_nof_symbols = 1;
    h->cfg.symbol_index = 0;
    for (int port = 0; port != h->cfg.nof_ports; ++port) {
        h->cfg.port_indices[port] = port;
    }
    bool ok = enqueue_puxch_demod_ci16_uncaptured(h, h->d_input_ci16, 1.0f, h->d_warmup_output, true) &&
              check_cuda(cudaStreamSynchronize(h->stream));
    h->cfg = saved_cfg;
    h->stream = saved_stream;
    return ok;
}

static bool prime_current_config(ocudu_lowphy_puxch_rx_handle* h, float input_scale, cudaStream_t stream)
{
    if ((h == nullptr) || (h->d_input_ci16 == nullptr) || (h->d_warmup_output == nullptr) ||
        !std::isfinite(input_scale) || (input_scale <= 0.0f)) {
        return false;
    }

    cudaStream_t saved_stream = h->stream;
    h->stream = (stream != nullptr) ? stream : ((h->owned_stream != nullptr) ? h->owned_stream : h->stream);
    if ((h->stream == nullptr) || !ensure_buffers(h) || !ensure_plan(h) || !ensure_window_phase(h)) {
        h->stream = saved_stream;
        return false;
    }

    if (!check_cuda(cudaMemsetAsync(h->d_input_ci16, 0, h->window_ci16_count * sizeof(int16_t), h->stream))) {
        h->stream = saved_stream;
        return false;
    }

    const bool ok = enqueue_puxch_demod_ci16(h, h->d_input_ci16, input_scale, h->d_warmup_output, true, false) &&
                    check_cuda(cudaStreamSynchronize(h->stream));
    h->stream = saved_stream;
    return ok;
}

} // namespace

int ocudu_lowphy_puxch_rx_create(const ocudu_lowphy_puxch_rx_config_t* config,
                                 ocudu_lowphy_puxch_rx_handle_t** handle)
{
    if ((config == nullptr) || (handle == nullptr) || !validate_config(*config)) {
        return 0;
    }

    auto* h = new ocudu_lowphy_puxch_rx_handle();
    h->cfg = *config;
    if (!check_cuda(create_realtime_stream(&h->owned_stream)) ||
        !check_cuda(cudaEventCreateWithFlags(&h->done_event, cudaEventDisableTiming))) {
        ocudu_lowphy_puxch_rx_destroy(h);
        return 0;
    }
    h->stream = h->owned_stream;

    if (!ensure_buffers(h) || !ensure_plan(h) || !ensure_window_phase(h) || !warmup_handle(h)) {
        ocudu_lowphy_puxch_rx_destroy(h);
        return 0;
    }

    *handle = h;
    return 1;
}

void ocudu_lowphy_puxch_rx_destroy(ocudu_lowphy_puxch_rx_handle_t* handle)
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

int ocudu_lowphy_puxch_rx_update_config(ocudu_lowphy_puxch_rx_handle_t* handle,
                                        const ocudu_lowphy_puxch_rx_config_t* config)
{
    if ((handle == nullptr) || (config == nullptr) || !validate_config(*config)) {
        return 0;
    }

    bool plan_changed = (handle->cfg.dft_size != config->dft_size) || (handle->cfg.nof_ports != config->nof_ports);
    handle->cfg = *config;
    if (plan_changed) {
        destroy_fft_plan(handle);
    }
    if (!ensure_buffers(handle) || !ensure_plan(handle) || !ensure_window_phase(handle)) {
        return 0;
    }
    return !plan_changed || warmup_handle(handle);
}

int ocudu_lowphy_puxch_rx_prime_config(ocudu_lowphy_puxch_rx_handle_t* handle,
                                       const ocudu_lowphy_puxch_rx_config_t* config,
                                       float input_scale,
                                       void* external_stream)
{
    if ((handle == nullptr) || (config == nullptr) || !validate_config(*config)) {
        return 0;
    }

    bool plan_changed = (handle->cfg.dft_size != config->dft_size) || (handle->cfg.nof_ports != config->nof_ports);
    handle->cfg = *config;
    if (plan_changed) {
        destroy_fft_plan(handle);
    }
    if (!ensure_buffers(handle) || !ensure_plan(handle) || !ensure_window_phase(handle)) {
        return 0;
    }

    return prime_current_config(handle, input_scale, static_cast<cudaStream_t>(external_stream)) ? 1 : 0;
}

int ocudu_lowphy_puxch_rx_process_ci16(ocudu_lowphy_puxch_rx_handle_t* handle,
                                       const void* input_ci16,
                                       float input_scale,
                                       void* output_cbf16,
                                       void* external_stream)
{
    if ((handle == nullptr) || (input_ci16 == nullptr) || (output_cbf16 == nullptr) || (handle->cfg.nof_ports != 1)) {
        return 0;
    }
    const bool record_completion_event = (external_stream == nullptr);
    handle->stream = (external_stream != nullptr) ? static_cast<cudaStream_t>(external_stream) : handle->owned_stream;
    if (!ensure_buffers(handle) || !ensure_plan(handle) || !ensure_window_phase(handle)) {
        return 0;
    }

    const int16_t* d_input = static_cast<const int16_t*>(input_ci16);
    bool input_is_fft_window = false;
    if (!handle->cfg.input_is_device) {
        const void* input_ports[1] = {input_ci16};
        bool direct_copy_enqueued = false;
        if (runtime_host_registration_enabled()) {
            direct_copy_enqueued = copy_registered_ci16_windows_to_device(handle, input_ports);
            if (!direct_copy_enqueued) {
                (void)cudaGetLastError();
            }
        }
        if (!direct_copy_enqueued) {
            int      staging_slot = -1;
            int16_t* staged       = acquire_ci16_staging_slot(handle, &staging_slot);
            if (staged == nullptr) {
                return 0;
            }
            copy_puxch_fft_window_to_staging(handle, staged, input_ports);
            if (!copy_staged_ci16_to_device(handle, staged, handle->window_ci16_count, staging_slot)) {
                return 0;
            }
        }
        d_input = handle->d_input_ci16;
        input_is_fft_window = true;
    }

    return enqueue_puxch_demod_ci16(
               handle, d_input, input_scale, output_cbf16, input_is_fft_window, record_completion_event)
               ? 1
               : 0;
}

int ocudu_lowphy_puxch_rx_process_ci16_ports(ocudu_lowphy_puxch_rx_handle_t* handle,
                                             const void* const* input_ports_ci16,
                                             float input_scale,
                                             void* output_cbf16,
                                             void* external_stream)
{
    if ((handle == nullptr) || (input_ports_ci16 == nullptr) || (output_cbf16 == nullptr)) {
        return 0;
    }
    const bool record_completion_event = (external_stream == nullptr);
    handle->stream = (external_stream != nullptr) ? static_cast<cudaStream_t>(external_stream) : handle->owned_stream;
    if (!ensure_buffers(handle) || !ensure_plan(handle) || !ensure_window_phase(handle)) {
        return 0;
    }

    if (handle->cfg.input_is_device) {
        return 0;
    }

    for (int port = 0; port != handle->cfg.nof_ports; ++port) {
        if (input_ports_ci16[port] == nullptr) {
            return 0;
        }
    }
    bool direct_copy_enqueued = false;
    if (runtime_host_registration_enabled()) {
        direct_copy_enqueued = copy_registered_ci16_windows_to_device(handle, input_ports_ci16);
        if (!direct_copy_enqueued) {
            (void)cudaGetLastError();
        }
    }
    if (!direct_copy_enqueued) {
        int      staging_slot = -1;
        int16_t* staged       = acquire_ci16_staging_slot(handle, &staging_slot);
        if (staged == nullptr) {
            return 0;
        }
        copy_puxch_fft_window_to_staging(handle, staged, input_ports_ci16);
        if (!copy_staged_ci16_to_device(handle, staged, handle->window_ci16_count, staging_slot)) {
            return 0;
        }
    }

    return enqueue_puxch_demod_ci16(
               handle, handle->d_input_ci16, input_scale, output_cbf16, true, record_completion_event)
               ? 1
               : 0;
}

int ocudu_lowphy_puxch_rx_synchronize(ocudu_lowphy_puxch_rx_handle_t* handle)
{
    if ((handle == nullptr) || (handle->done_event == nullptr)) {
        return 0;
    }
    if (!handle->done_event_recorded) {
        return cudaStreamSynchronize(handle->stream ? handle->stream : handle->owned_stream) == cudaSuccess;
    }
    cudaError_t status;
    while ((status = cudaEventQuery(handle->done_event)) == cudaErrorNotReady) {
        sched_yield();
    }
    return status == cudaSuccess;
}

void* ocudu_lowphy_puxch_rx_get_completion_event(ocudu_lowphy_puxch_rx_handle_t* handle)
{
    return (handle == nullptr) ? nullptr : static_cast<void*>(handle->done_event);
}

void* ocudu_lowphy_puxch_rx_get_stream(ocudu_lowphy_puxch_rx_handle_t* handle)
{
    return (handle == nullptr) ? nullptr : static_cast<void*>(handle->stream ? handle->stream : handle->owned_stream);
}
