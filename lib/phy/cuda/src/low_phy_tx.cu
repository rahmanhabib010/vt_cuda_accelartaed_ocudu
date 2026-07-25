// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "low_phy_tx.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda.h>
#include <stdint.h>
#include <sched.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include "vkFFT.h"

#ifndef OCUDU_PHY_CUDA_ENABLE_VKFFT
#error "OCUDU PHY CUDA low-PHY TX requires the bundled VkFFT backend."
#endif

namespace {

static constexpr int THREADS = 256;
static constexpr int HOST_GRID_STAGING_SLOTS = 8;
static constexpr size_t DEVICE_GRAPH_CACHE_LIMIT = 64;

struct tx_postprocess_config {
    int    cp_lengths[OCUDU_LOWPHY_TX_MAX_SYMBOLS];
    int    symbol_offsets[OCUDU_LOWPHY_TX_MAX_SYMBOLS];
    float2 phases[OCUDU_LOWPHY_TX_MAX_SYMBOLS];
    int    dft_size;
    int    nof_symbols;
    int    nof_samples;
    float  ofdm_scale;
    float  amplitude_gain;
    float  clipping_ceiling;
    int    clipping_enabled;
};

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
    static const bool enabled = env_flag_enabled("OCUDU_LOWPHY_TX_RUNTIME_HOST_REGISTRATION");
    return enabled;
}

static bool cuda_graphs_enabled()
{
    static const bool enabled = env_flag_enabled("OCUDU_LOWPHY_TX_CUDA_GRAPHS");
    return enabled;
}

static std::mutex& graph_capture_mutex()
{
    static std::mutex mutex;
    return mutex;
}

__device__ __forceinline__ float bf16_to_float(uint16_t value)
{
    return __uint_as_float(static_cast<unsigned>(value) << 16);
}

__device__ __forceinline__ cuFloatComplex cbf16_to_cu(uint32_t packed)
{
    return make_cuFloatComplex(bf16_to_float(static_cast<uint16_t>(packed & 0xffffU)),
                               bf16_to_float(static_cast<uint16_t>(packed >> 16)));
}

__device__ __forceinline__ int16_t quantize_i16(float value)
{
    value = fminf(fmaxf(value, -32768.0f), 32767.0f);
    return static_cast<int16_t>(lrintf(value));
}

__global__ void prepare_ifft_input_kernel(cuFloatComplex* __restrict__ d_freq,
                                          const uint32_t* __restrict__ d_grid,
                                          int                         dft_size,
                                          int                         rg_size,
                                          int                         nof_symbols,
                                          int                         nof_subc,
                                          int                         nof_batches)
{
    int idx       = blockIdx.x * blockDim.x + threadIdx.x;
    int total_res = nof_batches * dft_size;
    if (idx >= total_res) {
        return;
    }

    int freq_k = idx % dft_size;
    int batch  = idx / dft_size;
    int port   = batch / nof_symbols;
    int sym    = batch - port * nof_symbols;

    int half_grid = rg_size / 2;
    int grid_k    = -1;
    if (freq_k >= dft_size - half_grid) {
        grid_k = freq_k - (dft_size - half_grid);
    } else if (freq_k < half_grid) {
        grid_k = freq_k + half_grid;
    }

    cuFloatComplex value = make_cuFloatComplex(0.0F, 0.0F);
    if (grid_k >= 0) {
        int grid_idx = (port * nof_symbols + sym) * nof_subc + grid_k;
        value        = cbf16_to_cu(d_grid[grid_idx]);
    }
    d_freq[batch * dft_size + freq_k] = value;
}

__global__ void postprocess_ifft_to_sc16_kernel(int16_t* __restrict__ d_output_sc16,
                                                const cuFloatComplex* __restrict__ d_time,
                                                tx_postprocess_config              cfg)
{
    int sample_sym = blockIdx.x * blockDim.x + threadIdx.x;
    int batch      = blockIdx.y;
    int port       = batch / cfg.nof_symbols;
    int symbol     = batch - port * cfg.nof_symbols;

    int symbol_offset = cfg.symbol_offsets[symbol];
    int next_offset   = (symbol + 1 < cfg.nof_symbols) ? cfg.symbol_offsets[symbol + 1] : cfg.nof_samples;
    int symbol_size   = next_offset - symbol_offset;
    if (sample_sym >= symbol_size) {
        return;
    }

    int cp_len         = cfg.cp_lengths[symbol];
    int src_idx        = (sample_sym < cp_len) ? (cfg.dft_size - cp_len + sample_sym) : (sample_sym - cp_len);
    int sample_in_port = symbol_offset + sample_sym;

    cuFloatComplex x = d_time[batch * cfg.dft_size + src_idx];
    float2 phase     = cfg.phases[symbol];

    float re = (x.x * phase.x - x.y * phase.y) * cfg.ofdm_scale * cfg.amplitude_gain;
    float im = (x.x * phase.y + x.y * phase.x) * cfg.ofdm_scale * cfg.amplitude_gain;

    if (cfg.clipping_enabled) {
        float mag2 = re * re + im * im;
        float lim2 = cfg.clipping_ceiling * cfg.clipping_ceiling;
        if ((mag2 > lim2) && (mag2 > 0.0f)) {
            float s = cfg.clipping_ceiling * rsqrtf(mag2);
            re *= s;
            im *= s;
        }
    }

    int out_idx              = (port * cfg.nof_samples + sample_in_port) * 2;
    d_output_sc16[out_idx]   = quantize_i16(re * 32767.0f);
    d_output_sc16[out_idx + 1] = quantize_i16(im * 32767.0f);
}

} // namespace

struct ocudu_lowphy_tx_handle {
    ocudu_lowphy_tx_config_t cfg{};
    cudaStream_t             owned_stream = nullptr;
    cudaStream_t             stream       = nullptr;
    cudaEvent_t              done_event   = nullptr;

    cuFloatComplex* d_freq       = nullptr;
    uint32_t*       d_grid_stage = nullptr;
    int16_t*        d_output     = nullptr;
    size_t          d_freq_count = 0;
    size_t          d_grid_count = 0;
    size_t          d_output_i16 = 0;

    uint32_t*   h_grid_staging[HOST_GRID_STAGING_SLOTS] = {};
    cudaEvent_t h_grid_ready_events[HOST_GRID_STAGING_SLOTS] = {};
    bool        h_grid_event_recorded[HOST_GRID_STAGING_SLOTS] = {};
    size_t      h_grid_staging_count = 0;
    int         h_grid_next_slot = 0;

    int16_t* h_output_staging = nullptr;
    size_t   h_output_staging_i16 = 0;
    void*    pending_output_ports[OCUDU_LOWPHY_TX_MAX_PORTS] = {};
    size_t   pending_output_port_bytes = 0;
    int      pending_output_nof_ports = 0;
    bool     pending_output_ports_are_contiguous = false;
    bool     pending_output_valid = false;
    bool     done_event_recorded = false;

    struct host_registration {
        const void* ptr = nullptr;
        size_t      bytes = 0;
        bool        owned = false;
    };
    std::vector<host_registration> registered_hosts;
    std::vector<host_registration> unregistered_hosts;

    struct device_graph_entry {
        const void*     d_grid = nullptr;
        cudaStream_t    stream = nullptr;
        ocudu_lowphy_tx_config_t cfg{};
        cudaGraph_t     graph = nullptr;
        cudaGraphExec_t exec = nullptr;
    };
    std::vector<device_graph_entry> device_graphs;
    size_t                          next_device_graph_replacement = 0;
    bool                            device_graph_capture_failed = false;

    VkFFTApplication vkfft_app{};
    CUdevice         vkfft_device = 0;
    cudaStream_t     vkfft_stream = nullptr;
    void*            vkfft_buffer = nullptr;
    pfUINT           vkfft_buffer_size = 0;
    bool             vkfft_valid = false;
};

namespace {

static bool wait_event_yielding(cudaEvent_t event)
{
    if (event == nullptr) {
        return false;
    }
    cudaError_t status;
    while ((status = cudaEventQuery(event)) == cudaErrorNotReady) {
        sched_yield();
    }
    return status == cudaSuccess;
}

static bool flush_pending_output(ocudu_lowphy_tx_handle* h, bool wait_for_device)
{
    if (h == nullptr) {
        return true;
    }

    if (!h->pending_output_valid) {
        if (wait_for_device && h->done_event_recorded && !wait_event_yielding(h->done_event)) {
            return false;
        }
        h->done_event_recorded = false;
        return true;
    }
    if (wait_for_device && (!h->done_event_recorded || !wait_event_yielding(h->done_event))) {
        return false;
    }
    if ((h->h_output_staging == nullptr) || (h->pending_output_nof_ports <= 0) ||
        (h->pending_output_port_bytes == 0)) {
        h->pending_output_valid = false;
        h->done_event_recorded = false;
        return false;
    }

    if (h->pending_output_ports_are_contiguous) {
        if (h->pending_output_ports[0] == nullptr) {
            h->pending_output_valid = false;
            h->done_event_recorded = false;
            return false;
        }
        std::memcpy(h->pending_output_ports[0],
                    h->h_output_staging,
                    h->pending_output_port_bytes * static_cast<size_t>(h->pending_output_nof_ports));
    } else {
        size_t port_i16 = h->pending_output_port_bytes / sizeof(int16_t);
        for (int port = 0; port != h->pending_output_nof_ports; ++port) {
            if (h->pending_output_ports[port] == nullptr) {
                h->pending_output_valid = false;
                h->done_event_recorded = false;
                return false;
            }
            std::memcpy(h->pending_output_ports[port],
                        h->h_output_staging + static_cast<size_t>(port) * port_i16,
                        h->pending_output_port_bytes);
        }
    }

    for (void*& ptr : h->pending_output_ports) {
        ptr = nullptr;
    }
    h->pending_output_port_bytes = 0;
    h->pending_output_nof_ports = 0;
    h->pending_output_ports_are_contiguous = false;
    h->pending_output_valid = false;
    h->done_event_recorded = false;
    return true;
}

static void destroy_device_graph_entry(ocudu_lowphy_tx_handle::device_graph_entry& entry)
{
    if (entry.exec != nullptr) {
        cudaGraphExecDestroy(entry.exec);
        entry.exec = nullptr;
    }
    if (entry.graph != nullptr) {
        cudaGraphDestroy(entry.graph);
        entry.graph = nullptr;
    }
    entry.d_grid = nullptr;
    entry.stream = nullptr;
    std::memset(&entry.cfg, 0, sizeof(entry.cfg));
}

static void destroy_device_graphs(ocudu_lowphy_tx_handle* h)
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

static bool contains_registered_range(const std::vector<ocudu_lowphy_tx_handle::host_registration>& registrations,
                                      const void*                                                   ptr,
                                      size_t                                                        bytes)
{
    if ((ptr == nullptr) || (bytes == 0)) {
        return false;
    }
    const auto* begin = static_cast<const uint8_t*>(ptr);
    const auto* end   = begin + bytes;
    return std::any_of(registrations.begin(), registrations.end(), [begin, end](const auto& entry) {
        if (entry.ptr == nullptr) {
            return false;
        }
        const auto* entry_begin = static_cast<const uint8_t*>(entry.ptr);
        const auto* entry_end   = entry_begin + entry.bytes;
        return (begin >= entry_begin) && (end <= entry_end);
    });
}

static bool register_host_range(ocudu_lowphy_tx_handle* h, const void* ptr, size_t bytes)
{
    if ((h == nullptr) || (ptr == nullptr) || (bytes == 0)) {
        return false;
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

static void release_buffers(ocudu_lowphy_tx_handle* h)
{
    if (h->stream != nullptr) {
        cudaStreamSynchronize(h->stream);
    }
    (void)flush_pending_output(h, false);

    for (int slot = 0; slot != HOST_GRID_STAGING_SLOTS; ++slot) {
        if (h->h_grid_ready_events[slot] != nullptr) {
            cudaEventDestroy(h->h_grid_ready_events[slot]);
            h->h_grid_ready_events[slot] = nullptr;
        }
        if (h->h_grid_staging[slot] != nullptr) {
            cudaFreeHost(h->h_grid_staging[slot]);
            h->h_grid_staging[slot] = nullptr;
        }
        h->h_grid_event_recorded[slot] = false;
    }
    h->h_grid_staging_count = 0;
    h->h_grid_next_slot = 0;

    if (h->h_output_staging != nullptr) {
        cudaFreeHost(h->h_output_staging);
        h->h_output_staging = nullptr;
    }
    h->h_output_staging_i16 = 0;

    if (h->d_freq) {
        cudaFree(h->d_freq);
        h->d_freq = nullptr;
    }
    if (h->d_grid_stage) {
        cudaFree(h->d_grid_stage);
        h->d_grid_stage = nullptr;
    }
    if (h->d_output) {
        cudaFree(h->d_output);
        h->d_output = nullptr;
    }
    h->d_freq_count = 0;
    h->d_grid_count = 0;
    h->d_output_i16 = 0;
}

static void destroy_fft_plans(ocudu_lowphy_tx_handle* h)
{
    destroy_device_graphs(h);
    if (h->vkfft_valid) {
        deleteVkFFT(&h->vkfft_app);
        std::memset(&h->vkfft_app, 0, sizeof(h->vkfft_app));
        h->vkfft_valid = false;
    }
}

static bool validate_config(const ocudu_lowphy_tx_config_t& cfg)
{
    if ((cfg.dft_size <= 0) || (cfg.rg_size <= 0) || (cfg.nof_ports <= 0) ||
        (cfg.nof_symbols <= 0) || (cfg.nof_samples <= 0)) {
        return false;
    }
    if ((cfg.nof_ports > OCUDU_LOWPHY_TX_MAX_PORTS) ||
        (cfg.nof_symbols > OCUDU_LOWPHY_TX_MAX_SYMBOLS) ||
        (cfg.rg_size > cfg.dft_size) || ((cfg.rg_size % 2) != 0)) {
        return false;
    }
    for (int i = 0; i != cfg.nof_symbols; ++i) {
        if ((cfg.cp_lengths[i] < 0) || (cfg.cp_lengths[i] >= cfg.dft_size) || (cfg.symbol_offsets[i] < 0)) {
            return false;
        }
        if ((i == 0) && (cfg.symbol_offsets[i] != 0)) {
            return false;
        }
        int symbol_end = (i + 1 < cfg.nof_symbols) ? cfg.symbol_offsets[i + 1] : cfg.nof_samples;
        if ((symbol_end <= cfg.symbol_offsets[i]) || (symbol_end > cfg.nof_samples)) {
            return false;
        }
    }
    return true;
}

static bool ensure_plan(ocudu_lowphy_tx_handle* h)
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

    h->vkfft_stream      = h->stream;
    h->vkfft_buffer      = h->d_freq;
    h->vkfft_buffer_size = static_cast<pfUINT>(h->d_freq_count * sizeof(cuFloatComplex));

    VkFFTConfiguration configuration = {};
    configuration.FFTdim               = 1;
    configuration.size[0]              = static_cast<pfUINT>(h->cfg.dft_size);
    configuration.device               = &h->vkfft_device;
    configuration.stream               = &h->vkfft_stream;
    configuration.num_streams          = 1;
    configuration.numberBatches        = h->cfg.nof_ports * h->cfg.nof_symbols;
    configuration.bufferSize           = &h->vkfft_buffer_size;
    configuration.buffer               = &h->vkfft_buffer;
    configuration.makeInversePlanOnly  = 1;
    configuration.normalize            = 0;
    configuration.disableReorderFourStep = 1;
    configuration.useLUT               = 1;
    configuration.aimThreads           = 128;

    if (initializeVkFFT(&h->vkfft_app, configuration) != VKFFT_SUCCESS) {
        std::memset(&h->vkfft_app, 0, sizeof(h->vkfft_app));
        return false;
    }
    h->vkfft_valid = true;
    return true;
}

static bool ensure_buffers(ocudu_lowphy_tx_handle* h)
{
    size_t batches      = static_cast<size_t>(h->cfg.nof_ports) * h->cfg.nof_symbols;
    size_t freq_count   = batches * h->cfg.dft_size;
    size_t grid_count   = batches * h->cfg.rg_size;
    size_t output_i16   = static_cast<size_t>(h->cfg.nof_ports) * h->cfg.nof_samples * 2U;

    if ((h->d_freq_count == freq_count) && (h->d_grid_count == grid_count) && (h->d_output_i16 == output_i16) &&
        (h->h_grid_staging_count == grid_count) && (h->h_output_staging_i16 == output_i16) &&
        h->d_freq && h->d_grid_stage && h->d_output && h->h_output_staging) {
        bool host_grid_ready = true;
        for (int slot = 0; slot != HOST_GRID_STAGING_SLOTS; ++slot) {
            host_grid_ready = host_grid_ready && (h->h_grid_staging[slot] != nullptr) &&
                              (h->h_grid_ready_events[slot] != nullptr);
        }
        if (host_grid_ready) {
            return true;
        }
    }

    if (!flush_pending_output(h, true)) {
        return false;
    }

    destroy_fft_plans(h);
    release_buffers(h);

    if (!check_cuda(cudaMalloc(&h->d_freq, freq_count * sizeof(cuFloatComplex))) ||
        !check_cuda(cudaMalloc(&h->d_grid_stage, grid_count * sizeof(uint32_t))) ||
        !check_cuda(cudaMalloc(&h->d_output, output_i16 * sizeof(int16_t))) ||
        !check_cuda(cudaHostAlloc(&h->h_output_staging,
                                  output_i16 * sizeof(int16_t),
                                  cudaHostAllocDefault))) {
        release_buffers(h);
        return false;
    }

    for (int slot = 0; slot != HOST_GRID_STAGING_SLOTS; ++slot) {
        if (!check_cuda(cudaHostAlloc(&h->h_grid_staging[slot],
                                      grid_count * sizeof(uint32_t),
                                      cudaHostAllocWriteCombined)) ||
            !check_cuda(cudaEventCreateWithFlags(&h->h_grid_ready_events[slot], cudaEventDisableTiming))) {
            release_buffers(h);
            return false;
        }
    }

    h->d_freq_count = freq_count;
    h->d_grid_count = grid_count;
    h->d_output_i16 = output_i16;
    h->h_grid_staging_count = grid_count;
    h->h_output_staging_i16 = output_i16;
    return true;
}

static tx_postprocess_config make_postprocess_config(const ocudu_lowphy_tx_config_t& cfg)
{
    tx_postprocess_config out = {};
    for (int i = 0; i != cfg.nof_symbols; ++i) {
        out.cp_lengths[i]     = cfg.cp_lengths[i];
        out.symbol_offsets[i] = cfg.symbol_offsets[i];
        out.phases[i]         = make_float2(cfg.phase_re[i], cfg.phase_im[i]);
    }
    out.dft_size         = cfg.dft_size;
    out.nof_symbols      = cfg.nof_symbols;
    out.nof_samples      = cfg.nof_samples;
    out.ofdm_scale       = cfg.ofdm_scale;
    out.amplitude_gain   = cfg.amplitude_gain;
    out.clipping_ceiling = cfg.clipping_ceiling;
    out.clipping_enabled = cfg.clipping_enabled;
    return out;
}

static uint32_t* acquire_grid_staging_slot(ocudu_lowphy_tx_handle* h, int* slot_index)
{
    int slot = h->h_grid_next_slot;
    h->h_grid_next_slot = (h->h_grid_next_slot + 1) % HOST_GRID_STAGING_SLOTS;
    if (h->h_grid_event_recorded[slot] && !wait_event_yielding(h->h_grid_ready_events[slot])) {
        return nullptr;
    }
    h->h_grid_event_recorded[slot] = false;
    if (slot_index != nullptr) {
        *slot_index = slot;
    }
    return h->h_grid_staging[slot];
}

static bool enqueue_device_path_uncaptured(ocudu_lowphy_tx_handle* h, const void* d_grid_cbf16)
{
    int batches = h->cfg.nof_ports * h->cfg.nof_symbols;
    int grid_blocks = (batches * h->cfg.dft_size + THREADS - 1) / THREADS;
    int max_symbol_samples = 0;
    for (int symbol = 0; symbol != h->cfg.nof_symbols; ++symbol) {
        int symbol_end = (symbol + 1 < h->cfg.nof_symbols) ? h->cfg.symbol_offsets[symbol + 1] : h->cfg.nof_samples;
        max_symbol_samples = std::max(max_symbol_samples, symbol_end - h->cfg.symbol_offsets[symbol]);
    }
    int out_blocks = (max_symbol_samples + THREADS - 1) / THREADS;

    prepare_ifft_input_kernel<<<grid_blocks, THREADS, 0, h->stream>>>(h->d_freq,
                                                                      static_cast<const uint32_t*>(d_grid_cbf16),
                                                                      h->cfg.dft_size,
                                                                      h->cfg.rg_size,
                                                                      h->cfg.nof_symbols,
                                                                      h->cfg.rg_size,
                                                                      batches);
    if (!check_cuda(cudaGetLastError())) {
        return false;
    }

    VkFFTLaunchParams launch_params = {};
    launch_params.buffer = &h->vkfft_buffer;
    if (VkFFTAppend(&h->vkfft_app, 1, &launch_params) != VKFFT_SUCCESS) {
        return false;
    }

    tx_postprocess_config post_cfg = make_postprocess_config(h->cfg);
    dim3 out_grid(out_blocks, batches);
    postprocess_ifft_to_sc16_kernel<<<out_grid, THREADS, 0, h->stream>>>(h->d_output,
                                                                         h->d_freq,
                                                                         post_cfg);
    return check_cuda(cudaGetLastError());
}

static ocudu_lowphy_tx_handle::device_graph_entry* find_device_graph(ocudu_lowphy_tx_handle* h,
                                                                     const void*             d_grid_cbf16)
{
    for (auto& entry : h->device_graphs) {
        if ((entry.d_grid == d_grid_cbf16) && (entry.stream == h->stream) &&
            (std::memcmp(&entry.cfg, &h->cfg, sizeof(entry.cfg)) == 0) && (entry.exec != nullptr)) {
            return &entry;
        }
    }
    return nullptr;
}

static ocudu_lowphy_tx_handle::device_graph_entry* reserve_device_graph_slot(ocudu_lowphy_tx_handle* h)
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

static bool capture_device_graph(ocudu_lowphy_tx_handle* h, const void* d_grid_cbf16)
{
    cudaGraph_t     captured_graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    if (cudaStreamBeginCapture(h->stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        (void)cudaGetLastError();
        h->device_graph_capture_failed = true;
        return false;
    }

    bool        enqueue_ok = enqueue_device_path_uncaptured(h, d_grid_cbf16);
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

    auto* entry = reserve_device_graph_slot(h);
    entry->d_grid = d_grid_cbf16;
    entry->stream = h->stream;
    entry->cfg = h->cfg;
    entry->graph = captured_graph;
    entry->exec = graph_exec;
    return true;
}

static bool enqueue_device_path(ocudu_lowphy_tx_handle* h, const void* d_grid_cbf16)
{
    if (!cuda_graphs_enabled() || h->device_graph_capture_failed) {
        return enqueue_device_path_uncaptured(h, d_grid_cbf16);
    }

    if (auto* entry = find_device_graph(h, d_grid_cbf16); entry != nullptr) {
        if (cudaGraphLaunch(entry->exec, h->stream) == cudaSuccess) {
            return true;
        }
        destroy_device_graph_entry(*entry);
        (void)cudaGetLastError();
    }

    {
        std::lock_guard<std::mutex> lock(graph_capture_mutex());
        if (auto* entry = find_device_graph(h, d_grid_cbf16); entry != nullptr) {
            if (cudaGraphLaunch(entry->exec, h->stream) == cudaSuccess) {
                return true;
            }
            destroy_device_graph_entry(*entry);
            (void)cudaGetLastError();
        }
        if (capture_device_graph(h, d_grid_cbf16)) {
            if (auto* entry = find_device_graph(h, d_grid_cbf16); entry != nullptr) {
                if (cudaGraphLaunch(entry->exec, h->stream) == cudaSuccess) {
                    return true;
                }
                destroy_device_graph_entry(*entry);
                (void)cudaGetLastError();
            }
        }
    }

    return enqueue_device_path_uncaptured(h, d_grid_cbf16);
}

static bool warmup_handle(ocudu_lowphy_tx_handle* h)
{
    if ((h == nullptr) || (h->d_grid_stage == nullptr)) {
        return false;
    }

    cudaStream_t saved_stream = h->stream;
    h->stream = (h->owned_stream != nullptr) ? h->owned_stream : h->stream;
    if ((h->stream == nullptr) || !ensure_buffers(h) || !ensure_plan(h)) {
        h->stream = saved_stream;
        return false;
    }

    if (!check_cuda(cudaMemsetAsync(h->d_grid_stage, 0, h->d_grid_count * sizeof(uint32_t), h->stream))) {
        h->stream = saved_stream;
        return false;
    }

    bool ok = enqueue_device_path_uncaptured(h, h->d_grid_stage) && check_cuda(cudaStreamSynchronize(h->stream));
    h->stream = saved_stream;
    return ok;
}

static bool copy_output_to_host(ocudu_lowphy_tx_handle* h, void* const* h_output_ports)
{
    size_t port_bytes = static_cast<size_t>(h->cfg.nof_samples) * 2U * sizeof(int16_t);
    for (int port = 0; port != h->cfg.nof_ports; ++port) {
        if (h_output_ports[port] == nullptr) {
            return false;
        }
    }

    bool ports_are_contiguous = true;
    for (int port = 1; port != h->cfg.nof_ports; ++port) {
        const char* previous = static_cast<const char*>(h_output_ports[port - 1]);
        ports_are_contiguous =
            ports_are_contiguous && (static_cast<const void*>(previous + port_bytes) == h_output_ports[port]);
    }

    if (ports_are_contiguous) {
        size_t total_bytes = port_bytes * static_cast<size_t>(h->cfg.nof_ports);
        bool direct_copy_available = contains_registered_range(h->registered_hosts, h_output_ports[0], total_bytes);
        if (!direct_copy_available && runtime_host_registration_enabled()) {
            direct_copy_available = register_host_range(h, h_output_ports[0], total_bytes);
        }
        if (direct_copy_available) {
            if (!check_cuda(cudaMemcpyAsync(h_output_ports[0],
                                            h->d_output,
                                            total_bytes,
                                            cudaMemcpyDeviceToHost,
                                            h->stream))) {
                return false;
            }
            if (!check_cuda(cudaEventRecord(h->done_event, h->stream))) {
                return false;
            }
            h->pending_output_valid = false;
            h->done_event_recorded = true;
            return true;
        }
    } else {
        bool direct_copy_available = true;
        for (int port = 0; port != h->cfg.nof_ports; ++port) {
            bool port_registered = contains_registered_range(h->registered_hosts, h_output_ports[port], port_bytes);
            if (!port_registered && runtime_host_registration_enabled()) {
                port_registered = register_host_range(h, h_output_ports[port], port_bytes);
            }
            direct_copy_available = direct_copy_available && port_registered;
        }
        if (direct_copy_available) {
            for (int port = 0; port != h->cfg.nof_ports; ++port) {
                if (!check_cuda(cudaMemcpyAsync(h_output_ports[port],
                                                h->d_output + static_cast<size_t>(port) * h->cfg.nof_samples * 2U,
                                                port_bytes,
                                                cudaMemcpyDeviceToHost,
                                                h->stream))) {
                    return false;
                }
            }
            if (!check_cuda(cudaEventRecord(h->done_event, h->stream))) {
                return false;
            }
            h->pending_output_valid = false;
            h->done_event_recorded = true;
            return true;
        }
    }

    if (!check_cuda(cudaMemcpyAsync(h->h_output_staging,
                                    h->d_output,
                                    port_bytes * static_cast<size_t>(h->cfg.nof_ports),
                                    cudaMemcpyDeviceToHost,
                                    h->stream))) {
        return false;
    }
    if (!check_cuda(cudaEventRecord(h->done_event, h->stream))) {
        return false;
    }

    for (int port = 0; port != h->cfg.nof_ports; ++port) {
        h->pending_output_ports[port] = h_output_ports[port];
    }
    h->pending_output_port_bytes = port_bytes;
    h->pending_output_nof_ports = h->cfg.nof_ports;
    h->pending_output_ports_are_contiguous = ports_are_contiguous;
    h->pending_output_valid = true;
    h->done_event_recorded = true;
    return true;
}

static bool process_device_grid(ocudu_lowphy_tx_handle* h, const void* d_grid_cbf16, void* const* h_output_ports)
{
    if ((h == nullptr) || (d_grid_cbf16 == nullptr) || (h_output_ports == nullptr)) {
        return false;
    }
    if (!flush_pending_output(h, true)) {
        return false;
    }
    if (!ensure_buffers(h) || !ensure_plan(h)) {
        return false;
    }
    if (!enqueue_device_path(h, d_grid_cbf16)) {
        return false;
    }
    return copy_output_to_host(h, h_output_ports);
}

} // namespace

int ocudu_lowphy_tx_create(const ocudu_lowphy_tx_config_t* config, ocudu_lowphy_tx_handle_t** handle)
{
    if ((config == nullptr) || (handle == nullptr) || !validate_config(*config)) {
        return 0;
    }

    auto* h = new ocudu_lowphy_tx_handle();
    h->cfg = *config;
    if (!check_cuda(create_realtime_stream(&h->owned_stream)) ||
        !check_cuda(cudaEventCreateWithFlags(&h->done_event, cudaEventDisableTiming))) {
        ocudu_lowphy_tx_destroy(h);
        return 0;
    }
    h->stream = h->owned_stream;

    if (!ensure_buffers(h) || !ensure_plan(h) || !warmup_handle(h)) {
        ocudu_lowphy_tx_destroy(h);
        return 0;
    }

    *handle = h;
    return 1;
}

void ocudu_lowphy_tx_destroy(ocudu_lowphy_tx_handle_t* handle)
{
    if (handle == nullptr) {
        return;
    }

    (void)flush_pending_output(handle, true);
    for (const auto& registration : handle->registered_hosts) {
        if (registration.owned && (registration.ptr != nullptr)) {
            cudaHostUnregister(const_cast<void*>(registration.ptr));
        }
    }
    handle->registered_hosts.clear();
    handle->unregistered_hosts.clear();
    destroy_fft_plans(handle);
    release_buffers(handle);
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

int ocudu_lowphy_tx_update_config(ocudu_lowphy_tx_handle_t* handle, const ocudu_lowphy_tx_config_t* config)
{
    if ((handle == nullptr) || (config == nullptr) || !validate_config(*config)) {
        return 0;
    }
    if (!flush_pending_output(handle, true)) {
        return 0;
    }

    bool plan_changed   = (handle->cfg.dft_size != config->dft_size) ||
                        (handle->cfg.nof_ports != config->nof_ports) ||
                        (handle->cfg.nof_symbols != config->nof_symbols);
    bool config_changed = std::memcmp(&handle->cfg, config, sizeof(*config)) != 0;
    handle->cfg = *config;
    if (plan_changed) {
        destroy_fft_plans(handle);
    } else if (!config_changed) {
        return ensure_buffers(handle) && ensure_plan(handle);
    }
    if (!ensure_buffers(handle) || !ensure_plan(handle)) {
        return 0;
    }
    return !plan_changed || warmup_handle(handle);
}

int ocudu_lowphy_tx_register_host_output(ocudu_lowphy_tx_handle_t* handle, void* ptr, size_t bytes)
{
    if ((handle == nullptr) || (ptr == nullptr) || (bytes == 0)) {
        return 0;
    }
    return register_host_range(handle, ptr, bytes) ? 1 : 0;
}

int ocudu_lowphy_tx_process(ocudu_lowphy_tx_handle_t* handle,
                            const void*               d_grid_cbf16,
                            void* const*              h_output_ports,
                            void*                     external_stream)
{
    if (handle == nullptr) {
        return 0;
    }
    handle->stream = (external_stream != nullptr) ? static_cast<cudaStream_t>(external_stream) : handle->owned_stream;
    return process_device_grid(handle, d_grid_cbf16, h_output_ports) ? 1 : 0;
}

int ocudu_lowphy_tx_process_host_grid(ocudu_lowphy_tx_handle_t* handle,
                                      const void*               h_grid_cbf16,
                                      void* const*              h_output_ports,
                                      void*                     external_stream)
{
    if ((handle == nullptr) || (h_grid_cbf16 == nullptr)) {
        return 0;
    }
    handle->stream = (external_stream != nullptr) ? static_cast<cudaStream_t>(external_stream) : handle->owned_stream;
    if (!flush_pending_output(handle, true) || !ensure_buffers(handle)) {
        return 0;
    }

    size_t grid_bytes = handle->d_grid_count * sizeof(uint32_t);
    if (runtime_host_registration_enabled() && register_host_range(handle, h_grid_cbf16, grid_bytes)) {
        if (!check_cuda(cudaMemcpyAsync(
                handle->d_grid_stage, h_grid_cbf16, grid_bytes, cudaMemcpyHostToDevice, handle->stream))) {
            return 0;
        }
    } else {
        int grid_slot = -1;
        uint32_t* h_grid_staging = acquire_grid_staging_slot(handle, &grid_slot);
        if ((h_grid_staging == nullptr) || (grid_slot < 0)) {
            return 0;
        }
        std::memcpy(h_grid_staging, h_grid_cbf16, grid_bytes);
        if (!check_cuda(cudaMemcpyAsync(
                handle->d_grid_stage, h_grid_staging, grid_bytes, cudaMemcpyHostToDevice, handle->stream))) {
            return 0;
        }
        if (!check_cuda(cudaEventRecord(handle->h_grid_ready_events[grid_slot], handle->stream))) {
            return 0;
        }
        handle->h_grid_event_recorded[grid_slot] = true;
    }
    return process_device_grid(handle, handle->d_grid_stage, h_output_ports) ? 1 : 0;
}

int ocudu_lowphy_tx_synchronize(ocudu_lowphy_tx_handle_t* handle)
{
    if ((handle == nullptr) || (handle->done_event == nullptr)) {
        return 0;
    }
    return flush_pending_output(handle, true) ? 1 : 0;
}

void* ocudu_lowphy_tx_get_completion_event(ocudu_lowphy_tx_handle_t* handle)
{
    return (handle == nullptr) ? nullptr : static_cast<void*>(handle->done_event);
}

void* ocudu_lowphy_tx_get_stream(ocudu_lowphy_tx_handle_t* handle)
{
    return (handle == nullptr) ? nullptr : static_cast<void*>(handle->stream ? handle->stream : handle->owned_stream);
}
