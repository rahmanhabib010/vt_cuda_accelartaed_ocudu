/*
 * OCUDU PHY CUDA - CUDA-accelerated 5G NR PHY processing
 *
 * GPU Sounding Reference Signal channel estimator.
 */

#include "srs_estimator.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include <new>
#include <sched.h>
#include <cstring>
#include "vkFFT.h"

#ifndef OCUDU_PHY_CUDA_ENABLE_VKFFT
#error "OCUDU PHY CUDA SRS acceleration requires the bundled VkFFT backend."
#endif

struct srs_estimator_device_result {
    srs_estimator_time_alignment_t time_alignment;
    srs_estimator_metrics_t        metrics;
};

struct srs_estimator_ctx {
    srs_estimator_config_t config = {};

    cuFloatComplex* d_sequences = nullptr;
    cuFloatComplex* d_lse = nullptr;
    cuFloatComplex* d_idft = nullptr;
    cuFloatComplex* d_twiddles = nullptr;
    cuFloatComplex* d_noise_help = nullptr;
    cuFloatComplex* d_coeff = nullptr;
    float*          d_correlation = nullptr;
    srs_estimator_time_alignment_t* d_time_alignment_per_tx = nullptr;
    float*          d_epre_partials = nullptr;
    float*          d_noise_partials = nullptr;
    float*          d_rsrp_partials = nullptr;
    srs_estimator_device_result* d_result = nullptr;
    srs_estimator_device_result* h_result = nullptr;

    cudaGraph_t     graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    const void*     graph_grid_cbf16 = nullptr;

    size_t sequences_capacity = 0;
    size_t lse_capacity = 0;
    size_t idft_capacity = 0;
    size_t twiddles_capacity = 0;
    size_t noise_help_capacity = 0;
    size_t coeff_capacity = 0;
    size_t correlation_capacity = 0;
    size_t time_alignment_per_tx_capacity = 0;
    size_t epre_partials_capacity = 0;
    size_t noise_partials_capacity = 0;
    size_t rsrp_partials_capacity = 0;

    VkFFTApplication vkfft_app{};
    CUdevice         vkfft_device = 0;
    cudaStream_t     vkfft_stream = nullptr;
    void*            vkfft_buffer = nullptr;
    pfUINT           vkfft_buffer_size = 0;
    int              vkfft_dft_size = 0;
    int              vkfft_batch = 0;
    bool             vkfft_valid = false;

    bool twiddles_valid = false;
    int  twiddle_dft_size = 0;
    int  twiddle_sequence_length = 0;
    int  twiddle_window_size = 0;
};

namespace {

constexpr float TWO_PI = 6.2831853071795864769f;
constexpr int   CEXP_TABLE_SIZE = 1024;
constexpr int   SRS_ESTIMATOR_THREADS = 256;

__device__ __forceinline__ cuFloatComplex cbf16_to_fp32(unsigned int packed)
{
    unsigned int real_bits = packed & 0xFFFFU;
    unsigned int imag_bits = (packed >> 16U) & 0xFFFFU;
    unsigned int real_fp32 = real_bits << 16U;
    unsigned int imag_fp32 = imag_bits << 16U;
    float        real_f    = __uint_as_float(real_fp32);
    float        imag_f    = __uint_as_float(imag_fp32);
    return make_cuFloatComplex(real_f, imag_f);
}

__device__ __forceinline__ cuFloatComplex cmul(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ cuFloatComplex cconj(cuFloatComplex a)
{
    return make_cuFloatComplex(a.x, -a.y);
}

__device__ __forceinline__ cuFloatComplex csub(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float norm2(cuFloatComplex a)
{
    return a.x * a.x + a.y * a.y;
}

__host__ __device__ __forceinline__ float srs_to_db(float value)
{
    return value > 0.0f ? 10.0f * log10f(value) : -INFINITY;
}

__device__ __forceinline__ int round_half_away_from_zero(float value)
{
    return (value >= 0.0f) ? static_cast<int>(floorf(value + 0.5f)) : static_cast<int>(ceilf(value - 0.5f));
}

__device__ __forceinline__ cuFloatComplex quantized_phase(float phase)
{
    int index = round_half_away_from_zero(static_cast<float>(CEXP_TABLE_SIZE) * phase / TWO_PI);
    index &= (CEXP_TABLE_SIZE - 1);

    float s;
    float c;
    __sincosf(TWO_PI * static_cast<float>(index) / static_cast<float>(CEXP_TABLE_SIZE), &s, &c);
    return make_cuFloatComplex(c, s);
}

bool reserve_device(void** ptr, size_t& capacity, size_t required)
{
    if (required == 0) {
        return false;
    }
    if (required <= capacity) {
        return true;
    }

    if (*ptr != nullptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
        capacity = 0;
    }

    if (cudaMalloc(ptr, required) != cudaSuccess) {
        *ptr = nullptr;
        return false;
    }

    capacity = required;
    return true;
}

void destroy_graph(srs_estimator_ctx* ctx)
{
    if (ctx == nullptr) {
        return;
    }
    if (ctx->graph_exec != nullptr) {
        cudaGraphExecDestroy(ctx->graph_exec);
        ctx->graph_exec = nullptr;
    }
    if (ctx->graph != nullptr) {
        cudaGraphDestroy(ctx->graph);
        ctx->graph = nullptr;
    }
    ctx->graph_grid_cbf16 = nullptr;
}

void destroy_vkfft_plan(srs_estimator_ctx* ctx)
{
    if (ctx == nullptr || !ctx->vkfft_valid) {
        return;
    }

    deleteVkFFT(&ctx->vkfft_app);
    std::memset(&ctx->vkfft_app, 0, sizeof(ctx->vkfft_app));
    ctx->vkfft_stream = nullptr;
    ctx->vkfft_buffer = nullptr;
    ctx->vkfft_buffer_size = 0;
    ctx->vkfft_dft_size = 0;
    ctx->vkfft_batch = 0;
    ctx->vkfft_valid = false;
}

bool validate_config(const srs_estimator_config_t& cfg)
{
    if (cfg.nof_rx_ports <= 0 || cfg.nof_rx_ports > SRS_ESTIMATOR_MAX_RX_PORTS) {
        return false;
    }
    if (cfg.nof_tx_ports <= 0 || cfg.nof_tx_ports > SRS_ESTIMATOR_MAX_TX_PORTS) {
        return false;
    }
    if (cfg.nof_symbols <= 0 || cfg.sequence_length <= 0 ||
        cfg.sequence_length > SRS_ESTIMATOR_MAX_SEQUENCE_LENGTH) {
        return false;
    }
    if (cfg.comb_size <= 0 || cfg.grid_nof_ports <= 0 || cfg.grid_nof_symbols <= 0 ||
        cfg.grid_nof_subcarriers <= 0) {
        return false;
    }
    if (cfg.start_symbol < 0 || cfg.start_symbol + cfg.nof_symbols > cfg.grid_nof_symbols) {
        return false;
    }
    if (cfg.dft_size <= 0 || cfg.dft_size > SRS_ESTIMATOR_MAX_DFT_SIZE) {
        return false;
    }
    if (cfg.correlation_window_size < 0 || cfg.correlation_window_size > cfg.dft_size / 2) {
        return false;
    }
    if (cfg.ta_max_samples <= 0 || cfg.ta_max_samples > cfg.dft_size / 2) {
        return false;
    }
    for (int tx = 0; tx != cfg.nof_tx_ports; ++tx) {
        int last_subcarrier = cfg.mapping_initial_subcarrier[tx] + (cfg.sequence_length - 1) * cfg.comb_size;
        if (cfg.mapping_initial_subcarrier[tx] < 0 || last_subcarrier >= cfg.grid_nof_subcarriers) {
            return false;
        }
    }
    for (int rx = 0; rx != cfg.nof_rx_ports; ++rx) {
        if (cfg.rx_ports[rx] < 0 || cfg.rx_ports[rx] >= cfg.grid_nof_ports) {
            return false;
        }
    }
    return true;
}

bool uses_windowed_correlation(const srs_estimator_config_t& cfg)
{
    return (cfg.correlation_window_size > 0) && (cfg.correlation_window_size * 2 < cfg.dft_size);
}

std::mutex& graph_capture_mutex()
{
    static std::mutex mutex;
    return mutex;
}

cudaError_t stream_synchronize_yielding(cudaStream_t stream)
{
    cudaError_t err;
    int         query_count = 0;
    while ((err = cudaStreamQuery(stream)) == cudaErrorNotReady) {
        // SRS estimation is normally a short low-latency operation. Yielding too early can hand the CPU thread back
        // to the OS scheduler and turn a tens-of-microseconds GPU completion into a hundreds-of-microseconds outlier.
        if (++query_count >= 1048576) {
            sched_yield();
            query_count = 0;
        }
    }
    return err;
}

bool ensure_capacity(srs_estimator_ctx* ctx)
{
    const srs_estimator_config_t& cfg = ctx->config;

    size_t nof_seq    = static_cast<size_t>(cfg.nof_tx_ports) * cfg.sequence_length;
    size_t nof_lse    = static_cast<size_t>(cfg.nof_tx_ports) * cfg.nof_rx_ports * cfg.sequence_length;
    size_t nof_idft   = static_cast<size_t>(cfg.nof_tx_ports) * cfg.nof_rx_ports * cfg.dft_size;
    size_t nof_twiddles = uses_windowed_correlation(cfg)
                              ? static_cast<size_t>(cfg.correlation_window_size) * 2U * cfg.sequence_length
                              : 1U;
    size_t nof_noise  = static_cast<size_t>(2) * cfg.nof_rx_ports * cfg.sequence_length;
    size_t nof_coeff  = static_cast<size_t>(cfg.nof_rx_ports) * cfg.nof_tx_ports;
    size_t nof_corr   = static_cast<size_t>(cfg.nof_tx_ports) * cfg.dft_size;
    size_t nof_sequence_blocks =
        static_cast<size_t>((cfg.sequence_length + SRS_ESTIMATOR_THREADS - 1) / SRS_ESTIMATOR_THREADS);
    size_t nof_epre_partials = static_cast<size_t>(cfg.nof_tx_ports) * cfg.nof_rx_ports * nof_sequence_blocks;
    size_t nof_noise_partials =
        static_cast<size_t>(cfg.nof_rx_ports) * ((cfg.interleaved_pilots != 0) ? 2U : 1U);

    if (nof_idft * sizeof(cuFloatComplex) > ctx->idft_capacity) {
        destroy_vkfft_plan(ctx);
    }

    return reserve_device(reinterpret_cast<void**>(&ctx->d_sequences), ctx->sequences_capacity,
                          nof_seq * sizeof(cuFloatComplex)) &&
           reserve_device(reinterpret_cast<void**>(&ctx->d_lse), ctx->lse_capacity, nof_lse * sizeof(cuFloatComplex)) &&
           reserve_device(reinterpret_cast<void**>(&ctx->d_idft), ctx->idft_capacity,
                          nof_idft * sizeof(cuFloatComplex)) &&
           reserve_device(reinterpret_cast<void**>(&ctx->d_twiddles), ctx->twiddles_capacity,
                          nof_twiddles * sizeof(cuFloatComplex)) &&
           reserve_device(reinterpret_cast<void**>(&ctx->d_noise_help), ctx->noise_help_capacity,
                          nof_noise * sizeof(cuFloatComplex)) &&
           reserve_device(reinterpret_cast<void**>(&ctx->d_coeff), ctx->coeff_capacity,
                          nof_coeff * sizeof(cuFloatComplex)) &&
           reserve_device(reinterpret_cast<void**>(&ctx->d_correlation), ctx->correlation_capacity,
                          nof_corr * sizeof(float)) &&
           reserve_device(reinterpret_cast<void**>(&ctx->d_time_alignment_per_tx),
                          ctx->time_alignment_per_tx_capacity,
                          static_cast<size_t>(cfg.nof_tx_ports) * sizeof(srs_estimator_time_alignment_t)) &&
           reserve_device(reinterpret_cast<void**>(&ctx->d_epre_partials),
                          ctx->epre_partials_capacity,
                          nof_epre_partials * sizeof(float)) &&
           reserve_device(reinterpret_cast<void**>(&ctx->d_noise_partials),
                          ctx->noise_partials_capacity,
                          nof_noise_partials * sizeof(float)) &&
           reserve_device(reinterpret_cast<void**>(&ctx->d_rsrp_partials),
                          ctx->rsrp_partials_capacity,
                          nof_coeff * sizeof(float)) &&
           (ctx->d_result != nullptr ||
            cudaMalloc(reinterpret_cast<void**>(&ctx->d_result), sizeof(srs_estimator_device_result)) ==
                cudaSuccess) &&
           (ctx->h_result != nullptr ||
            cudaHostAlloc(reinterpret_cast<void**>(&ctx->h_result),
                          sizeof(srs_estimator_device_result),
                          cudaHostAllocDefault) == cudaSuccess);
}

bool ensure_vkfft_plan(srs_estimator_ctx* ctx, cudaStream_t stream)
{
    const srs_estimator_config_t& cfg = ctx->config;
    int batch = cfg.nof_tx_ports * cfg.nof_rx_ports;

    if (ctx->vkfft_valid && ctx->vkfft_dft_size == cfg.dft_size && ctx->vkfft_batch == batch &&
        ctx->vkfft_stream == stream && ctx->vkfft_buffer == ctx->d_idft) {
        return true;
    }

    destroy_vkfft_plan(ctx);

    int cuda_device = 0;
    if ((cudaGetDevice(&cuda_device) != cudaSuccess) || (cuDeviceGet(&ctx->vkfft_device, cuda_device) != CUDA_SUCCESS)) {
        return false;
    }

    ctx->vkfft_stream = stream;
    ctx->vkfft_buffer = ctx->d_idft;
    ctx->vkfft_buffer_size = static_cast<pfUINT>(static_cast<size_t>(batch) * cfg.dft_size * sizeof(cuFloatComplex));

    VkFFTConfiguration configuration = {};
    configuration.FFTdim = 1;
    configuration.size[0] = static_cast<pfUINT>(cfg.dft_size);
    configuration.device = &ctx->vkfft_device;
    configuration.stream = &ctx->vkfft_stream;
    configuration.num_streams = 1;
    configuration.numberBatches = batch;
    configuration.bufferSize = &ctx->vkfft_buffer_size;
    configuration.buffer = &ctx->vkfft_buffer;
    configuration.makeInversePlanOnly = 1;
    configuration.normalize = 0;
    configuration.disableReorderFourStep = 1;
    configuration.useLUT = 1;
    configuration.aimThreads = 128;

    if (initializeVkFFT(&ctx->vkfft_app, configuration) != VKFFT_SUCCESS) {
        std::memset(&ctx->vkfft_app, 0, sizeof(ctx->vkfft_app));
        ctx->vkfft_stream = nullptr;
        ctx->vkfft_buffer = nullptr;
        ctx->vkfft_buffer_size = 0;
        return false;
    }

    ctx->vkfft_dft_size = cfg.dft_size;
    ctx->vkfft_batch = batch;
    ctx->vkfft_valid = true;
    return true;
}

__global__ void srs_extract_lse_partial_kernel(const unsigned int* __restrict__ d_grid_cbf16,
                                               const cuFloatComplex* __restrict__ d_sequences,
                                               cuFloatComplex* __restrict__ d_lse,
                                               cuFloatComplex* __restrict__ d_noise_help,
                                               float* __restrict__ d_epre_partials,
                                               srs_estimator_config_t cfg)
{
    int tx = blockIdx.x;
    int rx = blockIdx.y;
    int n  = threadIdx.x + blockIdx.z * blockDim.x;
    if (tx >= cfg.nof_tx_ports || rx >= cfg.nof_rx_ports) {
        return;
    }

    __shared__ float partial_power[SRS_ESTIMATOR_THREADS];

    bool contributes_noise = (tx == 0) || ((cfg.interleaved_pilots != 0) && (tx == 1));
    float power_sum = 0.0f;

    if (n < cfg.sequence_length) {
        int rx_port = cfg.rx_ports[rx];
        int k       = cfg.mapping_initial_subcarrier[tx] + n * cfg.comb_size;
        int stride  = cfg.grid_nof_symbols * cfg.grid_nof_subcarriers;

        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        for (int s = 0; s != cfg.nof_symbols; ++s) {
            int symbol = cfg.start_symbol + s;
            int idx    = rx_port * stride + symbol * cfg.grid_nof_subcarriers + k;
            cuFloatComplex y = cbf16_to_fp32(d_grid_cbf16[idx]);
            sum.x += y.x;
            sum.y += y.y;
            power_sum += norm2(y);
        }

        cuFloatComplex seq = d_sequences[tx * cfg.sequence_length + n];
        cuFloatComplex lse = cmul(sum, cconj(seq));
        float          scale = 1.0f / static_cast<float>(cfg.nof_symbols);
        lse.x *= scale;
        lse.y *= scale;
        d_lse[(tx * cfg.nof_rx_ports + rx) * cfg.sequence_length + n] = lse;

        if (contributes_noise) {
            int set = (cfg.interleaved_pilots != 0) ? (tx & 1) : 0;
            d_noise_help[(set * cfg.nof_rx_ports + rx) * cfg.sequence_length + n] = sum;
        }
    }

    partial_power[threadIdx.x] = contributes_noise ? power_sum : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            partial_power[threadIdx.x] += partial_power[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int nof_sequence_blocks = (cfg.sequence_length + blockDim.x - 1) / blockDim.x;
        int index = (tx * cfg.nof_rx_ports + rx) * nof_sequence_blocks + blockIdx.z;
        d_epre_partials[index] = partial_power[0];
    }
}

__global__ void srs_pack_lse_for_idft_kernel(const cuFloatComplex* __restrict__ d_lse,
                                             cuFloatComplex* __restrict__ d_idft,
                                             srs_estimator_config_t cfg)
{
    int path = blockIdx.x;
    int n = threadIdx.x + blockIdx.y * blockDim.x;
    int nof_paths = cfg.nof_tx_ports * cfg.nof_rx_ports;
    if (path >= nof_paths || n >= cfg.sequence_length) {
        return;
    }

    d_idft[path * cfg.dft_size + n] = d_lse[path * cfg.sequence_length + n];
}

__global__ void srs_generate_twiddles_kernel(cuFloatComplex* __restrict__ d_twiddles, srs_estimator_config_t cfg)
{
    int window_index = blockIdx.x;
    int n = threadIdx.x + blockIdx.y * blockDim.x;
    int window_size = cfg.correlation_window_size;
    if (window_index >= 2 * window_size || n >= cfg.sequence_length) {
        return;
    }

    int k = (window_index < window_size) ? window_index : (cfg.dft_size - (2 * window_size - window_index));
    float phase = TWO_PI * static_cast<float>(k) * static_cast<float>(n) / static_cast<float>(cfg.dft_size);
    float s;
    float c;
    sincosf(phase, &s, &c);
    d_twiddles[window_index * cfg.sequence_length + n] = make_cuFloatComplex(c, s);
}

__global__ void srs_correlation_window_kernel(const cuFloatComplex* __restrict__ d_lse,
                                              const cuFloatComplex* __restrict__ d_twiddles,
                                              float* __restrict__ d_correlation,
                                              srs_estimator_config_t cfg)
{
    int window_index = blockIdx.x;
    int tx = blockIdx.y;
    int tid = threadIdx.x;
    int window_size = cfg.correlation_window_size;
    if (tx >= cfg.nof_tx_ports || window_index >= 2 * window_size) {
        return;
    }

    int k = (window_index < window_size) ? window_index : (cfg.dft_size - (2 * window_size - window_index));

    __shared__ float partial_re[256];
    __shared__ float partial_im[256];

    float correlation = 0.0f;
    for (int rx = 0; rx != cfg.nof_rx_ports; ++rx) {
        float sum_re = 0.0f;
        float sum_im = 0.0f;

        for (int n = tid; n < cfg.sequence_length; n += blockDim.x) {
            cuFloatComplex h = d_lse[(tx * cfg.nof_rx_ports + rx) * cfg.sequence_length + n];
            cuFloatComplex w = d_twiddles[window_index * cfg.sequence_length + n];
            sum_re += h.x * w.x - h.y * w.y;
            sum_im += h.x * w.y + h.y * w.x;
        }

        partial_re[tid] = sum_re;
        partial_im[tid] = sum_im;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial_re[tid] += partial_re[tid + offset];
                partial_im[tid] += partial_im[tid + offset];
            }
            __syncthreads();
        }

        if (tid == 0) {
            correlation += partial_re[0] * partial_re[0] + partial_im[0] * partial_im[0];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_correlation[tx * cfg.dft_size + k] = correlation;
    }
}

__global__ void srs_correlation_from_idft_kernel(const cuFloatComplex* __restrict__ d_idft,
                                                 float* __restrict__ d_correlation,
                                                 srs_estimator_config_t cfg)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = blockIdx.y;
    if (k >= cfg.dft_size || tx >= cfg.nof_tx_ports) {
        return;
    }

    float correlation = 0.0f;
    for (int rx = 0; rx != cfg.nof_rx_ports; ++rx) {
        cuFloatComplex h = d_idft[(tx * cfg.nof_rx_ports + rx) * cfg.dft_size + k];
        correlation += norm2(h);
    }
    d_correlation[tx * cfg.dft_size + k] = correlation;
}

__device__ float srs_fractional_peak(const float* __restrict__ correlation, int dft_size, int idx, int nof_taps)
{
    if (dft_size == SRS_ESTIMATOR_MAX_DFT_SIZE) {
        return 0.0f;
    }

    float num = 0.0f;
    float den = 0.0f;
    float correction = 1.0f;
    for (int i = 0; i != nof_taps; ++i) {
        int sample_index = (idx + i + dft_size - nof_taps / 2) % dft_size;
        float sample = correlation[sample_index];

        if (nof_taps == 5) {
            constexpr float num_weights[5] = {-0.4f, -0.2f, 0.0f, 0.2f, 0.4f};
            constexpr float den_weights[5] = {0.571429f, -0.285714f, -0.571429f, -0.285714f, 0.571429f};
            num += num_weights[i] * sample;
            den += den_weights[i] * sample;
        } else {
            constexpr float num_weights[3] = {-0.5f, 0.0f, 0.5f};
            constexpr float den_weights[3] = {0.5f, -1.0f, 0.5f};
            correction = 0.5f;
            num += num_weights[i] * sample;
            den += den_weights[i] * sample;
        }
    }

    float result = -correction * num / den;
    return (isnan(result) || isinf(result) || fabsf(result) > 1.0f) ? 0.0f : result;
}

__global__ void srs_estimate_ta_kernel(const float* __restrict__ d_correlation,
                                       srs_estimator_time_alignment_t* __restrict__ d_time_alignment_per_tx,
                                       srs_estimator_config_t cfg)
{
    int tx = blockIdx.x;
    int tid = threadIdx.x;
    if (tx >= cfg.nof_tx_ports) {
        return;
    }

    __shared__ float delay_values[256];
    __shared__ int   delay_indices[256];
    __shared__ float advance_values[256];
    __shared__ int   advance_indices[256];

    const float* correlation = d_correlation + tx * cfg.dft_size;
    int window = cfg.ta_max_samples;

    float delay_value = -FLT_MAX;
    int   delay_index = 0;
    float advance_value = -FLT_MAX;
    int   advance_index = 0;

    for (int k = tid; k < window; k += blockDim.x) {
        float value = correlation[k];
        if (value > delay_value) {
            delay_value = value;
            delay_index = k;
        }
    }

    for (int k = tid; k < window; k += blockDim.x) {
        float value = correlation[cfg.dft_size - window + k];
        if (value > advance_value) {
            advance_value = value;
            advance_index = k;
        }
    }

    delay_values[tid] = delay_value;
    delay_indices[tid] = delay_index;
    advance_values[tid] = advance_value;
    advance_indices[tid] = advance_index;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (delay_values[tid + offset] > delay_values[tid]) {
                delay_values[tid] = delay_values[tid + offset];
                delay_indices[tid] = delay_indices[tid + offset];
            }
            if (advance_values[tid + offset] > advance_values[tid]) {
                advance_values[tid] = advance_values[tid + offset];
                advance_indices[tid] = advance_indices[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int idx = -(window - advance_indices[0]);
        if (delay_values[0] >= advance_values[0]) {
            idx = delay_indices[0];
        }

        int nof_taps = (window > 2) ? 5 : 3;
        float fractional_sample_index = srs_fractional_peak(correlation, cfg.dft_size, idx, nof_taps);
        float sampling_rate_hz =
            static_cast<float>(cfg.dft_size * cfg.scs_khz * 1000 * cfg.comb_size);

        d_time_alignment_per_tx[tx] = srs_estimator_time_alignment_t{
            (static_cast<float>(idx) + fractional_sample_index) / sampling_rate_hz,
            1.0f / sampling_rate_hz,
            -static_cast<float>(window) / sampling_rate_hz,
            static_cast<float>(window) / sampling_rate_hz};
    }
}

__global__ void srs_coeff_partial_kernel(const cuFloatComplex* __restrict__ d_lse,
                                         cuFloatComplex* __restrict__ d_coeff,
                                         float* __restrict__ d_rsrp_partials,
                                         float time_alignment_s,
                                         const srs_estimator_time_alignment_t* __restrict__ d_time_alignment,
                                         int use_device_time_alignment,
                                         srs_estimator_config_t cfg)
{
    int rx  = blockIdx.x;
    int tx  = blockIdx.y;
    int tid = threadIdx.x;
    if (rx >= cfg.nof_rx_ports || tx >= cfg.nof_tx_ports) {
        return;
    }

    __shared__ float partial_re[SRS_ESTIMATOR_THREADS];
    __shared__ float partial_im[SRS_ESTIMATOR_THREADS];

    if (use_device_time_alignment != 0) {
        float combined_time_alignment = 0.0f;
        for (int tx_index = 0; tx_index != cfg.nof_tx_ports; ++tx_index) {
            combined_time_alignment += d_time_alignment[tx_index].time_alignment_s;
        }
        time_alignment_s = combined_time_alignment / static_cast<float>(cfg.nof_tx_ports);
    }

    float phase_shift_subcarrier =
        TWO_PI * time_alignment_s * static_cast<float>(cfg.scs_khz * 1000) * static_cast<float>(cfg.comb_size);
    float phase_shift_offset =
        phase_shift_subcarrier * static_cast<float>(cfg.mapping_initial_subcarrier[tx]) /
        static_cast<float>(cfg.comb_size);

    float sum_re = 0.0f;
    float sum_im = 0.0f;
    for (int n = tid; n < cfg.sequence_length; n += blockDim.x) {
        cuFloatComplex h = d_lse[(tx * cfg.nof_rx_ports + rx) * cfg.sequence_length + n];
        cuFloatComplex c = quantized_phase(static_cast<float>(n) * phase_shift_subcarrier + phase_shift_offset);
        cuFloatComplex compensated = cmul(h, c);
        sum_re += compensated.x;
        sum_im += compensated.y;
    }

    partial_re[tid] = sum_re;
    partial_im[tid] = sum_im;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial_re[tid] += partial_re[tid + offset];
            partial_im[tid] += partial_im[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float inv_len = 1.0f / static_cast<float>(cfg.sequence_length);
        cuFloatComplex coeff = make_cuFloatComplex(partial_re[0] * inv_len, partial_im[0] * inv_len);
        d_coeff[rx * cfg.nof_tx_ports + tx] = coeff;
        d_rsrp_partials[rx * cfg.nof_tx_ports + tx] = norm2(coeff);
    }
}

__global__ void srs_noise_partial_kernel(const cuFloatComplex* __restrict__ d_noise_help,
                                         const cuFloatComplex* __restrict__ d_sequences,
                                         const cuFloatComplex* __restrict__ d_coeff,
                                         float* __restrict__ d_noise_partials,
                                         float time_alignment_s,
                                         const srs_estimator_time_alignment_t* __restrict__ d_time_alignment,
                                         int use_device_time_alignment,
                                         srs_estimator_config_t cfg)
{
    int rx  = blockIdx.x;
    int set = blockIdx.y;
    int tid = threadIdx.x;
    int nof_sets = (cfg.interleaved_pilots != 0) ? 2 : 1;
    if (rx >= cfg.nof_rx_ports || set >= nof_sets) {
        return;
    }

    __shared__ float partial[SRS_ESTIMATOR_THREADS];

    int reference_tx = (cfg.interleaved_pilots != 0) ? set : 0;
    if (use_device_time_alignment != 0) {
        float combined_time_alignment = 0.0f;
        for (int tx_index = 0; tx_index != cfg.nof_tx_ports; ++tx_index) {
            combined_time_alignment += d_time_alignment[tx_index].time_alignment_s;
        }
        time_alignment_s = combined_time_alignment / static_cast<float>(cfg.nof_tx_ports);
    }

    float phase_shift_subcarrier =
        TWO_PI * time_alignment_s * static_cast<float>(cfg.scs_khz * 1000) * static_cast<float>(cfg.comb_size);
    float phase_shift_offset =
        phase_shift_subcarrier * static_cast<float>(cfg.mapping_initial_subcarrier[reference_tx]) /
        static_cast<float>(cfg.comb_size);

    float local_sum = 0.0f;
    for (int n = tid; n < cfg.sequence_length; n += blockDim.x) {
        cuFloatComplex residual = d_noise_help[(set * cfg.nof_rx_ports + rx) * cfg.sequence_length + n];
        cuFloatComplex c = quantized_phase(static_cast<float>(n) * phase_shift_subcarrier + phase_shift_offset);
        residual = cmul(residual, c);

        for (int tx = 0; tx != cfg.nof_tx_ports; ++tx) {
            if ((cfg.interleaved_pilots != 0) && ((tx & 1) != set)) {
                continue;
            }
            cuFloatComplex seq = d_sequences[tx * cfg.sequence_length + n];
            cuFloatComplex coeff = d_coeff[rx * cfg.nof_tx_ports + tx];
            cuFloatComplex recovered = cmul(seq, coeff);
            recovered.x *= static_cast<float>(cfg.nof_symbols);
            recovered.y *= static_cast<float>(cfg.nof_symbols);
            residual = csub(residual, recovered);
        }
        local_sum += norm2(residual);
    }

    partial[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_noise_partials[rx * nof_sets + set] = partial[0];
    }
}

__global__ void srs_finalize_result_kernel(const cuFloatComplex* __restrict__ d_coeff,
                                           const float* __restrict__ d_epre_partials,
                                           const float* __restrict__ d_noise_partials,
                                           const float* __restrict__ d_rsrp_partials,
                                           const srs_estimator_time_alignment_t* __restrict__ d_time_alignment_per_tx,
                                           int use_device_time_alignment,
                                           float time_alignment_s,
                                           srs_estimator_device_result* __restrict__ d_result,
                                           srs_estimator_config_t cfg)
{
    int tid = threadIdx.x;

    __shared__ float epre_sum;
    __shared__ float noise_sum;
    __shared__ float rsrp_sum;
    __shared__ float inv_noise_std;
    __shared__ float partial[SRS_ESTIMATOR_THREADS];

    int nof_sequence_blocks = (cfg.sequence_length + blockDim.x - 1) / blockDim.x;
    int nof_epre_partials = cfg.nof_tx_ports * cfg.nof_rx_ports * nof_sequence_blocks;
    float local_sum = 0.0f;
    for (int i = tid; i < nof_epre_partials; i += blockDim.x) {
        local_sum += d_epre_partials[i];
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        epre_sum = partial[0];
    }
    __syncthreads();

    int nof_sets = (cfg.interleaved_pilots != 0) ? 2 : 1;
    int nof_noise_partials = cfg.nof_rx_ports * nof_sets;
    local_sum = 0.0f;
    for (int i = tid; i < nof_noise_partials; i += blockDim.x) {
        local_sum += d_noise_partials[i];
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        noise_sum = partial[0];
    }
    __syncthreads();

    int nof_coeff = cfg.nof_rx_ports * cfg.nof_tx_ports;
    local_sum = 0.0f;
    for (int i = tid; i < nof_coeff; i += blockDim.x) {
        local_sum += d_rsrp_partials[i];
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        rsrp_sum = partial[0];
    }
    __syncthreads();

    if (tid == 0) {
        srs_estimator_time_alignment_t time_alignment =
            srs_estimator_time_alignment_t{time_alignment_s, 0.0f, 0.0f, 0.0f};
        if (use_device_time_alignment != 0) {
            float combined_time_alignment = 0.0f;
            float resolution = 0.0f;
            float min_value = FLT_MIN;
            float max_value = FLT_MAX;
            for (int tx = 0; tx != cfg.nof_tx_ports; ++tx) {
                combined_time_alignment += d_time_alignment_per_tx[tx].time_alignment_s;
                min_value = fmaxf(min_value, d_time_alignment_per_tx[tx].min_s);
                max_value = fminf(max_value, d_time_alignment_per_tx[tx].max_s);
                resolution = fmaxf(resolution, d_time_alignment_per_tx[tx].resolution_s);
            }
            time_alignment = srs_estimator_time_alignment_t{
                combined_time_alignment / static_cast<float>(cfg.nof_tx_ports), resolution, min_value, max_value};
        }
        d_result->time_alignment = time_alignment;

        int   correction_factor = (cfg.interleaved_pilots != 0) ? 2 : 1;
        int   nof_estimates = (cfg.interleaved_pilots != 0) ? 2 : cfg.nof_tx_ports;
        float noise_denom = static_cast<float>((cfg.nof_symbols * cfg.sequence_length - nof_estimates) *
                                               correction_factor * cfg.nof_rx_ports);
        d_result->metrics.noise_variance = noise_sum / noise_denom;
        d_result->metrics.epre_linear = epre_sum / static_cast<float>(cfg.sequence_length * cfg.nof_symbols *
                                                                      correction_factor * cfg.nof_rx_ports);
        d_result->metrics.rsrp_linear = rsrp_sum / static_cast<float>(cfg.nof_tx_ports * cfg.nof_rx_ports);
        d_result->metrics.epre_dB = srs_to_db(d_result->metrics.epre_linear);
        d_result->metrics.rsrp_dB = srs_to_db(d_result->metrics.rsrp_linear);

        float noise_std =
            fmaxf(sqrtf(d_result->metrics.noise_variance), sqrtf(d_result->metrics.rsrp_linear) * 0.01f);
        inv_noise_std = (noise_std > 0.0f) ? (1.0f / noise_std) : 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < nof_coeff; i += blockDim.x) {
        d_result->metrics.coeff[i].real = d_coeff[i].x * inv_noise_std;
        d_result->metrics.coeff[i].imag = d_coeff[i].y * inv_noise_std;
    }
}

nr_ldpc_status_t launch_fast_auto_ta_operations(srs_estimator_ctx* handle, const void* d_grid_cbf16, cudaStream_t stream)
{
    const srs_estimator_config_t& cfg = handle->config;
    constexpr int                 threads = SRS_ESTIMATOR_THREADS;

    dim3 blocks_lse(cfg.nof_tx_ports, cfg.nof_rx_ports, (cfg.sequence_length + threads - 1) / threads);
    srs_extract_lse_partial_kernel<<<blocks_lse, threads, 0, stream>>>(static_cast<const unsigned int*>(d_grid_cbf16),
                                                                      handle->d_sequences,
                                                                      handle->d_lse,
                                                                      handle->d_noise_help,
                                                                      handle->d_epre_partials,
                                                                      cfg);

    if (uses_windowed_correlation(cfg)) {
        dim3 blocks_corr(cfg.correlation_window_size * 2, cfg.nof_tx_ports);
        srs_correlation_window_kernel<<<blocks_corr, threads, 0, stream>>>(
            handle->d_lse, handle->d_twiddles, handle->d_correlation, cfg);
    } else {
        size_t idft_bytes =
            static_cast<size_t>(cfg.nof_tx_ports) * cfg.nof_rx_ports * cfg.dft_size * sizeof(cuFloatComplex);
        if (cudaMemsetAsync(handle->d_idft, 0, idft_bytes, stream) != cudaSuccess) {
            return NR_LDPC_ERROR_CUDA_FAILED;
        }

        dim3 blocks_pack(cfg.nof_tx_ports * cfg.nof_rx_ports, (cfg.sequence_length + threads - 1) / threads);
        srs_pack_lse_for_idft_kernel<<<blocks_pack, threads, 0, stream>>>(handle->d_lse, handle->d_idft, cfg);

        if (!ensure_vkfft_plan(handle, stream)) {
            return NR_LDPC_ERROR_CUDA_FAILED;
        }
        VkFFTLaunchParams launch_params = {};
        launch_params.buffer = &handle->vkfft_buffer;
        if (VkFFTAppend(&handle->vkfft_app, 1, &launch_params) != VKFFT_SUCCESS) {
            return NR_LDPC_ERROR_CUDA_FAILED;
        }

        dim3 blocks_corr((cfg.dft_size + threads - 1) / threads, cfg.nof_tx_ports);
        srs_correlation_from_idft_kernel<<<blocks_corr, threads, 0, stream>>>(
            handle->d_idft, handle->d_correlation, cfg);
    }

    srs_estimate_ta_kernel<<<cfg.nof_tx_ports, threads, 0, stream>>>(
        handle->d_correlation, handle->d_time_alignment_per_tx, cfg);

    dim3 blocks_coeff(cfg.nof_rx_ports, cfg.nof_tx_ports);
    srs_coeff_partial_kernel<<<blocks_coeff, threads, 0, stream>>>(
        handle->d_lse, handle->d_coeff, handle->d_rsrp_partials, 0.0f, handle->d_time_alignment_per_tx, 1, cfg);

    int  nof_sets = (cfg.interleaved_pilots != 0) ? 2 : 1;
    dim3 blocks_noise(cfg.nof_rx_ports, nof_sets);
    srs_noise_partial_kernel<<<blocks_noise, threads, 0, stream>>>(handle->d_noise_help,
                                                                   handle->d_sequences,
                                                                   handle->d_coeff,
                                                                   handle->d_noise_partials,
                                                                   0.0f,
                                                                   handle->d_time_alignment_per_tx,
                                                                   1,
                                                                   cfg);

    srs_finalize_result_kernel<<<1, threads, 0, stream>>>(handle->d_coeff,
                                                          handle->d_epre_partials,
                                                          handle->d_noise_partials,
                                                          handle->d_rsrp_partials,
                                                          handle->d_time_alignment_per_tx,
                                                          1,
                                                          0.0f,
                                                          handle->d_result,
                                                          cfg);
    return NR_LDPC_SUCCESS;
}

} // namespace

nr_ldpc_status_t srs_estimator_create(srs_estimator_handle_t* handle)
{
    if (handle == nullptr) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    srs_estimator_ctx* ctx = new (std::nothrow) srs_estimator_ctx();
    if (ctx == nullptr) {
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }

    *handle = ctx;
    return NR_LDPC_SUCCESS;
}

void srs_estimator_destroy(srs_estimator_handle_t handle)
{
    if (handle == nullptr) {
        return;
    }

    destroy_graph(handle);
    destroy_vkfft_plan(handle);
    cudaFree(handle->d_sequences);
    cudaFree(handle->d_lse);
    cudaFree(handle->d_idft);
    cudaFree(handle->d_twiddles);
    cudaFree(handle->d_noise_help);
    cudaFree(handle->d_coeff);
    cudaFree(handle->d_correlation);
    cudaFree(handle->d_time_alignment_per_tx);
    cudaFree(handle->d_epre_partials);
    cudaFree(handle->d_noise_partials);
    cudaFree(handle->d_rsrp_partials);
    cudaFree(handle->d_result);
    cudaFreeHost(handle->h_result);
    delete handle;
}

nr_ldpc_status_t srs_estimator_configure(srs_estimator_handle_t        handle,
                                         const srs_estimator_config_t* cfg,
                                         const srs_estimator_cf_t*     sequences,
                                         void*                         external_stream)
{
    if (handle == nullptr || cfg == nullptr || sequences == nullptr) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }
    if (!validate_config(*cfg)) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(external_stream);
    destroy_graph(handle);
    handle->config = *cfg;
    if (uses_windowed_correlation(*cfg)) {
        destroy_vkfft_plan(handle);
    }
    if (!ensure_capacity(handle) || (!uses_windowed_correlation(*cfg) && !ensure_vkfft_plan(handle, stream))) {
        return NR_LDPC_ERROR_ALLOC_FAILED;
    }

    size_t nof_sequences = static_cast<size_t>(cfg->nof_tx_ports) * cfg->sequence_length;
    cudaError_t err = cudaMemcpyAsync(handle->d_sequences,
                                      sequences,
                                      nof_sequences * sizeof(cuFloatComplex),
                                      cudaMemcpyHostToDevice,
                                      stream);
    if (err != cudaSuccess) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    if (uses_windowed_correlation(*cfg) &&
        (!handle->twiddles_valid || handle->twiddle_dft_size != cfg->dft_size ||
         handle->twiddle_sequence_length != cfg->sequence_length ||
         handle->twiddle_window_size != cfg->correlation_window_size)) {
        constexpr int threads = 256;
        dim3 blocks(cfg->correlation_window_size * 2, (cfg->sequence_length + threads - 1) / threads);
        srs_generate_twiddles_kernel<<<blocks, threads, 0, stream>>>(handle->d_twiddles, *cfg);
        if (cudaGetLastError() != cudaSuccess) {
            handle->twiddles_valid = false;
            return NR_LDPC_ERROR_CUDA_FAILED;
        }
        handle->twiddle_dft_size = cfg->dft_size;
        handle->twiddle_sequence_length = cfg->sequence_length;
        handle->twiddle_window_size = cfg->correlation_window_size;
        handle->twiddles_valid = true;
    }

    return NR_LDPC_SUCCESS;
}

nr_ldpc_status_t srs_estimator_estimate_auto_ta(srs_estimator_handle_t          handle,
                                                const void*                     d_grid_cbf16,
                                                float                           max_time_alignment_s,
                                                srs_estimator_time_alignment_t* time_alignment,
                                                srs_estimator_metrics_t*        metrics,
                                                void*                           external_stream)
{
    (void)max_time_alignment_s;
    if (handle == nullptr || d_grid_cbf16 == nullptr || time_alignment == nullptr || metrics == nullptr ||
        handle->d_result == nullptr || handle->h_result == nullptr) {
        return NR_LDPC_ERROR_INVALID_CONFIG;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(external_stream);
    const srs_estimator_config_t& cfg = handle->config;
    bool graph_launched = false;

    if (uses_windowed_correlation(cfg)) {
        if ((handle->graph_exec != nullptr) && (handle->graph_grid_cbf16 == d_grid_cbf16)) {
            if (cudaGraphLaunch(handle->graph_exec, stream) == cudaSuccess) {
                graph_launched = true;
            } else {
                destroy_graph(handle);
            }
        }

        if (!graph_launched) {
            std::lock_guard<std::mutex> lock(graph_capture_mutex());
            destroy_graph(handle);
            cudaGraph_t captured_graph = nullptr;
            if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                nr_ldpc_status_t launch_status = launch_fast_auto_ta_operations(handle, d_grid_cbf16, stream);
                cudaError_t      copy_status   = cudaMemcpyAsync(handle->h_result,
                                                            handle->d_result,
                                                            sizeof(srs_estimator_device_result),
                                                            cudaMemcpyDeviceToHost,
                                                            stream);
                cudaError_t      capture_status = cudaStreamEndCapture(stream, &captured_graph);
                if (launch_status == NR_LDPC_SUCCESS && copy_status == cudaSuccess && capture_status == cudaSuccess &&
                    captured_graph != nullptr &&
                    cudaGraphInstantiate(&handle->graph_exec, captured_graph, nullptr, nullptr, 0) == cudaSuccess) {
                    handle->graph           = captured_graph;
                    handle->graph_grid_cbf16 = d_grid_cbf16;
                    if (cudaGraphLaunch(handle->graph_exec, stream) == cudaSuccess) {
                        graph_launched = true;
                    } else {
                        destroy_graph(handle);
                    }
                } else if (captured_graph != nullptr) {
                    cudaGraphDestroy(captured_graph);
                }
            }
        }
    }

    if (!graph_launched) {
        nr_ldpc_status_t launch_status = launch_fast_auto_ta_operations(handle, d_grid_cbf16, stream);
        if (launch_status != NR_LDPC_SUCCESS) {
            return launch_status;
        }

        cudaError_t err = cudaMemcpyAsync(
            handle->h_result, handle->d_result, sizeof(srs_estimator_device_result), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            return NR_LDPC_ERROR_CUDA_FAILED;
        }
    }

    if (cudaPeekAtLastError() != cudaSuccess) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }
    if (stream_synchronize_yielding(stream) != cudaSuccess) {
        return NR_LDPC_ERROR_CUDA_FAILED;
    }

    *time_alignment = handle->h_result->time_alignment;
    *metrics        = handle->h_result->metrics;

    return NR_LDPC_SUCCESS;
}
