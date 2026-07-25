/**
 * @file cpu_baseline_benchmark.cpp
 * @brief CPU baseline benchmark for comparison with GPU OCUDU PHY CUDA
 *
 * Measures CPU-only performance for:
 *   - LDPC encoding (reference implementation)
 *   - LDPC decoding (reference implementation)
 *   - Modulation/Demodulation
 *   - Scrambling/Descrambling
 *
 * Outputs JSON for integration with OCUDU PHY CUDA perflog
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

// ============================================================================
// CPU Power Estimation
// ============================================================================

struct CpuTimes {
    uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
    uint64_t total_active() const { return user + nice + system + irq + softirq + steal; }
    uint64_t total() const { return total_active() + idle + iowait; }
};

CpuTimes read_cpu_times() {
    CpuTimes t = {0};
    FILE* fp = fopen("/proc/stat", "r");
    if (fp) {
        char line[256];
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "cpu %lu %lu %lu %lu %lu %lu %lu %lu",
                   &t.user, &t.nice, &t.system, &t.idle,
                   &t.iowait, &t.irq, &t.softirq, &t.steal);
        }
        fclose(fp);
    }
    return t;
}

struct CpuPowerResult {
    double utilization_pct;
    double estimated_watts;
    bool valid;
};

// GH200 power estimation
constexpr double TDP_WATTS = 250.0;
constexpr double IDLE_WATTS = 15.0;

// ============================================================================
// Timing Utilities
// ============================================================================

struct StageTiming {
    std::vector<double> samples_us;
    double min_us = 0, max_us = 0, avg_us = 0;
    double p50_us = 0, p90_us = 0, p99_us = 0;

    void compute_stats() {
        if (samples_us.empty()) return;
        std::sort(samples_us.begin(), samples_us.end());
        min_us = samples_us.front();
        max_us = samples_us.back();
        avg_us = std::accumulate(samples_us.begin(), samples_us.end(), 0.0) / samples_us.size();
        size_t n = samples_us.size();
        p50_us = samples_us[n / 2];
        p90_us = samples_us[n * 90 / 100];
        p99_us = samples_us[n * 99 / 100];
    }
};

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }
};

// ============================================================================
// Reference CPU Implementations (Simple/Naive for Baseline Comparison)
// ============================================================================

/**
 * Simple LDPC encoding simulation (not real LDPC, just representative computation)
 * Real LDPC would require the actual parity check matrix operations
 */
void cpu_ldpc_encode_reference(const uint8_t* input, uint8_t* output,
                                int K_bits, int N_bits, int iterations) {
    // Simulate LDPC encoding workload: matrix-vector multiplication pattern
    // This is a representative computational load, not actual LDPC
    for (int iter = 0; iter < iterations; iter++) {
        // Systematic bits (copy)
        memcpy(output, input, K_bits / 8);

        // Simulate parity generation with XOR operations
        int parity_bits = N_bits - K_bits;
        uint8_t* parity = output + K_bits / 8;

        for (int p = 0; p < parity_bits / 8; p++) {
            uint8_t val = 0;
            // Simulate sparse matrix multiplication
            for (int i = 0; i < K_bits / 8; i += 4) {
                val ^= input[i % (K_bits / 8)];
            }
            parity[p] = val ^ (p & 0xFF);
        }
    }
}

/**
 * Simple LDPC decoding simulation (min-sum-like workload)
 */
void cpu_ldpc_decode_reference(const int8_t* llrs, uint8_t* output,
                                int N_bits, int K_bits, int max_iterations) {
    // Simulate belief propagation workload
    std::vector<float> beliefs(N_bits);
    std::vector<float> messages(N_bits * 10);  // Simulate edge messages

    // Convert LLRs to beliefs
    for (int i = 0; i < N_bits; i++) {
        beliefs[i] = llrs[i] / 8.0f;
    }

    // Simulate iterations of message passing
    for (int iter = 0; iter < max_iterations; iter++) {
        // Check node update (simplified)
        for (int c = 0; c < N_bits / 10; c++) {
            float min1 = 100.0f, min2 = 100.0f;
            int sign = 1;
            for (int v = 0; v < 10; v++) {
                float abs_val = fabsf(messages[c * 10 + v]);
                sign *= (messages[c * 10 + v] >= 0) ? 1 : -1;
                if (abs_val < min1) { min2 = min1; min1 = abs_val; }
                else if (abs_val < min2) { min2 = abs_val; }
            }
            for (int v = 0; v < 10; v++) {
                float val = (fabsf(messages[c * 10 + v]) == min1) ? min2 : min1;
                int s = sign * ((messages[c * 10 + v] >= 0) ? 1 : -1);
                messages[c * 10 + v] = s * val * 0.75f;  // Scaling factor
            }
        }

        // Variable node update (simplified)
        for (int v = 0; v < N_bits; v++) {
            beliefs[v] = llrs[v] / 8.0f;
            for (int e = 0; e < 3; e++) {
                beliefs[v] += messages[(v * 3 + e) % messages.size()];
            }
        }
    }

    // Hard decision
    for (int i = 0; i < K_bits / 8; i++) {
        uint8_t byte = 0;
        for (int b = 0; b < 8; b++) {
            if (beliefs[i * 8 + b] < 0) byte |= (1 << b);
        }
        output[i] = byte;
    }
}

/**
 * CPU 64-QAM soft demodulation
 */
void cpu_demodulate_64qam(const float* symbols_re, const float* symbols_im,
                          int8_t* llrs, int num_symbols, float noise_var) {
    // 64-QAM constellation points: {-7, -5, -3, -1, 1, 3, 5, 7} / sqrt(42)
    const float scale = 1.0f / sqrtf(42.0f);
    float inv_var = 1.0f / noise_var;

    for (int s = 0; s < num_symbols; s++) {
        float re = symbols_re[s];
        float im = symbols_im[s];

        // Approximate LLR calculation for 64-QAM
        // LLR = ln(P(bit=0)/P(bit=1)) approximated by distance differences

        // Bit 0 (MSB of I): sign of real part
        float llr0 = -4.0f * re * scale * inv_var;

        // Bit 1: |re| > 4*scale
        float llr1 = (4.0f * scale - fabsf(re)) * 4.0f * inv_var;

        // Bit 2: |re| decision boundary at 2*scale
        float llr2 = (2.0f * scale - fabsf(fabsf(re) - 4.0f * scale)) * 4.0f * inv_var;

        // Bits 3-5: same for imaginary part
        float llr3 = -4.0f * im * scale * inv_var;
        float llr4 = (4.0f * scale - fabsf(im)) * 4.0f * inv_var;
        float llr5 = (2.0f * scale - fabsf(fabsf(im) - 4.0f * scale)) * 4.0f * inv_var;

        // Quantize to int8
        auto quantize = [](float v) -> int8_t {
            v = fmaxf(-127.0f, fminf(127.0f, v * 16.0f));
            return (int8_t)v;
        };

        llrs[s * 6 + 0] = quantize(llr0);
        llrs[s * 6 + 1] = quantize(llr1);
        llrs[s * 6 + 2] = quantize(llr2);
        llrs[s * 6 + 3] = quantize(llr3);
        llrs[s * 6 + 4] = quantize(llr4);
        llrs[s * 6 + 5] = quantize(llr5);
    }
}

/**
 * CPU 64-QAM modulation
 */
void cpu_modulate_64qam(const uint8_t* bits, float* symbols_re, float* symbols_im,
                        int num_symbols) {
    const float scale = 1.0f / sqrtf(42.0f);
    // Gray-coded mapping for 64-QAM
    const float map[] = {-7, -5, -1, -3, 7, 5, 1, 3};

    for (int s = 0; s < num_symbols; s++) {
        int byte_idx = (s * 6) / 8;
        int bit_offset = (s * 6) % 8;

        // Extract 6 bits (may span two bytes)
        uint16_t word = bits[byte_idx] | (bits[byte_idx + 1] << 8);
        int idx = (word >> bit_offset) & 0x3F;

        int i_idx = (idx >> 3) & 0x7;
        int q_idx = idx & 0x7;

        symbols_re[s] = map[i_idx] * scale;
        symbols_im[s] = map[q_idx] * scale;
    }
}

/**
 * CPU scrambling (XOR with LFSR sequence)
 */
void cpu_scramble(const uint8_t* input, uint8_t* output, int num_bytes,
                  uint32_t c_init) {
    // Gold sequence LFSR
    uint32_t x1 = 1;
    uint32_t x2 = c_init;

    // Advance LFSR by 1600 positions (as per 3GPP)
    for (int i = 0; i < 1600; i++) {
        x1 = (((x1 >> 3) ^ x1) & 1) | (x1 << 1);
        x2 = (((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1) | (x2 << 1);
    }

    // Generate scrambling sequence and apply
    for (int b = 0; b < num_bytes; b++) {
        uint8_t scr_byte = 0;
        for (int bit = 0; bit < 8; bit++) {
            uint32_t c = ((x1 >> 3) ^ x1 ^ (x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
            scr_byte |= (c << bit);
            x1 = (((x1 >> 3) ^ x1) & 1) | (x1 << 1);
            x2 = (((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1) | (x2 << 1);
        }
        output[b] = input[b] ^ scr_byte;
    }
}

// ============================================================================
// Benchmark Results Structure
// ============================================================================

struct CpuBenchmarkResult {
    // Configuration
    const char* bandwidth;
    int nof_prb;
    int tbs_bits;

    // Per-stage timings
    StageTiming ldpc_encode;
    StageTiming ldpc_decode;
    StageTiming modulation;
    StageTiming demodulation;
    StageTiming scrambling;

    // Aggregates
    double tx_total_us;
    double rx_total_us;
    double tx_throughput_mbps;
    double rx_throughput_mbps;

    // Power
    double power_watts;
    double tx_efficiency_mbps_per_watt;
    double rx_efficiency_mbps_per_watt;
};

// ============================================================================
// Multi-threaded Throughput Benchmark
// ============================================================================

struct MultiThreadResult {
    double throughput_mbps;
    double power_watts;
    double efficiency_mbps_per_watt;
    int num_threads;
    int batch_size;
};

MultiThreadResult benchmark_cpu_multithread_rx(int nof_prb, int num_threads, int batch_per_thread, int iterations) {
    MultiThreadResult result = {};
    result.num_threads = num_threads;
    result.batch_size = batch_per_thread;

    // Configuration matching GPU benchmark
    const int Qm = 6;  // 64-QAM
    const int data_symbols = 12;
    const int nof_data_re = nof_prb * 12 * data_symbols;
    const int nof_llrs = nof_data_re * Qm;

    // TBS calculation
    float code_rate = 0.5f;
    int tbs = std::min((int)(nof_llrs * code_rate * 0.9f), 8448);

    // LDPC parameters
    int K_bits = tbs;
    int N_bits = (int)(K_bits / code_rate);
    N_bits = ((N_bits + 383) / 384) * 384;

    // Pre-allocate per-thread buffers
    struct ThreadData {
        std::vector<float> symbols_re;
        std::vector<float> symbols_im;
        std::vector<int8_t> llrs;
        std::vector<uint8_t> decoded;
    };

    std::vector<ThreadData> thread_data(num_threads);
    for (int t = 0; t < num_threads; t++) {
        thread_data[t].symbols_re.resize(nof_data_re * batch_per_thread);
        thread_data[t].symbols_im.resize(nof_data_re * batch_per_thread);
        thread_data[t].llrs.resize(nof_llrs * batch_per_thread);
        thread_data[t].decoded.resize((K_bits / 8 + 1) * batch_per_thread);

        // Initialize with random data
        std::mt19937 rng(42 + t);
        for (auto& s : thread_data[t].symbols_re) s = (rng() / (float)UINT32_MAX - 0.5f) * 2.0f;
        for (auto& s : thread_data[t].symbols_im) s = (rng() / (float)UINT32_MAX - 0.5f) * 2.0f;
    }

    CpuTimes cpu_start = read_cpu_times();
    auto wall_start = std::chrono::high_resolution_clock::now();

    // Run parallel workload
    std::vector<std::thread> threads;
    std::atomic<int> total_slots_processed{0};

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            ThreadData& td = thread_data[t];
            for (int iter = 0; iter < iterations; iter++) {
                for (int b = 0; b < batch_per_thread; b++) {
                    int offset_re = b * nof_data_re;
                    int offset_llr = b * nof_llrs;
                    int offset_dec = b * (K_bits / 8 + 1);

                    // Demodulation
                    cpu_demodulate_64qam(
                        td.symbols_re.data() + offset_re,
                        td.symbols_im.data() + offset_re,
                        td.llrs.data() + offset_llr,
                        nof_data_re, 0.1f);

                    // LDPC Decode (5 iterations to match GPU)
                    cpu_ldpc_decode_reference(
                        td.llrs.data() + offset_llr,
                        td.decoded.data() + offset_dec,
                        N_bits, K_bits, 5);

                    total_slots_processed++;
                }
            }
        });
    }

    for (auto& t : threads) t.join();

    auto wall_end = std::chrono::high_resolution_clock::now();
    CpuTimes cpu_end = read_cpu_times();

    double wall_time_us = std::chrono::duration<double, std::micro>(wall_end - wall_start).count();
    int total_slots = total_slots_processed.load();

    result.throughput_mbps = (double)tbs * total_slots / wall_time_us;

    // Power estimation
    uint64_t active_delta = cpu_end.total_active() - cpu_start.total_active();
    uint64_t total_delta = cpu_end.total() - cpu_start.total();
    if (total_delta > 0) {
        double utilization = (double)active_delta / total_delta;
        result.power_watts = IDLE_WATTS + (TDP_WATTS - IDLE_WATTS) * utilization;
        result.efficiency_mbps_per_watt = result.throughput_mbps / result.power_watts;
    }

    return result;
}

MultiThreadResult benchmark_cpu_multithread_tx(int nof_prb, int num_threads, int batch_per_thread, int iterations) {
    MultiThreadResult result = {};
    result.num_threads = num_threads;
    result.batch_size = batch_per_thread;

    const int Qm = 6;
    const int data_symbols = 12;
    const int nof_data_re = nof_prb * 12 * data_symbols;
    const int nof_llrs = nof_data_re * Qm;

    float code_rate = 0.5f;
    int tbs = std::min((int)(nof_llrs * code_rate * 0.9f), 8448);

    int K_bits = tbs;
    int N_bits = (int)(K_bits / code_rate);
    N_bits = ((N_bits + 383) / 384) * 384;

    struct ThreadData {
        std::vector<uint8_t> info_bits;
        std::vector<uint8_t> encoded;
        std::vector<uint8_t> scrambled;
        std::vector<float> symbols_re;
        std::vector<float> symbols_im;
    };

    std::vector<ThreadData> thread_data(num_threads);
    for (int t = 0; t < num_threads; t++) {
        thread_data[t].info_bits.resize((K_bits / 8 + 1) * batch_per_thread);
        thread_data[t].encoded.resize((N_bits / 8 + 1) * batch_per_thread);
        thread_data[t].scrambled.resize((N_bits / 8 + 1) * batch_per_thread);
        thread_data[t].symbols_re.resize(nof_data_re * batch_per_thread);
        thread_data[t].symbols_im.resize(nof_data_re * batch_per_thread);

        std::mt19937 rng(42 + t);
        for (auto& b : thread_data[t].info_bits) b = rng();
    }

    CpuTimes cpu_start = read_cpu_times();
    auto wall_start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    std::atomic<int> total_slots_processed{0};

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            ThreadData& td = thread_data[t];
            uint32_t c_init = 0x1234 * 65536 + 100 + t;

            for (int iter = 0; iter < iterations; iter++) {
                for (int b = 0; b < batch_per_thread; b++) {
                    int offset_info = b * (K_bits / 8 + 1);
                    int offset_enc = b * (N_bits / 8 + 1);
                    int offset_re = b * nof_data_re;

                    cpu_ldpc_encode_reference(
                        td.info_bits.data() + offset_info,
                        td.encoded.data() + offset_enc,
                        K_bits, N_bits, 1);

                    cpu_scramble(
                        td.encoded.data() + offset_enc,
                        td.scrambled.data() + offset_enc,
                        N_bits / 8, c_init);

                    cpu_modulate_64qam(
                        td.scrambled.data() + offset_enc,
                        td.symbols_re.data() + offset_re,
                        td.symbols_im.data() + offset_re,
                        nof_data_re);

                    total_slots_processed++;
                }
            }
        });
    }

    for (auto& t : threads) t.join();

    auto wall_end = std::chrono::high_resolution_clock::now();
    CpuTimes cpu_end = read_cpu_times();

    double wall_time_us = std::chrono::duration<double, std::micro>(wall_end - wall_start).count();
    int total_slots = total_slots_processed.load();

    result.throughput_mbps = (double)tbs * total_slots / wall_time_us;

    uint64_t active_delta = cpu_end.total_active() - cpu_start.total_active();
    uint64_t total_delta = cpu_end.total() - cpu_start.total();
    if (total_delta > 0) {
        double utilization = (double)active_delta / total_delta;
        result.power_watts = IDLE_WATTS + (TDP_WATTS - IDLE_WATTS) * utilization;
        result.efficiency_mbps_per_watt = result.throughput_mbps / result.power_watts;
    }

    return result;
}

// ============================================================================
// Single-thread CPU Benchmark (for latency comparison)
// ============================================================================

CpuBenchmarkResult benchmark_cpu_pipeline(int nof_prb, int iterations) {
    CpuBenchmarkResult result = {};
    result.bandwidth = (nof_prb <= 51) ? "20MHz" : (nof_prb <= 106) ? "40MHz" : "100MHz";
    result.nof_prb = nof_prb;

    // Configuration matching GPU benchmark
    const int Qm = 6;  // 64-QAM
    const int data_symbols = 12;
    const int nof_data_re = nof_prb * 12 * data_symbols;
    const int nof_llrs = nof_data_re * Qm;

    // TBS calculation (simplified)
    float code_rate = 0.5f;
    int tbs = std::min((int)(nof_llrs * code_rate * 0.9f), 8448);
    result.tbs_bits = tbs;

    // LDPC parameters (BG2 approximation)
    int K_bits = tbs;
    int N_bits = (int)(K_bits / code_rate);
    N_bits = ((N_bits + 383) / 384) * 384;  // Align to lifting size

    // Allocate buffers
    std::vector<uint8_t> info_bits(K_bits / 8 + 1);
    std::vector<uint8_t> encoded_bits(N_bits / 8 + 1);
    std::vector<uint8_t> scrambled_bits(N_bits / 8 + 1);
    std::vector<float> symbols_re(nof_data_re);
    std::vector<float> symbols_im(nof_data_re);
    std::vector<int8_t> llrs(nof_llrs);
    std::vector<uint8_t> decoded_bits(K_bits / 8 + 1);

    // Initialize random data
    std::mt19937 rng(42);
    for (auto& b : info_bits) b = rng();
    for (auto& s : symbols_re) s = (rng() / (float)UINT32_MAX - 0.5f) * 2.0f;
    for (auto& s : symbols_im) s = (rng() / (float)UINT32_MAX - 0.5f) * 2.0f;

    // Scrambling init value
    uint32_t c_init = 0x1234 * 65536 + 100;

    // Warmup
    for (int w = 0; w < 5; w++) {
        cpu_ldpc_encode_reference(info_bits.data(), encoded_bits.data(), K_bits, N_bits, 1);
        cpu_scramble(encoded_bits.data(), scrambled_bits.data(), N_bits / 8, c_init);
        cpu_modulate_64qam(scrambled_bits.data(), symbols_re.data(), symbols_im.data(), nof_data_re);
        cpu_demodulate_64qam(symbols_re.data(), symbols_im.data(), llrs.data(), nof_data_re, 0.1f);
    }

    // Reserve timing vectors
    result.ldpc_encode.samples_us.reserve(iterations);
    result.ldpc_decode.samples_us.reserve(iterations);
    result.modulation.samples_us.reserve(iterations);
    result.demodulation.samples_us.reserve(iterations);
    result.scrambling.samples_us.reserve(iterations);

    Timer timer;

    // Measure CPU utilization during benchmarks
    CpuTimes cpu_start = read_cpu_times();

    // TX Pipeline benchmark
    double tx_time_total = 0;
    for (int i = 0; i < iterations; i++) {
        // LDPC Encode
        timer.start();
        cpu_ldpc_encode_reference(info_bits.data(), encoded_bits.data(), K_bits, N_bits, 1);
        double t = timer.elapsed_us();
        result.ldpc_encode.samples_us.push_back(t);
        tx_time_total += t;

        // Scrambling
        timer.start();
        cpu_scramble(encoded_bits.data(), scrambled_bits.data(), N_bits / 8, c_init);
        t = timer.elapsed_us();
        result.scrambling.samples_us.push_back(t);
        tx_time_total += t;

        // Modulation
        timer.start();
        cpu_modulate_64qam(scrambled_bits.data(), symbols_re.data(), symbols_im.data(), nof_data_re);
        t = timer.elapsed_us();
        result.modulation.samples_us.push_back(t);
        tx_time_total += t;
    }

    // RX Pipeline benchmark
    double rx_time_total = 0;
    for (int i = 0; i < iterations; i++) {
        // Demodulation
        timer.start();
        cpu_demodulate_64qam(symbols_re.data(), symbols_im.data(), llrs.data(), nof_data_re, 0.1f);
        double t = timer.elapsed_us();
        result.demodulation.samples_us.push_back(t);
        rx_time_total += t;

        // LDPC Decode
        timer.start();
        cpu_ldpc_decode_reference(llrs.data(), decoded_bits.data(), N_bits, K_bits, 5);
        t = timer.elapsed_us();
        result.ldpc_decode.samples_us.push_back(t);
        rx_time_total += t;
    }

    CpuTimes cpu_end = read_cpu_times();

    // Compute per-stage statistics
    result.ldpc_encode.compute_stats();
    result.ldpc_decode.compute_stats();
    result.modulation.compute_stats();
    result.demodulation.compute_stats();
    result.scrambling.compute_stats();

    // Compute aggregates
    result.tx_total_us = result.ldpc_encode.avg_us + result.scrambling.avg_us + result.modulation.avg_us;
    result.rx_total_us = result.demodulation.avg_us + result.ldpc_decode.avg_us;

    result.tx_throughput_mbps = (double)tbs / result.tx_total_us;
    result.rx_throughput_mbps = (double)tbs / result.rx_total_us;

    // Power estimation
    uint64_t active_delta = cpu_end.total_active() - cpu_start.total_active();
    uint64_t total_delta = cpu_end.total() - cpu_start.total();
    if (total_delta > 0) {
        double utilization = (double)active_delta / total_delta;
        result.power_watts = IDLE_WATTS + (TDP_WATTS - IDLE_WATTS) * utilization * 0.85;
        result.tx_efficiency_mbps_per_watt = result.tx_throughput_mbps / result.power_watts;
        result.rx_efficiency_mbps_per_watt = result.rx_throughput_mbps / result.power_watts;
    }

    return result;
}

// ============================================================================
// JSON Output
// ============================================================================

void write_json_output(const char* filename, const std::vector<CpuBenchmarkResult>& results) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }

    char hostname[256] = "unknown";
    FILE* hp = popen("hostname", "r");
    if (hp) {
        if (fgets(hostname, sizeof(hostname), hp)) hostname[strcspn(hostname, "\n")] = 0;
        pclose(hp);
    }

    time_t now = time(nullptr);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", localtime(&now));

    int cpu_cores = std::thread::hardware_concurrency();

    fprintf(f, "{\n");
    fprintf(f, "  \"timestamp\": \"%s\",\n", timestamp);
    fprintf(f, "  \"hostname\": \"%s\",\n", hostname);
    fprintf(f, "  \"cpu_cores\": %d,\n", cpu_cores);
    fprintf(f, "  \"benchmark_type\": \"cpu_baseline\",\n");
    fprintf(f, "  \"cpu_pipeline_results\": [\n");

    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"bandwidth\": \"%s\",\n", r.bandwidth);
        fprintf(f, "      \"nof_prb\": %d,\n", r.nof_prb);
        fprintf(f, "      \"tbs_bits\": %d,\n", r.tbs_bits);
        fprintf(f, "      \"tx\": {\n");
        fprintf(f, "        \"throughput_mbps\": %.2f,\n", r.tx_throughput_mbps);
        fprintf(f, "        \"total_latency_us\": %.2f,\n", r.tx_total_us);
        fprintf(f, "        \"efficiency_mbps_per_watt\": %.3f,\n", r.tx_efficiency_mbps_per_watt);
        fprintf(f, "        \"stages\": {\n");
        fprintf(f, "          \"ldpc_encode\": { \"avg_us\": %.2f, \"min_us\": %.2f, \"max_us\": %.2f, \"p50_us\": %.2f, \"p99_us\": %.2f },\n",
                r.ldpc_encode.avg_us, r.ldpc_encode.min_us, r.ldpc_encode.max_us, r.ldpc_encode.p50_us, r.ldpc_encode.p99_us);
        fprintf(f, "          \"scrambling\": { \"avg_us\": %.2f, \"min_us\": %.2f, \"max_us\": %.2f, \"p50_us\": %.2f, \"p99_us\": %.2f },\n",
                r.scrambling.avg_us, r.scrambling.min_us, r.scrambling.max_us, r.scrambling.p50_us, r.scrambling.p99_us);
        fprintf(f, "          \"modulation\": { \"avg_us\": %.2f, \"min_us\": %.2f, \"max_us\": %.2f, \"p50_us\": %.2f, \"p99_us\": %.2f }\n",
                r.modulation.avg_us, r.modulation.min_us, r.modulation.max_us, r.modulation.p50_us, r.modulation.p99_us);
        fprintf(f, "        }\n");
        fprintf(f, "      },\n");
        fprintf(f, "      \"rx\": {\n");
        fprintf(f, "        \"throughput_mbps\": %.2f,\n", r.rx_throughput_mbps);
        fprintf(f, "        \"total_latency_us\": %.2f,\n", r.rx_total_us);
        fprintf(f, "        \"efficiency_mbps_per_watt\": %.3f,\n", r.rx_efficiency_mbps_per_watt);
        fprintf(f, "        \"stages\": {\n");
        fprintf(f, "          \"demodulation\": { \"avg_us\": %.2f, \"min_us\": %.2f, \"max_us\": %.2f, \"p50_us\": %.2f, \"p99_us\": %.2f },\n",
                r.demodulation.avg_us, r.demodulation.min_us, r.demodulation.max_us, r.demodulation.p50_us, r.demodulation.p99_us);
        fprintf(f, "          \"ldpc_decode\": { \"avg_us\": %.2f, \"min_us\": %.2f, \"max_us\": %.2f, \"p50_us\": %.2f, \"p99_us\": %.2f }\n",
                r.ldpc_decode.avg_us, r.ldpc_decode.min_us, r.ldpc_decode.max_us, r.ldpc_decode.p50_us, r.ldpc_decode.p99_us);
        fprintf(f, "        }\n");
        fprintf(f, "      },\n");
        fprintf(f, "      \"power_watts\": %.1f\n", r.power_watts);
        fprintf(f, "    }%s\n", (i < results.size() - 1) ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);

    printf("CPU baseline results written to: %s\n", filename);
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  -o <file>  Output JSON file (default: cpu_baseline.json)\n");
    printf("  -n <iter>  Number of iterations (default: 100)\n");
    printf("  -h         Show help\n");
}

int main(int argc, char** argv) {
    const char* output_file = "cpu_baseline.json";
    int iterations = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("================================================================================\n");
    printf("CPU Baseline Benchmark (Reference Implementation)\n");
    printf("================================================================================\n");
    printf("Iterations: %d\n", iterations);
    printf("CPU Cores: %d\n", std::thread::hardware_concurrency());
    printf("\n");

    std::vector<CpuBenchmarkResult> results;

    // Test all bandwidths: 20, 40, 100 MHz
    int bandwidths[] = {51, 106, 273};
    const char* bw_names[] = {"20MHz", "40MHz", "100MHz"};

    for (int bw_idx = 0; bw_idx < 3; bw_idx++) {
        printf("Benchmarking %s (%d PRBs)...\n", bw_names[bw_idx], bandwidths[bw_idx]);
        CpuBenchmarkResult r = benchmark_cpu_pipeline(bandwidths[bw_idx], iterations);
        results.push_back(r);

        printf("  TX: %.1f Mbps @ %.1f W = %.2f Mbps/W\n",
               r.tx_throughput_mbps, r.power_watts, r.tx_efficiency_mbps_per_watt);
        printf("      Stages: LDPC=%.1fus, Scr=%.1fus, Mod=%.1fus\n",
               r.ldpc_encode.avg_us, r.scrambling.avg_us, r.modulation.avg_us);
        printf("  RX: %.1f Mbps @ %.1f W = %.2f Mbps/W\n",
               r.rx_throughput_mbps, r.power_watts, r.rx_efficiency_mbps_per_watt);
        printf("      Stages: Demod=%.1fus, LDPC=%.1fus\n",
               r.demodulation.avg_us, r.ldpc_decode.avg_us);
        printf("\n");
    }

    // Summary - Single Thread
    printf("================================================================================\n");
    printf("SUMMARY - Single-Thread (Latency)\n");
    printf("================================================================================\n");
    for (const auto& r : results) {
        printf("%s: TX %.1f Mbps @ %.1f us, RX %.1f Mbps @ %.1f us\n",
               r.bandwidth, r.tx_throughput_mbps, r.tx_total_us,
               r.rx_throughput_mbps, r.rx_total_us);
    }
    printf("\n");

    // Multi-threaded throughput benchmark
    printf("================================================================================\n");
    printf("Multi-Threaded Throughput Benchmark\n");
    printf("================================================================================\n");

    int num_cores = std::thread::hardware_concurrency();
    int test_threads[] = {1, 4, 8, 16, 32, num_cores};
    int num_thread_configs = 6;

    // Find best multi-thread config for 20MHz
    printf("\n=== RX Multi-Thread Sweep (20MHz, 8448 TBS) ===\n");
    printf("Threads   Batch   Throughput   Efficiency   Speedup\n");
    printf("------------------------------------------------------\n");

    double single_thread_rx = results[0].rx_throughput_mbps;
    MultiThreadResult best_rx = {};
    best_rx.throughput_mbps = 0;

    for (int i = 0; i < num_thread_configs; i++) {
        int threads = test_threads[i];
        if (threads > num_cores) continue;

        // Test with batch=10 per thread
        MultiThreadResult mr = benchmark_cpu_multithread_rx(51, threads, 10, 5);
        double speedup = mr.throughput_mbps / single_thread_rx;
        printf("%4d      %4d    %8.1f     %6.2f       %.1fx\n",
               threads, mr.batch_size, mr.throughput_mbps, mr.efficiency_mbps_per_watt, speedup);

        if (mr.throughput_mbps > best_rx.throughput_mbps) {
            best_rx = mr;
        }
    }

    printf("\n=== TX Multi-Thread Sweep (20MHz, 8448 TBS) ===\n");
    printf("Threads   Batch   Throughput   Efficiency   Speedup\n");
    printf("------------------------------------------------------\n");

    double single_thread_tx = results[0].tx_throughput_mbps;
    MultiThreadResult best_tx = {};
    best_tx.throughput_mbps = 0;

    for (int i = 0; i < num_thread_configs; i++) {
        int threads = test_threads[i];
        if (threads > num_cores) continue;

        MultiThreadResult mr = benchmark_cpu_multithread_tx(51, threads, 10, 5);
        double speedup = mr.throughput_mbps / single_thread_tx;
        printf("%4d      %4d    %8.1f     %6.2f       %.1fx\n",
               threads, mr.batch_size, mr.throughput_mbps, mr.efficiency_mbps_per_watt, speedup);

        if (mr.throughput_mbps > best_tx.throughput_mbps) {
            best_tx = mr;
        }
    }

    // Final summary
    printf("\n================================================================================\n");
    printf("FINAL SUMMARY (20MHz, 8448 TBS, 5 LDPC iterations)\n");
    printf("================================================================================\n");
    printf("Single-Thread:\n");
    printf("  TX: %.1f Mbps @ %.1f us latency\n", results[0].tx_throughput_mbps, results[0].tx_total_us);
    printf("  RX: %.1f Mbps @ %.1f us latency\n", results[0].rx_throughput_mbps, results[0].rx_total_us);
    printf("\nMulti-Thread Peak (%d cores available):\n", num_cores);
    printf("  TX: %.1f Mbps @ %.1f W = %.2f Mbps/W (%d threads)\n",
           best_tx.throughput_mbps, best_tx.power_watts, best_tx.efficiency_mbps_per_watt, best_tx.num_threads);
    printf("  RX: %.1f Mbps @ %.1f W = %.2f Mbps/W (%d threads)\n",
           best_rx.throughput_mbps, best_rx.power_watts, best_rx.efficiency_mbps_per_watt, best_rx.num_threads);
    printf("\n");

    // Write JSON
    write_json_output(output_file, results);

    return 0;
}
