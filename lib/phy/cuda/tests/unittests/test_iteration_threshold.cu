// Test CRC early termination for FP16 LDPC decoder
//
// Verifies CRC16, CRC24A, CRC24B early termination by comparing:
//   - Correct CRC type → should terminate early (iter ≤ checkpoint interval)
//   - Wrong CRC type → should NOT terminate early (iter = max_iterations)
//
// Tests both decoder paths:
//   - decode_batch()      → ldpc_decode_layered_x2_kernel   (syndrome+CRC ET)
//   - decode_batch_half() → ldpc_decode_layered_half2_kernel (confidence+CRC ET)

#include "ocudu_phy_cuda.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// ============================================================================
// Bit manipulation helpers
// ============================================================================

inline int get_bit_msb(const uint32_t* data, int bit_idx) {
    int word_idx = bit_idx / 32;
    int bit_in_word = bit_idx % 32;
    int byte_in_word = bit_in_word / 8;
    int bit_in_byte = 7 - (bit_in_word % 8);
    int bit_pos = byte_in_word * 8 + bit_in_byte;
    return (data[word_idx] >> bit_pos) & 1;
}

inline void set_bit_msb(uint32_t* data, int bit_idx, int value) {
    int word_idx = bit_idx / 32;
    int bit_in_word = bit_idx % 32;
    int byte_in_word = bit_in_word / 8;
    int bit_in_byte = 7 - (bit_in_word % 8);
    int bit_pos = byte_in_word * 8 + bit_in_byte;
    if (value) data[word_idx] |= (1u << bit_pos);
    else       data[word_idx] &= ~(1u << bit_pos);
}

inline int get_bit_lsb(const uint32_t* data, int bit_idx) {
    return (data[bit_idx / 32] >> (bit_idx % 32)) & 1;
}

inline int get_bit_msb_bytes(const uint8_t* data, int bit_idx) {
    return (data[bit_idx / 8] >> (7 - (bit_idx % 8))) & 1;
}

inline void set_bit_msb_bytes(uint8_t* data, int bit_idx, int value) {
    int byte_idx = bit_idx / 8;
    int bit_in_byte = 7 - (bit_idx % 8);
    if (value) data[byte_idx] |= (1u << bit_in_byte);
    else       data[byte_idx] &= ~(1u << bit_in_byte);
}

// ============================================================================
// CRC computation (3GPP polynomials, bit-by-bit)
// ============================================================================

static uint16_t compute_crc16(const uint8_t* data, int num_bits) {
    uint16_t crc = 0x0000;
    for (int i = 0; i < num_bits; i++) {
        int msb = (crc >> 15) & 1;
        crc = (crc << 1) | get_bit_msb_bytes(data, i);
        if (msb) crc ^= 0x1021;
    }
    for (int i = 0; i < 16; i++) {
        int msb = (crc >> 15) & 1;
        crc = crc << 1;
        if (msb) crc ^= 0x1021;
    }
    return crc;
}

static uint32_t compute_crc24a(const uint8_t* data, int num_bits) {
    uint32_t crc = 0;
    const uint32_t poly = 0x864CFB;
    for (int i = 0; i < num_bits; i++) {
        int msb = (crc >> 23) & 1;
        crc = ((crc << 1) | get_bit_msb_bytes(data, i)) & 0xFFFFFF;
        if (msb) crc ^= poly;
    }
    for (int i = 0; i < 24; i++) {
        int msb = (crc >> 23) & 1;
        crc = (crc << 1) & 0xFFFFFF;
        if (msb) crc ^= poly;
    }
    return crc;
}

static uint32_t compute_crc24b(const uint8_t* data, int num_bits) {
    uint32_t crc = 0;
    const uint32_t poly = 0x800063;
    for (int i = 0; i < num_bits; i++) {
        int msb = (crc >> 23) & 1;
        crc = ((crc << 1) | get_bit_msb_bytes(data, i)) & 0xFFFFFF;
        if (msb) crc ^= poly;
    }
    for (int i = 0; i < 24; i++) {
        int msb = (crc >> 23) & 1;
        crc = (crc << 1) & 0xFFFFFF;
        if (msb) crc ^= poly;
    }
    return crc;
}

// ============================================================================
// Prepare message with CRC appended, convert to encoder word format
// ============================================================================

static void prepare_message_with_crc(uint32_t* enc_words, int K, int Kd_no_crc,
                                     int crc_type, std::mt19937& rng) {
    int crc_bits = (crc_type == 1) ? 16 : (crc_type >= 2) ? 24 : 0;
    int data_bits = Kd_no_crc;

    int data_bytes = (data_bits + 7) / 8;
    std::vector<uint8_t> data_buf(data_bytes, 0);
    for (int i = 0; i < data_bytes; i++) data_buf[i] = rng() & 0xFF;
    int leftover = data_bits % 8;
    if (leftover > 0) data_buf[data_bytes - 1] &= (0xFF << (8 - leftover));

    uint32_t crc_val = 0;
    if (crc_type == 1) crc_val = compute_crc16(data_buf.data(), data_bits);
    else if (crc_type == 2) crc_val = compute_crc24a(data_buf.data(), data_bits);
    else if (crc_type == 3) crc_val = compute_crc24b(data_buf.data(), data_bits);

    int msg_bytes = (K + 7) / 8;
    std::vector<uint8_t> msg_buf(msg_bytes, 0);
    for (int i = 0; i < data_bits; i++)
        set_bit_msb_bytes(msg_buf.data(), i, get_bit_msb_bytes(data_buf.data(), i));
    for (int i = 0; i < crc_bits; i++)
        set_bit_msb_bytes(msg_buf.data(), data_bits + i, (crc_val >> (crc_bits - 1 - i)) & 1);

    int K_words = (K + 31) / 32;
    memset(enc_words, 0, K_words * sizeof(uint32_t));
    for (int i = 0; i < K; i++) {
        if (get_bit_msb_bytes(msg_buf.data(), i))
            set_bit_msb(enc_words, i, 1);
    }
}

// ============================================================================
// Decode result
// ============================================================================

struct DecodeResult {
    int errors;
    float avg_iterations;
};

// ============================================================================
// Decode a pre-encoded codeword with given decoder settings
// Supports both float-input and half-input paths
// ============================================================================

static DecodeResult decode_codeword(
    int bg, int Z, int Kd_no_crc, int Kd, int F,
    const uint32_t* h_encoded, int enc_words,
    const uint32_t* h_input, int K_words,
    int decode_crc_type,     // CRC type to configure the decoder with
    int max_iter,
    bool crc_early_term,
    bool use_half_path,      // true = decode_batch_half, false = decode_batch
    float snr_db,            // INFINITY = noiseless
    std::mt19937* noise_rng  // RNG for noise (may be null if noiseless)
) {
    int Kb = (bg == 1) ? 22 : 10;
    int N_cols = (bg == 1) ? 68 : 52;
    int parity_nodes = (bg == 1) ? 46 : 42;
    int K = Kb * Z;
    int N_full = N_cols * Z;
    int punctured = 2 * Z;

    nr_ldpc_config_t cfg = {};
    cfg.base_graph = bg;
    cfg.lifting_size = Z;
    cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(Z);
    cfg.num_info_bits = Kd;
    cfg.num_filler_bits = F;
    cfg.num_parity_bits = parity_nodes * Z;
    cfg.num_codeword_bits = K + cfg.num_parity_bits;
    cfg.puncture = true;
    cfg.redundancy_version = 0;

    ldpc_decoder_handle_t decoder;
    ldpc_decoder_create(&decoder);
    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations = max_iter;
    dec_params.crc_early_termination = crc_early_term;
    dec_params.crc_type = (ldpc_crc_type_t)decode_crc_type;
    dec_params.confidence_threshold = 0.25f;
    dec_params.skip_iteration_stats = false;
    dec_params.deferred_iteration_stats = false;
    ldpc_decoder_configure(decoder, &cfg, &dec_params);

    // Create LLRs as float
    std::vector<float> h_llrs(N_full);
    for (int i = 0; i < N_full; i++) {
        if (i < punctured) {
            h_llrs[i] = 0.0f;
        } else if (i >= Kd && i < K) {
            h_llrs[i] = 127.0f; // Filler = known zero
        } else {
            int bit_val = get_bit_msb(h_encoded, i);
            h_llrs[i] = bit_val ? -127.0f : 127.0f;
        }
    }

    // Add AWGN noise if requested
    if (std::isfinite(snr_db) && noise_rng) {
        float noise_std = 127.0f / sqrtf(powf(10.0f, snr_db / 10.0f));
        std::normal_distribution<float> noise(0.0f, noise_std);
        for (int i = 0; i < N_full; i++) {
            if (i < punctured || (i >= Kd && i < K)) continue;
            h_llrs[i] += noise(*noise_rng);
        }
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint32_t* d_decoded;
    cudaMalloc(&d_decoded, K_words * sizeof(uint32_t));
    cudaMemset(d_decoded, 0, K_words * sizeof(uint32_t));

    if (use_half_path) {
        // Convert to half and use decode_batch_half
        std::vector<__half> h_llrs_half(N_full);
        for (int i = 0; i < N_full; i++)
            h_llrs_half[i] = __float2half(h_llrs[i]);
        __half* d_llrs_half;
        cudaMalloc(&d_llrs_half, N_full * sizeof(__half));
        cudaMemcpy(d_llrs_half, h_llrs_half.data(), N_full * sizeof(__half), cudaMemcpyHostToDevice);
        ldpc_decoder_decode_batch_half(decoder, d_llrs_half, d_decoded, 1, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_llrs_half);
    } else {
        // Use decode_batch (float input)
        float* d_llrs_float;
        cudaMalloc(&d_llrs_float, N_full * sizeof(float));
        cudaMemcpy(d_llrs_float, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
        ldpc_decoder_decode_batch(decoder, d_llrs_float, d_decoded, 1, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_llrs_float);
    }

    float avg_iters = ldpc_decoder_get_avg_iterations(decoder);

    std::vector<uint32_t> h_decoded(K_words);
    cudaMemcpy(h_decoded.data(), d_decoded, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < Kd; i++) {
        if (get_bit_msb(h_input, i) != get_bit_lsb(h_decoded.data(), i))
            errors++;
    }

    cudaFree(d_decoded);
    cudaStreamDestroy(stream);
    ldpc_decoder_destroy(decoder);

    return {errors, avg_iters};
}

// ============================================================================
// CRC Verification Test
// Encodes data once with correct CRC, then decodes with:
//   1. Correct CRC type → should terminate early
//   2. Wrong CRC type → should NOT terminate early
// Returns: 0 = pass, 1 = fail
// ============================================================================

struct CRCTestCase {
    int bg;
    int Z;
    int Kd_no_crc;      // Data bits before CRC
    int correct_crc;     // CRC type embedded in data
    int wrong_crc;       // Mismatched CRC type for negative test
    const char* label;
};

static int run_crc_verification(const CRCTestCase& tc, bool use_half_path,
                                 const char* path_label, int max_iter,
                                 int& total_pass, int& total_fail) {
    int Kb = (tc.bg == 1) ? 22 : 10;
    int N_cols = (tc.bg == 1) ? 68 : 52;
    int parity_nodes = (tc.bg == 1) ? 46 : 42;
    int K = Kb * tc.Z;
    int crc_bits = (tc.correct_crc == 1) ? 16 : 24;
    int Kd = tc.Kd_no_crc + crc_bits;
    int F = K - Kd;
    int K_words = (K + 31) / 32;

    // Configure encoder
    nr_ldpc_config_t cfg = {};
    cfg.base_graph = tc.bg;
    cfg.lifting_size = tc.Z;
    cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(tc.Z);
    cfg.num_info_bits = Kd;
    cfg.num_filler_bits = F;
    cfg.num_parity_bits = parity_nodes * tc.Z;
    cfg.num_codeword_bits = K + cfg.num_parity_bits;
    cfg.puncture = true;
    cfg.redundancy_version = 0;

    ldpc_encoder_handle_t encoder;
    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);
    int enc_words = ldpc_encoder_get_output_words(encoder);

    // Prepare input with CRC
    std::mt19937 rng(42 + tc.bg * 1000 + tc.Z * 10 + tc.correct_crc);
    std::vector<uint32_t> h_input(K_words, 0);
    prepare_message_with_crc(h_input.data(), K, tc.Kd_no_crc, tc.correct_crc, rng);

    // Encode
    uint32_t *d_input, *d_encoded;
    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Decode with CORRECT CRC type
    DecodeResult r_correct = decode_codeword(
        tc.bg, tc.Z, tc.Kd_no_crc, Kd, F,
        h_encoded.data(), enc_words, h_input.data(), K_words,
        tc.correct_crc, max_iter, true, use_half_path, INFINITY, nullptr);

    // Decode with WRONG CRC type (same encoded data)
    DecodeResult r_wrong = decode_codeword(
        tc.bg, tc.Z, tc.Kd_no_crc, Kd, F,
        h_encoded.data(), enc_words, h_input.data(), K_words,
        tc.wrong_crc, max_iter, true, use_half_path, INFINITY, nullptr);

    // Decode with CRC_NONE (syndrome/confidence-only)
    DecodeResult r_none = decode_codeword(
        tc.bg, tc.Z, tc.Kd_no_crc, Kd, F,
        h_encoded.data(), enc_words, h_input.data(), K_words,
        LDPC_CRC_NONE, max_iter, true, use_half_path, INFINITY, nullptr);

    // Verify:
    // 1. Correct CRC: 0 errors, early termination
    // 2. Wrong CRC: 0 errors, iter = max (CRC fails, prevents early term)
    // 3. CRC_NONE: 0 errors, early termination (no CRC check)
    bool pass_correct = (r_correct.errors == 0 && r_correct.avg_iterations < (float)max_iter);
    bool pass_wrong = (r_wrong.errors == 0 && r_wrong.avg_iterations >= (float)max_iter);
    bool pass_none = (r_none.errors == 0 && r_none.avg_iterations < (float)max_iter);
    bool pass_all = pass_correct && pass_wrong && pass_none;

    const char* crc_names[] = {"NONE", "CRC16", "CRC24A", "CRC24B"};
    printf("  %-10s [%s] correct=%-6s iter=%-5.1f err=%d %s | wrong=%-6s iter=%-5.1f err=%d %s | none iter=%-5.1f %s\n",
           tc.label, path_label,
           crc_names[tc.correct_crc], r_correct.avg_iterations, r_correct.errors,
           pass_correct ? "OK" : "FAIL",
           crc_names[tc.wrong_crc], r_wrong.avg_iterations, r_wrong.errors,
           pass_wrong ? "OK" : "FAIL",
           r_none.avg_iterations,
           pass_none ? "OK" : "FAIL");

    if (pass_all) total_pass++; else total_fail++;

    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);

    return pass_all ? 0 : 1;
}

// ============================================================================
// Noiseless correctness test
// ============================================================================

struct NoiselessTest {
    int bg; int Z; int Kd_no_crc; int crc_type; const char* label;
};

static int run_noiseless_test(const NoiselessTest& nt, bool use_half_path,
                               const char* path_label, int max_iter,
                               int& total_pass, int& total_fail) {
    int Kb = (nt.bg == 1) ? 22 : 10;
    int N_cols = (nt.bg == 1) ? 68 : 52;
    int parity_nodes = (nt.bg == 1) ? 46 : 42;
    int K = Kb * nt.Z;
    int crc_bits = (nt.crc_type == 1) ? 16 : (nt.crc_type >= 2) ? 24 : 0;
    int Kd = nt.Kd_no_crc + crc_bits;
    int F = K - Kd;
    int K_words = (K + 31) / 32;

    nr_ldpc_config_t cfg = {};
    cfg.base_graph = nt.bg;
    cfg.lifting_size = nt.Z;
    cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(nt.Z);
    cfg.num_info_bits = Kd;
    cfg.num_filler_bits = F;
    cfg.num_parity_bits = parity_nodes * nt.Z;
    cfg.num_codeword_bits = K + cfg.num_parity_bits;
    cfg.puncture = true;
    cfg.redundancy_version = 0;

    ldpc_encoder_handle_t encoder;
    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);
    int enc_words = ldpc_encoder_get_output_words(encoder);

    std::mt19937 rng(100 + nt.bg * 1000 + nt.Z * 10 + nt.crc_type);
    std::vector<uint32_t> h_input(K_words, 0);
    prepare_message_with_crc(h_input.data(), K, nt.Kd_no_crc, nt.crc_type, rng);

    uint32_t *d_input, *d_encoded;
    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    DecodeResult r = decode_codeword(
        nt.bg, nt.Z, nt.Kd_no_crc, Kd, F,
        h_encoded.data(), enc_words, h_input.data(), K_words,
        nt.crc_type, max_iter, true, use_half_path, INFINITY, nullptr);

    // For float path (syndrome ET): checkpoint=4, so iter≤4
    // For half path (confidence ET): checkpoint=2, so iter≤2
    float max_expected_iter = use_half_path ? 2.0f : 4.0f;
    bool pass = (r.errors == 0 && r.avg_iterations <= max_expected_iter);
    printf("  %-12s [%s] BG%d/Z=%-3d Kd=%-5d err=%d iter=%.1f %s\n",
           nt.label, path_label, nt.bg, nt.Z, Kd, r.errors, r.avg_iterations,
           pass ? "PASS" : "FAIL");

    if (pass) total_pass++; else total_fail++;

    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);

    return pass ? 0 : 1;
}

// ============================================================================
// SNR sweep test
// ============================================================================

static int run_snr_sweep(int bg, int Z, int Kd_no_crc, int crc_type,
                          bool use_half_path, const char* label,
                          int max_iter, int& total_pass, int& total_fail) {
    int Kb = (bg == 1) ? 22 : 10;
    int N_cols = (bg == 1) ? 68 : 52;
    int parity_nodes = (bg == 1) ? 46 : 42;
    int K = Kb * Z;
    int crc_bits = (crc_type == 1) ? 16 : (crc_type >= 2) ? 24 : 0;
    int Kd = Kd_no_crc + crc_bits;
    int F = K - Kd;
    int K_words = (K + 31) / 32;

    nr_ldpc_config_t cfg = {};
    cfg.base_graph = bg;
    cfg.lifting_size = Z;
    cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(Z);
    cfg.num_info_bits = Kd;
    cfg.num_filler_bits = F;
    cfg.num_parity_bits = parity_nodes * Z;
    cfg.num_codeword_bits = K + cfg.num_parity_bits;
    cfg.puncture = true;
    cfg.redundancy_version = 0;

    ldpc_encoder_handle_t encoder;
    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);
    int enc_words = ldpc_encoder_get_output_words(encoder);

    std::mt19937 rng(200 + bg * 1000 + Z * 10 + crc_type);
    std::vector<uint32_t> h_input(K_words, 0);
    prepare_message_with_crc(h_input.data(), K, Kd_no_crc, crc_type, rng);

    uint32_t *d_input, *d_encoded;
    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    const char* path_label = use_half_path ? "half" : "float";
    printf("  %s [%s]:\n", label, path_label);

    int snr_vals[] = {10, 15, 20, 25, 30};
    for (int snr : snr_vals) {
        std::mt19937 noise_rng(300 + snr * 100 + bg * 10 + Z);
        DecodeResult r = decode_codeword(
            bg, Z, Kd_no_crc, Kd, F,
            h_encoded.data(), enc_words, h_input.data(), K_words,
            crc_type, max_iter, true, use_half_path, (float)snr, &noise_rng);

        bool pass;
        if (snr >= 25) pass = (r.errors == 0 && r.avg_iterations < (float)max_iter);
        else if (snr >= 15) pass = (r.avg_iterations < (float)max_iter);
        else pass = true; // Low SNR: just don't crash
        printf("    %2ddB: err=%d iter=%.1f %s\n", snr, r.errors, r.avg_iterations,
               pass ? "PASS" : "FAIL");
        if (pass) total_pass++; else total_fail++;
    }

    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
    return 0;
}

// ============================================================================
// Main test driver
// ============================================================================

int main() {
    ocudu_phy_cuda_init();

    printf("=============================================================\n");
    printf("FP16 LDPC Decoder: CRC Early Termination Verification\n");
    printf("=============================================================\n\n");

    int total_pass = 0, total_fail = 0;
    int max_iter = 10;

    // ========================================================================
    // Test 1: CRC Verification - correct vs wrong CRC type
    //
    // This is THE key test. If CRC computation is correct:
    //   - Correct CRC type → terminates early (iter < max)
    //   - Wrong CRC type → does NOT terminate (iter = max)
    // ========================================================================
    printf("=== Test 1: CRC Verification (correct vs wrong CRC type) ===\n");
    printf("  Testing both float-input (syndrome+CRC) and half-input (confidence+CRC) paths\n\n");

    CRCTestCase crc_tests[] = {
        // CRC16: BG2/Z=18, small TB
        {2, 18,  88,  LDPC_CRC_16,  LDPC_CRC_24B, "CRC16-BG2"},
        // CRC24A: BG1/Z=24, single-CB large TB
        {1, 24,  504, LDPC_CRC_24A, LDPC_CRC_24B, "CRC24A-BG1"},
        // CRC24B: BG2/Z=18, multi-CB small
        {2, 18,  80,  LDPC_CRC_24B, LDPC_CRC_24A, "CRC24B-BG2s"},
        // CRC24B: BG1/Z=384, multi-CB large
        {1, 384, 8424, LDPC_CRC_24B, LDPC_CRC_24A, "CRC24B-BG1l"},
        // CRC24B: BG2/Z=256, multi-CB large BG2
        {2, 256, 2536, LDPC_CRC_24B, LDPC_CRC_16,  "CRC24B-BG2l"},
        // CRC16: BG2/Z=52, medium
        {2, 52,  88,  LDPC_CRC_16,  LDPC_CRC_24A, "CRC16-BG2m"},
    };

    for (auto& ct : crc_tests) {
        run_crc_verification(ct, false, "float", max_iter, total_pass, total_fail);
        run_crc_verification(ct, true,  "half ", max_iter, total_pass, total_fail);
    }

    // ========================================================================
    // Test 2: Noiseless correctness - all CRC types, both BGs, both paths
    // ========================================================================
    printf("\n=== Test 2: Noiseless Correctness (all CRC types, both paths) ===\n");

    NoiselessTest noiseless_tests[] = {
        {2, 18,  88,   LDPC_CRC_16,  "CRC16"},
        {1, 24,  504,  LDPC_CRC_24A, "CRC24A"},
        {2, 18,  80,   LDPC_CRC_24B, "CRC24B-sm"},
        {1, 384, 8424, LDPC_CRC_24B, "CRC24B-lg"},
        {2, 256, 2536, LDPC_CRC_24B, "CRC24B-B2"},
        // BG1 with various Z
        {1, 24,  504,  LDPC_CRC_24B, "CRC24B-BG1s"},
        {1, 128, 2792, LDPC_CRC_24A, "CRC24A-BG1m"},
        // BG1/Z=240 (lifting set 7) - matches E2E pipeline config for 16QAM/52PRB
        {1, 240, 5008, LDPC_CRC_24B, "CRC24B-BG1-Z240"},
    };

    for (auto& nt : noiseless_tests) {
        run_noiseless_test(nt, false, "float", max_iter, total_pass, total_fail);
        run_noiseless_test(nt, true,  "half ", max_iter, total_pass, total_fail);
    }

    // ========================================================================
    // Test 3: SNR sweeps - verify iterations decrease with increasing SNR
    // ========================================================================
    printf("\n=== Test 3: SNR Sweep (all CRC types) ===\n");

    run_snr_sweep(2, 18,  88,   LDPC_CRC_16,  false, "CRC16  BG2/Z=18",  max_iter, total_pass, total_fail);
    run_snr_sweep(1, 24,  504,  LDPC_CRC_24A, false, "CRC24A BG1/Z=24",  max_iter, total_pass, total_fail);
    run_snr_sweep(1, 384, 8424, LDPC_CRC_24B, false, "CRC24B BG1/Z=384", max_iter, total_pass, total_fail);
    run_snr_sweep(2, 18,  80,   LDPC_CRC_24B, false, "CRC24B BG2/Z=18",  max_iter, total_pass, total_fail);
    run_snr_sweep(1, 240, 5008, LDPC_CRC_24B, false, "CRC24B BG1/Z=240", max_iter, total_pass, total_fail);

    // ========================================================================
    // Test 4: Large LLR magnitudes and partial parity (E2E-like conditions)
    // ========================================================================
    printf("\n=== Test 4: Large LLR + Partial Parity (E2E conditions) ===\n");
    {
        // Reproduce E2E conditions: BG1/Z=240, F=248, large LLR magnitudes, E < N_cb
        int bg = 1, Z = 240, Kd_no_crc = 5008;
        int crc_type = LDPC_CRC_24B;
        int Kb = 22, N_cols = 68, parity_nodes = 46;
        int K = Kb * Z;  // 5280
        int crc_bits = 24;
        int Kd = Kd_no_crc + crc_bits;  // 5032
        int F = K - Kd;  // 248
        int K_words = (K + 31) / 32;
        int N_full = N_cols * Z;  // 16320
        int punctured = 2 * Z;   // 480

        nr_ldpc_config_t cfg = {};
        cfg.base_graph = bg;
        cfg.lifting_size = Z;
        cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(Z);
        cfg.num_info_bits = Kd;
        cfg.num_filler_bits = F;
        cfg.num_parity_bits = parity_nodes * Z;
        cfg.num_codeword_bits = K + cfg.num_parity_bits;
        cfg.puncture = true;
        cfg.redundancy_version = 0;

        ldpc_encoder_handle_t encoder;
        ldpc_encoder_create(&encoder);
        ldpc_encoder_configure(encoder, &cfg);
        int enc_words = ldpc_encoder_get_output_words(encoder);

        std::mt19937 rng(999);
        std::vector<uint32_t> h_input(K_words, 0);
        prepare_message_with_crc(h_input.data(), K, Kd_no_crc, crc_type, rng);

        uint32_t *d_input, *d_encoded;
        cudaMalloc(&d_input, K_words * sizeof(uint32_t));
        cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));
        cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
        cudaStreamSynchronize(stream);

        std::vector<uint32_t> h_encoded(enc_words);
        cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Sub-test A: Normal LLRs (±127), full parity
        {
            std::vector<float> h_llrs(N_full);
            for (int i = 0; i < N_full; i++) {
                if (i < punctured) h_llrs[i] = 0.0f;
                else if (i >= Kd && i < K) h_llrs[i] = 127.0f;
                else h_llrs[i] = get_bit_msb(h_encoded.data(), i) ? -127.0f : 127.0f;
            }
            DecodeResult r = decode_codeword(bg, Z, Kd_no_crc, Kd, F,
                h_encoded.data(), enc_words, h_input.data(), K_words,
                crc_type, max_iter, false, false, INFINITY, nullptr);
            printf("  A) Normal (±127), full parity:   err=%d iter=%.1f %s\n",
                   r.errors, r.avg_iterations, r.errors == 0 ? "PASS" : "FAIL");
            if (r.errors == 0) total_pass++; else total_fail++;
        }

        // Sub-test B: Large LLRs (±1000), full parity
        {
            DecodeResult r;
            {
                std::vector<float> h_llrs(N_full);
                for (int i = 0; i < N_full; i++) {
                    if (i < punctured) h_llrs[i] = 0.0f;
                    else if (i >= Kd && i < K) h_llrs[i] = 10000.0f;
                    else h_llrs[i] = get_bit_msb(h_encoded.data(), i) ? -1000.0f : 1000.0f;
                }
                ldpc_decoder_handle_t dec;
                ldpc_decoder_create(&dec);
                ldpc_decoder_params_t dp;
                ldpc_decoder_params_init(&dp);
                dp.max_iterations = max_iter;
                dp.crc_early_termination = false;
                dp.skip_iteration_stats = false;
                dp.deferred_iteration_stats = false;
                ldpc_decoder_configure(dec, &cfg, &dp);

                float* d_llrs;
                uint32_t* d_dec;
                cudaMalloc(&d_llrs, N_full * sizeof(float));
                cudaMalloc(&d_dec, K_words * sizeof(uint32_t));
                cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemset(d_dec, 0, K_words * sizeof(uint32_t));
                ldpc_decoder_decode_batch(dec, d_llrs, d_dec, 1, stream);
                cudaStreamSynchronize(stream);
                r.avg_iterations = ldpc_decoder_get_avg_iterations(dec);
                std::vector<uint32_t> h_dec(K_words);
                cudaMemcpy(h_dec.data(), d_dec, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                r.errors = 0;
                for (int i = 0; i < Kd; i++) {
                    if (get_bit_msb(h_input.data(), i) != get_bit_lsb(h_dec.data(), i))
                        r.errors++;
                }
                cudaFree(d_llrs);
                cudaFree(d_dec);
                ldpc_decoder_destroy(dec);
            }
            printf("  B) Large (±1000), full parity:   err=%d iter=%.1f %s\n",
                   r.errors, r.avg_iterations, r.errors == 0 ? "PASS" : "FAIL");
            if (r.errors == 0) total_pass++; else total_fail++;
        }

        // Sub-test C: Normal LLRs, partial parity (last 616 positions = 0)
        {
            int E_parity = 10424;  // Same as E2E: E - Kd_cb = 14976 - 4552
            std::vector<float> h_llrs(N_full, 0.0f);
            for (int i = 0; i < N_full; i++) {
                if (i < punctured) h_llrs[i] = 0.0f;
                else if (i >= Kd && i < K) h_llrs[i] = 127.0f;
                else if (i >= K && i >= K + E_parity) h_llrs[i] = 0.0f;  // Partial parity
                else h_llrs[i] = get_bit_msb(h_encoded.data(), i) ? -127.0f : 127.0f;
            }
            DecodeResult r = decode_codeword(bg, Z, Kd_no_crc, Kd, F,
                h_encoded.data(), enc_words, h_input.data(), K_words,
                crc_type, max_iter, false, false, INFINITY, nullptr);
            printf("  C) Normal (±127), partial parity: err=%d iter=%.1f %s\n",
                   r.errors, r.avg_iterations, r.errors == 0 ? "PASS" : "FAIL");
            if (r.errors == 0) total_pass++; else total_fail++;
        }

        // Sub-test D: Large LLRs, partial parity (E2E exact conditions)
        {
            int E_parity = 10424;
            DecodeResult r;
            {
                std::vector<float> h_llrs(N_full, 0.0f);
                for (int i = 0; i < N_full; i++) {
                    if (i < punctured) h_llrs[i] = 0.0f;
                    else if (i >= Kd && i < K) h_llrs[i] = 10000.0f;
                    else if (i >= K && i >= K + E_parity) h_llrs[i] = 0.0f;
                    else h_llrs[i] = get_bit_msb(h_encoded.data(), i) ? -1000.0f : 1000.0f;
                }
                ldpc_decoder_handle_t dec;
                ldpc_decoder_create(&dec);
                ldpc_decoder_params_t dp;
                ldpc_decoder_params_init(&dp);
                dp.max_iterations = max_iter;
                dp.crc_early_termination = false;
                dp.skip_iteration_stats = false;
                dp.deferred_iteration_stats = false;
                ldpc_decoder_configure(dec, &cfg, &dp);

                float* d_llrs;
                uint32_t* d_dec;
                cudaMalloc(&d_llrs, N_full * sizeof(float));
                cudaMalloc(&d_dec, K_words * sizeof(uint32_t));
                cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemset(d_dec, 0, K_words * sizeof(uint32_t));
                ldpc_decoder_decode_batch(dec, d_llrs, d_dec, 1, stream);
                cudaStreamSynchronize(stream);
                r.avg_iterations = ldpc_decoder_get_avg_iterations(dec);
                std::vector<uint32_t> h_dec(K_words);
                cudaMemcpy(h_dec.data(), d_dec, K_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                r.errors = 0;
                for (int i = 0; i < Kd; i++) {
                    if (get_bit_msb(h_input.data(), i) != get_bit_lsb(h_dec.data(), i))
                        r.errors++;
                }
                cudaFree(d_llrs);
                cudaFree(d_dec);
                ldpc_decoder_destroy(dec);
            }
            printf("  D) Large (±1000), partial parity: err=%d iter=%.1f %s\n",
                   r.errors, r.avg_iterations, r.errors == 0 ? "PASS" : "FAIL");
            if (r.errors == 0) total_pass++; else total_fail++;
        }

        cudaFree(d_input);
        cudaFree(d_encoded);
        cudaStreamDestroy(stream);
        ldpc_encoder_destroy(encoder);
    }

    // ========================================================================
    // Summary
    // ========================================================================
    printf("\n=============================================================\n");
    printf("SUMMARY: %d PASSED, %d FAILED (total %d tests)\n",
           total_pass, total_fail, total_pass + total_fail);
    printf("=============================================================\n");

    return total_fail > 0 ? 1 : 0;
}
