// Compare GPU encoder output with CPU reference
#include "ocudu_phy_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

// Include srsRAN CPU encoder (we'll link against it)
extern "C" {
    // Forward declare srsRAN ldpc encoder functions if available
}

// Simplified CPU LDPC encoder for BG2 validation
// Based on 3GPP TS 38.212 Section 5.3.2

// BG2 row pointers (42 rows)
static const int bg2_row_ptr[] = {
    0, 8, 18, 26, 36, 40, 46, 52, 58, 62, 67, 72,
    77, 81, 86, 91, 95, 100, 105, 109, 113, 117, 121, 124,
    128, 132, 135, 140, 143, 147, 150, 155, 158, 162, 166, 170,
    174, 178, 181, 185, 189, 193, 197
};

// BG2 column indices
static const int bg2_col[] = {
    0,  1,  2,  3,  6,  9, 10, 11,
    0,  3,  4,  5,  6,  7,  8,  9, 11, 12,
    0,  1,  3,  4,  8, 10, 12, 13,
    1,  2,  4,  5,  6,  7,  8,  9, 10, 13,
    0,  1, 11, 14,
    0,  1,  5,  7, 11, 15,
    0,  5,  7,  9, 11, 16,
    1,  5,  7, 11, 13, 17,
    0,  1, 12, 18,
    1,  8, 10, 11, 19,
    0,  1,  6,  7, 20,
    0,  7,  9, 13, 21,
    1,  3, 11, 22,
    0,  1,  8, 13, 23,
    1,  6, 11, 13, 24,
    0, 10, 11, 25,
    1,  9, 11, 12, 26,
    1,  5, 11, 12, 27,
    0,  6,  7, 28,
    0,  1, 10, 29,
    1,  4, 11, 30,
    0,  8, 13, 31,
    1,  2, 32,
    0,  3,  5, 33,
    1,  2,  9, 34,
    0,  5, 35,
    2,  7, 12, 13, 36,
    0,  6, 37,
    1,  2,  5, 38,
    0,  4, 39,
    2,  5,  7,  9, 40,
    1, 13, 41,
    0,  5, 12, 42,
    2,  7, 10, 43,
    0, 12, 13, 44,
    1,  5, 11, 45,
    0,  2,  7, 46,
    10, 13, 47,
    1,  5, 11, 48,
    0,  7, 12, 49,
    2, 10, 13, 50,
    1,  5, 11, 51
};

// BG2 shift values for all 8 lifting sets (197 shifts per set)
// Matching the GPU encoder's shift tables exactly
static const int16_t bg2_shifts[8][197] = {
    // Lifting set 0 (Z_ref=256: Z=2,4,8,16,32,64,128,256)
    {  9, 117, 204,  26, 189, 205,   0,   0, 167, 166, 253, 125, 226, 156, 224, 252,
       0,   0,  81, 114,  44,  52, 240,   1,   0,   0,   8,  58, 158, 104, 209,  54,
      18, 128,   0,   0, 179, 214,  71,   0, 231,  41, 194, 159, 103,   0, 155, 228,
      45,  28, 158,   0, 129, 147, 140,   3, 116,   0, 142,  94, 230,   0, 203, 205,
      61, 247,   0,  11, 185,   0, 117,   0,  11, 236, 210,  56,   0,  63, 111,  14,
       0,  83,   2,  38, 222,   0, 115, 145,   3, 232,   0,  51, 175, 213,   0, 203,
     142,   8, 242,   0, 254, 124, 114,  64,   0, 220, 194,  50,   0,  87,  20, 185,
       0,  26, 105,  29,   0,  76,  42, 210,   0, 222,  63,   0,  23, 235, 238,   0,
      46, 139,   8,   0, 228, 156,   0,  29, 143, 160, 122,   0,   8, 151,   0,  98,
     101, 135,   0,  18,  28,   0,  71, 240,   9,  84,   0, 106,   1,   0, 242,  44,
     166,   0, 132, 164, 235,   0, 147,  85,  36,   0,  57,  40,  63,   0, 140,  38,
     154,   0, 219, 151,   0,  31,  66,  38,   0, 239, 172,  34,   0,   0,  75, 120,
       0, 129, 229, 118,   0},
    // Lifting set 1 (Z_ref=384)
    {174,  97, 166,  66,  71, 172,   0,   0,  27,  36,  48,  92,  31, 187, 185,   3,
       0,   0,  25, 114, 117, 110, 114,   1,   0,   0, 136, 175, 113,  72, 123, 118,
      28, 186,   0,   0,  72,  74,  29,   0,  10,  44, 121,  80,  48,   0, 129,  92,
     100,  49, 184,   0,  80, 186,  16, 102, 143,   0, 118,  70, 152,   0,  28, 132,
     185, 178,   0,  59, 104,  22,  52,   0,  32,  92, 174, 154,   0,  39,  93,  11,
       0,  49, 125,  35, 166,   0,  19, 118,  21, 163,   0,  68,  63,  81,   0,  87,
     177, 135,  64,   0, 158,  23,   9,   6,   0, 186,   6,  46,   0,  58,  42, 156,
       0,  76,  61, 153,   0, 157, 175,  67,   0,  20,  52,   0, 106,  86,  95,   0,
     182, 153,  64,   0,  45,  21,   0,  67, 137,  55,  85,   0, 103,  50,   0,  70,
     111, 168,   0, 110,  17,   0, 120, 154,  52,  56,   0,   3, 170,   0,  84,   8,
      17,   0, 165, 179, 124,   0, 173, 177,  12,   0,  77, 184,  18,   0,  25, 151,
     170,   0,  37,  31,   0,  84, 151, 190,   0,  93, 132,  57,   0, 103, 107, 163,
       0, 147,   7,  60,   0},
    // Lifting set 2 (Z_ref=320)
    {  0,   0,   0,   0,   0,   0,   0,   0, 137, 124,   0,   0,  88,   0,   0,  55,
       0,   0,  20,  94,  99,   9, 108,   1,   0,   0,  38,  15, 102, 146,  12,  57,
      53,  46,   0,   0,   0, 136, 157,   0,   0, 131, 142, 141,  64,   0,   0, 124,
      99,  45, 148,   0,   0,  45, 148,  96,  78,   0,   0,  65,  87,   0,   0,  97,
      51,  85,   0,   0,  17, 156,  20,   0,   0,   7,   4,   2,   0,   0, 113,  48,
       0,   0, 112, 102,  26,   0,   0, 138,  57,  27,   0,   0,  73,  99,   0,   0,
      79, 111, 143,   0,   0,  24, 109,  18,   0,   0,  18,  86,   0,   0, 158, 154,
       0,   0, 148, 104,   0,   0,  17,  33,   0,   0,   4,   0,   0,  75, 158,   0,
       0,  69,  87,   0,   0,  65,   0,   0, 100,  13,   7,   0,   0,  32,   0,   0,
     126, 110,   0,   0, 154,   0,   0,  35,  51, 134,   0,   0,  20,   0,   0,  20,
     122,   0,   0,  88,  13,   0,   0,  19,  78,   0,   0, 157,   6,   0,   0,  63,
      82,   0,   0, 144,   0,   0,  93,  19,   0,   0,  24, 138,   0,   0,  36, 143,
       0,   0,   2,  55,   0},
    // Lifting set 3 (Z_ref=224)
    { 72, 110,  23, 181,  95,   8,   1,   0,  53, 156, 115, 156, 115, 200,  29,  31,
       0,   0, 152, 131,  46, 191,  91,   0,   0,   0, 185,   6,  36, 124, 124, 110,
     156, 133,   1,   0, 200,  16, 101,   0, 185, 138, 170, 219, 193,   0, 123,  55,
      31, 222, 209,   0, 103,  13, 105, 150, 181,   0, 147,  43, 152,   0,   2,  30,
     184,  83,   0, 174, 150,   8,  56,   0,  99, 138, 110,  99,   0,  46, 217, 109,
       0,  37, 113, 143, 140,   0,  36,  95,  40, 116,   0, 116, 200, 110,   0,  75,
     158, 134,  97,   0,  48, 132, 206,   2,   0,  68,  16, 156,   0,  35, 138,  86,
       0,   6,  20, 141,   0,  80,  43,  81,   0,  49,   1,   0, 156,  54, 134,   0,
     153,  88,  63,   0, 211,  94,   0,  90,   6, 221,   6,   0,  27, 118,   0, 216,
     212, 193,   0, 108,  61,   0, 106,  44, 185, 176,   0, 147, 182,   0, 108,  21,
     110,   0,  71,  12, 109,   0,  29, 201,  69,   0,  91, 165,  55,   0,   1, 175,
      83,   0,  40,  12,   0,  37,  97,  46,   0, 106, 181, 154,   0,  98,  35,  36,
       0, 120, 101,  81,   0},
    // Lifting set 4 (Z_ref=288: Z=9,18,36,72,144,288)
    {  3,  26,  53,  35, 115, 127,   0,   0,  19,  94, 104,  66,  84,  98,  69,  50,
       0,   0,  95, 106,  92, 110, 111,   1,   0,   0, 120, 121,  22,   4,  73,  49,
     128,  79,   0,   0,  42,  24,  51,   0,  40, 140,  84, 137,  71,   0, 109,  87,
     107, 133, 139,   0,  97, 135,  35, 108,  65,   0,  70,  69,  88,   0,  97,  40,
      24,  49,   0,  46,  41, 101,  96,   0,  28,  30, 116,  64,   0,  33, 122, 131,
       0,  76,  37,  62,  47,   0, 143,  51, 130,  97,   0, 139,  96, 128,   0,  48,
       9,  28,   8,   0, 120,  43,  65,  42,   0,  17, 106, 142,   0,  79,  28,  41,
       0,   2, 103,  78,   0,  91,  75,  81,   0,  54, 132,   0,  68, 115,  56,   0,
      30,  42, 101,   0, 128,  63,   0, 142,  28, 100, 133,   0,  13,  10,   0, 106,
      77,  43,   0, 133,  25,   0,  87,  56, 104,  70,   0,  80, 139,   0,  32,  89,
      71,   0, 135,   6,   2,   0,  37,  25, 114,   0,  60, 137,  93,   0, 121, 129,
      26,   0,  97,  56,   0,   1,  70,   1,   0, 119,  32, 142,   0,   6,  73, 102,
       0,  48,  47,  19,   0},
    // Lifting set 5-7 (placeholder - not used in this test)
    {0}, {0}, {0}
};

// Get lifting set index from Z value
int get_lifting_set_idx_cpu(int Z) {
    static const int set_bases[8] = {2, 3, 5, 7, 9, 11, 13, 15};
    for (int i = 0; i < 8; i++) {
        int base = set_bases[i];
        if (Z % base == 0) {
            int ratio = Z / base;
            if (ratio > 0 && (ratio & (ratio - 1)) == 0) {
                return i;
            }
        }
    }
    return 0;
}

// Get actual shift value for Z (using 3GPP formula: shift = shift_ref % Z)
int get_scaled_shift(int edge_idx, int Z) {
    int set_idx = get_lifting_set_idx_cpu(Z);
    int raw_shift = bg2_shifts[set_idx][edge_idx];
    if (raw_shift == 0) return 0;  // No connection or zero shift (both are valid)
    return raw_shift % Z;
}

// Simple CPU encoder for comparison
void cpu_encode_bg2(const uint8_t* input_bits, uint8_t* output_bits, int Z) {
    const int Kb = 10;
    const int M = 42;
    const int N = 52;

    // Clear output
    memset(output_bits, 0, N * Z);

    // Copy systematic bits (columns 0 to Kb-1)
    for (int col = 0; col < Kb; col++) {
        for (int z = 0; z < Z; z++) {
            output_bits[col * Z + z] = input_bits[col * Z + z];
        }
    }

    // Compute syndromes for first 4 rows
    std::vector<uint8_t> syndrome(M * Z, 0);
    for (int row = 0; row < 4; row++) {
        int row_start = bg2_row_ptr[row];
        int row_end = bg2_row_ptr[row + 1];

        for (int z = 0; z < Z; z++) {
            int syn = 0;
            for (int e = row_start; e < row_end; e++) {
                int col = bg2_col[e];
                if (col >= Kb) continue;

                int shift = get_scaled_shift(e, Z);
                if (shift < 0) continue;

                int src_z = (z + shift) % Z;
                syn ^= output_bits[col * Z + src_z];
            }
            syndrome[row * Z + z] = syn;
        }
    }

    // Compute p0 = XOR of all syndromes at shifted position
    // For lifting set 4 (not i3 or i7), p0[z] = XOR of syndromes at z-1
    for (int z = 0; z < Z; z++) {
        int src_z = (z - 1 + Z) % Z;
        int p0 = syndrome[0 * Z + src_z] ^ syndrome[1 * Z + src_z] ^
                 syndrome[2 * Z + src_z] ^ syndrome[3 * Z + src_z];
        output_bits[Kb * Z + z] = p0;
    }

    // Compute p1 = s0 ^ p0_shifted
    for (int z = 0; z < Z; z++) {
        int p0_z = output_bits[Kb * Z + z];  // No shift for set 4
        output_bits[(Kb + 1) * Z + z] = syndrome[0 * Z + z] ^ p0_z;
    }

    // Compute p2 = s1 ^ p1
    for (int z = 0; z < Z; z++) {
        output_bits[(Kb + 2) * Z + z] = syndrome[1 * Z + z] ^ output_bits[(Kb + 1) * Z + z];
    }

    // Compute p3 = s3 ^ p0_shifted
    for (int z = 0; z < Z; z++) {
        int p0_z = output_bits[Kb * Z + z];
        output_bits[(Kb + 3) * Z + z] = syndrome[3 * Z + z] ^ p0_z;
    }

    // Compute remaining parity (rows 4 to M-1)
    for (int row = 4; row < M; row++) {
        int row_start = bg2_row_ptr[row];
        int row_end = bg2_row_ptr[row + 1];

        int parity_col = -1;
        for (int z = 0; z < Z; z++) {
            int parity = 0;

            for (int e = row_start; e < row_end; e++) {
                int col = bg2_col[e];
                int shift = get_scaled_shift(e, Z);
                if (shift < 0) continue;

                if (col >= Kb + 4) {
                    parity_col = col;
                } else {
                    int src_z = (z + shift) % Z;
                    parity ^= output_bits[col * Z + src_z];
                }
            }

            if (parity_col >= 0) {
                output_bits[parity_col * Z + z] = parity;
            }
        }
    }
}

// Helper: Get MSB-first bit position within 32-bit word
inline int msb_first_pos(int linear_pos) {
    int bit_in_word = linear_pos % 32;
    int byte_in_word = bit_in_word / 8;
    int bit_in_byte = 7 - (bit_in_word % 8);
    return byte_in_word * 8 + bit_in_byte;
}

void compare_encoders(int Z, uint32_t pattern) {
    const int Kb = 10;
    const int K = Kb * Z;
    const int K_words = (K + 31) / 32;
    const int N = 52;
    const int N_bits = N * Z;

    printf("\n=== Z=%d, pattern=0x%08X ===\n", Z, pattern);

    // Create input bits (linear ordering)
    std::vector<uint8_t> input_bits(K, 0);
    for (int i = 0; i < K; i++) {
        input_bits[i] = (pattern >> (i % 32)) & 1;
    }

    // CPU encode (using linear bit ordering)
    std::vector<uint8_t> cpu_output(N_bits, 0);
    cpu_encode_bg2(input_bits.data(), cpu_output.data(), Z);

    // GPU encode
    nr_ldpc_config_t cfg = {};
    cfg.base_graph = 2;
    cfg.lifting_size = Z;
    cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(Z);
    cfg.num_info_bits = K;
    cfg.num_filler_bits = 0;
    cfg.num_parity_bits = 42 * Z;
    cfg.num_codeword_bits = K + cfg.num_parity_bits;
    cfg.puncture = true;
    cfg.redundancy_version = 0;

    ldpc_encoder_handle_t encoder;
    ldpc_encoder_create(&encoder);
    ldpc_encoder_configure(encoder, &cfg);

    int enc_words = ldpc_encoder_get_output_words(encoder);

    uint32_t *d_input, *d_encoded;
    cudaMalloc(&d_input, K_words * sizeof(uint32_t));
    cudaMalloc(&d_encoded, enc_words * sizeof(uint32_t));

    // Pack input for GPU using MSB-first byte ordering (matching GPU encoder's expectation)
    std::vector<uint32_t> h_input(K_words, 0);
    for (int i = 0; i < K; i++) {
        if (input_bits[i]) {
            int word_idx = i / 32;
            int bit_pos = msb_first_pos(i);
            h_input[word_idx] |= (1u << bit_pos);
        }
    }

    cudaMemcpy(d_input, h_input.data(), K_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemset(d_encoded, 0, enc_words * sizeof(uint32_t));
    ldpc_encoder_encode(encoder, d_input, d_encoded, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> h_encoded(enc_words);
    cudaMemcpy(h_encoded.data(), d_encoded, enc_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Unpack GPU output using MSB-first byte ordering (matching GPU encoder's output)
    std::vector<uint8_t> gpu_output(N_bits, 0);
    for (int i = 0; i < N_bits; i++) {
        int word_idx = i / 32;
        int bit_pos = msb_first_pos(i);
        gpu_output[i] = (h_encoded[word_idx] >> bit_pos) & 1;
    }

    // Compare systematic portion
    int sys_diff = 0;
    printf("Systematic portion (first %d bits):\n", K);
    for (int i = 0; i < K; i++) {
        if (cpu_output[i] != gpu_output[i]) {
            if (sys_diff < 20) {
                printf("  bit %d: CPU=%d GPU=%d (col=%d z=%d)\n",
                       i, cpu_output[i], gpu_output[i], i/Z, i%Z);
            }
            sys_diff++;
        }
    }
    printf("  Systematic differences: %d/%d\n", sys_diff, K);

    // Compare parity portion
    int par_diff = 0;
    for (int i = K; i < N_bits; i++) {
        if (cpu_output[i] != gpu_output[i]) {
            par_diff++;
        }
    }
    printf("  Parity differences: %d/%d\n", par_diff, N_bits - K);

    if (sys_diff == 0 && par_diff == 0) {
        printf("  PASS - GPU matches CPU\n");
    } else {
        printf("  FAIL - GPU differs from CPU\n");
    }

    cudaFree(d_input);
    cudaFree(d_encoded);
    cudaStreamDestroy(stream);
    ldpc_encoder_destroy(encoder);
}

int main() {
    ocudu_phy_cuda_init();

    printf("Comparing GPU vs CPU encoder:\n");

    compare_encoders(18, 0x00000001);  // Single bit
    compare_encoders(18, 0xFFFFFFFF);  // All ones
    compare_encoders(32, 0x00000001);  // Single bit, Z=32
    compare_encoders(32, 0xFFFFFFFF);  // All ones, Z=32

    return 0;
}
