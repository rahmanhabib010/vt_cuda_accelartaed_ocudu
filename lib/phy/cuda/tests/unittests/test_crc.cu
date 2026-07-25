/**
 * Quick CRC24A parallel implementation test
 * Verifies the GF(2) shift-and-combine math is correct
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CRC24A_POLY 0x864CFB

// Forward declaration of GPU CRC function
extern "C" {
    typedef int nr_ldpc_status_t;
    nr_ldpc_status_t crc24a_compute(const uint8_t* d_data, int num_bits,
                                    uint32_t* d_crc, cudaStream_t stream);
    nr_ldpc_status_t crc24a_compute_batch(const uint8_t* d_tb_data,
                                          uint32_t* d_crc_out,
                                          int num_tbs,
                                          int tb_size_bits,
                                          int tb_stride_bytes,
                                          cudaStream_t stream);
}

// Reference sequential CRC24A (known good)
uint32_t crc24a_reference(const uint8_t* data, int num_bytes) {
    uint32_t crc = 0;
    for (int i = 0; i < num_bytes; i++) {
        crc ^= ((uint32_t)data[i] << 16);
        for (int j = 0; j < 8; j++) {
            if (crc & 0x800000) {
                crc = (crc << 1) ^ CRC24A_POLY;
            } else {
                crc <<= 1;
            }
        }
    }
    return crc & 0xFFFFFF;
}

uint32_t crc24a_reference_bits(const uint8_t* data, int num_bits) {
    uint32_t crc = 0;
    int num_full_bytes = num_bits / 8;
    int remaining_bits = num_bits % 8;

    for (int i = 0; i < num_full_bytes; i++) {
        crc ^= ((uint32_t)data[i] << 16);
        for (int j = 0; j < 8; j++) {
            if (crc & 0x800000) {
                crc = (crc << 1) ^ CRC24A_POLY;
            } else {
                crc <<= 1;
            }
            crc &= 0xFFFFFF;
        }
    }

    if (remaining_bits > 0) {
        uint8_t byte = data[num_full_bytes];
        for (int b = 0; b < remaining_bits; b++) {
            uint8_t data_bit = (byte >> (7 - b)) & 1;
            uint32_t msb = (crc >> 23) & 1;
            crc = (crc << 1) | data_bit;
            if (msb) {
                crc ^= CRC24A_POLY;
            }
            crc &= 0xFFFFFF;
        }
        for (int b = 0; b < (8 - remaining_bits); b++) {
            uint32_t msb = (crc >> 23) & 1;
            crc <<= 1;
            if (msb) {
                crc ^= CRC24A_POLY;
            }
            crc &= 0xFFFFFF;
        }
    }

    return crc & 0xFFFFFF;
}

// GF(2^24) multiplication: (a * b) mod G(x)
uint32_t gf24_multiply(uint32_t a, uint32_t b) {
    uint32_t result = 0;
    while (b) {
        if (b & 1) result ^= a;
        b >>= 1;
        if (a & 0x800000) {
            a = ((a << 1) ^ CRC24A_POLY) & 0xFFFFFF;
        } else {
            a = (a << 1) & 0xFFFFFF;
        }
    }
    return result & 0xFFFFFF;
}

// Compute x^n mod G(x) where n is number of BITS
uint32_t compute_x_power(int n_bits) {
    uint32_t result = 1;  // x^0 = 1
    for (int i = 0; i < n_bits; i++) {
        if (result & 0x800000) {
            result = ((result << 1) ^ CRC24A_POLY) & 0xFFFFFF;
        } else {
            result = (result << 1) & 0xFFFFFF;
        }
    }
    return result;
}

// Shift CRC by n_bytes: crc * x^(n_bytes*8) mod G(x)
uint32_t crc24a_shift_bytes(uint32_t crc, int n_bytes) {
    if (crc == 0 || n_bytes == 0) return crc;
    uint32_t x_power = compute_x_power(n_bytes * 8);
    return gf24_multiply(crc, x_power);
}

// Parallel CRC using 2 chunks
uint32_t crc24a_parallel_2chunks(const uint8_t* data, int num_bytes) {
    int chunk_size = num_bytes / 2;
    int chunk1_size = chunk_size;
    int chunk2_size = num_bytes - chunk_size;

    // Compute CRC of each chunk independently
    uint32_t crc1 = crc24a_reference(data, chunk1_size);
    uint32_t crc2 = crc24a_reference(data + chunk1_size, chunk2_size);

    // Combine: CRC(A || B) = shift(CRC(A), len(B)*8) XOR CRC(B)
    uint32_t shifted_crc1 = crc24a_shift_bytes(crc1, chunk2_size);
    uint32_t final_crc = shifted_crc1 ^ crc2;

    return final_crc;
}

// Parallel CRC using N chunks
uint32_t crc24a_parallel_nchunks(const uint8_t* data, int num_bytes, int num_chunks) {
    int base_chunk_size = num_bytes / num_chunks;
    int remainder = num_bytes % num_chunks;

    uint32_t* chunk_crcs = (uint32_t*)malloc(num_chunks * sizeof(uint32_t));
    int* chunk_sizes = (int*)malloc(num_chunks * sizeof(int));

    // Compute CRC of each chunk
    int offset = 0;
    for (int i = 0; i < num_chunks; i++) {
        chunk_sizes[i] = base_chunk_size + (i < remainder ? 1 : 0);
        chunk_crcs[i] = crc24a_reference(data + offset, chunk_sizes[i]);
        offset += chunk_sizes[i];
    }

    // Compute suffix sums (bytes remaining after each chunk)
    int* suffix_sums = (int*)malloc(num_chunks * sizeof(int));
    suffix_sums[num_chunks - 1] = 0;
    for (int i = num_chunks - 2; i >= 0; i--) {
        suffix_sums[i] = suffix_sums[i + 1] + chunk_sizes[i + 1];
    }

    // Combine all chunks
    uint32_t final_crc = 0;
    for (int i = 0; i < num_chunks; i++) {
        uint32_t shifted = crc24a_shift_bytes(chunk_crcs[i], suffix_sums[i]);
        final_crc ^= shifted;
    }

    free(chunk_crcs);
    free(chunk_sizes);
    free(suffix_sums);

    return final_crc;
}

// Precomputed x^(8*2^i) powers for O(log n) shift
uint32_t x_powers[21];

void init_x_powers() {
    // x^8 = shift 1 through 8 bits
    uint32_t x_8 = 1;
    for (int i = 0; i < 8; i++) {
        if (x_8 & 0x800000) {
            x_8 = ((x_8 << 1) ^ CRC24A_POLY) & 0xFFFFFF;
        } else {
            x_8 = (x_8 << 1) & 0xFFFFFF;
        }
    }
    x_powers[0] = x_8;  // x^8

    // x^(8*2^i) = (x^(8*2^(i-1)))^2
    for (int i = 1; i < 21; i++) {
        x_powers[i] = gf24_multiply(x_powers[i-1], x_powers[i-1]);
    }
}

// Fast shift using precomputed powers - O(log n)
uint32_t crc24a_shift_bytes_fast(uint32_t crc, int n_bytes) {
    if (crc == 0 || n_bytes == 0) return crc;

    // Compute x^(n*8) using binary decomposition
    uint32_t x_power = 0;
    bool first = true;

    for (int i = 0; i < 21 && n_bytes > 0; i++) {
        if (n_bytes & 1) {
            if (first) {
                x_power = x_powers[i];
                first = false;
            } else {
                x_power = gf24_multiply(x_power, x_powers[i]);
            }
        }
        n_bytes >>= 1;
    }

    if (first) return crc;  // n_bytes was 0
    return gf24_multiply(crc, x_power);
}

// Parallel CRC using fast shift
uint32_t crc24a_parallel_fast(const uint8_t* data, int num_bytes, int num_chunks) {
    int base_chunk_size = num_bytes / num_chunks;
    int remainder = num_bytes % num_chunks;

    uint32_t* chunk_crcs = (uint32_t*)malloc(num_chunks * sizeof(uint32_t));
    int* chunk_sizes = (int*)malloc(num_chunks * sizeof(int));

    // Compute CRC of each chunk
    int offset = 0;
    for (int i = 0; i < num_chunks; i++) {
        chunk_sizes[i] = base_chunk_size + (i < remainder ? 1 : 0);
        chunk_crcs[i] = crc24a_reference(data + offset, chunk_sizes[i]);
        offset += chunk_sizes[i];
    }

    // Compute suffix sums
    int* suffix_sums = (int*)malloc(num_chunks * sizeof(int));
    suffix_sums[num_chunks - 1] = 0;
    for (int i = num_chunks - 2; i >= 0; i--) {
        suffix_sums[i] = suffix_sums[i + 1] + chunk_sizes[i + 1];
    }

    // Combine using FAST shift
    uint32_t final_crc = 0;
    for (int i = 0; i < num_chunks; i++) {
        uint32_t shifted = crc24a_shift_bytes_fast(chunk_crcs[i], suffix_sums[i]);
        final_crc ^= shifted;
    }

    free(chunk_crcs);
    free(chunk_sizes);
    free(suffix_sums);

    return final_crc;
}

int main() {
    printf("=== CRC24A Parallel Implementation Test ===\n\n");

    // Initialize precomputed powers
    init_x_powers();

    // Test 1: Verify x^n computation
    printf("Test 1: x^n computation\n");
    printf("  x^8  = 0x%06X (expected: 0x000100)\n", compute_x_power(8));
    printf("  x^16 = 0x%06X (expected: 0x010000)\n", compute_x_power(16));
    printf("  x^24 = 0x%06X (expected: 0x%06X = G(x) XOR x^24 = poly)\n",
           compute_x_power(24), CRC24A_POLY);
    printf("\n");

    // Test 2: Small data parallel CRC
    printf("Test 2: Small data (16 bytes, 2 chunks)\n");
    uint8_t small_data[16] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                              0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10};
    uint32_t ref_crc = crc24a_reference(small_data, 16);
    uint32_t par_crc = crc24a_parallel_2chunks(small_data, 16);
    printf("  Reference CRC: 0x%06X\n", ref_crc);
    printf("  Parallel CRC:  0x%06X\n", par_crc);
    printf("  Match: %s\n\n", (ref_crc == par_crc) ? "YES" : "NO <<<< BUG!");

    // Test 3: Larger data with multiple chunks
    printf("Test 3: Larger data (1000 bytes)\n");
    uint8_t* large_data = (uint8_t*)malloc(1000);
    for (int i = 0; i < 1000; i++) large_data[i] = (uint8_t)(i & 0xFF);

    ref_crc = crc24a_reference(large_data, 1000);
    printf("  Reference CRC: 0x%06X\n", ref_crc);

    for (int chunks = 2; chunks <= 8; chunks++) {
        par_crc = crc24a_parallel_nchunks(large_data, 1000, chunks);
        printf("  Parallel (%d chunks): 0x%06X %s\n", chunks, par_crc,
               (ref_crc == par_crc) ? "OK" : "FAIL!");
    }
    printf("\n");

    // Test 4: 90KB data (40MHz TB size)
    printf("Test 4: 90KB data (40MHz TB size)\n");
    int tb_size = 90000;
    uint8_t* tb_data = (uint8_t*)malloc(tb_size);
    for (int i = 0; i < tb_size; i++) tb_data[i] = (uint8_t)(i * 17 & 0xFF);

    ref_crc = crc24a_reference(tb_data, tb_size);
    printf("  Reference CRC: 0x%06X\n", ref_crc);

    for (int chunks = 4; chunks <= 32; chunks *= 2) {
        par_crc = crc24a_parallel_nchunks(tb_data, tb_size, chunks);
        printf("  Parallel (%2d chunks): 0x%06X %s\n", chunks, par_crc,
               (ref_crc == par_crc) ? "OK" : "FAIL!");
    }

    free(large_data);
    free(tb_data);

    // Test 5: Verify fast shift matches slow shift
    printf("\nTest 5: Fast shift vs slow shift\n");
    printf("  x_powers[0] (x^8):  0x%06X (expected: 0x000100)\n", x_powers[0]);
    printf("  x_powers[1] (x^16): 0x%06X (expected: 0x010000)\n", x_powers[1]);
    printf("  x_powers[2] (x^32): 0x%06X\n", x_powers[2]);

    uint32_t test_crc = 0x123456;
    for (int bytes = 1; bytes <= 10000; bytes *= 10) {
        uint32_t slow = crc24a_shift_bytes(test_crc, bytes);
        uint32_t fast = crc24a_shift_bytes_fast(test_crc, bytes);
        printf("  shift(0x%06X, %5d bytes): slow=0x%06X fast=0x%06X %s\n",
               test_crc, bytes, slow, fast, (slow == fast) ? "OK" : "FAIL!");
    }

    // Test 6: Fast parallel CRC on 90KB
    printf("\nTest 6: Fast parallel CRC on 90KB\n");
    tb_data = (uint8_t*)malloc(tb_size);
    for (int i = 0; i < tb_size; i++) tb_data[i] = (uint8_t)(i * 17 & 0xFF);

    ref_crc = crc24a_reference(tb_data, tb_size);
    printf("  Reference CRC: 0x%06X\n", ref_crc);

    for (int chunks = 4; chunks <= 32; chunks *= 2) {
        par_crc = crc24a_parallel_fast(tb_data, tb_size, chunks);
        printf("  Fast parallel (%2d chunks): 0x%06X %s\n", chunks, par_crc,
               (ref_crc == par_crc) ? "OK" : "FAIL!");
    }

    free(tb_data);

    // Test 7: GPU kernel test with timing
    printf("\nTest 7: GPU parallel CRC kernel with timing\n");

    // Allocate GPU memory
    uint8_t* d_data;
    uint32_t* d_crc;
    cudaMalloc(&d_data, tb_size);
    cudaMalloc(&d_crc, sizeof(uint32_t));

    // Create test data
    tb_data = (uint8_t*)malloc(tb_size);
    for (int i = 0; i < tb_size; i++) tb_data[i] = (uint8_t)(i * 17 & 0xFF);

    // Copy to GPU
    cudaMemcpy(d_data, tb_data, tb_size, cudaMemcpyHostToDevice);

    // Compute reference on CPU
    ref_crc = crc24a_reference(tb_data, tb_size);

    // Warmup
    crc24a_compute(d_data, tb_size * 8, d_crc, 0);
    cudaDeviceSynchronize();

    // Time GPU CRC
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iters = 100;
    cudaEventRecord(start);
    for (int i = 0; i < num_iters; i++) {
        crc24a_compute(d_data, tb_size * 8, d_crc, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_us = (milliseconds * 1000.0f) / num_iters;

    // Get GPU result
    uint32_t gpu_crc;
    cudaMemcpy(&gpu_crc, d_crc, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("  Data size: %d bytes (%.1f KB)\n", tb_size, tb_size / 1024.0f);
    printf("  Reference CRC: 0x%06X\n", ref_crc);
    printf("  GPU CRC:       0x%06X %s\n", gpu_crc, (ref_crc == gpu_crc) ? "OK" : "FAIL!");
    printf("  GPU CRC time:  %.1f µs (avg of %d iterations)\n", avg_us, num_iters);
    printf("  Throughput:    %.1f GB/s\n", (tb_size / 1e9) / (avg_us / 1e6));

    printf("\nTest 8: Batch GPU CRC kernel correctness\n");
    const int batch_tbs = 6;
    const int bit_sizes[] = {1024, 8449, 20405, 65536, 205167, 360000};
    const int max_bits = 360000;
    const int stride = (max_bits + 7) / 8 + 16;
    uint8_t* batch_data = (uint8_t*)calloc(batch_tbs * stride, 1);
    uint32_t ref_batch[batch_tbs];
    for (int tb = 0; tb < batch_tbs; tb++) {
        uint8_t* tb_ptr = batch_data + tb * stride;
        int bytes = (bit_sizes[tb] + 7) / 8;
        for (int i = 0; i < bytes; i++) {
            tb_ptr[i] = (uint8_t)((tb * 73 + i * 29 + 0x5A) & 0xFF);
        }
        if ((bit_sizes[tb] % 8) != 0) {
            int keep = bit_sizes[tb] % 8;
            tb_ptr[bytes - 1] &= (uint8_t)(0xFF << (8 - keep));
        }
        ref_batch[tb] = crc24a_reference_bits(tb_ptr, bit_sizes[tb]);
    }

    uint8_t* d_batch_data;
    uint32_t* d_batch_crc;
    cudaMalloc(&d_batch_data, batch_tbs * stride);
    cudaMalloc(&d_batch_crc, batch_tbs * sizeof(uint32_t));
    cudaMemcpy(d_batch_data, batch_data, batch_tbs * stride, cudaMemcpyHostToDevice);

    bool batch_ok = true;
    for (int tb = 0; tb < batch_tbs; tb++) {
        crc24a_compute_batch(d_batch_data + tb * stride, d_batch_crc + tb, 1, bit_sizes[tb], stride, 0);
    }
    cudaDeviceSynchronize();

    uint32_t gpu_batch[batch_tbs] = {};
    cudaMemcpy(gpu_batch, d_batch_crc, batch_tbs * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (int tb = 0; tb < batch_tbs; tb++) {
        bool ok = (gpu_batch[tb] == ref_batch[tb]);
        batch_ok = batch_ok && ok;
        printf("  TB %d %6d bits: ref=0x%06X gpu=0x%06X %s\n",
               tb, bit_sizes[tb], ref_batch[tb], gpu_batch[tb], ok ? "OK" : "FAIL!");
    }

    const int uniform_bits = 205167;
    for (int tb = 0; tb < batch_tbs; tb++) {
        uint8_t* tb_ptr = batch_data + tb * stride;
        int bytes = (uniform_bits + 7) / 8;
        memset(tb_ptr, 0, stride);
        for (int i = 0; i < bytes; i++) {
            tb_ptr[i] = (uint8_t)((tb * 101 + i * 37 + 0x33) & 0xFF);
        }
        int keep = uniform_bits % 8;
        tb_ptr[bytes - 1] &= (uint8_t)(0xFF << (8 - keep));
        ref_batch[tb] = crc24a_reference_bits(tb_ptr, uniform_bits);
    }

    cudaMemcpy(d_batch_data, batch_data, batch_tbs * stride, cudaMemcpyHostToDevice);
    crc24a_compute_batch(d_batch_data, d_batch_crc, batch_tbs, uniform_bits, stride, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_batch, d_batch_crc, batch_tbs * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (int tb = 0; tb < batch_tbs; tb++) {
        bool ok = (gpu_batch[tb] == ref_batch[tb]);
        batch_ok = batch_ok && ok;
        printf("  Uniform TB %d %6d bits: ref=0x%06X gpu=0x%06X %s\n",
               tb, uniform_bits, ref_batch[tb], gpu_batch[tb], ok ? "OK" : "FAIL!");
    }

    if (!batch_ok) {
        printf("  Batch CRC correctness: FAIL\n");
        return 1;
    }
    printf("  Batch CRC correctness: OK\n");

    cudaFree(d_batch_data);
    cudaFree(d_batch_crc);
    free(batch_data);

    // Test different sizes
    printf("\nTest 9: CRC timing for different TB sizes\n");
    int sizes[] = {11000, 22000, 45000, 90000, 180000};
    for (int s = 0; s < 5; s++) {
        int size = sizes[s];
        if (size > tb_size) {
            cudaFree(d_data);
            cudaMalloc(&d_data, size);
            tb_data = (uint8_t*)realloc(tb_data, size);
            for (int i = 0; i < size; i++) tb_data[i] = (uint8_t)(i * 17 & 0xFF);
            cudaMemcpy(d_data, tb_data, size, cudaMemcpyHostToDevice);
            tb_size = size;
        }

        cudaEventRecord(start);
        for (int i = 0; i < num_iters; i++) {
            crc24a_compute(d_data, size * 8, d_crc, 0);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        avg_us = (milliseconds * 1000.0f) / num_iters;

        printf("  %6d bytes (%5.1f KB): %.1f µs (%.1f GB/s)\n",
               size, size/1024.0f, avg_us, (size/1e9)/(avg_us/1e6));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_crc);
    free(tb_data);

    printf("\n=== Test Complete ===\n");
    return 0;
}
