// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "pdsch_resource_grid_mapper_cuda.h"
#include <cuda_runtime.h>

using namespace ocudu;

namespace {

__device__ uint16_t float_to_bf16_bits(float value)
{
  uint32_t as_u32 = __float_as_uint(value);
  as_u32 += 0x7fffU + ((as_u32 >> 16U) & 1U);
  return static_cast<uint16_t>(as_u32 >> 16U);
}

__global__ void map_one_layer_all_ports_int8_to_bf16_grid_kernel(const int8_t* __restrict__ d_symbols_int8,
                                                                 uint16_t* __restrict__ d_grid_bf16,
                                                                 unsigned nof_re,
                                                                 unsigned nof_ports,
                                                                 float    weight_real)
{
  unsigned i_re = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_re >= nof_re) {
    return;
  }

  float    real      = static_cast<float>(d_symbols_int8[2U * i_re]) * weight_real;
  float    imag      = static_cast<float>(d_symbols_int8[2U * i_re + 1U]) * weight_real;
  uint16_t real_bf16 = float_to_bf16_bits(real);
  uint16_t imag_bf16 = float_to_bf16_bits(imag);

  for (unsigned i_port = 0; i_port != nof_ports; ++i_port) {
    unsigned out_index         = 2U * (i_port * nof_re + i_re);
    d_grid_bf16[out_index]     = real_bf16;
    d_grid_bf16[out_index + 1] = imag_bf16;
  }
}

__global__ void map_layers_int8_to_bf16_real_grid_kernel(const int8_t* __restrict__ d_symbols_int8,
                                                         const uint32_t* __restrict__ d_re_offsets,
                                                         uint16_t* __restrict__ d_grid_bf16,
                                                         unsigned nof_re,
                                                         unsigned nof_ports,
                                                         unsigned nof_layers,
                                                         unsigned nof_grid_re_per_port,
                                                         float    weight_real)
{
  unsigned i_re = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_re >= nof_re) {
    return;
  }

  uint32_t re_offset = d_re_offsets[i_re];

  if (nof_layers == 1) {
    float    real      = static_cast<float>(d_symbols_int8[2U * i_re]) * weight_real;
    float    imag      = static_cast<float>(d_symbols_int8[2U * i_re + 1U]) * weight_real;
    uint16_t real_bf16 = float_to_bf16_bits(real);
    uint16_t imag_bf16 = float_to_bf16_bits(imag);

    for (unsigned i_port = 0; i_port != nof_ports; ++i_port) {
      unsigned out_index         = 2U * (i_port * nof_grid_re_per_port + re_offset);
      d_grid_bf16[out_index]     = real_bf16;
      d_grid_bf16[out_index + 1] = imag_bf16;
    }
    return;
  }

  for (unsigned i_layer = 0; i_layer != nof_layers; ++i_layer) {
    unsigned symbol_index = i_re * nof_layers + i_layer;
    float    real         = static_cast<float>(d_symbols_int8[2U * symbol_index]) * weight_real;
    float    imag         = static_cast<float>(d_symbols_int8[2U * symbol_index + 1U]) * weight_real;
    unsigned out_index    = 2U * (i_layer * nof_grid_re_per_port + re_offset);
    d_grid_bf16[out_index]     = float_to_bf16_bits(real);
    d_grid_bf16[out_index + 1] = float_to_bf16_bits(imag);
  }
}

} // namespace

bool ocudu::pdsch_map_one_layer_all_ports_int8_to_bf16_grid(const int8_t* d_symbols_int8,
                                                            uint16_t*     d_grid_bf16,
                                                            unsigned      nof_re,
                                                            unsigned      nof_ports,
                                                            float         weight_real,
                                                            void*         stream)
{
  if ((d_symbols_int8 == nullptr) || (d_grid_bf16 == nullptr) || (nof_re == 0) || (nof_ports == 0)) {
    return false;
  }

  static constexpr unsigned block_size  = 256;
  unsigned                  grid_size   = (nof_re + block_size - 1U) / block_size;
  auto                      cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  map_one_layer_all_ports_int8_to_bf16_grid_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
      d_symbols_int8, d_grid_bf16, nof_re, nof_ports, weight_real);
  return cudaGetLastError() == cudaSuccess;
}

bool ocudu::pdsch_map_one_layer_all_ports_int8_to_bf16_real_grid(const int8_t*   d_symbols_int8,
                                                                 const uint32_t* d_re_offsets,
                                                                 uint16_t*       d_grid_bf16,
                                                                 unsigned        nof_re,
                                                                 unsigned        nof_ports,
                                                                 unsigned        nof_grid_re_per_port,
                                                                 float           weight_real,
                                                                 void*           stream)
{
  return pdsch_map_layers_int8_to_bf16_real_grid(
      d_symbols_int8, d_re_offsets, d_grid_bf16, nof_re, nof_ports, 1, nof_grid_re_per_port, weight_real, stream);
}

bool ocudu::pdsch_map_layers_int8_to_bf16_real_grid(const int8_t*   d_symbols_int8,
                                                    const uint32_t* d_re_offsets,
                                                    uint16_t*       d_grid_bf16,
                                                    unsigned        nof_re,
                                                    unsigned        nof_ports,
                                                    unsigned        nof_layers,
                                                    unsigned        nof_grid_re_per_port,
                                                    float           weight_real,
                                                    void*           stream)
{
  if ((d_symbols_int8 == nullptr) || (d_re_offsets == nullptr) || (d_grid_bf16 == nullptr) || (nof_re == 0) ||
      (nof_ports == 0) || (nof_layers == 0) || (nof_grid_re_per_port == 0)) {
    return false;
  }
  if ((nof_layers > 1) && (nof_layers != nof_ports)) {
    return false;
  }

  static constexpr unsigned block_size  = 256;
  unsigned                  grid_size   = (nof_re + block_size - 1U) / block_size;
  auto                      cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  map_layers_int8_to_bf16_real_grid_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
      d_symbols_int8, d_re_offsets, d_grid_bf16, nof_re, nof_ports, nof_layers, nof_grid_re_per_port, weight_real);
  return cudaGetLastError() == cudaSuccess;
}
