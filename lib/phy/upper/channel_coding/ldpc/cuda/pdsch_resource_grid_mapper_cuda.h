// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include <cstdint>

namespace ocudu {

/// \brief Maps one-layer INT8 PDSCH symbols into a device-resident BF16 resource-grid scratch buffer.
///
/// This is the first GPU-resident mapper primitive. It intentionally maps a
/// dense, already-compacted PDSCH symbol span into a compact per-port scratch
/// grid, not the host resource_grid layout. The scratch layout is:
/// `port * nof_re + re`, with each RE stored as two BF16 uint16 words
/// `(real, imag)`.
///
/// \param[in]  d_symbols_int8 Device pointer to interleaved INT8 IQ symbols.
/// \param[out] d_grid_bf16    Device pointer to compact BF16 IQ grid scratch.
/// \param[in]  nof_re         Number of resource elements in the compact data allocation.
/// \param[in]  nof_ports      Number of output ports.
/// \param[in]  weight_real    Real precoding weight applied to all ports.
/// \param[in]  stream         Opaque cudaStream_t. If null, CUDA default stream is used.
/// \return True if the CUDA launch was accepted.
bool pdsch_map_one_layer_all_ports_int8_to_bf16_grid(const int8_t* d_symbols_int8,
                                                     uint16_t*     d_grid_bf16,
                                                     unsigned      nof_re,
                                                     unsigned      nof_ports,
                                                     float         weight_real,
                                                     void*         stream);

/// \brief Maps INT8 PDSCH symbols into real device resource-grid coordinates.
///
/// The output layout is host resource-grid compatible:
/// `port * nof_grid_re_per_port + symbol * nof_subc + subcarrier`, with each
/// RE stored as two BF16 uint16 words `(real, imag)`. The RE coordinate list is
/// already compacted in host mapper order.
///
/// For one layer, the same layer is replicated to all output ports. For
/// multi-layer diagonal precoding, `nof_layers` must equal `nof_ports`, and
/// layer `i` is mapped to port `i`. Input symbols are RE-major and layer
/// interleaved: `symbols[re * nof_layers + layer]`.
///
/// \param[in]  d_symbols_int8       Device pointer to interleaved INT8 IQ symbols.
/// \param[in]  d_re_offsets         Device pointer to linear RE offsets within one port.
/// \param[out] d_grid_bf16          Device pointer to full BF16 IQ grid.
/// \param[in]  nof_re               Number of data RE to map.
/// \param[in]  nof_ports            Number of output ports.
/// \param[in]  nof_grid_re_per_port Number of RE in one full port grid.
/// \param[in]  weight_real          Real precoding weight applied to all ports.
/// \param[in]  stream               Opaque cudaStream_t. If null, CUDA default stream is used.
/// \return True if the CUDA launch was accepted.
bool pdsch_map_one_layer_all_ports_int8_to_bf16_real_grid(const int8_t*   d_symbols_int8,
                                                          const uint32_t* d_re_offsets,
                                                          uint16_t*       d_grid_bf16,
                                                          unsigned        nof_re,
                                                          unsigned        nof_ports,
                                                          unsigned        nof_grid_re_per_port,
                                                          float           weight_real,
                                                          void*           stream);

/// \brief Maps INT8 PDSCH symbols from one or more layers into real device resource-grid coordinates.
///
/// For one layer, the same layer is replicated to all output ports. For multi-layer diagonal precoding, \c nof_layers
/// must equal \c nof_ports, and layer \c i is mapped to port \c i.
///
/// \param[in]  d_symbols_int8       Device pointer to interleaved INT8 IQ symbols.
/// \param[in]  d_re_offsets         Device pointer to linear RE offsets within one port.
/// \param[out] d_grid_bf16          Device pointer to full BF16 IQ grid.
/// \param[in]  nof_re               Number of data RE to map.
/// \param[in]  nof_ports            Number of output ports.
/// \param[in]  nof_layers           Number of PDSCH layers in the input symbol stream.
/// \param[in]  nof_grid_re_per_port Number of RE in one full port grid.
/// \param[in]  weight_real          Real precoding weight applied to the mapped ports.
/// \param[in]  stream               Opaque cudaStream_t. If null, CUDA default stream is used.
/// \return True if the CUDA launch was accepted.
bool pdsch_map_layers_int8_to_bf16_real_grid(const int8_t*   d_symbols_int8,
                                             const uint32_t* d_re_offsets,
                                             uint16_t*       d_grid_bf16,
                                             unsigned        nof_re,
                                             unsigned        nof_ports,
                                             unsigned        nof_layers,
                                             unsigned        nof_grid_re_per_port,
                                             float           weight_real,
                                             void*           stream);

} // namespace ocudu
