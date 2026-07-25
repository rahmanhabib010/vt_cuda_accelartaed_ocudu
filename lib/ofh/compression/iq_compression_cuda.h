// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/ofh/compression/compression_params.h"
#include "ocudu/ofh/compression/iq_compressor.h"
#include "ocudu/ofh/compression/iq_decompressor.h"
#include <memory>
#include <mutex>
#include <vector>

struct ocudu_ofh_compression_handle;

namespace ocudu {
namespace ofh {

bool is_iq_compression_cuda_available();

class iq_compression_cuda : public iq_compressor, public iq_decompressor
{
public:
  explicit iq_compression_cuda(compression_type type_, float iq_scaling_ = 1.0F);
  ~iq_compression_cuda() override;

  iq_compression_cuda(const iq_compression_cuda&)            = delete;
  iq_compression_cuda& operator=(const iq_compression_cuda&) = delete;

  void compress(span<uint8_t> buffer, span<const cbf16_t> iq_data, const ru_compression_params& params) override;

  bool compress_device_symbol(span<uint8_t>                buffer,
                              const resource_grid_reader&  grid,
                              unsigned                     port,
                              unsigned                     symbol,
                              unsigned                     start_prb,
                              unsigned                     nof_prbs,
                              const ru_compression_params& params) override;

  bool compress_device_symbols(span<uint8_t>                buffer,
                               unsigned                     port_stride_bytes,
                               const resource_grid_reader&  grid,
                               unsigned                     first_port,
                               unsigned                     nof_ports,
                               unsigned                     symbol,
                               unsigned                     start_prb,
                               unsigned                     nof_prbs,
                               const ru_compression_params& params) override;

  bool compress_device_symbols_to_buffers(span<span<uint8_t>>          buffers,
                                          const resource_grid_reader&  grid,
                                          unsigned                     first_port,
                                          unsigned                     nof_ports,
                                          unsigned                     symbol,
                                          unsigned                     start_prb,
                                          unsigned                     nof_prbs,
                                          const ru_compression_params& params) override;

  bool compress_device_symbol_batch(span<uint8_t>                buffer,
                                    unsigned                     symbol_stride_bytes,
                                    unsigned                     port_stride_bytes,
                                    const resource_grid_reader&  grid,
                                    unsigned                     first_port,
                                    unsigned                     nof_ports,
                                    unsigned                     first_symbol,
                                    unsigned                     nof_symbols,
                                    unsigned                     start_prb,
                                    unsigned                     nof_prbs,
                                    const ru_compression_params& params) override;

  bool compress_device_symbol_batch_to_buffers(span<span<uint8_t>>          buffers,
                                               const resource_grid_reader&  grid,
                                               unsigned                     first_port,
                                               unsigned                     nof_ports,
                                               unsigned                     first_symbol,
                                               unsigned                     nof_symbols,
                                               unsigned                     start_prb,
                                               unsigned                     nof_prbs,
                                               const ru_compression_params& params) override;

  void
  decompress(span<cbf16_t> iq_data, span<const uint8_t> compressed_data, const ru_compression_params& params) override;

  bool supports_resource_grid_decompression() const override { return true; }

  bool supports_prach_buffer_decompression() const override { return true; }

  bool decompress_to_resource_grid(resource_grid_writer&        grid,
                                   unsigned                     port,
                                   unsigned                     symbol,
                                   unsigned                     start_prb,
                                   unsigned                     nof_prbs,
                                   span<const uint8_t>          compressed_data,
                                   const ru_compression_params& params) override;

  bool decompress_to_prach_buffer(prach_buffer&                buffer,
                                  unsigned                     port,
                                  unsigned                     td_occasion,
                                  unsigned                     fd_occasion,
                                  unsigned                     symbol,
                                  unsigned                     start_re,
                                  unsigned                     input_start_re,
                                  unsigned                     nof_re,
                                  unsigned                     nof_prbs,
                                  span<const uint8_t>          compressed_data,
                                  const ru_compression_params& params) override;

private:
  ocudu_ofh_compression_handle* acquire_handle();
  void                          release_handle(ocudu_ofh_compression_handle* acquired_handle);

  compression_type                           type;
  float                                      iq_scaling;
  std::vector<ocudu_ofh_compression_handle*> handles;
  std::mutex                                 mutex;
};

} // namespace ofh
} // namespace ocudu
