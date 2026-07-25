// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "iq_compression_cuda.h"
#include "ocudu/ofh/compression/compression_properties.h"
#include "ocudu/phy/support/prach_buffer.h"
#include "ocudu/phy/support/resource_grid_reader.h"
#include "ocudu/phy/support/resource_grid_writer.h"
#include "ocudu/support/error_handling.h"
#include "ocudu/support/ocudu_assert.h"

#ifdef ENABLE_CUDA
#include "ofh_compression.h"
#endif

using namespace ocudu;
using namespace ofh;

static int to_cuda_compression_type(compression_type type)
{
  switch (type) {
    case compression_type::none:
      return 0;
    case compression_type::BFP:
      return 1;
    default:
      return -1;
  }
}

bool ocudu::ofh::is_iq_compression_cuda_available()
{
#ifdef ENABLE_CUDA
  return ocudu_ofh_compression_available() != 0;
#else
  return false;
#endif
}

iq_compression_cuda::iq_compression_cuda(compression_type type_, float iq_scaling_) :
  type(type_), iq_scaling(iq_scaling_)
{
#ifdef ENABLE_CUDA
  report_fatal_error_if_not((type == compression_type::none) || (type == compression_type::BFP),
                            "CUDA OFH compression supports none and BFP only.");
  ocudu_ofh_compression_handle* initial_handle = nullptr;
  report_fatal_error_if_not(ocudu_ofh_compression_create(&initial_handle) != 0,
                            "Failed to create CUDA OFH compression backend.");
  handles.push_back(initial_handle);
#else
  report_fatal_error("CUDA OFH compression backend is not available in this build.");
#endif
}

iq_compression_cuda::~iq_compression_cuda()
{
#ifdef ENABLE_CUDA
  for (ocudu_ofh_compression_handle* handle : handles) {
    ocudu_ofh_compression_destroy(handle);
  }
#endif
}

ocudu_ofh_compression_handle* iq_compression_cuda::acquire_handle()
{
#ifdef ENABLE_CUDA
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!handles.empty()) {
      ocudu_ofh_compression_handle* handle = handles.back();
      handles.pop_back();
      return handle;
    }
  }

  ocudu_ofh_compression_handle* handle = nullptr;
  if (ocudu_ofh_compression_create(&handle) == 0) {
    return nullptr;
  }
  return handle;
#else
  return nullptr;
#endif
}

void iq_compression_cuda::release_handle(ocudu_ofh_compression_handle* acquired_handle)
{
#ifdef ENABLE_CUDA
  if (acquired_handle == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex);
  handles.push_back(acquired_handle);
#else
  (void)acquired_handle;
#endif
}

void iq_compression_cuda::compress(span<uint8_t>                buffer,
                                         span<const cbf16_t>          iq_data,
                                         const ru_compression_params& params)
{
#ifdef ENABLE_CUDA
  ocudu_assert(params.type == type, "CUDA OFH compression object used with unexpected compression type.");
  unsigned nof_prbs = iq_data.size() / NOF_SUBCARRIERS_PER_RB;
  unsigned prb_size = get_compressed_prb_size(params).value();
  ocudu_assert(buffer.size() >= prb_size * nof_prbs, "Output buffer does not have enough space for compressed PRBs.");
  if (nof_prbs == 0) {
    return;
  }
  ocudu_ofh_compression_handle* handle = acquire_handle();
  report_fatal_error_if_not(handle != nullptr, "Failed to create CUDA OFH compression backend.");
  bool success = ocudu_ofh_compress(handle,
                                    to_cuda_compression_type(type),
                                    buffer.data(),
                                    iq_data.data(),
                                    nof_prbs,
                                    params.data_width,
                                    iq_scaling) != 0;
  release_handle(handle);
  report_fatal_error_if_not(success, "CUDA OFH compression failed.");
#else
  (void)buffer;
  (void)iq_data;
  (void)params;
  report_fatal_error("CUDA OFH compression backend is not available in this build.");
#endif
}

bool iq_compression_cuda::compress_device_symbol(span<uint8_t>                buffer,
                                                       const resource_grid_reader&  grid,
                                                       unsigned                     port,
                                                       unsigned                     symbol,
                                                       unsigned                     start_prb,
                                                       unsigned                     nof_prbs,
                                                       const ru_compression_params& params)
{
#ifdef ENABLE_CUDA
  if (!grid.supports_device_grid_reading()) {
    return false;
  }
  if ((params.type != type) || (nof_prbs == 0)) {
    return false;
  }

  unsigned prb_size = get_compressed_prb_size(params).value();
  if (buffer.size() < prb_size * nof_prbs) {
    return false;
  }

  ocudu_ofh_compression_handle* handle = acquire_handle();
  if (handle == nullptr) {
    return false;
  }

  void* stream = ocudu_ofh_compression_get_stream(handle);
  bool  success =
      grid.prepare_device_grid_reading(stream) && (ocudu_ofh_compress_device_grid(handle,
                                                                                  to_cuda_compression_type(type),
                                                                                  buffer.data(),
                                                                                  grid.get_device_grid_cbf16(),
                                                                                  grid.get_nof_symbols(),
                                                                                  grid.get_nof_subc(),
                                                                                  port,
                                                                                  symbol,
                                                                                  start_prb,
                                                                                  nof_prbs,
                                                                                  params.data_width,
                                                                                  iq_scaling) != 0);
  release_handle(handle);
  return success;
#else
  (void)buffer;
  (void)grid;
  (void)port;
  (void)symbol;
  (void)start_prb;
  (void)nof_prbs;
  (void)params;
  return false;
#endif
}

bool iq_compression_cuda::compress_device_symbols(span<uint8_t>                buffer,
                                                        unsigned                     port_stride_bytes,
                                                        const resource_grid_reader&  grid,
                                                        unsigned                     first_port,
                                                        unsigned                     nof_ports,
                                                        unsigned                     symbol,
                                                        unsigned                     start_prb,
                                                        unsigned                     nof_prbs,
                                                        const ru_compression_params& params)
{
#ifdef ENABLE_CUDA
  if (!grid.supports_device_grid_reading()) {
    return false;
  }
  if ((params.type != type) || (nof_ports == 0) || (nof_prbs == 0)) {
    return false;
  }

  unsigned prb_size   = get_compressed_prb_size(params).value();
  unsigned port_bytes = prb_size * nof_prbs;
  if ((port_stride_bytes < port_bytes) ||
      (buffer.size() < static_cast<size_t>(nof_ports - 1U) * port_stride_bytes + port_bytes)) {
    return false;
  }

  ocudu_ofh_compression_handle* handle = acquire_handle();
  if (handle == nullptr) {
    return false;
  }

  void* stream = ocudu_ofh_compression_get_stream(handle);
  bool  success =
      grid.prepare_device_grid_reading(stream) && (ocudu_ofh_compress_device_grid_ports(handle,
                                                                                        to_cuda_compression_type(type),
                                                                                        buffer.data(),
                                                                                        port_stride_bytes,
                                                                                        grid.get_device_grid_cbf16(),
                                                                                        grid.get_nof_symbols(),
                                                                                        grid.get_nof_subc(),
                                                                                        first_port,
                                                                                        nof_ports,
                                                                                        symbol,
                                                                                        start_prb,
                                                                                        nof_prbs,
                                                                                        params.data_width,
                                                                                        iq_scaling) != 0);
  release_handle(handle);
  return success;
#else
  (void)buffer;
  (void)port_stride_bytes;
  (void)grid;
  (void)first_port;
  (void)nof_ports;
  (void)symbol;
  (void)start_prb;
  (void)nof_prbs;
  (void)params;
  return false;
#endif
}

bool iq_compression_cuda::compress_device_symbols_to_buffers(span<span<uint8_t>>          buffers,
                                                                   const resource_grid_reader&  grid,
                                                                   unsigned                     first_port,
                                                                   unsigned                     nof_ports,
                                                                   unsigned                     symbol,
                                                                   unsigned                     start_prb,
                                                                   unsigned                     nof_prbs,
                                                                   const ru_compression_params& params)
{
#ifdef ENABLE_CUDA
  if (!grid.supports_device_grid_reading()) {
    return false;
  }
  if ((params.type != type) || (nof_ports == 0) || (nof_prbs == 0) || (buffers.size() < nof_ports)) {
    return false;
  }

  unsigned                        prb_size   = get_compressed_prb_size(params).value();
  unsigned                        port_bytes = prb_size * nof_prbs;
  thread_local std::vector<void*> output_buffers;
  output_buffers.resize(nof_ports);
  for (unsigned i_port = 0; i_port != nof_ports; ++i_port) {
    if (buffers[i_port].size() < port_bytes) {
      return false;
    }
    output_buffers[i_port] = buffers[i_port].data();
  }

  ocudu_ofh_compression_handle* handle = acquire_handle();
  if (handle == nullptr) {
    return false;
  }

  void* stream  = ocudu_ofh_compression_get_stream(handle);
  bool  success = grid.prepare_device_grid_reading(stream) &&
                 (ocudu_ofh_compress_device_grid_ports_to_host_buffers(handle,
                                                                       to_cuda_compression_type(type),
                                                                       output_buffers.data(),
                                                                       nof_ports,
                                                                       port_bytes,
                                                                       grid.get_device_grid_cbf16(),
                                                                       grid.get_nof_symbols(),
                                                                       grid.get_nof_subc(),
                                                                       first_port,
                                                                       nof_ports,
                                                                       symbol,
                                                                       start_prb,
                                                                       nof_prbs,
                                                                       params.data_width,
                                                                       iq_scaling) != 0);
  release_handle(handle);
  return success;
#else
  (void)buffers;
  (void)grid;
  (void)first_port;
  (void)nof_ports;
  (void)symbol;
  (void)start_prb;
  (void)nof_prbs;
  (void)params;
  return false;
#endif
}

bool iq_compression_cuda::compress_device_symbol_batch(span<uint8_t>                buffer,
                                                             unsigned                     symbol_stride_bytes,
                                                             unsigned                     port_stride_bytes,
                                                             const resource_grid_reader&  grid,
                                                             unsigned                     first_port,
                                                             unsigned                     nof_ports,
                                                             unsigned                     first_symbol,
                                                             unsigned                     nof_symbols,
                                                             unsigned                     start_prb,
                                                             unsigned                     nof_prbs,
                                                             const ru_compression_params& params)
{
#ifdef ENABLE_CUDA
  if (!grid.supports_device_grid_reading()) {
    return false;
  }
  if ((params.type != type) || (nof_ports == 0) || (nof_symbols == 0) || (nof_prbs == 0)) {
    return false;
  }

  unsigned prb_size      = get_compressed_prb_size(params).value();
  unsigned port_bytes    = prb_size * nof_prbs;
  size_t   required_size = static_cast<size_t>(nof_symbols - 1U) * symbol_stride_bytes +
                         static_cast<size_t>(nof_ports - 1U) * port_stride_bytes + port_bytes;
  if ((port_stride_bytes < port_bytes) ||
      (symbol_stride_bytes < static_cast<size_t>(nof_ports - 1U) * port_stride_bytes + port_bytes) ||
      (buffer.size() < required_size)) {
    return false;
  }

  ocudu_ofh_compression_handle* handle = acquire_handle();
  if (handle == nullptr) {
    return false;
  }

  void* stream  = ocudu_ofh_compression_get_stream(handle);
  bool  success = grid.prepare_device_grid_reading(stream) &&
                 (ocudu_ofh_compress_device_grid_symbol_batch(handle,
                                                              to_cuda_compression_type(type),
                                                              buffer.data(),
                                                              symbol_stride_bytes,
                                                              port_stride_bytes,
                                                              grid.get_device_grid_cbf16(),
                                                              grid.get_nof_symbols(),
                                                              grid.get_nof_subc(),
                                                              first_port,
                                                              nof_ports,
                                                              first_symbol,
                                                              nof_symbols,
                                                              start_prb,
                                                              nof_prbs,
                                                              params.data_width,
                                                              iq_scaling) != 0);
  release_handle(handle);
  return success;
#else
  (void)buffer;
  (void)symbol_stride_bytes;
  (void)port_stride_bytes;
  (void)grid;
  (void)first_port;
  (void)nof_ports;
  (void)first_symbol;
  (void)nof_symbols;
  (void)start_prb;
  (void)nof_prbs;
  (void)params;
  return false;
#endif
}

bool iq_compression_cuda::compress_device_symbol_batch_to_buffers(span<span<uint8_t>>          buffers,
                                                                        const resource_grid_reader&  grid,
                                                                        unsigned                     first_port,
                                                                        unsigned                     nof_ports,
                                                                        unsigned                     first_symbol,
                                                                        unsigned                     nof_symbols,
                                                                        unsigned                     start_prb,
                                                                        unsigned                     nof_prbs,
                                                                        const ru_compression_params& params)
{
#ifdef ENABLE_CUDA
  if (!grid.supports_device_grid_reading()) {
    return false;
  }
  if ((params.type != type) || (nof_ports == 0) || (nof_symbols == 0) || (nof_prbs == 0)) {
    return false;
  }

  unsigned nof_buffers = nof_symbols * nof_ports;
  if (buffers.size() < nof_buffers) {
    return false;
  }

  unsigned                        prb_size   = get_compressed_prb_size(params).value();
  unsigned                        port_bytes = prb_size * nof_prbs;
  thread_local std::vector<void*> output_buffers;
  output_buffers.resize(nof_buffers);
  for (unsigned i_buffer = 0; i_buffer != nof_buffers; ++i_buffer) {
    if (buffers[i_buffer].size() < port_bytes) {
      return false;
    }
    output_buffers[i_buffer] = buffers[i_buffer].data();
  }

  ocudu_ofh_compression_handle* handle = acquire_handle();
  if (handle == nullptr) {
    return false;
  }

  void* stream  = ocudu_ofh_compression_get_stream(handle);
  bool  success = grid.prepare_device_grid_reading(stream) &&
                 (ocudu_ofh_compress_device_grid_symbol_batch_to_host_buffers(handle,
                                                                              to_cuda_compression_type(type),
                                                                              output_buffers.data(),
                                                                              nof_buffers,
                                                                              port_bytes,
                                                                              grid.get_device_grid_cbf16(),
                                                                              grid.get_nof_symbols(),
                                                                              grid.get_nof_subc(),
                                                                              first_port,
                                                                              nof_ports,
                                                                              first_symbol,
                                                                              nof_symbols,
                                                                              start_prb,
                                                                              nof_prbs,
                                                                              params.data_width,
                                                                              iq_scaling) != 0);
  release_handle(handle);
  return success;
#else
  (void)buffers;
  (void)grid;
  (void)first_port;
  (void)nof_ports;
  (void)first_symbol;
  (void)nof_symbols;
  (void)start_prb;
  (void)nof_prbs;
  (void)params;
  return false;
#endif
}

void iq_compression_cuda::decompress(span<cbf16_t>                iq_data,
                                           span<const uint8_t>          compressed_data,
                                           const ru_compression_params& params)
{
#ifdef ENABLE_CUDA
  ocudu_assert(params.type == type, "CUDA OFH decompression object used with unexpected compression type.");
  unsigned nof_prbs = iq_data.size() / NOF_SUBCARRIERS_PER_RB;
  unsigned prb_size = get_compressed_prb_size(params).value();
  ocudu_assert(compressed_data.size() >= nof_prbs * prb_size,
               "Input does not contain enough bytes to decompress {} PRBs",
               nof_prbs);
  if (nof_prbs == 0) {
    return;
  }
  ocudu_ofh_compression_handle* handle = acquire_handle();
  report_fatal_error_if_not(handle != nullptr, "Failed to create CUDA OFH compression backend.");
  bool success = ocudu_ofh_decompress(handle,
                                      to_cuda_compression_type(type),
                                      iq_data.data(),
                                      compressed_data.data(),
                                      nof_prbs,
                                      params.data_width) != 0;
  release_handle(handle);
  report_fatal_error_if_not(success, "CUDA OFH decompression failed.");
#else
  (void)iq_data;
  (void)compressed_data;
  (void)params;
  report_fatal_error("CUDA OFH compression backend is not available in this build.");
#endif
}

bool iq_compression_cuda::decompress_to_resource_grid(resource_grid_writer&        grid,
                                                            unsigned                     port,
                                                            unsigned                     symbol,
                                                            unsigned                     start_prb,
                                                            unsigned                     nof_prbs,
                                                            span<const uint8_t>          compressed_data,
                                                            const ru_compression_params& params)
{
#ifdef ENABLE_CUDA
  if (!grid.supports_device_grid_mapping()) {
    return false;
  }
  if ((params.type != type) || (nof_prbs == 0)) {
    return false;
  }

  unsigned prb_size = get_compressed_prb_size(params).value();
  if (compressed_data.size() < prb_size * nof_prbs) {
    return false;
  }

  ocudu_ofh_compression_handle* handle = acquire_handle();
  if (handle == nullptr) {
    return false;
  }

  void* stream = ocudu_ofh_compression_get_stream(handle);
  if (!grid.prepare_device_grid_mapping(stream)) {
    release_handle(handle);
    return false;
  }

  bool success = ocudu_ofh_decompress_to_device_grid_async(handle,
                                                           to_cuda_compression_type(type),
                                                           grid.get_device_grid_bf16(),
                                                           compressed_data.data(),
                                                           grid.get_nof_symbols(),
                                                           grid.get_nof_subc(),
                                                           port,
                                                           symbol,
                                                           start_prb,
                                                           nof_prbs,
                                                           params.data_width) != 0;
  if (success) {
    success = grid.on_device_grid_mapping_enqueued(stream);
  }

  if (!success) {
    (void)ocudu_ofh_compression_synchronize(handle);
    (void)grid.cancel_device_grid_mapping();
  }

  release_handle(handle);
  return success;
#else
  (void)grid;
  (void)port;
  (void)symbol;
  (void)start_prb;
  (void)nof_prbs;
  (void)compressed_data;
  (void)params;
  return false;
#endif
}

bool iq_compression_cuda::decompress_to_prach_buffer(prach_buffer&                buffer,
                                                           unsigned                     port,
                                                           unsigned                     td_occasion,
                                                           unsigned                     fd_occasion,
                                                           unsigned                     symbol,
                                                           unsigned                     start_re,
                                                           unsigned                     input_start_re,
                                                           unsigned                     nof_re,
                                                           unsigned                     nof_prbs,
                                                           span<const uint8_t>          compressed_data,
                                                           const ru_compression_params& params)
{
#ifdef ENABLE_CUDA
  if (!buffer.supports_device_prach_buffer_mapping()) {
    return false;
  }
  if ((params.type != type) || (nof_prbs == 0) || (nof_re == 0)) {
    return false;
  }

  unsigned prb_size = get_compressed_prb_size(params).value();
  if (compressed_data.size() < prb_size * nof_prbs) {
    return false;
  }

  ocudu_ofh_compression_handle* handle = acquire_handle();
  if (handle == nullptr) {
    return false;
  }

  void* stream = ocudu_ofh_compression_get_stream(handle);
  if (!buffer.prepare_device_prach_buffer_mapping(stream)) {
    release_handle(handle);
    return false;
  }

  unsigned output_offset_re = buffer.get_device_prach_symbol_offset(port, td_occasion, fd_occasion, symbol) + start_re;
  bool     success          = ocudu_ofh_decompress_to_device_prach_buffer_async(handle,
                                                                   to_cuda_compression_type(type),
                                                                   buffer.get_device_prach_buffer_cbf16(),
                                                                   output_offset_re,
                                                                   compressed_data.data(),
                                                                   input_start_re,
                                                                   nof_re,
                                                                   nof_prbs,
                                                                   params.data_width) != 0;
  if (success) {
    success = buffer.on_device_prach_buffer_mapping_enqueued(stream);
  }

  if (!success) {
    (void)ocudu_ofh_compression_synchronize(handle);
    (void)buffer.cancel_device_prach_buffer_mapping();
  }

  release_handle(handle);
  return success;
#else
  (void)buffer;
  (void)port;
  (void)td_occasion;
  (void)fd_occasion;
  (void)symbol;
  (void)start_re;
  (void)input_start_re;
  (void)nof_re;
  (void)nof_prbs;
  (void)compressed_data;
  (void)params;
  return false;
#endif
}
