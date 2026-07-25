// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ocudu_ofh_compression_handle ocudu_ofh_compression_handle_t;

enum {
    OCUDU_OFH_COMPRESSION_TYPE_NONE = 0,
    OCUDU_OFH_COMPRESSION_TYPE_BFP  = 1
};

int  ocudu_ofh_compression_available(void);
int  ocudu_ofh_compression_create(ocudu_ofh_compression_handle_t** handle);
void ocudu_ofh_compression_destroy(ocudu_ofh_compression_handle_t* handle);
void* ocudu_ofh_compression_get_stream(ocudu_ofh_compression_handle_t* handle);
int  ocudu_ofh_compression_synchronize(ocudu_ofh_compression_handle_t* handle);

int ocudu_ofh_compress(ocudu_ofh_compression_handle_t* handle,
                       int                             compression_type,
                       void*                           output_bytes,
                       const void*                     input_cbf16,
                       unsigned                        nof_prbs,
                       unsigned                        data_width,
                       float                           iq_scaling);

int ocudu_ofh_decompress(ocudu_ofh_compression_handle_t* handle,
                         int                             compression_type,
                         void*                           output_cbf16,
                         const void*                     input_bytes,
                         unsigned                        nof_prbs,
                         unsigned                        data_width);

int ocudu_ofh_compress_device_grid(ocudu_ofh_compression_handle_t* handle,
                                   int                             compression_type,
                                   void*                           output_bytes,
                                   const void*                     input_grid_cbf16,
                                   unsigned                        nof_symbols,
                                   unsigned                        nof_subc,
                                   unsigned                        port,
                                   unsigned                        symbol,
                                   unsigned                        start_prb,
                                   unsigned                        nof_prbs,
                                   unsigned                        data_width,
                                   float                           iq_scaling);

int ocudu_ofh_decompress_to_device_grid(ocudu_ofh_compression_handle_t* handle,
                                        int                             compression_type,
                                        void*                           output_grid_cbf16,
                                        const void*                     input_bytes,
                                        unsigned                        nof_symbols,
                                        unsigned                        nof_subc,
                                        unsigned                        port,
                                        unsigned                        symbol,
                                        unsigned                        start_prb,
                                        unsigned                        nof_prbs,
                                        unsigned                        data_width);

int ocudu_ofh_decompress_to_device_grid_async(ocudu_ofh_compression_handle_t* handle,
                                              int                             compression_type,
                                              void*                           output_grid_cbf16,
                                              const void*                     input_bytes,
                                              unsigned                        nof_symbols,
                                              unsigned                        nof_subc,
                                              unsigned                        port,
                                              unsigned                        symbol,
                                              unsigned                        start_prb,
                                              unsigned                        nof_prbs,
                                              unsigned                        data_width);

int ocudu_ofh_decompress_to_device_prach_buffer_async(ocudu_ofh_compression_handle_t* handle,
                                                      int                             compression_type,
                                                      void*                           output_prach_cbf16,
                                                      unsigned                        output_offset_re,
                                                      const void*                     input_bytes,
                                                      unsigned                        input_start_re,
                                                      unsigned                        nof_re,
                                                      unsigned                        nof_prbs,
                                                      unsigned                        data_width);

int ocudu_ofh_compress_device_grid_ports(ocudu_ofh_compression_handle_t* handle,
                                         int                             compression_type,
                                         void*                           output_bytes,
                                         unsigned                        output_port_stride_bytes,
                                         const void*                     input_grid_cbf16,
                                         unsigned                        nof_symbols,
                                         unsigned                        nof_subc,
                                         unsigned                        first_port,
                                         unsigned                        nof_ports,
                                         unsigned                        symbol,
                                         unsigned                        start_prb,
                                         unsigned                        nof_prbs,
                                         unsigned                        data_width,
                                         float                           iq_scaling);

int ocudu_ofh_compress_device_grid_ports_to_host_buffers(ocudu_ofh_compression_handle_t* handle,
                                                         int                             compression_type,
                                                         void* const*                    output_host_buffers,
                                                         unsigned                        nof_output_buffers,
                                                         unsigned                        output_buffer_size_bytes,
                                                         const void*                     input_grid_cbf16,
                                                         unsigned                        nof_symbols,
                                                         unsigned                        nof_subc,
                                                         unsigned                        first_port,
                                                         unsigned                        nof_ports,
                                                         unsigned                        symbol,
                                                         unsigned                        start_prb,
                                                         unsigned                        nof_prbs,
                                                         unsigned                        data_width,
                                                         float                           iq_scaling);

int ocudu_ofh_compress_device_grid_ports_to_device(ocudu_ofh_compression_handle_t* handle,
                                                   int                             compression_type,
                                                   void*                           output_device_bytes,
                                                   unsigned                        output_port_stride_bytes,
                                                   const void*                     input_grid_cbf16,
                                                   unsigned                        nof_symbols,
                                                   unsigned                        nof_subc,
                                                   unsigned                        first_port,
                                                   unsigned                        nof_ports,
                                                   unsigned                        symbol,
                                                   unsigned                        start_prb,
                                                   unsigned                        nof_prbs,
                                                   unsigned                        data_width,
                                                   float                           iq_scaling);

int ocudu_ofh_compress_device_grid_ports_to_device_async(ocudu_ofh_compression_handle_t* handle,
                                                         int                             compression_type,
                                                         void*                           output_device_bytes,
                                                         unsigned                        output_port_stride_bytes,
                                                         const void*                     input_grid_cbf16,
                                                         unsigned                        nof_symbols,
                                                         unsigned                        nof_subc,
                                                         unsigned                        first_port,
                                                         unsigned                        nof_ports,
                                                         unsigned                        symbol,
                                                         unsigned                        start_prb,
                                                         unsigned                        nof_prbs,
                                                         unsigned                        data_width,
                                                         float                           iq_scaling);

int ocudu_ofh_compress_device_grid_symbol_batch(ocudu_ofh_compression_handle_t* handle,
                                                int                             compression_type,
                                                void*                           output_bytes,
                                                unsigned                        output_symbol_stride_bytes,
                                                unsigned                        output_port_stride_bytes,
                                                const void*                     input_grid_cbf16,
                                                unsigned                        nof_grid_symbols,
                                                unsigned                        nof_subc,
                                                unsigned                        first_port,
                                                unsigned                        nof_ports,
                                                unsigned                        first_symbol,
                                                unsigned                        nof_symbols,
                                                unsigned                        start_prb,
                                                unsigned                        nof_prbs,
                                                unsigned                        data_width,
                                                float                           iq_scaling);

int ocudu_ofh_compress_device_grid_symbol_batch_to_device(ocudu_ofh_compression_handle_t* handle,
                                                          int                             compression_type,
                                                          void*                           output_device_bytes,
                                                          unsigned                        output_symbol_stride_bytes,
                                                          unsigned                        output_port_stride_bytes,
                                                          const void*                     input_grid_cbf16,
                                                          unsigned                        nof_grid_symbols,
                                                          unsigned                        nof_subc,
                                                          unsigned                        first_port,
                                                          unsigned                        nof_ports,
                                                          unsigned                        first_symbol,
                                                          unsigned                        nof_symbols,
                                                          unsigned                        start_prb,
                                                          unsigned                        nof_prbs,
                                                          unsigned                        data_width,
                                                          float                           iq_scaling);

int ocudu_ofh_compress_device_grid_symbol_batch_to_host_buffers(ocudu_ofh_compression_handle_t* handle,
                                                                int                             compression_type,
                                                                void* const*                    output_host_buffers,
                                                                unsigned                        nof_output_buffers,
                                                                unsigned                        output_buffer_size_bytes,
                                                                const void*                     input_grid_cbf16,
                                                                unsigned                        nof_grid_symbols,
                                                                unsigned                        nof_subc,
                                                                unsigned                        first_port,
                                                                unsigned                        nof_ports,
                                                                unsigned                        first_symbol,
                                                                unsigned                        nof_symbols,
                                                                unsigned                        start_prb,
                                                                unsigned                        nof_prbs,
                                                                unsigned                        data_width,
                                                                float                           iq_scaling);

int ocudu_ofh_compress_device_grid_symbol_batch_to_device_async(ocudu_ofh_compression_handle_t* handle,
                                                                int                             compression_type,
                                                                void*                           output_device_bytes,
                                                                unsigned                        output_symbol_stride_bytes,
                                                                unsigned                        output_port_stride_bytes,
                                                                const void*                     input_grid_cbf16,
                                                                unsigned                        nof_grid_symbols,
                                                                unsigned                        nof_subc,
                                                                unsigned                        first_port,
                                                                unsigned                        nof_ports,
                                                                unsigned                        first_symbol,
                                                                unsigned                        nof_symbols,
                                                                unsigned                        start_prb,
                                                                unsigned                        nof_prbs,
                                                                unsigned                        data_width,
                                                                float                           iq_scaling);

int ocudu_ofh_decompress_to_device_grid_ports(ocudu_ofh_compression_handle_t* handle,
                                              int                             compression_type,
                                              void*                           output_grid_cbf16,
                                              const void*                     input_bytes,
                                              unsigned                        input_port_stride_bytes,
                                              unsigned                        nof_symbols,
                                              unsigned                        nof_subc,
                                              unsigned                        first_port,
                                              unsigned                        nof_ports,
                                              unsigned                        symbol,
                                              unsigned                        start_prb,
                                              unsigned                        nof_prbs,
                                              unsigned                        data_width);

int ocudu_ofh_decompress_device_bytes_to_device_grid_ports(ocudu_ofh_compression_handle_t* handle,
                                                           int                             compression_type,
                                                           void*                           output_grid_cbf16,
                                                           const void*                     input_device_bytes,
                                                           unsigned                        input_port_stride_bytes,
                                                           unsigned                        nof_symbols,
                                                           unsigned                        nof_subc,
                                                           unsigned                        first_port,
                                                           unsigned                        nof_ports,
                                                           unsigned                        symbol,
                                                           unsigned                        start_prb,
                                                           unsigned                        nof_prbs,
                                                           unsigned                        data_width);

int ocudu_ofh_decompress_to_device_grid_symbol_batch(ocudu_ofh_compression_handle_t* handle,
                                                     int                             compression_type,
                                                     void*                           output_grid_cbf16,
                                                     const void*                     input_bytes,
                                                     unsigned                        input_symbol_stride_bytes,
                                                     unsigned                        input_port_stride_bytes,
                                                     unsigned                        nof_grid_symbols,
                                                     unsigned                        nof_subc,
                                                     unsigned                        first_port,
                                                     unsigned                        nof_ports,
                                                     unsigned                        first_symbol,
                                                     unsigned                        nof_symbols,
                                                     unsigned                        start_prb,
                                                     unsigned                        nof_prbs,
                                                     unsigned                        data_width);

int ocudu_ofh_decompress_device_bytes_to_device_grid_symbol_batch(ocudu_ofh_compression_handle_t* handle,
                                                                  int                             compression_type,
                                                                  void*                           output_grid_cbf16,
                                                                  const void*                     input_device_bytes,
                                                                  unsigned                        input_symbol_stride_bytes,
                                                                  unsigned                        input_port_stride_bytes,
                                                                  unsigned                        nof_grid_symbols,
                                                                  unsigned                        nof_subc,
                                                                  unsigned                        first_port,
                                                                  unsigned                        nof_ports,
                                                                  unsigned                        first_symbol,
                                                                  unsigned                        nof_symbols,
                                                                  unsigned                        start_prb,
                                                                  unsigned                        nof_prbs,
                                                                  unsigned                        data_width);

#ifdef __cplusplus
}
#endif
