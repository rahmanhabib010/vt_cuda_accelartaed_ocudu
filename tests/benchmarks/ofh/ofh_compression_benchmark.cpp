// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/ofh/compression/compression_factory.h"
#include "ocudu/ofh/compression/compression_properties.h"
#include "ocudu/ran/bs_channel_bandwidth.h"
#include "ocudu/ran/cyclic_prefix.h"
#include "ocudu/ran/resource_block.h"
#include "ocudu/ran/slot_point.h"
#include "ocudu/support/benchmark_utils.h"
#ifdef ENABLE_CUDA
#include "ofh_compression.h"
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <cstdlib>
#include <getopt.h>
#include <random>

using namespace ocudu;

// Random generator.
static std::mt19937 rgen(0);

static unsigned             nof_repetitions   = 10000;
static bool                 silent            = false;
static std::string          method            = "bfp";
static std::string          impl_type         = "auto";
static std::string          grid_mode         = "host";
static unsigned             nof_ports         = 1;
static unsigned             nof_batch_symbols = 1;
static bs_channel_bandwidth bw                = ocudu::bs_channel_bandwidth::MHz20;
static subcarrier_spacing   scs               = subcarrier_spacing::kHz30;

static void usage(const char* prog)
{
  fmt::print("Usage: {} [-R repetitions] [-T compression type] [-F factory type] [-s silent]\n", prog);
  fmt::print("\t-R Repetitions [Default {}]\n", nof_repetitions);
  fmt::print("\t-T Type of compression [{{'none', 'bfp'}}, default is {}]\n", method);
  fmt::print("\t-F Select compression factory [Default {}]\n", impl_type);
  fmt::print("\t-G Grid mode [{{'host', 'device', 'device-batch', 'device-host-buffer-batch', "
             "'device-device-batch', 'device-symbol-batch', 'device-host-buffer-symbol-batch', "
             "'device-device-symbol-batch', 'device-device-symbol-batch-async'}}, default is {}]\n",
             grid_mode);
  fmt::print("\t-B Channel bandwidth [Default {}]\n", fmt::underlying(bw));
  fmt::print("\t-C Subcarrier spacing. [Default {}]\n", to_string(scs));
  fmt::print("\t-N Number of ports [from 1 to 4, default {}]\n", nof_ports);
  fmt::print("\t-Y Number of OFDM symbols to batch [Default {}]\n", nof_batch_symbols);
  fmt::print("\t-h Show this message\n");
}

static bool validate_bw(unsigned bandwidth)
{
  switch (bandwidth) {
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz5):
      bw = bs_channel_bandwidth::MHz5;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz10):
      bw = bs_channel_bandwidth::MHz10;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz15):
      bw = bs_channel_bandwidth::MHz15;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz20):
      bw = bs_channel_bandwidth::MHz20;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz25):
      bw = bs_channel_bandwidth::MHz25;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz30):
      bw = bs_channel_bandwidth::MHz30;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz40):
      bw = bs_channel_bandwidth::MHz40;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz50):
      bw = bs_channel_bandwidth::MHz50;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz60):
      bw = bs_channel_bandwidth::MHz60;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz70):
      bw = bs_channel_bandwidth::MHz70;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz80):
      bw = bs_channel_bandwidth::MHz80;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz90):
      bw = bs_channel_bandwidth::MHz90;
      break;
    case bs_channel_bandwidth_to_MHz(bs_channel_bandwidth::MHz100):
      bw = bs_channel_bandwidth::MHz100;
      break;
    default:
      return false;
  }
  return true;
}

static void parse_args(int argc, char** argv)
{
  int  opt         = 0;
  bool invalid_arg = false;
  while ((opt = getopt(argc, argv, "R:T:F:G:B:C:N:Y:sh")) != -1) {
    switch (opt) {
      case 'R':
        nof_repetitions = std::strtol(optarg, nullptr, 10);
        break;
      case 'T':
        method = std::string(optarg);
        break;
      case 'F':
        impl_type = std::string(optarg);
        break;
      case 'G':
        grid_mode = std::string(optarg);
        if ((grid_mode != "host") && (grid_mode != "device") && (grid_mode != "device-batch") &&
            (grid_mode != "device-host-buffer-batch") && (grid_mode != "device-device-batch") &&
            (grid_mode != "device-symbol-batch") && (grid_mode != "device-host-buffer-symbol-batch") &&
            (grid_mode != "device-device-symbol-batch") && (grid_mode != "device-device-symbol-batch-async")) {
          fmt::print("Invalid grid mode\n");
          invalid_arg = true;
        }
        break;
      case 'B':
        if (optarg != nullptr) {
          if (!validate_bw(std::strtol(optarg, nullptr, 10))) {
            fmt::print("Invalid bandwidth\n");
            invalid_arg = true;
          }
        }
        break;
      case 'C':
        if (optarg != nullptr) {
          scs = to_subcarrier_spacing(std::string(optarg));
          if (scs == subcarrier_spacing::invalid) {
            fmt::print("Invalid subcarrier spacing\n");
            invalid_arg = true;
          }
        }
        break;
      case 'N':
        nof_ports = std::strtol(optarg, nullptr, 10);
        if ((nof_ports < 1) || (nof_ports > 4)) {
          fmt::print("Invalid number of ports\n");
          invalid_arg = true;
        }
        break;
      case 'Y':
        nof_batch_symbols = std::strtol(optarg, nullptr, 10);
        if ((nof_batch_symbols < 1) || (nof_batch_symbols > 14)) {
          fmt::print("Invalid number of symbols\n");
          invalid_arg = true;
        }
        break;
      case 's':
        silent = (!silent);
        break;
      case 'h':
      default:
        usage(argv[0]);
        std::exit(0);
    }
    if (invalid_arg) {
      usage(argv[0]);
      std::exit(0);
    }
  }
}

#ifdef ENABLE_CUDA
static int to_cuda_compression_type(ofh::compression_type type)
{
  return (type == ofh::compression_type::BFP) ? OCUDU_OFH_COMPRESSION_TYPE_BFP : OCUDU_OFH_COMPRESSION_TYPE_NONE;
}

static void check_cuda(cudaError_t status, const char* operation)
{
  if (status != cudaSuccess) {
    fmt::print(stderr, "{} failed: {}\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

static void check_cuda(bool success, const char* operation)
{
  if (!success) {
    fmt::print(stderr, "{} failed\n", operation);
    std::exit(1);
  }
}
#endif

int main(int argc, char** argv)
{
  parse_args(argc, argv);

  std::uniform_real_distribution<float> dist(-1.0, +1.0);

  ocudulog::basic_logger& logger = ocudulog::fetch_basic_logger("TEST", false);
  logger.set_level(ocudulog::basic_levels::none);

  std::size_t nof_prbs           = get_max_Nprb(bw, scs, frequency_range::FR1);
  double      symbol_duration_us = 1e3 / (get_nsymb_per_slot(cyclic_prefix::NORMAL) * get_nof_slots_per_subframe(scs));

  std::unique_ptr<ofh::iq_compressor> compressor =
      create_iq_compressor(ofh::to_compression_type(method), logger, 0.27, impl_type);
  ocudu_assert(compressor != nullptr, "Failed to create OFH compressor");

  std::unique_ptr<ofh::iq_decompressor> decompressor =
      create_iq_decompressor(ofh::to_compression_type(method), logger, impl_type);
  ocudu_assert(decompressor != nullptr, "Failed to create OFH decompressor");

  fmt::memory_buffer meas_name;
  fmt::format_to(std::back_inserter(meas_name),
                 "OFH compression: Method={}, Symbol duration is {:>6.3f} us, Implementation is {}, PRBs = {}, Number "
                 "of ports = {}, Number of symbols = {}, Grid mode = {}",
                 method,
                 symbol_duration_us,
                 impl_type,
                 nof_prbs,
                 nof_ports,
                 nof_batch_symbols,
                 grid_mode);
  benchmarker perf_meas(to_string(meas_name), nof_repetitions);

  // Test for the most common bit width.
  for (unsigned bit_width : {8, 9, 10, 12, 14, 16}) {
    ofh::ru_compression_params params;
    params.type       = ofh::to_compression_type(method);
    params.data_width = bit_width;

    // Measurement description.
    std::string common_meas_name         = to_string(params.type) + "-" + std::to_string(bit_width) + "b";
    std::string meas_descr_compression   = common_meas_name + " compression";
    std::string meas_descr_decompression = common_meas_name + " decompression";

    std::vector<std::vector<cbf16_t>> test_data(nof_ports);
    std::vector<std::vector<cbf16_t>> decompressed_data(nof_ports);
    std::vector<std::vector<uint8_t>> compressed_data(nof_ports);

    unsigned comp_prb_size = ofh::get_compressed_prb_size(params).value();
    for (unsigned i = 0; i != nof_ports; ++i) {
      test_data[i].resize(static_cast<size_t>(nof_batch_symbols) * nof_prbs * NOF_SUBCARRIERS_PER_RB);
      decompressed_data[i].resize(static_cast<size_t>(nof_batch_symbols) * nof_prbs * NOF_SUBCARRIERS_PER_RB);
      compressed_data[i].resize(static_cast<size_t>(nof_batch_symbols) * nof_prbs * comp_prb_size);
    }

    // Generate input random data.
    for (unsigned i = 0; i != nof_ports; ++i) {
      std::generate(test_data[i].begin(), test_data[i].end(), [&]() { return cbf16_t{dist(rgen), dist(rgen)}; });
    }

#ifdef ENABLE_CUDA
    if ((grid_mode == "device") || (grid_mode == "device-batch") || (grid_mode == "device-host-buffer-batch") ||
        (grid_mode == "device-device-batch") || (grid_mode == "device-symbol-batch") ||
        (grid_mode == "device-host-buffer-symbol-batch") || (grid_mode == "device-device-symbol-batch") ||
        (grid_mode == "device-device-symbol-batch-async")) {
      if (impl_type != "cuda") {
        fmt::print(stderr, "Device grid mode requires -F cuda.\n");
        std::exit(1);
      }

      ocudu_ofh_compression_handle_t* cuda_handle = nullptr;
      check_cuda(ocudu_ofh_compression_create(&cuda_handle) != 0, "ocudu_ofh_compression_create");

      cbf16_t* device_input_grid  = nullptr;
      cbf16_t* device_output_grid = nullptr;
      size_t   symbol_re_count    = nof_prbs * NOF_SUBCARRIERS_PER_RB;
      size_t   port_re_count      = static_cast<size_t>(nof_batch_symbols) * symbol_re_count;
      size_t   grid_re_count      = static_cast<size_t>(nof_ports) * port_re_count;
      check_cuda(cudaMalloc(&device_input_grid, grid_re_count * sizeof(cbf16_t)), "cudaMalloc input grid");
      check_cuda(cudaMalloc(&device_output_grid, grid_re_count * sizeof(cbf16_t)), "cudaMalloc output grid");
      for (unsigned i = 0; i != nof_ports; ++i) {
        check_cuda(cudaMemcpy(device_input_grid + static_cast<size_t>(i) * port_re_count,
                              test_data[i].data(),
                              test_data[i].size() * sizeof(cbf16_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy input grid");
      }

      std::vector<uint8_t> batched_compressed_data;
      if ((grid_mode == "device-batch") || (grid_mode == "device-host-buffer-batch") ||
          (grid_mode == "device-device-batch") || (grid_mode == "device-symbol-batch") ||
          (grid_mode == "device-host-buffer-symbol-batch") || (grid_mode == "device-device-symbol-batch") ||
          (grid_mode == "device-device-symbol-batch-async")) {
        batched_compressed_data.resize(static_cast<size_t>(nof_batch_symbols) * nof_ports * nof_prbs * comp_prb_size);
      }
      uint8_t* device_compressed_data = nullptr;
      if ((grid_mode == "device-device-batch") || (grid_mode == "device-device-symbol-batch") ||
          (grid_mode == "device-device-symbol-batch-async")) {
        check_cuda(cudaMalloc(&device_compressed_data,
                              static_cast<size_t>(nof_batch_symbols) * nof_ports * nof_prbs * comp_prb_size),
                   "cudaMalloc compressed data");
      }

      unsigned port_stride_bytes   = nof_prbs * comp_prb_size;
      unsigned symbol_stride_bytes = nof_ports * port_stride_bytes;
      size_t   nof_iq_samples =
          static_cast<size_t>(nof_batch_symbols) * nof_ports * nof_prbs * NOF_SUBCARRIERS_PER_RB * 2U;
      std::vector<void*> host_batch_buffers(static_cast<size_t>(nof_batch_symbols) * nof_ports);
      for (unsigned symbol = 0; symbol != nof_batch_symbols; ++symbol) {
        for (unsigned i_port = 0; i_port != nof_ports; ++i_port) {
          host_batch_buffers[static_cast<size_t>(symbol) * nof_ports + i_port] =
              compressed_data[i_port].data() + static_cast<size_t>(symbol) * port_stride_bytes;
        }
      }

      perf_meas.new_measure(
          meas_descr_compression,
          nof_iq_samples,
          [&]() {
            if (grid_mode == "device-device-batch") {
              for (unsigned symbol = 0; symbol != nof_batch_symbols; ++symbol) {
                check_cuda(ocudu_ofh_compress_device_grid_ports_to_device(
                                     cuda_handle,
                                     to_cuda_compression_type(params.type),
                                     device_compressed_data + static_cast<size_t>(symbol) * symbol_stride_bytes,
                                     port_stride_bytes,
                                     device_input_grid,
                                     nof_batch_symbols,
                                     nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                     0,
                                     nof_ports,
                                     symbol,
                                     0,
                                     nof_prbs,
                                     params.data_width,
                                     0.27F) != 0,
                                 "ocudu_ofh_compress_device_grid_ports_to_device");
              }
            } else if (grid_mode == "device-device-symbol-batch") {
              check_cuda(
                  ocudu_ofh_compress_device_grid_symbol_batch_to_device(cuda_handle,
                                                                        to_cuda_compression_type(params.type),
                                                                        device_compressed_data,
                                                                        symbol_stride_bytes,
                                                                        port_stride_bytes,
                                                                        device_input_grid,
                                                                        nof_batch_symbols,
                                                                        nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                                                        0,
                                                                        nof_ports,
                                                                        0,
                                                                        nof_batch_symbols,
                                                                        0,
                                                                        nof_prbs,
                                                                        params.data_width,
                                                                        0.27F) != 0,
                  "ocudu_ofh_compress_device_grid_symbol_batch_to_device");
            } else if (grid_mode == "device-device-symbol-batch-async") {
              check_cuda(
                  ocudu_ofh_compress_device_grid_symbol_batch_to_device_async(cuda_handle,
                                                                              to_cuda_compression_type(params.type),
                                                                              device_compressed_data,
                                                                              symbol_stride_bytes,
                                                                              port_stride_bytes,
                                                                              device_input_grid,
                                                                              nof_batch_symbols,
                                                                              nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                                                              0,
                                                                              nof_ports,
                                                                              0,
                                                                              nof_batch_symbols,
                                                                              0,
                                                                              nof_prbs,
                                                                              params.data_width,
                                                                              0.27F) != 0,
                  "ocudu_ofh_compress_device_grid_symbol_batch_to_device_async");
            } else if (grid_mode == "device-host-buffer-batch") {
              for (unsigned symbol = 0; symbol != nof_batch_symbols; ++symbol) {
                check_cuda(ocudu_ofh_compress_device_grid_ports_to_host_buffers(
                                     cuda_handle,
                                     to_cuda_compression_type(params.type),
                                     host_batch_buffers.data() + static_cast<size_t>(symbol) * nof_ports,
                                     nof_ports,
                                     port_stride_bytes,
                                     device_input_grid,
                                     nof_batch_symbols,
                                     nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                     0,
                                     nof_ports,
                                     symbol,
                                     0,
                                     nof_prbs,
                                     params.data_width,
                                     0.27F) != 0,
                                 "ocudu_ofh_compress_device_grid_ports_to_host_buffers");
              }
            } else if (grid_mode == "device-host-buffer-symbol-batch") {
              check_cuda(
                  ocudu_ofh_compress_device_grid_symbol_batch_to_host_buffers(cuda_handle,
                                                                              to_cuda_compression_type(params.type),
                                                                              host_batch_buffers.data(),
                                                                              host_batch_buffers.size(),
                                                                              port_stride_bytes,
                                                                              device_input_grid,
                                                                              nof_batch_symbols,
                                                                              nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                                                              0,
                                                                              nof_ports,
                                                                              0,
                                                                              nof_batch_symbols,
                                                                              0,
                                                                              nof_prbs,
                                                                              params.data_width,
                                                                              0.27F) != 0,
                  "ocudu_ofh_compress_device_grid_symbol_batch_to_host_buffers");
            } else if (grid_mode == "device-batch") {
              for (unsigned symbol = 0; symbol != nof_batch_symbols; ++symbol) {
                check_cuda(ocudu_ofh_compress_device_grid_ports(
                                     cuda_handle,
                                     to_cuda_compression_type(params.type),
                                     batched_compressed_data.data() + static_cast<size_t>(symbol) * symbol_stride_bytes,
                                     port_stride_bytes,
                                     device_input_grid,
                                     nof_batch_symbols,
                                     nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                     0,
                                     nof_ports,
                                     symbol,
                                     0,
                                     nof_prbs,
                                     params.data_width,
                                     0.27F) != 0,
                                 "ocudu_ofh_compress_device_grid_ports");
              }
            } else if (grid_mode == "device-symbol-batch") {
              check_cuda(ocudu_ofh_compress_device_grid_symbol_batch(cuda_handle,
                                                                           to_cuda_compression_type(params.type),
                                                                           batched_compressed_data.data(),
                                                                           symbol_stride_bytes,
                                                                           port_stride_bytes,
                                                                           device_input_grid,
                                                                           nof_batch_symbols,
                                                                           nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                                                           0,
                                                                           nof_ports,
                                                                           0,
                                                                           nof_batch_symbols,
                                                                           0,
                                                                           nof_prbs,
                                                                           params.data_width,
                                                                           0.27F) != 0,
                               "ocudu_ofh_compress_device_grid_symbol_batch");
            } else {
              for (unsigned symbol = 0; symbol != nof_batch_symbols; ++symbol) {
                for (unsigned i = 0; i != nof_ports; ++i) {
                  check_cuda(ocudu_ofh_compress_device_grid(cuda_handle,
                                                                  to_cuda_compression_type(params.type),
                                                                  compressed_data[i].data() +
                                                                      static_cast<size_t>(symbol) * port_stride_bytes,
                                                                  device_input_grid,
                                                                  nof_batch_symbols,
                                                                  nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                                                  i,
                                                                  symbol,
                                                                  0,
                                                                  nof_prbs,
                                                                  params.data_width,
                                                                  0.27F) != 0,
                                   "ocudu_ofh_compress_device_grid");
                }
              }
            }
          },
          [&]() {
            if (grid_mode == "device-device-symbol-batch-async") {
              check_cuda(ocudu_ofh_compression_synchronize(cuda_handle) != 0,
                               "ocudu_ofh_compression_synchronize");
            }
          });
      perf_meas.new_measure(meas_descr_decompression, nof_iq_samples, [&]() {
        if (grid_mode == "device-device-batch") {
          for (unsigned symbol = 0; symbol != nof_batch_symbols; ++symbol) {
            check_cuda(ocudu_ofh_decompress_device_bytes_to_device_grid_ports(
                                 cuda_handle,
                                 to_cuda_compression_type(params.type),
                                 device_output_grid,
                                 device_compressed_data + static_cast<size_t>(symbol) * symbol_stride_bytes,
                                 port_stride_bytes,
                                 nof_batch_symbols,
                                 nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                 0,
                                 nof_ports,
                                 symbol,
                                 0,
                                 nof_prbs,
                                 params.data_width) != 0,
                             "ocudu_ofh_decompress_device_bytes_to_device_grid_ports");
          }
        } else if ((grid_mode == "device-device-symbol-batch") || (grid_mode == "device-device-symbol-batch-async")) {
          check_cuda(
              ocudu_ofh_decompress_device_bytes_to_device_grid_symbol_batch(cuda_handle,
                                                                            to_cuda_compression_type(params.type),
                                                                            device_output_grid,
                                                                            device_compressed_data,
                                                                            symbol_stride_bytes,
                                                                            port_stride_bytes,
                                                                            nof_batch_symbols,
                                                                            nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                                                            0,
                                                                            nof_ports,
                                                                            0,
                                                                            nof_batch_symbols,
                                                                            0,
                                                                            nof_prbs,
                                                                            params.data_width) != 0,
              "ocudu_ofh_decompress_device_bytes_to_device_grid_symbol_batch");
        } else if ((grid_mode == "device-batch") || (grid_mode == "device-host-buffer-batch")) {
          for (unsigned symbol = 0; symbol != nof_batch_symbols; ++symbol) {
            check_cuda(ocudu_ofh_decompress_to_device_grid_ports(
                                 cuda_handle,
                                 to_cuda_compression_type(params.type),
                                 device_output_grid,
                                 batched_compressed_data.data() + static_cast<size_t>(symbol) * symbol_stride_bytes,
                                 port_stride_bytes,
                                 nof_batch_symbols,
                                 nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                 0,
                                 nof_ports,
                                 symbol,
                                 0,
                                 nof_prbs,
                                 params.data_width) != 0,
                             "ocudu_ofh_decompress_to_device_grid_ports");
          }
        } else if ((grid_mode == "device-symbol-batch") || (grid_mode == "device-host-buffer-symbol-batch")) {
          check_cuda(ocudu_ofh_decompress_to_device_grid_symbol_batch(cuda_handle,
                                                                            to_cuda_compression_type(params.type),
                                                                            device_output_grid,
                                                                            batched_compressed_data.data(),
                                                                            symbol_stride_bytes,
                                                                            port_stride_bytes,
                                                                            nof_batch_symbols,
                                                                            nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                                                            0,
                                                                            nof_ports,
                                                                            0,
                                                                            nof_batch_symbols,
                                                                            0,
                                                                            nof_prbs,
                                                                            params.data_width) != 0,
                           "ocudu_ofh_decompress_to_device_grid_symbol_batch");
        } else {
          for (unsigned symbol = 0; symbol != nof_batch_symbols; ++symbol) {
            for (unsigned i = 0; i != nof_ports; ++i) {
              check_cuda(ocudu_ofh_decompress_to_device_grid(cuda_handle,
                                                                   to_cuda_compression_type(params.type),
                                                                   device_output_grid,
                                                                   compressed_data[i].data() +
                                                                       static_cast<size_t>(symbol) * port_stride_bytes,
                                                                   nof_batch_symbols,
                                                                   nof_prbs * NOF_SUBCARRIERS_PER_RB,
                                                                   i,
                                                                   symbol,
                                                                   0,
                                                                   nof_prbs,
                                                                   params.data_width) != 0,
                               "ocudu_ofh_decompress_to_device_grid");
            }
          }
        }
      });

      if (device_compressed_data != nullptr) {
        cudaFree(device_compressed_data);
      }
      cudaFree(device_output_grid);
      cudaFree(device_input_grid);
      ocudu_ofh_compression_destroy(cuda_handle);
      continue;
    }
#else
    ocudu_assert(grid_mode == "host", "Device grid mode requires a CUDA build.");
#endif

    // Measure performance.
    size_t nof_iq_samples = static_cast<size_t>(nof_batch_symbols) * nof_ports * nof_prbs * NOF_SUBCARRIERS_PER_RB * 2U;
    perf_meas.new_measure(meas_descr_compression, nof_iq_samples, [&]() {
      for (unsigned i = 0; i != nof_ports; ++i) {
        for (unsigned symbol = 0; symbol != nof_batch_symbols; ++symbol) {
          compressor->compress(
              span<uint8_t>(compressed_data[i])
                  .subspan(static_cast<size_t>(symbol) * nof_prbs * comp_prb_size, nof_prbs * comp_prb_size),
              span<const cbf16_t>(test_data[i])
                  .subspan(static_cast<size_t>(symbol) * nof_prbs * NOF_SUBCARRIERS_PER_RB,
                           nof_prbs * NOF_SUBCARRIERS_PER_RB),
              params);
        }
      }
    });
    perf_meas.new_measure(meas_descr_decompression, nof_iq_samples, [&]() {
      for (unsigned i = 0; i != nof_ports; ++i) {
        for (unsigned symbol = 0; symbol != nof_batch_symbols; ++symbol) {
          decompressor->decompress(
              span<cbf16_t>(decompressed_data[i])
                  .subspan(static_cast<size_t>(symbol) * nof_prbs * NOF_SUBCARRIERS_PER_RB,
                           nof_prbs * NOF_SUBCARRIERS_PER_RB),
              span<const uint8_t>(compressed_data[i])
                  .subspan(static_cast<size_t>(symbol) * nof_prbs * comp_prb_size, nof_prbs * comp_prb_size),
              params);
        }
      }
    });
  }

  if (!silent) {
    perf_meas.print_percentiles_time("microseconds", 1e-3);
    perf_meas.print_percentiles_throughput("samples");
  }
}
