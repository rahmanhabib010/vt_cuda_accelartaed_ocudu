// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ocudu/ofh/compression/compression_factory.h"
#include "iq_compression_bfp_impl.h"
#include "iq_compression_cuda.h"
#include "iq_compression_death_impl.h"
#include "iq_compression_none_impl.h"
#include "iq_compressor_selector.h"
#include "iq_decompressor_selector.h"
#include "ocudu/support/cpu_features.h"
#include "ocudu/support/error_handling.h"

#ifdef __x86_64__
#include "iq_compression_bfp_avx2.h"
#include "iq_compression_bfp_avx512.h"
#include "iq_compression_none_avx2.h"
#include "iq_compression_none_avx512.h"
#endif

#ifdef __ARM_NEON
#include "iq_compression_bfp_neon.h"
#include "iq_compression_none_neon.h"
#endif // __ARM_NEON

#include <cstdlib>

using namespace ocudu;
using namespace ofh;

static std::string resolve_compression_impl_type(const std::string& requested, const char* env_name)
{
  auto normalize_mode = [](const std::string& mode) {
    if (mode == "enabled") {
      return std::string("cuda");
    }
    if ((mode == "disabled") || (mode == "cpu") || (mode == "host")) {
      return std::string("cpu_auto");
    }
    return mode;
  };

  if (requested != "auto") {
    return normalize_mode(requested);
  }
  const char* env_value = std::getenv(env_name);
  if ((env_value != nullptr) && (env_value[0] != '\0')) {
    return normalize_mode(std::string(env_value));
  }
  env_value = std::getenv("OCUDU_OFH_COMPRESSION_IMPL");
  if ((env_value != nullptr) && (env_value[0] != '\0')) {
    return normalize_mode(std::string(env_value));
  }
#ifdef ENABLE_CUDA
  if (is_iq_compression_cuda_available()) {
    return "cuda";
  }
#endif
  return requested;
}

static bool is_cuda_compression_requested(const std::string& impl_type)
{
  return (impl_type == "cuda") || (impl_type == "gpu");
}

static bool is_cpu_auto_compression_requested(const std::string& impl_type)
{
  return (impl_type == "auto") || (impl_type == "cpu_auto");
}

std::unique_ptr<iq_compressor> ocudu::ofh::create_iq_compressor(compression_type        type,
                                                                ocudulog::basic_logger& logger,
                                                                float                   iq_scaling,
                                                                const std::string&      impl_type)
{
  std::string resolved_impl_type = resolve_compression_impl_type(impl_type, "OCUDU_OFH_TX_COMPRESSION_IMPL");
#ifndef ENABLE_CUDA
  report_fatal_error_if_not(!is_cuda_compression_requested(resolved_impl_type),
                            "CUDA OFH compression requested but unavailable.");
#endif

  switch (type) {
    case compression_type::none:
#ifdef ENABLE_CUDA
      if (is_cuda_compression_requested(resolved_impl_type)) {
        report_fatal_error_if_not(is_iq_compression_cuda_available(),
                                  "CUDA OFH compression requested but unavailable.");
        return std::make_unique<iq_compression_cuda>(type, iq_scaling);
      }
#endif
#ifdef __x86_64__
      {
        bool supports_avx2   = cpu_supports_feature(cpu_feature::avx2);
        bool supports_avx512 = cpu_supports_feature(cpu_feature::avx512f) &&
                               cpu_supports_feature(cpu_feature::avx512vl) &&
                               cpu_supports_feature(cpu_feature::avx512bw);
        if (((resolved_impl_type == "avx512") || is_cpu_auto_compression_requested(resolved_impl_type)) &&
            supports_avx512) {
          return std::make_unique<iq_compression_none_avx512>(logger, iq_scaling);
        }
        if (((resolved_impl_type == "avx2") || is_cpu_auto_compression_requested(resolved_impl_type)) &&
            supports_avx2) {
          return std::make_unique<iq_compression_none_avx2>(logger, iq_scaling);
        }
      }
#endif
#ifdef __ARM_NEON
      if ((resolved_impl_type == "neon") || is_cpu_auto_compression_requested(resolved_impl_type)) {
        return std::make_unique<iq_compression_none_neon>(logger, iq_scaling);
      }
#endif // __ARM_NEON
      return std::make_unique<iq_compression_none_impl>(logger, iq_scaling);
    case compression_type::BFP:
#ifdef ENABLE_CUDA
      if (is_cuda_compression_requested(resolved_impl_type)) {
        report_fatal_error_if_not(is_iq_compression_cuda_available(),
                                  "CUDA OFH compression requested but unavailable.");
        return std::make_unique<iq_compression_cuda>(type, iq_scaling);
      }
#endif
#ifdef __x86_64__
      {
        bool supports_avx2 = cpu_supports_feature(cpu_feature::avx2);
        bool supports_avx512 =
            cpu_supports_feature(cpu_feature::avx512f) && cpu_supports_feature(cpu_feature::avx512vl) &&
            cpu_supports_feature(cpu_feature::avx512bw) && cpu_supports_feature(cpu_feature::avx512dq) &&
            cpu_supports_feature(cpu_feature::avx512cd);
        if (((resolved_impl_type == "avx512") || is_cpu_auto_compression_requested(resolved_impl_type)) &&
            supports_avx512) {
          return std::make_unique<iq_compression_bfp_avx512>(logger, iq_scaling);
        }
        if (((resolved_impl_type == "avx2") || is_cpu_auto_compression_requested(resolved_impl_type)) &&
            supports_avx2) {
          return std::make_unique<iq_compression_bfp_avx2>(logger, iq_scaling);
        }
      }
#endif
#ifdef __ARM_NEON
      if ((resolved_impl_type == "neon") || is_cpu_auto_compression_requested(resolved_impl_type)) {
        return std::make_unique<iq_compression_bfp_neon>(logger, iq_scaling);
      }
#endif // __ARM_NEON
      return std::make_unique<iq_compression_bfp_impl>(logger, iq_scaling);
    case compression_type::block_scaling:
      return std::make_unique<iq_compression_death_impl>();
    case compression_type::mu_law:
      return std::make_unique<iq_compression_death_impl>();
    case compression_type::modulation:
      return std::make_unique<iq_compression_death_impl>();
    case compression_type::bfp_selective:
      return std::make_unique<iq_compression_death_impl>();
    case compression_type::mod_selective:
      return std::make_unique<iq_compression_death_impl>();
    default:
      report_fatal_error("Compression type '{}' is not implemented", to_string(type));
  }
}

std::unique_ptr<iq_decompressor>
ocudu::ofh::create_iq_decompressor(compression_type type, ocudulog::basic_logger& logger, const std::string& impl_type)
{
  std::string resolved_impl_type = resolve_compression_impl_type(impl_type, "OCUDU_OFH_RX_COMPRESSION_IMPL");
#ifndef ENABLE_CUDA
  report_fatal_error_if_not(!is_cuda_compression_requested(resolved_impl_type),
                            "CUDA OFH decompression requested but unavailable.");
#endif

  switch (type) {
    case compression_type::none:
#ifdef ENABLE_CUDA
      if (is_cuda_compression_requested(resolved_impl_type)) {
        report_fatal_error_if_not(is_iq_compression_cuda_available(),
                                  "CUDA OFH decompression requested but unavailable.");
        return std::make_unique<iq_compression_cuda>(type);
      }
#endif
#ifdef __x86_64__
      {
        bool supports_avx2 = cpu_supports_feature(cpu_feature::avx2);
        bool supports_avx512 =
            cpu_supports_feature(cpu_feature::avx512f) && cpu_supports_feature(cpu_feature::avx512vl) &&
            cpu_supports_feature(cpu_feature::avx512bw) && cpu_supports_feature(cpu_feature::avx512vbmi);
        if (((resolved_impl_type == "avx512") || is_cpu_auto_compression_requested(resolved_impl_type)) &&
            supports_avx512) {
          return std::make_unique<iq_compression_none_avx512>(logger);
        }
        if (((resolved_impl_type == "avx2") || is_cpu_auto_compression_requested(resolved_impl_type)) &&
            supports_avx2) {
          return std::make_unique<iq_compression_none_avx2>(logger);
        }
      }
#endif
#ifdef __ARM_NEON
      if ((resolved_impl_type == "neon") || is_cpu_auto_compression_requested(resolved_impl_type)) {
        return std::make_unique<iq_compression_none_neon>(logger);
      }
#endif // __ARM_NEON
      return std::make_unique<iq_compression_none_impl>(logger);
    case compression_type::BFP:
#ifdef ENABLE_CUDA
      if (is_cuda_compression_requested(resolved_impl_type)) {
        report_fatal_error_if_not(is_iq_compression_cuda_available(),
                                  "CUDA OFH decompression requested but unavailable.");
        return std::make_unique<iq_compression_cuda>(type);
      }
#endif
#ifdef __x86_64__
      {
        bool supports_avx2 = cpu_supports_feature(cpu_feature::avx2);
        bool supports_avx512 =
            cpu_supports_feature(cpu_feature::avx512f) && cpu_supports_feature(cpu_feature::avx512vl) &&
            cpu_supports_feature(cpu_feature::avx512bw) && cpu_supports_feature(cpu_feature::avx512vbmi);
        if (((resolved_impl_type == "avx512") || is_cpu_auto_compression_requested(resolved_impl_type)) &&
            supports_avx512) {
          return std::make_unique<iq_compression_bfp_avx512>(logger);
        }
        if (((resolved_impl_type == "avx2") || is_cpu_auto_compression_requested(resolved_impl_type)) &&
            supports_avx2) {
          return std::make_unique<iq_compression_bfp_avx2>(logger);
        }
      }
#endif
#ifdef __ARM_NEON
      if ((resolved_impl_type == "neon") || is_cpu_auto_compression_requested(resolved_impl_type)) {
        return std::make_unique<iq_compression_bfp_neon>(logger);
      }
#endif // __ARM_NEON
      return std::make_unique<iq_compression_bfp_impl>(logger);
    case compression_type::block_scaling:
      return std::make_unique<iq_compression_death_impl>();
    case compression_type::mu_law:
      return std::make_unique<iq_compression_death_impl>();
    case compression_type::modulation:
      return std::make_unique<iq_compression_death_impl>();
    case compression_type::bfp_selective:
      return std::make_unique<iq_compression_death_impl>();
    case compression_type::mod_selective:
      return std::make_unique<iq_compression_death_impl>();
    default:
      report_fatal_error("Compression type '{}' is not implemented", to_string(type));
  }
}

std::unique_ptr<iq_decompressor> ocudu::ofh::create_iq_decompressor_selector(
    std::array<std::unique_ptr<iq_decompressor>, NOF_COMPRESSION_TYPES_SUPPORTED> decompressors)
{
  return std::make_unique<iq_decompressor_selector>(std::move(decompressors));
}

std::unique_ptr<iq_compressor> ocudu::ofh::create_iq_compressor_selector(
    std::array<std::unique_ptr<iq_compressor>, NOF_COMPRESSION_TYPES_SUPPORTED> compressors)
{
  return std::make_unique<iq_compressor_selector>(std::move(compressors));
}
