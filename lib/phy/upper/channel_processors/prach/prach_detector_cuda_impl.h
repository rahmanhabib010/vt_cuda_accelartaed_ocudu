// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/phy/upper/channel_processors/prach/prach_detector.h"
#include "ocudu/phy/upper/channel_processors/prach/prach_generator.h"
#include <complex>
#include <memory>
#include <mutex>
#include <prach_detector.h>
#include <vector>

namespace ocudu {

class prach_detector_cuda_impl : public prach_detector
{
public:
  prach_detector_cuda_impl(std::unique_ptr<prach_generator> generator_,
                                 std::unique_ptr<prach_detector>  fallback_,
                                 bool                             force_gpu_path_);

  ~prach_detector_cuda_impl() override;

  prach_detector_cuda_impl(const prach_detector_cuda_impl&)            = delete;
  prach_detector_cuda_impl& operator=(const prach_detector_cuda_impl&) = delete;

  prach_detection_result detect(const prach_buffer& input, const configuration& config) override;

  bool is_gpu_available() const { return gpu_available; }

private:
  struct detector_geometry;

  bool should_use_gpu(const prach_buffer& input, const configuration& config, const detector_geometry& geometry) const;

  prach_detection_result detect_gpu(const prach_buffer& input, const configuration& config, const detector_geometry& geometry);

  void prepare_roots(const configuration& config, const detector_geometry& geometry);

  uint64_t make_root_cache_key(const configuration& config, const detector_geometry& geometry) const;

  std::unique_ptr<prach_generator> generator;
  std::unique_ptr<prach_detector>  fallback;
  bool                             force_gpu_path = false;
  bool                             gpu_available  = false;
  int                              device_id      = 0;
  void*                            stream         = nullptr;
  prach_detector_handle_t          handle         = nullptr;
  std::mutex                       detector_mutex;
  std::vector<uint32_t>            host_prach_packed;
  std::vector<std::complex<float>> host_roots;
  uint64_t                         cached_root_key = 0;
};

/// Returns true when the CUDA PRACH detector backend can be instantiated.
bool is_prach_detector_cuda_available();

} // namespace ocudu
