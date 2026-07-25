// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/phy/upper/sequence_generators/low_papr_sequence_generator.h"
#include "ocudu/phy/upper/signal_processors/srs/srs_estimator.h"
#include <memory>
#include <mutex>
#include <srs_estimator.h>
#include <vector>

namespace ocudu {

class pusch_device_grid_reader_cuda;

/// GPU-backed SRS estimator using CUDA kernels with software fallback.
class srs_estimator_cuda_impl : public srs_estimator
{
public:
  srs_estimator_cuda_impl(std::unique_ptr<low_papr_sequence_generator> sequence_generator_,
                             std::unique_ptr<srs_estimator>               fallback_,
                             unsigned                                     max_nof_prb_,
                             bool                                         force_gpu_path_ = false);

  ~srs_estimator_cuda_impl() override;

  srs_estimator_cuda_impl(const srs_estimator_cuda_impl&)            = delete;
  srs_estimator_cuda_impl& operator=(const srs_estimator_cuda_impl&) = delete;
  srs_estimator_cuda_impl(srs_estimator_cuda_impl&&)                 = delete;
  srs_estimator_cuda_impl& operator=(srs_estimator_cuda_impl&&)      = delete;

  srs_estimator_result estimate(const resource_grid_reader& grid, const srs_estimator_configuration& config) override;

  bool is_gpu_available() const { return gpu_available; }

private:
  std::unique_ptr<low_papr_sequence_generator> sequence_generator;
  std::unique_ptr<srs_estimator>               fallback;
  unsigned                                     max_nof_prb;

  bool                   gpu_available  = false;
  bool                   force_gpu_path = false;
  int                    device_id      = 0;
  void*                  stream         = nullptr;
  srs_estimator_handle_t handle         = nullptr;
  std::mutex             estimator_mutex;

  const resource_grid_reader*                       staged_grid_source = nullptr;
  std::unique_ptr<pusch_device_grid_reader_cuda> staged_grid_reader;

  bool                            gpu_configured = false;
  std::vector<int>                last_config_signature;
  std::vector<srs_estimator_cf_t> sequences;
};

/// Returns true when the CUDA SRS estimator backend can be instantiated.
bool is_srs_estimator_cuda_available();

} // namespace ocudu
