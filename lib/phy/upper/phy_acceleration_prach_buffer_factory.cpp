// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "phy_acceleration_prach_buffer_factory.h"
#include "phy_acceleration_runtime_options.h"
#include "ocudu/phy/support/support_factories.h"
#include "ocudu/ran/prach/prach_constants.h"

#ifdef ENABLE_CUDA
#include "prach_buffer_cuda_visible_impl.h"
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#endif

using namespace ocudu;

namespace {

#ifdef ENABLE_CUDA
bool should_use_managed_prach_buffer_auto()
{
  int device_id = 0;
  if (cudaGetDevice(&device_id) != cudaSuccess) {
    return false;
  }

  int managed_memory = 0;
  if (cudaDeviceGetAttribute(&managed_memory, cudaDevAttrManagedMemory, device_id) != cudaSuccess) {
    return false;
  }
  return managed_memory != 0;
}

bool mode_uses_managed_prach_buffer()
{
  const char* mode = std::getenv("OCUDU_UL_CUDA_VISIBLE_PRACH_BUFFER");
  if (mode == nullptr) {
    mode = phy_acceleration_cuda_visible_grid_mode("OCUDU_UL_CUDA_VISIBLE_GRID");
  }
  if (mode == nullptr) {
    return should_use_managed_prach_buffer_auto();
  }

  if (phy_acceleration_env_mode_is_managed(mode)) {
    return true;
  }
  if ((std::strcmp(mode, "pinned") == 0) || (std::strcmp(mode, "off") == 0) || (std::strcmp(mode, "disabled") == 0)) {
    return false;
  }
  return should_use_managed_prach_buffer_auto();
}
#endif

} // namespace

std::unique_ptr<prach_buffer> ocudu::create_phy_acceleration_prach_buffer_long(unsigned max_nof_ports,
                                                                               unsigned max_nof_fd_occasions,
                                                                               bool     enable_accelerated_buffer)
{
#ifdef ENABLE_CUDA
  if (enable_accelerated_buffer && mode_uses_managed_prach_buffer()) {
    auto buffer = std::make_unique<cuda_visible_prach_buffer>(max_nof_ports,
                                                              1,
                                                              max_nof_fd_occasions,
                                                              prach_constants::LONG_SEQUENCE_MAX_NOF_SYMBOLS,
                                                              prach_constants::LONG_SEQUENCE_LENGTH);
    if (buffer->is_valid()) {
      return buffer;
    }
  }
#else
  (void)enable_accelerated_buffer;
#endif

  return create_prach_buffer_long(max_nof_ports, max_nof_fd_occasions);
}

std::unique_ptr<prach_buffer> ocudu::create_phy_acceleration_prach_buffer_short(unsigned max_nof_ports,
                                                                                unsigned max_nof_td_occasions,
                                                                                unsigned max_nof_fd_occasions,
                                                                                bool     enable_accelerated_buffer)
{
#ifdef ENABLE_CUDA
  if (enable_accelerated_buffer && mode_uses_managed_prach_buffer()) {
    auto buffer = std::make_unique<cuda_visible_prach_buffer>(max_nof_ports,
                                                              max_nof_td_occasions,
                                                              max_nof_fd_occasions,
                                                              prach_constants::SHORT_SEQUENCE_MAX_NOF_SYMBOLS,
                                                              prach_constants::SHORT_SEQUENCE_LENGTH);
    if (buffer->is_valid()) {
      return buffer;
    }
  }
#else
  (void)enable_accelerated_buffer;
#endif

  return create_prach_buffer_short(max_nof_ports, max_nof_td_occasions, max_nof_fd_occasions);
}
