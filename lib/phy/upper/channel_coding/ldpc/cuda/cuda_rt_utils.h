// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <sched.h>

namespace ocudu {

enum class upper_phy_cuda_stream_priority_mode { normal, high, low };

inline upper_phy_cuda_stream_priority_mode parse_upper_phy_cuda_stream_priority(const char* value)
{
  if ((value == nullptr) || (*value == '\0')) {
    return upper_phy_cuda_stream_priority_mode::normal;
  }

  if ((std::strcmp(value, "1") == 0) || (std::strcmp(value, "true") == 0) || (std::strcmp(value, "on") == 0) ||
      (std::strcmp(value, "yes") == 0) || (std::strcmp(value, "high") == 0) || (std::strcmp(value, "realtime") == 0)) {
    return upper_phy_cuda_stream_priority_mode::high;
  }

  if ((std::strcmp(value, "low") == 0) || (std::strcmp(value, "lowest") == 0) ||
      (std::strcmp(value, "background") == 0)) {
    return upper_phy_cuda_stream_priority_mode::low;
  }

  return upper_phy_cuda_stream_priority_mode::normal;
}

inline upper_phy_cuda_stream_priority_mode get_upper_phy_cuda_stream_priority()
{
  const char* value = std::getenv("OCUDU_UPPER_PHY_CUDA_STREAM_PRIORITY");
  if ((value != nullptr) && (*value != '\0')) {
    return parse_upper_phy_cuda_stream_priority(value);
  }

  value = std::getenv("OCUDU_UPPER_PHY_CUDA_HIGH_PRIORITY");
  if ((value == nullptr) || (*value == '\0')) {
    return upper_phy_cuda_stream_priority_mode::normal;
  }

  return parse_upper_phy_cuda_stream_priority(value);
}

inline cudaError_t cudaStreamCreateUpperPhy(cudaStream_t* stream)
{
  const upper_phy_cuda_stream_priority_mode priority_mode = get_upper_phy_cuda_stream_priority();
  if (priority_mode == upper_phy_cuda_stream_priority_mode::normal) {
    return cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
  }

  int         least_priority    = 0;
  int         greatest_priority = 0;
  cudaError_t status            = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
  if (status == cudaSuccess) {
    const int requested_priority =
        (priority_mode == upper_phy_cuda_stream_priority_mode::high) ? greatest_priority : least_priority;
    status = cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking, requested_priority);
    if (status == cudaSuccess) {
      return cudaSuccess;
    }
    (void)cudaGetLastError();
  }

  return cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
}

/// \brief RT-friendly CUDA stream synchronization.
///
/// This function synchronizes on a CUDA stream while yielding the CPU to other threads.
/// When the calling thread has real-time (SCHED_FIFO/SCHED_RR) priority, a blocking
/// cudaStreamSynchronize() can cause priority inversion by preventing GPU driver threads
/// (which run at normal priority) from making progress.
///
/// This function uses cudaStreamQuery() in a loop with sched_yield() to allow other
/// threads to run while waiting for the GPU to complete.
///
/// \param stream The CUDA stream to synchronize on.
/// \return cudaSuccess on success, or an error code if the stream had an error.
inline cudaError_t cudaStreamSynchronizeYielding(cudaStream_t stream)
{
  cudaError_t err;
  while ((err = cudaStreamQuery(stream)) == cudaErrorNotReady) {
    sched_yield();
  }
  return err;
}

/// \brief RT-friendly CUDA event synchronization.
///
/// This function synchronizes on a CUDA event while yielding the CPU to other threads.
/// Similar to cudaStreamSynchronizeYielding, this avoids priority inversion by polling
/// the event status and yielding between checks.
///
/// \param event The CUDA event to synchronize on.
/// \return cudaSuccess on success, or an error code if the event had an error.
inline cudaError_t cudaEventSynchronizeYielding(cudaEvent_t event)
{
  cudaError_t err;
  while ((err = cudaEventQuery(event)) == cudaErrorNotReady) {
    sched_yield();
  }
  return err;
}

} // namespace ocudu
