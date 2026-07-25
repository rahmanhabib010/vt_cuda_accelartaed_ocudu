// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/support/tracing/event_tracing.h"

namespace ocudu {

namespace detail {

/// Emits an NVTX range when runtime NVTX tracing is enabled.
///
/// The implementation deliberately lives in ocudu_support and resolves NVTX dynamically. This keeps CUDA/NVTX headers
/// and link dependencies out of generic PHY code.
void nvtx_trace_push(const char* name);

/// Closes an NVTX range opened with \ref nvtx_trace_push.
void nvtx_trace_pop();

/// Returns true when the runtime NVTX backend is enabled and available.
bool is_nvtx_trace_enabled();

} // namespace detail

/// RAII helper for short host-side timing scopes.
///
/// The domain code should use implementation-neutral names such as "L1.UL.PUSCH.demodulate". When the compile-time
/// OCUDU event tracer is disabled, the file-trace side compiles down to a disabled tracer path. NVTX is opt-in at
/// runtime via OCUDU_NVTX_TRACE=1.
template <typename Tracer>
class scoped_trace
{
public:
  scoped_trace(Tracer& tracer_, const char* name_) : tracer(tracer_), name(name_), start_tp(tracer.now())
  {
    if (detail::is_nvtx_trace_enabled()) {
      nvtx_active = true;
      detail::nvtx_trace_push(name);
    }
  }

  scoped_trace(const scoped_trace&)            = delete;
  scoped_trace& operator=(const scoped_trace&) = delete;
  scoped_trace(scoped_trace&&)                 = delete;
  scoped_trace& operator=(scoped_trace&&)      = delete;

  ~scoped_trace()
  {
    if (tracer.is_enabled()) {
      tracer << trace_event(name, start_tp);
    }
    if (nvtx_active) {
      detail::nvtx_trace_pop();
    }
  }

private:
  Tracer&     tracer;
  const char* name;
  trace_point start_tp;
  bool        nvtx_active = false;
};

template <typename Tracer>
scoped_trace<Tracer> make_scoped_trace(Tracer& tracer, const char* name)
{
  return scoped_trace<Tracer>(tracer, name);
}

} // namespace ocudu
