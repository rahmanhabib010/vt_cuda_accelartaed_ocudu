// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ocudu/support/tracing/scoped_trace.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>

using namespace ocudu;

namespace {

using nvtx_push_fn = int (*)(const char*);
using nvtx_pop_fn  = int (*)();

bool is_env_enabled(const char* name)
{
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }

  return (std::strcmp(value, "0") != 0) && (std::strcmp(value, "false") != 0) &&
         (std::strcmp(value, "False") != 0) && (std::strcmp(value, "off") != 0) &&
         (std::strcmp(value, "OFF") != 0) && (std::strcmp(value, "no") != 0) &&
         (std::strcmp(value, "NO") != 0);
}

class nvtx_backend
{
public:
  bool enabled() const { return push != nullptr && pop != nullptr; }

  void initialize()
  {
    if (!is_env_enabled("OCUDU_NVTX_TRACE")) {
      return;
    }

    handle = ::dlopen("libnvToolsExt.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
      handle = ::dlopen("libnvToolsExt.so", RTLD_LAZY | RTLD_LOCAL);
    }
    if (handle == nullptr) {
      return;
    }

    push = reinterpret_cast<nvtx_push_fn>(::dlsym(handle, "nvtxRangePushA"));
    pop  = reinterpret_cast<nvtx_pop_fn>(::dlsym(handle, "nvtxRangePop"));
  }

  void range_push(const char* name) const
  {
    if (enabled()) {
      push(name);
    }
  }

  void range_pop() const
  {
    if (enabled()) {
      pop();
    }
  }

private:
  void*        handle = nullptr;
  nvtx_push_fn push   = nullptr;
  nvtx_pop_fn  pop    = nullptr;
};

const nvtx_backend& get_nvtx_backend()
{
  static nvtx_backend backend;
  static std::once_flag once;
  std::call_once(once, []() { backend.initialize(); });
  return backend;
}

} // namespace

bool ocudu::detail::is_nvtx_trace_enabled()
{
  return get_nvtx_backend().enabled();
}

void ocudu::detail::nvtx_trace_push(const char* name)
{
  get_nvtx_backend().range_push(name);
}

void ocudu::detail::nvtx_trace_pop()
{
  get_nvtx_backend().range_pop();
}
