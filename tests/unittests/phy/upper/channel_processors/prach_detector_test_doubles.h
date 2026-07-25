// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ocudu/phy/upper/channel_processors/prach/prach_detector.h"
#include <optional>

namespace ocudu {

class prach_detector_spy : public prach_detector
{
  bool                         detect_method_been_called = false;
  std::optional<configuration> last_config;

public:
  prach_detection_result detect(const prach_buffer& input, const configuration& config) override
  {
    detect_method_been_called = true;
    last_config               = config;

    return {};
  }

  bool has_detect_method_been_called() const { return detect_method_been_called; }

  const std::optional<configuration>& get_last_config() const { return last_config; }
};

} // namespace ocudu
