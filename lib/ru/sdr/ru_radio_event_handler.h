// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/radio/radio_event_notifier.h"
#include <array>
#include <chrono>
#include <mutex>

namespace ocudu {

/// Radio Unit radio event logger.
class ru_radio_logger_event_handler : public radio_event_notifier
{
public:
  explicit ru_radio_logger_event_handler(ocudulog::basic_logger& logger_) : logger(logger_) {}

  // See interface for documentation.
  void on_radio_rt_event(const event_description& description) override
  {
    if (description.type == radio_event_type::LATE || description.type == radio_event_type::UNDERFLOW ||
        description.type == radio_event_type::OVERFLOW) {
      unsigned nof_suppressed_events = 0;
      if (should_log_rt_event(description.type, nof_suppressed_events)) {
        if (nof_suppressed_events == 0) {
          logger.warning("Real-time failure in RF: {}", to_string(description.type));
        } else {
          logger.warning("Real-time failure in RF: {} ({} similar events suppressed)",
                         to_string(description.type),
                         nof_suppressed_events);
        }
      }
    }

    static const auto& log_format_debug = "Real-time failure in RF: Type={} Source={} Timestamp={}";

    if (description.timestamp.has_value()) {
      logger.debug(
          log_format_debug, to_string(description.type), to_string(description.source), *description.timestamp);
    } else {
      logger.debug(log_format_debug, to_string(description.type), to_string(description.source), "na");
    }
  }

private:
  struct rt_event_log_state {
    std::chrono::steady_clock::time_point last_log_time{};
    unsigned                              nof_suppressed_events = 0;
  };

  static std::optional<size_t> get_rt_event_index(radio_event_type type)
  {
    switch (type) {
      case radio_event_type::LATE:
        return 0;
      case radio_event_type::UNDERFLOW:
        return 1;
      case radio_event_type::OVERFLOW:
        return 2;
      case radio_event_type::UNDEFINED:
      case radio_event_type::START_OF_BURST:
      case radio_event_type::END_OF_BURST:
      case radio_event_type::OTHER:
        return std::nullopt;
    }
    return std::nullopt;
  }

  bool should_log_rt_event(radio_event_type type, unsigned& nof_suppressed_events)
  {
    static constexpr std::chrono::milliseconds min_log_interval{100};

    std::optional<size_t> index = get_rt_event_index(type);
    if (!index.has_value()) {
      return false;
    }

    const auto now = std::chrono::steady_clock::now();

    std::scoped_lock    lock(rt_event_log_mutex);
    rt_event_log_state& state = rt_event_log_states[*index];
    if ((state.last_log_time == std::chrono::steady_clock::time_point{}) ||
        (now - state.last_log_time >= min_log_interval)) {
      nof_suppressed_events       = state.nof_suppressed_events;
      state.nof_suppressed_events = 0;
      state.last_log_time         = now;
      return true;
    }

    ++state.nof_suppressed_events;
    return false;
  }

  ocudulog::basic_logger&           logger;
  std::mutex                        rt_event_log_mutex;
  std::array<rt_event_log_state, 3> rt_event_log_states;
};

/// Radio event dispatcher.
class ru_radio_event_dispatcher : public radio_event_notifier
{
  std::vector<radio_event_notifier*> handlers;

public:
  explicit ru_radio_event_dispatcher(std::vector<radio_event_notifier*> handlers_) : handlers(std::move(handlers_))
  {
    ocudu_assert(!handlers.empty(), "Empty list of radio event notifiers");
    ocudu_assert(std::all_of(handlers.begin(), handlers.end(), [](auto* handler) { return handler != nullptr; }),
                 "Invalid radio event notifier");
  }

  // See interface for documentation.
  void on_radio_rt_event(const event_description& description) override
  {
    for (auto* handler : handlers) {
      handler->on_radio_rt_event(description);
    }
  }
};

} // namespace ocudu
