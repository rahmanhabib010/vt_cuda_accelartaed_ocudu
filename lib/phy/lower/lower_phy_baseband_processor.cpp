// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "lower_phy_baseband_processor.h"
#include "ocudu/adt/interval.h"
#include "ocudu/instrumentation/traces/ru_traces.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/ran/slot_point_extended.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

using namespace ocudu;

static unsigned get_timing_warn_threshold_us()
{
  static const unsigned threshold_us = []() {
    const char* value = std::getenv("OCUDU_LOWER_PHY_TIMING_WARN_US");
    if ((value == nullptr) || (*value == '\0')) {
      return 0U;
    }

    char*         end    = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value) {
      return 0U;
    }

    return static_cast<unsigned>(std::min<unsigned long>(parsed, 1000000UL));
  }();

  return threshold_us;
}

static unsigned get_timing_warn_max_logs()
{
  static const unsigned max_logs = []() {
    static constexpr unsigned default_max_logs = 256;

    const char* value = std::getenv("OCUDU_LOWER_PHY_TIMING_WARN_MAX_LOGS");
    if ((value == nullptr) || (*value == '\0')) {
      return default_max_logs;
    }

    char*         end    = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value) {
      return default_max_logs;
    }

    return static_cast<unsigned>(std::min<unsigned long>(parsed, 1000000UL));
  }();

  return max_logs;
}

static unsigned get_timing_warn_max_logs_per_phase()
{
  static const unsigned max_logs = []() {
    const char* value = std::getenv("OCUDU_LOWER_PHY_TIMING_WARN_MAX_LOGS_PER_PHASE");
    if ((value == nullptr) || (*value == '\0')) {
      return get_timing_warn_max_logs();
    }

    char*         end    = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value) {
      return get_timing_warn_max_logs();
    }

    return static_cast<unsigned>(std::min<unsigned long>(parsed, 1000000UL));
  }();

  return max_logs;
}

static unsigned get_effective_timing_warn_threshold_us(const char* phase)
{
  const unsigned threshold_us = get_timing_warn_threshold_us();
  if (threshold_us == 0) {
    return 0;
  }

  // The pacing loop intentionally waits in short increments when TX is running ahead of RX. Logging every normal
  // sub-millisecond wait hides the actual slow baseband phases that cause OTA drops.
  if (std::strcmp(phase, "DL pacing wait") == 0) {
    return std::max(threshold_us, 1500U);
  }

  // Receive is a blocking radio call. Keep this available for pathological stalls without letting idle/startup waits
  // consume the logging budget.
  if (std::strcmp(phase, "UL radio receive") == 0) {
    return std::max(threshold_us, 200000U);
  }

  return threshold_us;
}

static std::atomic<unsigned>& get_timing_warn_counter(const char* phase)
{
  static std::atomic<unsigned> dl_pacing_wait{0};
  static std::atomic<unsigned> dl_system_time_throttle{0};
  static std::atomic<unsigned> dl_baseband_processing{0};
  static std::atomic<unsigned> dl_radio_transmit{0};
  static std::atomic<unsigned> ul_radio_receive{0};
  static std::atomic<unsigned> ul_baseband_processing{0};
  static std::atomic<unsigned> other{0};

  if (std::strcmp(phase, "DL pacing wait") == 0) {
    return dl_pacing_wait;
  }
  if (std::strcmp(phase, "DL system-time throttle") == 0) {
    return dl_system_time_throttle;
  }
  if (std::strcmp(phase, "DL baseband processing") == 0) {
    return dl_baseband_processing;
  }
  if (std::strcmp(phase, "DL radio transmit") == 0) {
    return dl_radio_transmit;
  }
  if (std::strcmp(phase, "UL radio receive") == 0) {
    return ul_radio_receive;
  }
  if (std::strcmp(phase, "UL baseband processing") == 0) {
    return ul_baseband_processing;
  }

  return other;
}

static void log_lower_phy_phase_if_slow(const char*                         phase,
                                        std::chrono::steady_clock::duration duration,
                                        baseband_gateway_timestamp          timestamp)
{
  const unsigned threshold_us = get_effective_timing_warn_threshold_us(phase);
  if (threshold_us == 0) {
    return;
  }

  const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
  if (duration_us.count() <= static_cast<std::chrono::microseconds::rep>(threshold_us)) {
    return;
  }

  std::atomic<unsigned>& nof_logs  = get_timing_warn_counter(phase);
  const unsigned         log_index = nof_logs.fetch_add(1, std::memory_order_relaxed);
  if (log_index >= get_timing_warn_max_logs_per_phase()) {
    return;
  }

  ocudulog::fetch_basic_logger("PHY").warning(
      "Lower-PHY timing: {} took {} us at baseband timestamp {}.", phase, duration_us.count(), timestamp);
}

static std::chrono::microseconds get_tx_pacing_guard()
{
  static const std::chrono::microseconds guard = []() {
    static constexpr unsigned default_guard_us = 250;
    static constexpr unsigned max_guard_us     = 5000;

    const char* value = std::getenv("OCUDU_LOWER_PHY_TX_PACING_GUARD_US");
    if ((value == nullptr) || (*value == '\0')) {
      return std::chrono::microseconds(default_guard_us);
    }

    char*         end    = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value) {
      return std::chrono::microseconds(default_guard_us);
    }

    return std::chrono::microseconds(
        static_cast<std::chrono::microseconds::rep>(std::min<unsigned long>(parsed, max_guard_us)));
  }();

  return guard;
}

static std::chrono::microseconds samples_to_microseconds(baseband_gateway_timestamp nof_samples, sampling_rate srate)
{
  const uint64_t srate_hz = srate.to_Hz<uint64_t>();
  return std::chrono::microseconds(divide_ceil(nof_samples * 1000000UL, srate_hz));
}

static baseband_gateway_timestamp microseconds_to_samples(std::chrono::microseconds duration, sampling_rate srate)
{
  const uint64_t srate_hz = srate.to_Hz<uint64_t>();
  return divide_ceil(static_cast<uint64_t>(duration.count()) * srate_hz, 1000000UL);
}

lower_phy_baseband_processor::lower_phy_baseband_processor(const lower_phy_baseband_processor_configuration& config,
                                                           const lower_phy_baseband_processor_dependencies&  deps) :
  srate(config.srate),
  nof_samples_in_all_hyper_frames(config.srate.to_kHz() * NOF_HYPER_SFNS * NOF_SFNS * NOF_SUBFRAMES_PER_FRAME),
  rx_buffer_size(config.rx_buffer_size),
  slot_duration(1000 / pow2(to_numerology_value(config.scs))),
  system_time_throttling_ratio(config.system_time_throttling),
  rx_executor(deps.rx_task_executor),
  tx_executor(deps.tx_task_executor),
  uplink_executor(deps.ul_task_executor),
  receiver(deps.receiver),
  transmitter(deps.transmitter),
  uplink_processor(deps.ul_bb_proc),
  downlink_processor(deps.dl_bb_proc),
  rx_buffers(config.nof_rx_buffers),
  tx_time_offset(config.tx_time_offset),
  rx_to_tx_max_delay(config.rx_to_tx_max_delay),
  tx_state(config.stop_nof_slots),
  rx_state(config.stop_nof_slots)
{
  static constexpr interval<float> system_time_throttling_range(0, 20);

  ocudu_assert(rx_buffer_size, "Invalid buffer size.");
  ocudu_assert(system_time_throttling_range.contains(config.system_time_throttling),
               "System time throttling (i.e., {}) is out of the range {}.",
               config.system_time_throttling,
               system_time_throttling_range);
  ocudu_assert(config.nof_rx_ports != 0, "Invalid number of receive ports.");
  ocudu_assert(config.nof_tx_ports != 0, "Invalid number of transmit ports.");

  // Create queue of receive buffers.
  while (!rx_buffers.full()) {
    rx_buffers.push_blocking(std::make_unique<baseband_gateway_buffer_dynamic>(config.nof_rx_ports, rx_buffer_size));
  }
}

void lower_phy_baseband_processor::start(baseband_gateway_timestamp init_time, baseband_gateway_timestamp sfn0_ref_time)
{
  // If it is required to start with system frame number 0, then set a time offset to start an SFN earlier.
  start_time_sfn0   = sfn0_ref_time;
  last_rx_timestamp = init_time;

  rx_state.start();
  report_fatal_error_if_not(rx_executor.defer([this]() { ul_process(); }), "Failed to execute initial uplink task.");

  tx_state.start();
  report_fatal_error_if_not(tx_executor.defer([this, init_time]() { dl_process(init_time + rx_to_tx_max_delay); }),
                            "Failed to execute initial downlink task.");
}

void lower_phy_baseband_processor::stop()
{
  rx_state.request_stop();
  tx_state.request_stop();
  rx_state.wait_stop();
  tx_state.wait_stop();
}

void lower_phy_baseband_processor::dl_process(baseband_gateway_timestamp timestamp)
{
  // Check if it is running, notify stop and return without enqueueing more tasks.
  if (!tx_state.on_process()) {
    return;
  }

  const bool timing_warnings_enabled = get_timing_warn_threshold_us() != 0;
  auto       phase_start             = std::chrono::steady_clock::time_point{};

  // Throttling mechanism to keep a maximum latency of one millisecond in the transmit buffer based on the latest
  // received timestamp.
  if (timing_warnings_enabled) {
    phase_start = std::chrono::steady_clock::now();
  }
  {
    const std::chrono::microseconds            tx_pacing_guard = get_tx_pacing_guard();
    static constexpr std::chrono::microseconds sleep_threshold{100};
    static constexpr std::chrono::microseconds max_sleep_step{10};
    static constexpr std::chrono::microseconds yield_threshold{20};
    const baseband_gateway_timestamp tx_pacing_guard_samples = microseconds_to_samples(tx_pacing_guard, srate);

    // Calculate maximum waiting time to avoid deadlock.
    std::chrono::microseconds timeout_duration = 2 * slot_duration;
    // Maximum time point to wait for.
    std::chrono::time_point<std::chrono::steady_clock> wait_until_tp =
        std::chrono::steady_clock::now() + timeout_duration;
    // Wait until one of these conditions is met:
    // - The reception timestamp reaches the desired value;
    // - The system time reaches the maximum waiting time; or
    // - The lower PHY was stopped.
    for (;;) {
      const baseband_gateway_timestamp latest_allowed_tx_timestamp =
          last_rx_timestamp.load(std::memory_order_acquire) + rx_to_tx_max_delay + tx_pacing_guard_samples;
      if (timestamp <= latest_allowed_tx_timestamp) {
        break;
      }

      const auto now = std::chrono::steady_clock::now();
      if (now >= wait_until_tp) {
        break;
      }

      const std::chrono::microseconds tx_ahead_time =
          samples_to_microseconds(timestamp - latest_allowed_tx_timestamp, srate);
      const std::chrono::microseconds time_to_timeout =
          std::chrono::duration_cast<std::chrono::microseconds>(wait_until_tp - now);

      if (tx_ahead_time > sleep_threshold) {
        // Keep sleeps short. Long sleeps reduce CPU burn but can overshoot the remaining radio timing margin.
        const std::chrono::microseconds sleep_for =
            std::min({time_to_timeout, max_sleep_step, tx_ahead_time - yield_threshold});
        std::this_thread::sleep_for(sleep_for);
      } else {
        std::this_thread::yield();
      }
    }
  }
  if (timing_warnings_enabled) {
    log_lower_phy_phase_if_slow("DL pacing wait", std::chrono::steady_clock::now() - phase_start, timestamp);
  }

  // Throttling mechanism to slow down the baseband processing.
  if (timing_warnings_enabled) {
    phase_start = std::chrono::steady_clock::now();
  }
  if ((system_time_throttling_ratio > 0.0) && (last_tx_time.has_value()) && (last_tx_buffer_size != 0)) {
    // Get current time and calculate the elapsed time since the last call.
    std::chrono::time_point<std::chrono::high_resolution_clock> now     = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds                                    elapsed = now - *last_tx_time;

    // Calculate the number of samples from the previous transmission to the next one and convert it seconds.
    float expected_elapsed_s = static_cast<double>(last_tx_buffer_size) / srate.to_Hz<float>();

    // Calculate the minimum elapsed time required to satisfy the throttling time.
    std::chrono::nanoseconds minimum_elapsed(
        static_cast<uint64_t>(expected_elapsed_s * 1e9 * system_time_throttling_ratio));

    if (elapsed < minimum_elapsed) {
      std::this_thread::sleep_until(*last_tx_time + minimum_elapsed);
    }
  }
  last_tx_time.emplace(std::chrono::high_resolution_clock::now());
  if (timing_warnings_enabled) {
    log_lower_phy_phase_if_slow("DL system-time throttle", std::chrono::steady_clock::now() - phase_start, timestamp);
  }

  // Process downlink buffer.
  if (timing_warnings_enabled) {
    phase_start = std::chrono::steady_clock::now();
  }
  downlink_processor_baseband::processing_result result =
      downlink_processor.process(apply_timestamp_sfn0_ref(timestamp));
  if (timing_warnings_enabled) {
    log_lower_phy_phase_if_slow("DL baseband processing", std::chrono::steady_clock::now() - phase_start, timestamp);
  }
  ocudu_assert(result.buffer, "The buffer must be valid.");

  // Set transmission timestamp.
  result.metadata.ts = timestamp + tx_time_offset;

  // Enqueue transmission.
  trace_point tx_tp = ru_tracer.now();

  // Transmit buffer.
  if (timing_warnings_enabled) {
    phase_start = std::chrono::steady_clock::now();
  }
  transmitter.transmit(result.buffer->get_reader(), result.metadata);
  if (timing_warnings_enabled) {
    log_lower_phy_phase_if_slow("DL radio transmit", std::chrono::steady_clock::now() - phase_start, timestamp);
  }

  ru_tracer << trace_event("transmit_baseband", tx_tp);

  // Update last buffer size.
  last_tx_buffer_size = result.buffer->get_nof_samples();

  // Enqueue DL process task.
  report_fatal_error_if_not(
      tx_executor.defer([this, new_timestamp = timestamp + last_tx_buffer_size]() { dl_process(new_timestamp); }),
      "Failed to execute downlink processing task");
}

void lower_phy_baseband_processor::ul_process()
{
  // Check if it is running, notify stop and return without enqueueing more tasks.
  if (!rx_state.on_process()) {
    return;
  }

  // Get receive buffer.
  std::unique_ptr<baseband_gateway_buffer_dynamic> rx_buffer = rx_buffers.pop_blocking();

  // Receive baseband.
  const bool rx_timing_warnings_enabled = get_timing_warn_threshold_us() != 0;
  auto       rx_phase_start             = std::chrono::steady_clock::time_point{};
  if (rx_timing_warnings_enabled) {
    rx_phase_start = std::chrono::steady_clock::now();
  }
  trace_point                         tp          = ru_tracer.now();
  baseband_gateway_receiver::metadata rx_metadata = receiver.receive(rx_buffer->get_writer());
  if (rx_timing_warnings_enabled) {
    log_lower_phy_phase_if_slow("UL radio receive", std::chrono::steady_clock::now() - rx_phase_start, rx_metadata.ts);
  }
  ru_tracer << trace_event("receive_baseband", tp);

  // Update last timestamp.
  last_rx_timestamp.store(rx_metadata.ts + rx_buffer->get_nof_samples(), std::memory_order_release);

  // Queue uplink buffer processing.
  report_fatal_error_if_not(uplink_executor.defer([this, ul_buffer = std::move(rx_buffer), rx_metadata]() mutable {
    trace_point ul_tp = ru_tracer.now();

    // Process UL.
    const bool ul_timing_warnings_enabled = get_timing_warn_threshold_us() != 0;
    auto       ul_phase_start             = std::chrono::steady_clock::time_point{};
    if (ul_timing_warnings_enabled) {
      ul_phase_start = std::chrono::steady_clock::now();
    }
    uplink_processor.process(ul_buffer->get_reader(), apply_timestamp_sfn0_ref(rx_metadata.ts));
    if (ul_timing_warnings_enabled) {
      log_lower_phy_phase_if_slow(
          "UL baseband processing", std::chrono::steady_clock::now() - ul_phase_start, rx_metadata.ts);
    }

    // Return buffer to receive.
    rx_buffers.push_blocking(std::move(ul_buffer));

    ru_tracer << trace_event("uplink_baseband", ul_tp);
  }),
                            "Failed to execute uplink processing task.");

  // Enqueue next iteration if it is running.
  report_fatal_error_if_not(rx_executor.defer([this]() { ul_process(); }), "Failed to execute receive task.");
}
