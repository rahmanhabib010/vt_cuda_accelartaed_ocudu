// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "fmt/format.h"
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

namespace ocudu {

/// \brief Thread-local latency collection utility for per-slot timing measurements.
///
/// This class provides lock-free recording of per-slot latencies by maintaining
/// separate storage per thread. Use this to measure true per-slot latency from
/// process() to wait_for_completion() in multi-threaded benchmarks.
class slot_latency_collector
{
public:
  using clock_type = std::chrono::high_resolution_clock;

  /// \brief Constructs a latency collector with pre-allocated storage.
  /// \param nof_threads Number of threads that will record latencies.
  /// \param batch_size_per_thread Number of slots each thread will process.
  slot_latency_collector(unsigned nof_threads, unsigned batch_size_per_thread) :
    thread_latencies(nof_threads), nof_threads_(nof_threads), batch_size_(batch_size_per_thread)
  {
    for (auto& vec : thread_latencies) {
      vec.reserve(batch_size_per_thread);
    }
  }

  /// \brief Records a single slot latency (lock-free per thread).
  /// \param thread_id The thread recording the latency.
  /// \param latency The measured latency in nanoseconds.
  void record(unsigned thread_id, std::chrono::nanoseconds latency)
  {
    if (thread_id < thread_latencies.size()) {
      thread_latencies[thread_id].push_back(static_cast<uint64_t>(latency.count()));
    }
  }

  /// \brief Clears all measurements for the next repetition.
  void clear()
  {
    for (auto& vec : thread_latencies) {
      vec.clear();
    }
  }

  /// \brief Aggregates all thread measurements into a sorted vector.
  /// \return Sorted vector of all latencies in nanoseconds.
  std::vector<uint64_t> aggregate_and_sort() const
  {
    std::vector<uint64_t> all_latencies;
    all_latencies.reserve(nof_threads_ * batch_size_);

    for (const auto& vec : thread_latencies) {
      all_latencies.insert(all_latencies.end(), vec.begin(), vec.end());
    }

    std::sort(all_latencies.begin(), all_latencies.end());
    return all_latencies;
  }

  /// \brief Gets the latency at a given percentile.
  /// \param sorted_latencies Sorted vector of latencies.
  /// \param percentile Percentile value (0.0 to 1.0).
  /// \return Latency at the given percentile in nanoseconds.
  static uint64_t get_percentile(const std::vector<uint64_t>& sorted_latencies, double percentile)
  {
    if (sorted_latencies.empty()) {
      return 0;
    }
    size_t index = static_cast<size_t>(percentile * static_cast<double>(sorted_latencies.size() - 1));
    return sorted_latencies[index];
  }

  /// \brief Prints latency percentiles (50th, 75th, 90th, 99th, 99.9th, max).
  /// \param description Description string for the measurement.
  void print_percentiles(const std::string& description) const
  {
    auto sorted = aggregate_and_sort();
    if (sorted.empty()) {
      fmt::print("{:<50}| No latency data collected\n", description);
      return;
    }

    // Convert from nanoseconds to microseconds for display.
    double p50   = static_cast<double>(get_percentile(sorted, 0.50)) / 1000.0;
    double p75   = static_cast<double>(get_percentile(sorted, 0.75)) / 1000.0;
    double p90   = static_cast<double>(get_percentile(sorted, 0.90)) / 1000.0;
    double p99   = static_cast<double>(get_percentile(sorted, 0.99)) / 1000.0;
    double p999  = static_cast<double>(get_percentile(sorted, 0.999)) / 1000.0;
    double p_max = static_cast<double>(sorted.back()) / 1000.0;

    fmt::print("{:<50}| p50={:>7.1f}us p75={:>7.1f}us p90={:>7.1f}us p99={:>7.1f}us p99.9={:>7.1f}us max={:>7.1f}us\n",
               description,
               p50,
               p75,
               p90,
               p99,
               p999,
               p_max);
  }

  /// \brief Returns the total number of recorded latencies.
  size_t size() const
  {
    size_t total = 0;
    for (const auto& vec : thread_latencies) {
      total += vec.size();
    }
    return total;
  }

private:
  std::vector<std::vector<uint64_t>> thread_latencies;
  unsigned                           nof_threads_;
  unsigned                           batch_size_;
};

} // namespace ocudu
