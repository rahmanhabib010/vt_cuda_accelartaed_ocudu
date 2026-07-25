// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ocudu/adt/span.h"
#include "ocudu/ocudulog/logger.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/phy/upper/channel_processors/pdsch/formatters.h"
#include "ocudu/phy/upper/channel_processors/pdsch/pdsch_processor.h"
#include "ocudu/support/memory_pool/bounded_object_pool.h"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace ocudu {

/// Shared wait state used by the PDSCH processor pool when it applies backpressure instead of dropping work.
struct pdsch_processor_pool_availability_state {
  void notify_available()
  {
    // Pair with waiters so notifications cannot race between a failed pool get() and the condition-variable wait.
    {
      std::lock_guard<std::mutex> lock(mutex);
    }
    available.notify_one();
  }

  std::mutex              mutex;
  std::condition_variable available;
};

/// PDSCH processor wrapper. It appends its identifier into the free list when the processing is finished.
class pdsch_processor_wrapper : public pdsch_processor, private pdsch_processor_notifier
{
public:
  /// Pool of PDSCH processor wrappers.
  using pool = bounded_unique_object_pool<pdsch_processor_wrapper>;

  /// Creates a PDSCH processor wrapper from another PDSCH processor.
  explicit pdsch_processor_wrapper(std::unique_ptr<pdsch_processor> processor_) : processor(std::move(processor_))
  {
    ocudu_assert(processor, "Invalid PDSCH processor.");
  }

  /// Creates a PDSCH processor wrapper from another PDSCH processor wrapper.
  pdsch_processor_wrapper(pdsch_processor_wrapper&& other) noexcept :
    unique_pool_ptr(std::move(other.unique_pool_ptr)),
    notifier(other.notifier),
    availability_state(std::move(other.availability_state)),
    processor(std::move(other.processor))
  {
    other.notifier = nullptr;
  }

  // See pdsch_processor interface for documentation.
  void process(resource_grid_writer&                                                            grid,
               pdsch_processor_notifier&                                                        notifier_,
               static_vector<shared_transport_block, pdsch_processor::MAX_NOF_TRANSPORT_BLOCKS> data,
               const pdsch_processor::pdu_t&                                                    pdu) override
  {
    // Save original notifier.
    notifier = &notifier_;

    // Process.
    processor->process(grid, *this, std::move(data), pdu);
  }

  /// Sets the unique pointer that will be released upon the completion of the PDSCH processing.
  void set_unique_ptr(pool::ptr unique_token_) { unique_pool_ptr = std::move(unique_token_); }

  /// Sets the availability state to notify when the wrapped processor is released back to the pool.
  void set_availability_state(std::shared_ptr<pdsch_processor_pool_availability_state> availability_state_)
  {
    availability_state = std::move(availability_state_);
  }

private:
  // See pdsch_processor_notifier for documentation.
  void on_finish_processing() override
  {
    ocudu_assert(notifier != nullptr, "Invalid notifier.");

    // Notify the completion of the processing.
    notifier->on_finish_processing();

    // Return the PDSCH processor identifier to the free list and wake any caller waiting for backpressure.
    unique_pool_ptr = nullptr;
    if (availability_state) {
      availability_state->notify_available();
    }
  }

  /// Processor identifier within the pool.
  pool::ptr unique_pool_ptr;
  /// Current PDSCH processor notifier.
  pdsch_processor_notifier* notifier = nullptr;
  /// Availability state to notify when this processor is returned to the pool.
  std::shared_ptr<pdsch_processor_pool_availability_state> availability_state;
  /// Wrapped PDSCH processor.
  std::unique_ptr<pdsch_processor> processor;
};

inline span<std::unique_ptr<pdsch_processor_wrapper>> attach_pdsch_processor_pool_availability_state(
    span<std::unique_ptr<pdsch_processor_wrapper>>                  processors,
    const std::shared_ptr<pdsch_processor_pool_availability_state>& availability_state)
{
  for (std::unique_ptr<pdsch_processor_wrapper>& processor : processors) {
    processor->set_availability_state(availability_state);
  }
  return processors;
}

/// \brief Asynchronous PDSCH processor pool.
///
/// It contains PDSCH processors that are asynchronously executed. The processing of a PDSCH transmission is dropped if
/// there are no free PDSCH processors available, unless backpressure mode is enabled.
class pdsch_processor_pool : public pdsch_processor
{
public:
  explicit pdsch_processor_pool(span<std::unique_ptr<pdsch_processor_wrapper>> processors_,
                                bool                                           block_on_exhaustion_ = false) :
    logger(ocudulog::fetch_basic_logger("PHY")),
    availability_state(std::make_shared<pdsch_processor_pool_availability_state>()),
    pool(attach_pdsch_processor_pool_availability_state(processors_, availability_state)),
    block_on_exhaustion(block_on_exhaustion_)
  {
  }

  void process(resource_grid_writer&                                           grid,
               pdsch_processor_notifier&                                       notifier,
               static_vector<shared_transport_block, MAX_NOF_TRANSPORT_BLOCKS> data,
               const pdu_t&                                                    pdu) override
  {
    // Get a processor from the pool.
    auto unique_processor = pool.get();

    // If no processor is available, either preserve legacy drop behavior or apply backpressure.
    if (!unique_processor) {
      if (!block_on_exhaustion) {
        logger.warning(pdu.slot.sfn(),
                       pdu.slot.slot_index(),
                       "Insufficient number of PDSCH processors. Dropping PDSCH {:s}.",
                       pdu);
        notifier.on_finish_processing();
        return;
      }
      unique_processor = wait_for_available_processor(pdu);
    }

    // Save reference to the processor.
    pdsch_processor& processor = *unique_processor;

    // Set unique pointer, it will be returned to the pool when the processing is completed.
    unique_processor->set_unique_ptr(std::move(unique_processor));

    // Process PDSCH.
    processor.process(grid, notifier, std::move(data), pdu);
  }

private:
  pdsch_processor_wrapper::pool::ptr wait_for_available_processor(const pdu_t& pdu)
  {
    std::unique_lock<std::mutex> lock(availability_state->mutex);

    if (!saturation_warning_logged) {
      logger.warning(pdu.slot.sfn(),
                     pdu.slot.slot_index(),
                     "PDSCH processor pool exhausted ({} processors); blocking until a processor is released. "
                     "Increase pdsch_acceleration_nof_lanes and max_pdsch_concurrency if this warning appears during OTA "
                     "traffic. PDSCH {:s}.",
                     pool.capacity(),
                     pdu);
      saturation_warning_logged = true;
    } else if (logger.debug.enabled()) {
      logger.debug(pdu.slot.sfn(),
                   pdu.slot.slot_index(),
                   "PDSCH processor pool exhausted; waiting for a free processor. PDSCH {:s}.",
                   pdu);
    }

    pdsch_processor_wrapper::pool::ptr unique_processor = pool.get();
    while (!unique_processor) {
      availability_state->available.wait(lock);
      unique_processor = pool.get();
    }
    return unique_processor;
  }

  ocudulog::basic_logger&                                  logger;
  std::shared_ptr<pdsch_processor_pool_availability_state> availability_state;
  pdsch_processor_wrapper::pool                            pool;
  bool                                                     block_on_exhaustion       = false;
  bool                                                     saturation_warning_logged = false;
};

} // namespace ocudu
