// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "pdsch_grid_output_strategy.h"
#include "phy_acceleration_runtime_options.h"
#include "ocudu/support/error_handling.h"

#ifdef ENABLE_CUDA
#include "cuda/pdsch_device_grid_writer_cuda.h"
#include <atomic>
#endif

using namespace ocudu;

namespace {

#ifdef ENABLE_CUDA
std::atomic<bool> logged_pdsch_direct_grid_path_once{false};
std::atomic<bool> logged_pdsch_sidecar_grid_path_once{false};
std::atomic<bool> logged_pdsch_host_grid_path_once{false};

bool use_direct_cuda_visible_pdsch_grid()
{
  // The direct CUDA-visible grid path avoids sidecar device-grid materialization and is the OTA-stable default on the
  // GB10/B210 CUDA-visible-grid setup. Keep the environment variable as an opt-out for A/B testing and platform debug.
  return phy_acceleration_env_flag_enabled("OCUDU_PDSCH_DIRECT_DEVICE_GRID", true);
}
#endif

class host_pdsch_grid_output_strategy final : public pdsch_grid_output_strategy
{
public:
  void configure(const resource_grid_context& /*context*/, shared_resource_grid& /*grid*/) override {}

  resource_grid_writer& get_pdsch_writer(shared_resource_grid& grid) override { return grid.get_writer(); }

  void before_send_grid() override {}
};

#ifdef ENABLE_CUDA
class cuda_pdsch_grid_output_strategy final : public pdsch_grid_output_strategy
{
public:
  cuda_pdsch_grid_output_strategy(ocudulog::basic_logger& logger_, bool use_device_grid_writer_) :
    logger(logger_), use_device_grid_writer(use_device_grid_writer_)
  {
  }

  void configure(const resource_grid_context& context, shared_resource_grid& grid) override
  {
    device_grid_writer.reset();

    if (use_device_grid_writer && grid.is_valid()) {
      resource_grid_writer& writer = grid.get_writer();
      if (writer.supports_device_grid_mapping() && use_direct_cuda_visible_pdsch_grid()) {
        log_direct_path(context);
        return;
      }

      device_grid_writer = std::make_unique<pdsch_device_grid_writer_cuda>(writer);
      if (!device_grid_writer->supports_device_grid_mapping()) {
        logger.warning(context.slot.sfn(),
                       context.slot.slot_index(),
                       "Failed to allocate PDSCH device-grid writer; falling back to host PDSCH grid mapping.");
        device_grid_writer.reset();
        return;
      }

      log_sidecar_path(context, writer.supports_device_grid_mapping());
      return;
    }

    if (grid.is_valid()) {
      log_host_path(context);
    }
  }

  resource_grid_writer& get_pdsch_writer(shared_resource_grid& grid) override
  {
    if (device_grid_writer) {
      return *device_grid_writer;
    }
    return grid.get_writer();
  }

  void before_send_grid() override
  {
    if (device_grid_writer) {
      report_fatal_error_if_not(device_grid_writer->materialize_nonzero_device_grid_to_host(),
                                "Failed to materialize PDSCH device grid into host resource grid before transmission.");
    }
    device_grid_writer.reset();
  }

private:
  void log_direct_path(const resource_grid_context& context)
  {
    if (!logged_direct_grid_path) {
      logger.info(context.slot.sfn(),
                  context.slot.slot_index(),
                  "PDSCH GPU grid path: direct CUDA-visible resource-grid writer with deferred host-read sync "
                  "(OCUDU_PDSCH_DIRECT_DEVICE_GRID enabled).");
      logged_direct_grid_path = true;
    }
    if (!logged_pdsch_direct_grid_path_once.exchange(true)) {
      logger.info(context.slot.sfn(),
                  context.slot.slot_index(),
                  "PDSCH GPU grid path selected: direct CUDA-visible resource-grid writer.");
    }
  }

  void log_sidecar_path(const resource_grid_context& context, bool writer_supports_device_grid)
  {
    if (!logged_sidecar_grid_path) {
      logger.info(context.slot.sfn(),
                  context.slot.slot_index(),
                  writer_supports_device_grid
                      ? "PDSCH GPU grid path: CUDA sidecar device-grid writer over CUDA-visible resource grid."
                      : "PDSCH GPU grid path: CUDA sidecar device-grid writer.");
      logged_sidecar_grid_path = true;
    }
    if (!logged_pdsch_sidecar_grid_path_once.exchange(true)) {
      logger.info(context.slot.sfn(),
                  context.slot.slot_index(),
                  writer_supports_device_grid
                      ? "PDSCH GPU grid path selected: sidecar device-grid writer over CUDA-visible grid."
                      : "PDSCH GPU grid path selected: sidecar device-grid writer.");
    }
  }

  void log_host_path(const resource_grid_context& context)
  {
    if (!logged_host_grid_path) {
      logger.info(context.slot.sfn(), context.slot.slot_index(), "PDSCH grid path: host resource-grid writer.");
      logged_host_grid_path = true;
    }
    if (!logged_pdsch_host_grid_path_once.exchange(true)) {
      logger.info(
          context.slot.sfn(), context.slot.slot_index(), "PDSCH grid path selected: host resource-grid writer.");
    }
  }

  ocudulog::basic_logger&                           logger;
  bool                                              use_device_grid_writer;
  std::unique_ptr<pdsch_device_grid_writer_cuda> device_grid_writer;
  bool                                              logged_direct_grid_path  = false;
  bool                                              logged_sidecar_grid_path = false;
  bool                                              logged_host_grid_path    = false;
};
#endif

} // namespace

std::unique_ptr<pdsch_grid_output_strategy> ocudu::create_pdsch_grid_output_strategy(ocudulog::basic_logger& logger,
                                                                                     bool use_device_grid_writer)
{
#ifdef ENABLE_CUDA
  return std::make_unique<cuda_pdsch_grid_output_strategy>(logger, use_device_grid_writer);
#else
  (void)logger;
  (void)use_device_grid_writer;
  return std::make_unique<host_pdsch_grid_output_strategy>();
#endif
}
