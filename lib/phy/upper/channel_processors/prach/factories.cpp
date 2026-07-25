// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "ocudu/phy/upper/channel_processors/prach/factories.h"
#include "prach_detector_generic_impl.h"
#include "prach_detector_pool.h"
#include "prach_generator_impl.h"
#include "ocudu/phy/support/support_formatters.h"
#include "ocudu/phy/upper/channel_processors/prach/formatters.h"
#include "ocudu/support/error_handling.h"
#ifdef ENABLE_CUDA
#include "prach_detector_cuda_impl.h"
#endif
#include <cstdlib>
#include <cstring>

using namespace ocudu;

namespace {

#ifdef ENABLE_CUDA
bool is_disabled_value(const char* value)
{
  return (std::strcmp(value, "0") == 0) || (std::strcmp(value, "false") == 0) || (std::strcmp(value, "FALSE") == 0) ||
         (std::strcmp(value, "off") == 0) || (std::strcmp(value, "OFF") == 0) || (std::strcmp(value, "disable") == 0) ||
         (std::strcmp(value, "DISABLE") == 0) || (std::strcmp(value, "disabled") == 0);
}

bool is_enabled_value(const char* value)
{
  return (std::strcmp(value, "1") == 0) || (std::strcmp(value, "true") == 0) || (std::strcmp(value, "TRUE") == 0) ||
         (std::strcmp(value, "on") == 0) || (std::strcmp(value, "ON") == 0) || (std::strcmp(value, "enable") == 0) ||
         (std::strcmp(value, "ENABLE") == 0) || (std::strcmp(value, "enabled") == 0);
}

std::string resolve_prach_acceleration_mode(const std::string& configured_mode)
{
  if (configured_mode != "auto") {
    return configured_mode;
  }
  const char* env_value = std::getenv("OCUDU_PRACH_ACCELERATION");
  if (env_value == nullptr) {
    return configured_mode;
  }
  if (is_disabled_value(env_value)) {
    return "disabled";
  }
  if (is_enabled_value(env_value)) {
    return "enabled";
  }
  if ((std::strcmp(env_value, "auto") == 0) || (std::strcmp(env_value, "AUTO") == 0)) {
    return "auto";
  }
  return configured_mode;
}
#else
std::string resolve_prach_acceleration_mode(const std::string& configured_mode)
{
  return configured_mode;
}
#endif

class prach_detector_factory_sw : public prach_detector_factory
{
private:
  std::shared_ptr<dft_processor_factory>   dft_factory;
  std::shared_ptr<prach_generator_factory> prach_gen_factory;
  unsigned                                 idft_long_size;
  unsigned                                 idft_short_size;

public:
  prach_detector_factory_sw(std::shared_ptr<dft_processor_factory>         dft_factory_,
                            std::shared_ptr<prach_generator_factory>       prach_gen_factory_,
                            const prach_detector_factory_sw_configuration& config) :
    dft_factory(std::move(dft_factory_)),
    prach_gen_factory(std::move(prach_gen_factory_)),
    idft_long_size(config.idft_long_size),
    idft_short_size(config.idft_short_size)
  {
    ocudu_assert(dft_factory, "Invalid DFT factory.");
    ocudu_assert(prach_gen_factory, "Invalid PRACH generator factory.");
  }

  std::unique_ptr<prach_detector> create() override
  {
    dft_processor::configuration idft_long_config;
    idft_long_config.size = idft_long_size;
    idft_long_config.dir  = dft_processor::direction::INVERSE;
    dft_processor::configuration idft_short_config;
    idft_short_config.size = idft_short_size;
    idft_short_config.dir  = dft_processor::direction::INVERSE;
    return std::make_unique<prach_detector_generic_impl>(
        dft_factory->create(idft_long_config), dft_factory->create(idft_short_config), prach_gen_factory->create());
  }

  std::unique_ptr<prach_detector_validator> create_validator() override
  {
    return std::make_unique<prach_detector_validator_impl>();
  }
};

#ifdef ENABLE_CUDA
class prach_detector_factory_cuda : public prach_detector_factory
{
public:
  prach_detector_factory_cuda(std::shared_ptr<dft_processor_factory>         dft_factory_,
                              std::shared_ptr<prach_generator_factory>       prach_gen_factory_,
                              const prach_detector_factory_sw_configuration& config_,
                              bool                                           force_gpu_path_) :
    dft_factory(std::move(dft_factory_)),
    prach_gen_factory(std::move(prach_gen_factory_)),
    config(config_),
    force_gpu_path(force_gpu_path_)
  {
    ocudu_assert(dft_factory, "Invalid DFT factory.");
    ocudu_assert(prach_gen_factory, "Invalid PRACH generator factory.");
  }

  std::unique_ptr<prach_detector> create() override
  {
    dft_processor::configuration idft_long_config;
    idft_long_config.size = config.idft_long_size;
    idft_long_config.dir  = dft_processor::direction::INVERSE;
    dft_processor::configuration idft_short_config;
    idft_short_config.size = config.idft_short_size;
    idft_short_config.dir  = dft_processor::direction::INVERSE;
    auto fallback          = std::make_unique<prach_detector_generic_impl>(
        dft_factory->create(idft_long_config), dft_factory->create(idft_short_config), prach_gen_factory->create());
    return std::make_unique<prach_detector_cuda_impl>(prach_gen_factory->create(), std::move(fallback), force_gpu_path);
  }

  std::unique_ptr<prach_detector_validator> create_validator() override
  {
    return std::make_unique<prach_detector_validator_impl>();
  }

private:
  std::shared_ptr<dft_processor_factory>   dft_factory;
  std::shared_ptr<prach_generator_factory> prach_gen_factory;
  prach_detector_factory_sw_configuration  config;
  bool                                     force_gpu_path;
};
#endif

class prach_detector_pool_factory : public prach_detector_factory
{
public:
  prach_detector_pool_factory(std::shared_ptr<prach_detector_factory> factory_, unsigned nof_concurrent_threads_) :
    factory(std::move(factory_)), nof_concurrent_threads(nof_concurrent_threads_)
  {
    ocudu_assert(factory, "Invalid PRACH detector factory.");
    ocudu_assert(nof_concurrent_threads > 1, "Number of concurrent threads must be greater than one.");
  }

  std::unique_ptr<prach_detector> create() override
  {
    if (!pool) {
      std::vector<std::unique_ptr<prach_detector>> detectors(nof_concurrent_threads);
      std::generate(detectors.begin(), detectors.end(), [this]() { return factory->create(); });
      pool = std::make_shared<prach_detector_pool::detector_pool>(detectors);
    }

    return std::make_unique<prach_detector_pool>(pool);
  }

  std::unique_ptr<prach_detector> create(ocudulog::basic_logger& logger, bool log_all_opportunities) override
  {
    if (!pool) {
      std::vector<std::unique_ptr<prach_detector>> detectors(nof_concurrent_threads);
      std::generate(detectors.begin(), detectors.end(), [this, &logger, log_all_opportunities]() {
        return factory->create(logger, log_all_opportunities);
      });
      pool = std::make_shared<prach_detector_pool::detector_pool>(detectors);
    }

    return std::make_unique<prach_detector_pool>(pool);
  }

  std::unique_ptr<prach_detector_validator> create_validator() override { return factory->create_validator(); }

private:
  std::shared_ptr<prach_detector_factory>             factory;
  std::shared_ptr<prach_detector_pool::detector_pool> pool;
  unsigned                                            nof_concurrent_threads;
};

class prach_generator_factory_sw : public prach_generator_factory
{
public:
  std::unique_ptr<prach_generator> create() override { return std::make_unique<prach_generator_impl>(); }
};

} // namespace

std::shared_ptr<prach_detector_factory>
ocudu::create_prach_detector_factory_sw(std::shared_ptr<dft_processor_factory>         dft_factory,
                                        std::shared_ptr<prach_generator_factory>       prach_gen_factory,
                                        const prach_detector_factory_sw_configuration& config)
{
  return std::make_shared<prach_detector_factory_sw>(std::move(dft_factory), std::move(prach_gen_factory), config);
}

std::shared_ptr<prach_detector_factory>
ocudu::create_prach_detector_factory_accelerated(std::shared_ptr<dft_processor_factory>         dft_factory,
                                                 std::shared_ptr<prach_generator_factory>       prach_gen_factory,
                                                 const prach_detector_factory_sw_configuration& config,
                                                 std::string                                    acceleration_mode)
{
  std::string resolved_mode = resolve_prach_detector_acceleration_mode(acceleration_mode);
#ifdef ENABLE_CUDA
  if (resolved_mode == "enabled") {
    report_fatal_error_if_not(is_prach_detector_acceleration_available(),
                              "Accelerated PRACH detector requested but not available.");
    return std::make_shared<prach_detector_factory_cuda>(
        std::move(dft_factory), std::move(prach_gen_factory), config, true);
  }
  if ((resolved_mode == "auto") && is_prach_detector_acceleration_available()) {
    return std::make_shared<prach_detector_factory_cuda>(
        std::move(dft_factory), std::move(prach_gen_factory), config, false);
  }
#else
  if (resolved_mode == "enabled") {
    report_fatal_error("Accelerated PRACH detector requested but CUDA support is not compiled.");
  }
#endif
  return create_prach_detector_factory_sw(std::move(dft_factory), std::move(prach_gen_factory), config);
}

bool ocudu::is_prach_detector_acceleration_available()
{
#ifdef ENABLE_CUDA
  return is_prach_detector_cuda_available();
#else
  return false;
#endif
}

std::string ocudu::resolve_prach_detector_acceleration_mode(const std::string& configured_mode)
{
  return resolve_prach_acceleration_mode(configured_mode);
}

std::shared_ptr<prach_detector_factory>
ocudu::create_prach_detector_pool_factory(std::shared_ptr<prach_detector_factory> factory,
                                          unsigned                                nof_concurrent_threads)
{
  return std::make_shared<prach_detector_pool_factory>(std::move(factory), nof_concurrent_threads);
}

std::shared_ptr<prach_generator_factory> ocudu::create_prach_generator_factory_sw()
{
  return std::make_shared<prach_generator_factory_sw>();
}

namespace {

class logging_prach_detector_decorator : public prach_detector
{
  template <typename Func>
  static std::chrono::nanoseconds time_execution(Func&& func)
  {
    auto start = std::chrono::steady_clock::now();
    func();
    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  }

public:
  logging_prach_detector_decorator(ocudulog::basic_logger&         logger_,
                                   bool                            log_all_opportunities_,
                                   std::unique_ptr<prach_detector> detector_) :
    logger(logger_), log_all_opportunities(log_all_opportunities_), detector(std::move(detector_))
  {
    ocudu_assert(detector, "Invalid detector.");
  }

  prach_detection_result detect(const prach_buffer& input, const configuration& config) override
  {
    prach_detection_result result;
    const auto&&           func = [&]() { result = detector->detect(input, config); };

    std::chrono::nanoseconds time_ns = time_execution(func);

    if (log_all_opportunities || !result.preambles.empty()) {
      if (logger.debug.enabled()) {
        // Detailed log information, including a list of all PRACH config and result fields.
        logger.debug(config.slot.sfn(),
                     config.slot.slot_index(),
                     "PRACH: {:s} {:s} {}\n  {:n}\n  {:n}",
                     config,
                     result,
                     time_ns,
                     config,
                     result);
      } else {
        // Single line log entry.
        logger.info(config.slot.sfn(), config.slot.slot_index(), "PRACH: {:s} {:s} {}", config, result, time_ns);
      }
    }

    return result;
  }

private:
  ocudulog::basic_logger&         logger;
  bool                            log_all_opportunities;
  std::unique_ptr<prach_detector> detector;
};

} // namespace

std::unique_ptr<prach_detector> prach_detector_factory::create(ocudulog::basic_logger& logger,
                                                               bool                    log_all_opportunities)
{
  return std::make_unique<logging_prach_detector_decorator>(logger, log_all_opportunities, create());
}
