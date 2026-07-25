// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "ocudu/phy/upper/signal_processors/srs/srs_estimator_factory.h"
#include "logging_srs_estimator_decorator.h"
#ifdef ENABLE_CUDA
#include "srs_estimator_cuda_impl.h"
#endif
#include "srs_estimator_generic_impl.h"
#include "srs_estimator_pool.h"
#include "srs_validator_generic_impl.h"
#include "ocudu/phy/support/time_alignment_estimator/time_alignment_estimator_factories.h"
#include "ocudu/support/error_handling.h"
#ifdef ENABLE_CUDA
#include <cstdlib>
#include <cstring>
#endif

using namespace ocudu;

namespace {

#ifdef ENABLE_CUDA
bool is_disabled_value(const char* value)
{
  return (std::strcmp(value, "0") == 0) || (std::strcmp(value, "false") == 0) || (std::strcmp(value, "FALSE") == 0) ||
         (std::strcmp(value, "off") == 0) || (std::strcmp(value, "OFF") == 0) || (std::strcmp(value, "disable") == 0) ||
         (std::strcmp(value, "DISABLE") == 0);
}

bool is_enabled_value(const char* value)
{
  return (std::strcmp(value, "1") == 0) || (std::strcmp(value, "true") == 0) || (std::strcmp(value, "TRUE") == 0) ||
         (std::strcmp(value, "on") == 0) || (std::strcmp(value, "ON") == 0) || (std::strcmp(value, "enable") == 0) ||
         (std::strcmp(value, "ENABLE") == 0);
}

std::string resolve_srs_acceleration_mode(const std::string& configured_mode)
{
  if (configured_mode != "auto") {
    return configured_mode;
  }

  const char* env_value = std::getenv("OCUDU_SRS_ACCELERATION");
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
std::string resolve_srs_acceleration_mode(const std::string& configured_mode)
{
  return configured_mode;
}
#endif

class srs_estimator_factory_generic : public srs_estimator_factory
{
public:
  srs_estimator_factory_generic(std::shared_ptr<low_papr_sequence_generator_factory> sequence_generator_factory_,
                                std::shared_ptr<time_alignment_estimator_factory>    ta_estimator_factory_,
                                unsigned                                             max_nof_prb_,
                                std::string                                          acceleration_mode_) :
    sequence_generator_factory(std::move(sequence_generator_factory_)),
    ta_estimator_factory(std::move(ta_estimator_factory_)),
    max_nof_prb(max_nof_prb_),
    acceleration_mode(std::move(acceleration_mode_))
  {
    ocudu_assert(sequence_generator_factory, "Invalid sequence generator factory.");
    ocudu_assert(ta_estimator_factory, "Invalid TA estimator factory.");
    ocudu_assert(max_nof_prb != 0, "Maximum number of PRB cannot be zero.");
  }

  std::unique_ptr<srs_estimator> create() override
  {
    auto create_software_estimator = [this]() {
      srs_estimator_generic_impl::dependencies deps;
      deps.sequence_generator = sequence_generator_factory->create();
      deps.ta_estimator       = ta_estimator_factory->create();

      return std::make_unique<srs_estimator_generic_impl>(std::move(deps), max_nof_prb);
    };

#ifdef ENABLE_CUDA
    std::string resolved_acceleration_mode = resolve_srs_estimator_acceleration_mode(acceleration_mode);
    if (resolved_acceleration_mode == "enabled") {
      report_fatal_error_if_not(is_srs_estimator_acceleration_available(),
                                "Accelerated SRS estimator requested but not available.");
      return std::make_unique<srs_estimator_cuda_impl>(
          sequence_generator_factory->create(), create_software_estimator(), max_nof_prb, true);
    }
    if ((resolved_acceleration_mode == "auto") && is_srs_estimator_acceleration_available()) {
      return std::make_unique<srs_estimator_cuda_impl>(
          sequence_generator_factory->create(), create_software_estimator(), max_nof_prb, false);
    }
#else
    if (resolve_srs_estimator_acceleration_mode(acceleration_mode) == "enabled") {
      report_fatal_error("Accelerated SRS estimator requested but CUDA support is not compiled.");
    }
#endif

    return create_software_estimator();
  }

  std::unique_ptr<srs_estimator_configuration_validator> create_validator() override
  {
    return std::make_unique<srs_validator_generic_impl>(max_nof_prb);
  }

private:
  std::shared_ptr<low_papr_sequence_generator_factory> sequence_generator_factory;
  std::shared_ptr<time_alignment_estimator_factory>    ta_estimator_factory;
  unsigned                                             max_nof_prb;
  std::string                                          acceleration_mode;
};

class srs_estimator_factory_pool : public srs_estimator_factory
{
public:
  srs_estimator_factory_pool(std::shared_ptr<srs_estimator_factory> factory_, unsigned nof_concurrent_threads_) :
    factory(std::move(factory_)), nof_concurrent_threads(nof_concurrent_threads_)
  {
    ocudu_assert(factory, "Invalid factory.");
    ocudu_assert(nof_concurrent_threads != 0, "Number of threads must be larger than 0.");
  }

  // See interface for documentation.
  std::unique_ptr<srs_estimator> create() override
  {
    if (!pool) {
      std::vector<std::unique_ptr<srs_estimator>> estimators(nof_concurrent_threads);

      for (auto& estimator : estimators) {
        estimator = factory->create();
      }

      pool = std::make_shared<srs_estimator_pool::estimator_pool>(estimators);
    }

    return std::make_unique<srs_estimator_pool>(pool);
  }

  // See interface for documentation.
  std::unique_ptr<srs_estimator_configuration_validator> create_validator() override
  {
    return factory->create_validator();
  }

private:
  std::shared_ptr<srs_estimator_factory>              factory;
  std::shared_ptr<srs_estimator_pool::estimator_pool> pool;
  unsigned                                            nof_concurrent_threads;
};

} // namespace

std::shared_ptr<srs_estimator_factory> ocudu::create_srs_estimator_generic_factory(
    std::shared_ptr<low_papr_sequence_generator_factory> sequence_generator_factory,
    std::shared_ptr<time_alignment_estimator_factory>    ta_estimator_factory,
    unsigned                                             max_nof_prb,
    std::string                                          acceleration_mode)
{
  return std::make_shared<srs_estimator_factory_generic>(std::move(sequence_generator_factory),
                                                         std::move(ta_estimator_factory),
                                                         max_nof_prb,
                                                         std::move(acceleration_mode));
}

bool ocudu::is_srs_estimator_acceleration_available()
{
#ifdef ENABLE_CUDA
  return is_srs_estimator_cuda_available();
#else
  return false;
#endif
}

std::string ocudu::resolve_srs_estimator_acceleration_mode(const std::string& configured_mode)
{
  return resolve_srs_acceleration_mode(configured_mode);
}

std::shared_ptr<srs_estimator_factory>
ocudu::create_srs_estimator_pool(std::shared_ptr<srs_estimator_factory> base_factory, unsigned max_nof_threads)
{
  return std::make_shared<srs_estimator_factory_pool>(std::move(base_factory), max_nof_threads);
}

std::unique_ptr<srs_estimator> srs_estimator_factory::create(ocudulog::basic_logger& logger)
{
  return std::make_unique<logging_srs_estimator_decorator>(logger, create());
}
