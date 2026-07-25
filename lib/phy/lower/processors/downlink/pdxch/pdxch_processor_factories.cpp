// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "ocudu/phy/lower/processors/downlink/pdxch/pdxch_processor_factories.h"
#include "pdxch_baseband_modulator_accelerator.h"
#include "pdxch_processor_factories_cuda.h"
#include "pdxch_processor_impl.h"
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

using namespace ocudu;

namespace {

class pdxch_processor_factory_sw : public pdxch_processor_factory
{
public:
  pdxch_processor_factory_sw(std::shared_ptr<ofdm_modulator_factory>       ofdm_mod_factory_,
                             std::shared_ptr<amplitude_controller_factory> amplitude_control_factory_) :
    ofdm_mod_factory(std::move(ofdm_mod_factory_)), amplitude_control_factory(std::move(amplitude_control_factory_))
  {
    ocudu_assert(ofdm_mod_factory, "Invalid OFDM modulator factory.");
    ocudu_assert(amplitude_control_factory, "Invalid amplitude control factory.");
  }

  std::unique_ptr<pdxch_processor> create(const pdxch_processor_configuration& config,
                                          task_executor&                       modulation_executor) override
  {
    ofdm_modulator_configuration mod_config = {.numerology     = to_numerology_value(config.scs),
                                               .bw_rb          = config.bandwidth_rb,
                                               .dft_size       = config.srate.get_dft_size(config.scs),
                                               .cp             = config.cp,
                                               .scale          = modulator_scaling,
                                               .center_freq_Hz = config.center_freq_Hz};

    pdxch_processor_impl::configuration pdxch_config = {
        .scs = config.scs, .cp = config.cp, .srate = config.srate, .nof_tx_ports = config.nof_tx_ports};

    return std::make_unique<pdxch_processor_impl>(ofdm_mod_factory->create_ofdm_symbol_modulator(mod_config),
                                                  amplitude_control_factory->create(),
                                                  modulation_executor,
                                                  pdxch_config);
  }

private:
  static constexpr float                        modulator_scaling = 1.0F;
  std::shared_ptr<ofdm_modulator_factory>       ofdm_mod_factory;
  std::shared_ptr<amplitude_controller_factory> amplitude_control_factory;
};

#ifdef ENABLE_CUDA
class pdxch_processor_factory_cuda : public pdxch_processor_factory
{
public:
  pdxch_processor_factory_cuda(std::shared_ptr<ofdm_modulator_factory>       ofdm_mod_factory_,
                               std::shared_ptr<amplitude_controller_factory> amplitude_control_factory_,
                               amplitude_controller_clipping_config          amplitude_config_) :
    ofdm_mod_factory(std::move(ofdm_mod_factory_)),
    amplitude_control_factory(std::move(amplitude_control_factory_)),
    amplitude_config(amplitude_config_)
  {
    ocudu_assert(ofdm_mod_factory, "Invalid OFDM modulator factory.");
    ocudu_assert(amplitude_control_factory, "Invalid amplitude control factory.");
  }

  std::unique_ptr<pdxch_processor> create(const pdxch_processor_configuration& config,
                                          task_executor&                       modulation_executor) override
  {
    ofdm_modulator_configuration mod_config = {.numerology     = to_numerology_value(config.scs),
                                               .bw_rb          = config.bandwidth_rb,
                                               .dft_size       = config.srate.get_dft_size(config.scs),
                                               .cp             = config.cp,
                                               .scale          = modulator_scaling,
                                               .center_freq_Hz = config.center_freq_Hz};

    pdxch_processor_impl::configuration pdxch_config = {
        .scs = config.scs, .cp = config.cp, .srate = config.srate, .nof_tx_ports = config.nof_tx_ports};

    auto accelerator_factory = [config, amplitude_config = amplitude_config]() {
      return create_pdxch_baseband_modulator_accelerator_cuda(config.scs,
                                                              config.cp,
                                                              config.srate,
                                                              config.bandwidth_rb,
                                                              config.center_freq_Hz,
                                                              config.nof_tx_ports,
                                                              amplitude_config);
    };

    return std::make_unique<pdxch_processor_impl>(ofdm_mod_factory->create_ofdm_symbol_modulator(mod_config),
                                                  amplitude_control_factory->create(),
                                                  modulation_executor,
                                                  pdxch_config,
                                                  std::move(accelerator_factory));
  }

private:
  static constexpr float                        modulator_scaling = 1.0F;
  std::shared_ptr<ofdm_modulator_factory>       ofdm_mod_factory;
  std::shared_ptr<amplitude_controller_factory> amplitude_control_factory;
  amplitude_controller_clipping_config          amplitude_config;
};
#endif

} // namespace

std::shared_ptr<pdxch_processor_factory>
ocudu::create_pdxch_processor_factory_sw(std::shared_ptr<ofdm_modulator_factory>       ofdm_mod_factory,
                                         std::shared_ptr<amplitude_controller_factory> amplitude_control_factory)
{
  return std::make_shared<pdxch_processor_factory_sw>(std::move(ofdm_mod_factory),
                                                      std::move(amplitude_control_factory));
}

bool ocudu::is_pdxch_processor_factory_cuda_available()
{
#ifdef ENABLE_CUDA
  int device_count = 0;
  if ((cudaGetDeviceCount(&device_count) != cudaSuccess) || (device_count == 0)) {
    cudaGetLastError();
    return false;
  }
  return true;
#else
  return false;
#endif
}

std::shared_ptr<pdxch_processor_factory>
ocudu::create_pdxch_processor_factory_cuda(std::shared_ptr<ofdm_modulator_factory>       ofdm_mod_factory,
                                           std::shared_ptr<amplitude_controller_factory> amplitude_control_factory,
                                           const amplitude_controller_clipping_config&   amplitude_config)
{
#ifdef ENABLE_CUDA
  return std::make_shared<pdxch_processor_factory_cuda>(
      std::move(ofdm_mod_factory), std::move(amplitude_control_factory), amplitude_config);
#else
  (void)amplitude_config;
  return create_pdxch_processor_factory_sw(std::move(ofdm_mod_factory), std::move(amplitude_control_factory));
#endif
}
