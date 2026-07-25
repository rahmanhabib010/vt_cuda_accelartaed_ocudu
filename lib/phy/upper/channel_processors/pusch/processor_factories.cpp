// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "logging_pusch_processor_decorator.h"
#include "pusch_codeblock_decoder.h"
#include "pusch_decoder_empty_impl.h"
#include "pusch_decoder_hw_impl.h"
#include "pusch_decoder_impl.h"
#include "pusch_processor_impl.h"
#include "pusch_processor_pool.h"
#include "pusch_processor_validator_impl.h"
#include "ulsch_demultiplex_impl.h"
#include "ocudu/phy/upper/channel_processors/pusch/factories.h"
#include "ocudu/ocudulog/ocudulog.h"
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <vector>

#ifdef ENABLE_CUDA
#include "../../channel_coding/ldpc/cuda/ldpc_cuda_factories.h"
#include "cuda/pusch_codeblock_decoder_cuda_batch.h"
#endif

using namespace ocudu;

namespace {

#ifdef ENABLE_CUDA
class bounded_pusch_batch_gpu_decoder_pool : public pusch_decoder_impl::batch_gpu_decoder_pool,
                                            public std::enable_shared_from_this<bounded_pusch_batch_gpu_decoder_pool>
{
public:
  bounded_pusch_batch_gpu_decoder_pool(std::shared_ptr<ldpc_rate_dematcher_factory> dematcher_factory_,
                                       std::shared_ptr<crc_calculator_factory>      crc_factory_,
                                       std::string                                  ldpc_decoder_algorithm_,
                                       unsigned                                     max_decoders_) :
    dematcher_factory(std::move(dematcher_factory_)),
    crc_factory(std::move(crc_factory_)),
    ldpc_decoder_algorithm(std::move(ldpc_decoder_algorithm_)),
    max_decoders(max_decoders_)
  {
    ocudu_assert(dematcher_factory, "Invalid LDPC dematcher factory.");
    ocudu_assert(crc_factory, "Invalid CRC calculator factory.");
    ocudu_assert(max_decoders > 0, "Invalid maximum number of batch GPU decoders.");
    entries.reserve(max_decoders);
  }

  decoder_ptr get() override
  {
    std::unique_lock<std::mutex> lock(mutex);

    for (unsigned i = 0, e = entries.size(); i != e; ++i) {
      if (!entries[i].busy) {
        entries[i].busy = true;
        return make_ptr(i);
      }
    }

    if (entries.size() < max_decoders) {
      const unsigned index = entries.size();
      entries.emplace_back(create_decoder(), true);
      return make_ptr(index);
    }

    if (!saturation_warning_logged) {
      ocudulog::fetch_basic_logger("PHY").warning(
          "PUSCH resident GPU decoder pool exhausted ({} decoders); blocking until a decoder is released. Increase "
          "the pool size or reduce PUSCH concurrency if this warning appears during OTA traffic.",
          max_decoders);
      saturation_warning_logged = true;
    }

    available.wait(lock, [this]() {
      return std::any_of(entries.begin(), entries.end(), [](const entry& item) { return !item.busy; });
    });

    for (unsigned i = 0, e = entries.size(); i != e; ++i) {
      if (!entries[i].busy) {
        entries[i].busy = true;
        return make_ptr(i);
      }
    }

    ocudu_assert(false, "PUSCH resident GPU decoder pool woke without an available decoder.");
    return decoder_ptr(nullptr, [](pusch_resident_codeblock_decoder*) {});
  }

  void warmup() override
  {
    std::vector<decoder_ptr> decoders;
    decoders.reserve(max_decoders);
    for (unsigned i = 0; i != max_decoders; ++i) {
      decoders.push_back(get());
    }
  }

private:
  struct entry {
    std::unique_ptr<pusch_codeblock_decoder_cuda_batch> decoder;
    bool                                                  busy = false;

    entry(std::unique_ptr<pusch_codeblock_decoder_cuda_batch> decoder_, bool busy_) :
      decoder(std::move(decoder_)), busy(busy_)
    {
    }
  };

  std::unique_ptr<pusch_codeblock_decoder_cuda_batch> create_decoder()
  {
    pusch_codeblock_decoder_cuda_batch::sch_crc batch_crcs;
    batch_crcs.crc16  = crc_factory->create(crc_generator_poly::CRC16);
    batch_crcs.crc24A = crc_factory->create(crc_generator_poly::CRC24A);
    batch_crcs.crc24B = crc_factory->create(crc_generator_poly::CRC24B);

    auto decoder =
        std::make_unique<pusch_codeblock_decoder_cuda_batch>(dematcher_factory->create(), std::move(batch_crcs));
    decoder->set_ldpc_decoder_algorithm(ldpc_decoder_algorithm);
    return decoder;
  }

  decoder_ptr make_ptr(unsigned index)
  {
    auto self = shared_from_this();
    return decoder_ptr(entries[index].decoder.get(),
                       [self, index](pusch_resident_codeblock_decoder*) { self->release(index); });
  }

  void release(unsigned index)
  {
    std::lock_guard<std::mutex> lock(mutex);
    ocudu_assert(index < entries.size(), "Invalid batch GPU decoder pool index.");
    ocudu_assert(entries[index].busy, "Releasing an idle batch GPU decoder.");
    entries[index].busy = false;
    available.notify_one();
  }

  std::shared_ptr<ldpc_rate_dematcher_factory> dematcher_factory;
  std::shared_ptr<crc_calculator_factory>      crc_factory;
  std::string                                  ldpc_decoder_algorithm;
  unsigned                                     max_decoders;
  bool                                         saturation_warning_logged = false;
  std::mutex                                   mutex;
  std::condition_variable                      available;
  std::vector<entry>                           entries;
};
#endif

/// \brief Factory for empty PUSCH decoders.
///
/// It creates \ref pusch_decoder_empty_impl instances.
class pusch_decoder_factory_empty : public pusch_decoder_factory
{
public:
  /// Constructs a factory taking the maximum number of PRB and layers.
  pusch_decoder_factory_empty(unsigned nof_prb_, unsigned nof_layers_) : nof_prb(nof_prb_), nof_layers(nof_layers_)
  {
    ocudu_assert(nof_prb > 0, "Invalid number of PRB.");
    ocudu_assert(nof_layers > 0, "Invalid number of layers.");
  }

  // See interface for documentation.
  std::unique_ptr<pusch_decoder> create() override
  {
    return std::make_unique<pusch_decoder_empty_impl>(nof_prb, nof_layers);
  }

private:
  unsigned nof_prb;
  unsigned nof_layers;
};

class pusch_decoder_factory_generic : public pusch_decoder_factory
{
public:
  explicit pusch_decoder_factory_generic(pusch_decoder_factory_sw_configuration config) :
    crc_factory(std::move(config.crc_factory)),
    dematcher_factory(std::move(config.dematcher_factory)),
    segmenter_factory(std::move(config.segmenter_factory)),
    executor(config.executor),
    nof_prb(config.nof_prb),
    nof_layers(config.nof_layers),
    enable_resident_acceleration_decoder(config.enable_resident_acceleration_decoder),
    ldpc_decoder_algorithm(config.ldpc_decoder_algorithm)
  {
    ocudu_assert(crc_factory, "Invalid CRC calculator factory.");
    ocudu_assert(config.decoder_factory, "Invalid LDPC decoder factory.");
    ocudu_assert(dematcher_factory, "Invalid LDPC dematcher factory.");
    ocudu_assert(segmenter_factory, "Invalid LDPC segmenter factory.");

    std::vector<std::unique_ptr<pusch_codeblock_decoder>> codeblock_decoders(
        std::max(1U, config.nof_pusch_decoder_threads));
    for (std::unique_ptr<pusch_codeblock_decoder>& codeblock_decoder : codeblock_decoders) {
      pusch_codeblock_decoder::sch_crc crcs1;
      crcs1.crc16  = crc_factory->create(crc_generator_poly::CRC16);
      crcs1.crc24A = crc_factory->create(crc_generator_poly::CRC24A);
      crcs1.crc24B = crc_factory->create(crc_generator_poly::CRC24B);

      codeblock_decoder = std::make_unique<pusch_codeblock_decoder>(
          dematcher_factory->create(), config.decoder_factory->create(), crcs1);
    }

    decoder_pool = std::make_unique<pusch_decoder_impl::codeblock_decoder_pool>(codeblock_decoders);

#ifdef ENABLE_CUDA
    if (enable_resident_acceleration_decoder && (nof_layers <= pusch_constants::MAX_NOF_LAYERS) &&
        is_cuda_available()) {
      unsigned nof_resident_acceleration_decoders = config.nof_resident_acceleration_decoders;
      if (nof_resident_acceleration_decoders == 0) {
        nof_resident_acceleration_decoders = std::max(1U, config.nof_pusch_decoder_threads);
      }
      if (nof_resident_acceleration_decoders > MAX_BATCH_GPU_DECODERS) {
        ocudulog::fetch_basic_logger("PHY").warning(
            "PUSCH resident GPU decoder pool requested {} decoders; capping at {}. Reduce PUSCH concurrency or "
            "increase the pool cap to avoid runtime blocking.",
            nof_resident_acceleration_decoders,
            MAX_BATCH_GPU_DECODERS);
        nof_resident_acceleration_decoders = MAX_BATCH_GPU_DECODERS;
      }
      batch_gpu_decoder_pool = std::make_shared<bounded_pusch_batch_gpu_decoder_pool>(
          dematcher_factory, crc_factory, ldpc_decoder_algorithm, nof_resident_acceleration_decoders);
    } else if (enable_resident_acceleration_decoder) {
      ocudulog::fetch_basic_logger("PHY").warning(
          "PUSCH resident GPU decoder disabled because CUDA is not available.");
    }
#endif
  }

  std::unique_ptr<pusch_decoder> create() override
  {
    pusch_decoder_impl::sch_crc crcs;
    crcs.crc16  = crc_factory->create(crc_generator_poly::CRC16);
    crcs.crc24A = crc_factory->create(crc_generator_poly::CRC24A);
    crcs.crc24B = crc_factory->create(crc_generator_poly::CRC24B);

#ifdef ENABLE_CUDA
    return std::make_unique<pusch_decoder_impl>(segmenter_factory->create(),
                                                decoder_pool,
                                                batch_gpu_decoder_pool,
                                                std::move(crcs),
                                                executor,
                                                nof_prb,
                                                nof_layers);
#else
    return std::make_unique<pusch_decoder_impl>(
        segmenter_factory->create(), decoder_pool, std::move(crcs), executor, nof_prb, nof_layers);
#endif
  }

  void warmup() override
  {
#ifdef ENABLE_CUDA
    if (batch_gpu_decoder_pool) {
      batch_gpu_decoder_pool->warmup();
    }
#endif
  }

private:
  std::shared_ptr<pusch_decoder_impl::codeblock_decoder_pool> decoder_pool;
  std::shared_ptr<crc_calculator_factory>                     crc_factory;
  std::shared_ptr<ldpc_rate_dematcher_factory>                dematcher_factory;
  std::shared_ptr<ldpc_segmenter_rx_factory>                  segmenter_factory;
  task_executor*                                              executor;
  unsigned                                                    nof_prb;
  unsigned                                                    nof_layers;
  bool                                                        enable_resident_acceleration_decoder;
  std::string                                                 ldpc_decoder_algorithm;
#ifdef ENABLE_CUDA
  std::shared_ptr<pusch_decoder_impl::batch_gpu_decoder_pool> batch_gpu_decoder_pool;
  /// Maximum number of batch GPU decoders to create (~48 MB each).
  static constexpr unsigned MAX_BATCH_GPU_DECODERS = 32;
#endif
};

/// HW-accelerated PUSCH decoder factory.
class pusch_decoder_factory_hw : public pusch_decoder_factory
{
public:
  explicit pusch_decoder_factory_hw(const pusch_decoder_factory_hw_configuration& config) :
    segmenter_factory(config.segmenter_factory), crc_factory(config.crc_factory), executor(config.executor)
  {
    ocudu_assert(segmenter_factory, "Invalid LDPC segmenter factory.");
    ocudu_assert(crc_factory, "Invalid CRC factory.");
    ocudu_assert(config.hw_decoder_factory, "Invalid hardware accelerator factory.");

    // Creates a vector of hardware decoders. These are shared for all the PUSCH decoders.
    std::vector<std::unique_ptr<hal::hw_accelerator_pusch_dec>> hw_decoders(
        std::max(1U, config.nof_pusch_decoder_threads));
    for (std::unique_ptr<hal::hw_accelerator_pusch_dec>& hw_decoder : hw_decoders) {
      hw_decoder = config.hw_decoder_factory->create();
    }

    // Creates the hardware decoder pool. The pool is common among all the PUSCH decoders.
    hw_decoder_pool = std::make_unique<pusch_decoder_hw_impl::hw_decoder_pool>(hw_decoders);
  }

  std::unique_ptr<pusch_decoder> create() override
  {
    pusch_decoder_hw_impl::sch_crc crc = {
        crc_factory->create(crc_generator_poly::CRC16),
        crc_factory->create(crc_generator_poly::CRC24A),
        crc_factory->create(crc_generator_poly::CRC24B),
    };
    return std::make_unique<pusch_decoder_hw_impl>(segmenter_factory->create(), crc, hw_decoder_pool, executor);
  }

private:
  std::shared_ptr<ldpc_segmenter_rx_factory>              segmenter_factory;
  std::shared_ptr<crc_calculator_factory>                 crc_factory;
  std::shared_ptr<pusch_decoder_hw_impl::hw_decoder_pool> hw_decoder_pool;
  task_executor*                                          executor;
};

class pusch_processor_factory_generic : public pusch_processor_factory
{
public:
  explicit pusch_processor_factory_generic(pusch_processor_factory_sw_configuration& config) :
    estimator_factory(config.estimator_factory),
    demodulator_factory(config.demodulator_factory),
    demux_factory(config.demux_factory),
    decoder_factory(config.decoder_factory),
    uci_dec_factory(config.uci_dec_factory),
    ch_estimate_dimensions(config.ch_estimate_dimensions),
    dec_nof_iterations(config.dec_nof_iterations),
    dec_enable_early_stop(config.dec_enable_early_stop),
    csi_sinr_calc_method(config.csi_sinr_calc_method)
  {
    ocudu_assert(estimator_factory, "Invalid channel estimation factory.");
    ocudu_assert(demodulator_factory, "Invalid demodulation factory.");
    ocudu_assert(demux_factory, "Invalid demux factory.");
    ocudu_assert(decoder_factory, "Invalid decoder factory.");
    ocudu_assert(uci_dec_factory, "Invalid UCI decoder factory.");

    // Create common dependencies.
    std::vector<std::unique_ptr<pusch_processor_impl::concurrent_dependencies>> dependencies(
        config.max_nof_concurrent_threads);
    std::generate(dependencies.begin(), dependencies.end(), [this]() {
      return std::make_unique<pusch_processor_impl::concurrent_dependencies>(estimator_factory->create(),
                                                                             demodulator_factory->create(),
                                                                             demux_factory->create(),
                                                                             uci_dec_factory->create(),
                                                                             ch_estimate_dimensions);
    });

    // Create common dependencies pool.
    dependencies_pool = std::make_shared<pusch_processor_impl::concurrent_dependencies_pool_type>(dependencies);

    // Warm decoder-side GPU resources after demodulator instances have initialized their CUDA handles.
    decoder_factory->warmup();
  }

  std::unique_ptr<pusch_processor> create() override
  {
    pusch_processor_impl::configuration config;
    config.dependencies_pool     = dependencies_pool;
    config.decoder               = decoder_factory->create();
    config.dec_nof_iterations    = dec_nof_iterations;
    config.dec_enable_early_stop = dec_enable_early_stop;
    config.csi_sinr_calc_method  = csi_sinr_calc_method;
    config.ce_dims               = ch_estimate_dimensions;
    return std::make_unique<pusch_processor_impl>(config);
  }

  std::unique_ptr<pusch_pdu_validator> create_validator() override
  {
    return std::make_unique<pusch_processor_validator_impl>(ch_estimate_dimensions);
  }

private:
  std::shared_ptr<pusch_processor_impl::concurrent_dependencies_pool_type> dependencies_pool;
  std::shared_ptr<dmrs_pusch_estimator_factory>                            estimator_factory;
  std::shared_ptr<pusch_demodulator_factory>                               demodulator_factory;
  std::shared_ptr<ulsch_demultiplex_factory>                               demux_factory;
  std::shared_ptr<pusch_decoder_factory>                                   decoder_factory;
  std::shared_ptr<uci_decoder_factory>                                     uci_dec_factory;
  pusch_processor::channel_size                                            ch_estimate_dimensions;
  unsigned                                                                 dec_nof_iterations;
  bool                                                                     dec_enable_early_stop;
  channel_state_information::sinr_type                                     csi_sinr_calc_method;
};

class pusch_processor_pool_factory : public pusch_processor_factory
{
public:
  pusch_processor_pool_factory(pusch_processor_pool_factory_config& config) :
    regular_factory(config.factory),
    uci_factory(config.uci_factory),
    nof_regular_processors(config.nof_regular_processors),
    nof_uci_processors(config.nof_uci_processors)
  {
    ocudu_assert(regular_factory, "Invalid PUSCH factory.");
    ocudu_assert(uci_factory, "Invalid PUSCH factory for UCI.");
  }

  std::unique_ptr<pusch_processor> create() override
  {
    if (nof_regular_processors <= 1) {
      return regular_factory->create();
    }

    std::vector<std::unique_ptr<pusch_processor_wrapper>> processors(nof_regular_processors);
    for (std::unique_ptr<pusch_processor_wrapper>& processor : processors) {
      processor = std::make_unique<pusch_processor_wrapper>(regular_factory->create());
    }

    if (!uci_processor_pool) {
      std::vector<std::unique_ptr<pusch_processor>> uci_processors(nof_uci_processors);
      for (std::unique_ptr<pusch_processor>& processor : uci_processors) {
        processor = uci_factory->create();
      }
      uci_processor_pool = std::make_shared<pusch_processor_pool::uci_processor_pool>(uci_processors);
    }

    return std::make_unique<pusch_processor_pool>(processors, uci_processor_pool);
  }

  std::unique_ptr<pusch_processor> create(ocudulog::basic_logger& logger) override
  {
    if (nof_regular_processors <= 1) {
      return regular_factory->create(logger);
    }

    std::vector<std::unique_ptr<pusch_processor_wrapper>> processors(nof_regular_processors);
    for (std::unique_ptr<pusch_processor_wrapper>& processor : processors) {
      processor = std::make_unique<pusch_processor_wrapper>(regular_factory->create(logger));
    }

    if (!uci_processor_pool) {
      std::vector<std::unique_ptr<pusch_processor>> uci_processors(nof_uci_processors);
      for (std::unique_ptr<pusch_processor>& processor : uci_processors) {
        processor = uci_factory->create(logger);
      }
      uci_processor_pool = std::make_shared<pusch_processor_pool::uci_processor_pool>(uci_processors);
    }

    return std::make_unique<pusch_processor_pool>(processors, uci_processor_pool);
  }

  std::unique_ptr<pusch_pdu_validator> create_validator() override { return regular_factory->create_validator(); }

private:
  std::shared_ptr<pusch_processor_factory>                  regular_factory;
  std::shared_ptr<pusch_processor_factory>                  uci_factory;
  std::shared_ptr<pusch_processor_pool::uci_processor_pool> uci_processor_pool;
  unsigned                                                  nof_regular_processors;
  unsigned                                                  nof_uci_processors;
};

class ulsch_demultiplex_factory_sw : public ulsch_demultiplex_factory
{
public:
  std::unique_ptr<ulsch_demultiplex> create() override { return std::make_unique<ulsch_demultiplex_impl>(); }
};

} // namespace

std::shared_ptr<pusch_decoder_factory> ocudu::create_pusch_decoder_empty_factory(unsigned nof_prb, unsigned nof_layers)
{
  return std::make_shared<pusch_decoder_factory_empty>(nof_prb, nof_layers);
}

std::shared_ptr<pusch_decoder_factory>
ocudu::create_pusch_decoder_factory_sw(pusch_decoder_factory_sw_configuration config)
{
  return std::make_shared<pusch_decoder_factory_generic>(std::move(config));
}

std::shared_ptr<pusch_decoder_factory>
ocudu::create_pusch_decoder_factory_hw(const pusch_decoder_factory_hw_configuration& config)
{
  return std::make_shared<pusch_decoder_factory_hw>(config);
}

std::shared_ptr<pusch_processor_factory>
ocudu::create_pusch_processor_factory_sw(pusch_processor_factory_sw_configuration& config)
{
  return std::make_shared<pusch_processor_factory_generic>(config);
}

std::shared_ptr<pusch_processor_factory> ocudu::create_pusch_processor_pool(pusch_processor_pool_factory_config& config)
{
  return std::make_shared<pusch_processor_pool_factory>(config);
}

std::unique_ptr<pusch_processor> pusch_processor_factory::create(ocudulog::basic_logger& logger)
{
  return std::make_unique<logging_pusch_processor_decorator>(logger, create());
}

std::shared_ptr<ulsch_demultiplex_factory> ocudu::create_ulsch_demultiplex_factory_sw()
{
  return std::make_shared<ulsch_demultiplex_factory_sw>();
}
