// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "lib/phy/lower/modulation/phase_compensation_lut.h"
#include "low_phy_tx.h"
#include "pdxch_baseband_modulator_accelerator.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/phy/lower/amplitude_controller/amplitude_controller_factories.h"
#include "ocudu/ran/resource_block.h"
#include "ocudu/support/units.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

using namespace ocudu;

namespace {

static_assert(sizeof(cbf16_t) == sizeof(uint32_t), "CUDA low-PHY TX expects packed complex BF16 resource elements.");

std::atomic<bool> logged_lowphy_tx_device_grid_path{false};
std::atomic<bool> logged_lowphy_tx_host_grid_path{false};
std::atomic<bool> logged_lowphy_tx_unavailable_path{false};

static bool env_flag_disabled(const char* name)
{
  const char* value = std::getenv(name);
  return value && ((std::strcmp(value, "0") == 0) || (std::strcmp(value, "false") == 0) ||
                   (std::strcmp(value, "off") == 0) || (std::strcmp(value, "no") == 0));
}

static unsigned get_lowphy_tx_metrics_period()
{
  static const unsigned period = []() {
    const char* value = std::getenv("OCUDU_LOWPHY_TX_METRICS_PERIOD");
    if ((value == nullptr) || (*value == '\0')) {
      return 1U;
    }

    char*         end    = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value) {
      return 1U;
    }

    return static_cast<unsigned>(std::min<unsigned long>(parsed, 1000000UL));
  }();

  return period;
}

static float dB_to_amplitude(float value)
{
  return std::pow(10.0F, value / 20.0F);
}

class pdxch_baseband_modulator_cuda : public pdxch_baseband_modulator_accelerator
{
public:
  pdxch_baseband_modulator_cuda(subcarrier_spacing                          scs_,
                                      cyclic_prefix                               cp_,
                                      sampling_rate                               srate_,
                                      unsigned                                    bandwidth_rb_,
                                      double                                      center_freq_Hz_,
                                      unsigned                                    nof_ports_,
                                      const amplitude_controller_clipping_config& amplitude_config_) :
    scs(scs_),
    cp(cp_),
    srate(srate_),
    bandwidth_rb(bandwidth_rb_),
    center_freq_Hz(center_freq_Hz_),
    nof_ports(nof_ports_),
    nof_symbols_per_slot(get_nsymb_per_slot(cp)),
    dft_size(srate.get_dft_size(scs)),
    rg_size(bandwidth_rb * NOF_SUBCARRIERS_PER_RB),
    amplitude_config(amplitude_config_),
    phase_table(scs, cp, dft_size, center_freq_Hz, true),
    next_center_freq_Hz(center_freq_Hz)
  {
    use_host_grid_staging = !env_flag_disabled("OCUDU_LOWPHY_TX_HOST_GRID_STAGING");
    warmup_handle();
  }

  ~pdxch_baseband_modulator_cuda() override
  {
    if (handle != nullptr) {
      ocudu_lowphy_tx_destroy(handle);
      handle = nullptr;
    }
  }

  bool enqueue(baseband_gateway_buffer_dynamic& output,
               const resource_grid_reader&      grid,
               unsigned                         i_symbol_sf_begin,
               span<const unsigned>             symbol_sizes_sf) override
  {
    if ((nof_ports != output.get_nof_channels()) || (symbol_sizes_sf.size() != nof_symbols_per_slot)) {
      return false;
    }

    update_phase_table_if_needed();

    ocudu_lowphy_tx_config_t cfg = make_config(i_symbol_sf_begin, symbol_sizes_sf);

    if (!ensure_handle(cfg)) {
      return false;
    }

    std::array<void*, OCUDU_LOWPHY_TX_MAX_PORTS> output_ports = {};
    last_output_ports.fill(nullptr);
    last_nof_samples  = cfg.nof_samples;
    last_nof_symbols  = nof_symbols_per_slot;
    last_output_valid = false;

    unsigned symbol_offset = 0;
    for (unsigned i_symbol = 0; i_symbol != nof_symbols_per_slot; ++i_symbol) {
      last_symbol_offsets[i_symbol] = symbol_offset;
      last_symbol_sizes[i_symbol]   = symbol_sizes_sf[i_symbol];
      symbol_offset += symbol_sizes_sf[i_symbol];
    }

    for (unsigned port = 0; port != nof_ports; ++port) {
      span<ci16_t> port_buffer = output[port];
      if (port_buffer.size() != static_cast<size_t>(cfg.nof_samples)) {
        return false;
      }
      output_ports[port]      = port_buffer.data();
      last_output_ports[port] = port_buffer.data();
    }

    auto submit = [this](bool success) {
      last_output_valid = success;
      return success;
    };

    void* stream = ocudu_lowphy_tx_get_stream(handle);
    if (grid.supports_device_grid_reading() && grid.prepare_device_grid_reading(stream)) {
      if (!logged_lowphy_tx_device_grid_path.exchange(true)) {
        ocudulog::fetch_basic_logger("PHY").info(
            "Lower-PHY TX GPU path selected: direct CUDA-visible downlink resource-grid reader.");
      }
      bool enqueued = ocudu_lowphy_tx_process(handle, grid.get_device_grid_cbf16(), output_ports.data(), stream) != 0;
      if (enqueued && !grid.on_device_grid_reading_enqueued(stream)) {
        (void)ocudu_lowphy_tx_synchronize(handle);
      }
      return submit(enqueued);
    }

    if (!use_host_grid_staging) {
      if (!logged_lowphy_tx_unavailable_path.exchange(true)) {
        ocudulog::fetch_basic_logger("PHY").warning(
            "Lower-PHY TX GPU path unavailable: resource grid is not CUDA-visible and host-grid staging is disabled.");
      }
      return false;
    }

    const void* host_grid = get_contiguous_host_grid(grid);
    if (host_grid == nullptr) {
      if (!logged_lowphy_tx_unavailable_path.exchange(true)) {
        ocudulog::fetch_basic_logger("PHY").warning(
            "Lower-PHY TX GPU path unavailable: downlink resource grid is not contiguous for host-grid staging.");
      }
      return false;
    }
    if (!logged_lowphy_tx_host_grid_path.exchange(true)) {
      ocudulog::fetch_basic_logger("PHY").info("Lower-PHY TX GPU path selected: host resource-grid staging fallback.");
    }
    return submit(ocudu_lowphy_tx_process_host_grid(handle, host_grid, output_ports.data(), stream) != 0);
  }

  bool wait() override { return (handle != nullptr) && (ocudu_lowphy_tx_synchronize(handle) != 0); }

  bool prepare_output_buffer(baseband_gateway_buffer_dynamic& output) override
  {
    if ((handle == nullptr) || (output.get_nof_channels() != nof_ports)) {
      return false;
    }

    span<ci16_t> first_port = output[0];
    if (first_port.data() == nullptr || first_port.empty()) {
      return false;
    }

    for (unsigned port = 1; port != nof_ports; ++port) {
      span<ci16_t> port_samples = output[port];
      if (port_samples.data() != first_port.data() + static_cast<size_t>(port) * first_port.size()) {
        return false;
      }
    }

    size_t total_bytes = static_cast<size_t>(first_port.size()) * nof_ports * sizeof(ci16_t);
    return ocudu_lowphy_tx_register_host_output(handle, first_port.data(), total_bytes) != 0;
  }

  void set_center_frequency(double center_freq_Hz_) override
  {
    next_center_freq_Hz.store(center_freq_Hz_, std::memory_order_relaxed);
  }

  lower_phy_baseband_metrics collect_metrics() const override
  {
    if (!last_output_valid || (last_nof_samples == 0) || (last_nof_symbols == 0)) {
      return {};
    }

    const unsigned metrics_period = get_lowphy_tx_metrics_period();
    if (metrics_period == 0) {
      return cached_metrics;
    }
    const uint64_t metrics_index = metrics_counter.fetch_add(1, std::memory_order_relaxed);
    if ((metrics_period > 1) && ((metrics_index % metrics_period) != 0)) {
      return cached_metrics;
    }

    static constexpr float ci16_to_cf_scale   = 1.0F / static_cast<float>(std::numeric_limits<int16_t>::max());
    static constexpr float clipping_threshold = 0.95F;

    float    peak_power            = 0.0F;
    double   avg_power_acc         = 0.0;
    unsigned nof_avg_terms         = 0;
    uint64_t nof_clipped_samples   = 0;
    uint64_t nof_processed_samples = 0;

    for (unsigned port = 0; port != nof_ports; ++port) {
      const ci16_t* port_samples = last_output_ports[port];
      if (port_samples == nullptr) {
        continue;
      }
      for (unsigned i_symbol = 0; i_symbol != last_nof_symbols; ++i_symbol) {
        const unsigned symbol_offset = last_symbol_offsets[i_symbol];
        const unsigned symbol_size   = last_symbol_sizes[i_symbol];
        double         symbol_power  = 0.0;
        for (unsigned i_sample = 0; i_sample != symbol_size; ++i_sample) {
          const ci16_t sample = port_samples[symbol_offset + i_sample];
          const float  re     = static_cast<float>(sample.real()) * ci16_to_cf_scale;
          const float  im     = static_cast<float>(sample.imag()) * ci16_to_cf_scale;
          const float  power  = re * re + im * im;
          symbol_power += power;
          peak_power = std::max(peak_power, power);
          nof_clipped_samples += (std::abs(re) > clipping_threshold || std::abs(im) > clipping_threshold) ? 1U : 0U;
        }
        if (symbol_size != 0) {
          avg_power_acc += symbol_power / static_cast<double>(symbol_size);
          ++nof_avg_terms;
          nof_processed_samples += symbol_size;
        }
      }
    }

    lower_phy_baseband_metrics metrics = {};
    metrics.avg_power  = (nof_avg_terms != 0) ? static_cast<float>(avg_power_acc / nof_avg_terms) : 0.0F;
    metrics.peak_power = peak_power;
    metrics.clipping =
        clipping_counters{.nof_clipped_samples = nof_clipped_samples, .nof_processed_samples = nof_processed_samples};
    cached_metrics = metrics;
    return metrics;
  }

private:
  ocudu_lowphy_tx_config_t make_config(unsigned i_symbol_sf_begin, span<const unsigned> symbol_sizes_sf)
  {
    ocudu_lowphy_tx_config_t cfg = {};
    cfg.dft_size                 = dft_size;
    cfg.rg_size                  = rg_size;
    cfg.nof_ports                = nof_ports;
    cfg.nof_symbols              = nof_symbols_per_slot;
    cfg.ofdm_scale               = 1.0F;
    cfg.amplitude_gain           = dB_to_amplitude(amplitude_config.input_gain_dB);
    cfg.clipping_enabled         = amplitude_config.enable_clipping ? 1 : 0;
    cfg.clipping_ceiling         = amplitude_config.full_scale_lin * dB_to_amplitude(amplitude_config.ceiling_dBFS);

    unsigned sample_offset = 0;
    for (unsigned i_symbol = 0; i_symbol != nof_symbols_per_slot; ++i_symbol) {
      unsigned i_symbol_sf         = i_symbol_sf_begin + i_symbol;
      unsigned cp_len              = cp.get_length(i_symbol_sf, scs).to_samples(srate.to_Hz());
      cfg.cp_lengths[i_symbol]     = cp_len;
      cfg.symbol_offsets[i_symbol] = sample_offset;
      cf_t phase                   = phase_table.get_coefficient(i_symbol_sf);
      cfg.phase_re[i_symbol]       = phase.real();
      cfg.phase_im[i_symbol]       = phase.imag();
      sample_offset += symbol_sizes_sf[i_symbol];
    }
    cfg.nof_samples = sample_offset;
    return cfg;
  }

  void warmup_handle()
  {
    std::array<unsigned, MAX_NSYMB_PER_SLOT> initial_symbol_sizes = {};
    for (unsigned i_symbol = 0; i_symbol != nof_symbols_per_slot; ++i_symbol) {
      unsigned cp_len                = cp.get_length(i_symbol, scs).to_samples(srate.to_Hz());
      initial_symbol_sizes[i_symbol] = dft_size + cp_len;
    }
    span<const unsigned>     symbol_sizes(initial_symbol_sizes.data(), nof_symbols_per_slot);
    ocudu_lowphy_tx_config_t cfg = make_config(0, symbol_sizes);
    if (handle == nullptr) {
      (void)ocudu_lowphy_tx_create(&cfg, &handle);
    }
  }

  void update_phase_table_if_needed()
  {
    double next_freq_Hz = next_center_freq_Hz.load(std::memory_order_relaxed);
    if (next_freq_Hz != center_freq_Hz) {
      phase_table    = phase_compensation_lut(scs, cp, dft_size, next_freq_Hz, true);
      center_freq_Hz = next_freq_Hz;
    }
  }

  bool ensure_handle(const ocudu_lowphy_tx_config_t& cfg)
  {
    if (handle == nullptr) {
      return ocudu_lowphy_tx_create(&cfg, &handle) != 0;
    }
    return ocudu_lowphy_tx_update_config(handle, &cfg) != 0;
  }

  const void* get_contiguous_host_grid(const resource_grid_reader& grid) const
  {
    span<const cbf16_t> base = grid.get_view(0, 0);
    if ((base.data() == nullptr) || (base.size() != rg_size)) {
      return nullptr;
    }

    const cbf16_t* base_ptr = base.data();
    for (unsigned port = 0; port != nof_ports; ++port) {
      for (unsigned symbol = 0; symbol != nof_symbols_per_slot; ++symbol) {
        span<const cbf16_t> view     = grid.get_view(port, symbol);
        const cbf16_t*      expected = base_ptr + (static_cast<size_t>(port) * nof_symbols_per_slot + symbol) * rg_size;
        if ((view.data() != expected) || (view.size() != rg_size)) {
          return nullptr;
        }
      }
    }
    return base_ptr;
  }

  subcarrier_spacing                                   scs;
  cyclic_prefix                                        cp;
  sampling_rate                                        srate;
  unsigned                                             bandwidth_rb;
  double                                               center_freq_Hz;
  unsigned                                             nof_ports;
  unsigned                                             nof_symbols_per_slot;
  unsigned                                             dft_size;
  unsigned                                             rg_size;
  amplitude_controller_clipping_config                 amplitude_config;
  phase_compensation_lut                               phase_table;
  std::atomic<double>                                  next_center_freq_Hz;
  ocudu_lowphy_tx_handle_t*                            handle                = nullptr;
  bool                                                 use_host_grid_staging = true;
  std::array<const ci16_t*, OCUDU_LOWPHY_TX_MAX_PORTS> last_output_ports     = {};
  std::array<unsigned, MAX_NSYMB_PER_SLOT>             last_symbol_offsets   = {};
  std::array<unsigned, MAX_NSYMB_PER_SLOT>             last_symbol_sizes     = {};
  unsigned                                             last_nof_samples      = 0;
  unsigned                                             last_nof_symbols      = 0;
  bool                                                 last_output_valid     = false;
  mutable std::atomic<uint64_t>                        metrics_counter{0};
  mutable lower_phy_baseband_metrics                   cached_metrics = {};
};

} // namespace

std::unique_ptr<pdxch_baseband_modulator_accelerator> ocudu::create_pdxch_baseband_modulator_accelerator_cuda(
    subcarrier_spacing                          scs,
    cyclic_prefix                               cp,
    sampling_rate                               srate,
    unsigned                                    bandwidth_rb,
    double                                      center_freq_Hz,
    unsigned                                    nof_ports,
    const amplitude_controller_clipping_config& amplitude_config)
{
  return std::make_unique<pdxch_baseband_modulator_cuda>(
      scs, cp, srate, bandwidth_rb, center_freq_Hz, nof_ports, amplitude_config);
}
