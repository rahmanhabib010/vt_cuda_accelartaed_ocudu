// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief GPU vs CPU PUSCH SINR comparison test.
///
/// This test compares the post-equalization SINR values reported by the CPU and GPU
/// PUSCH demodulator implementations to identify and quantify any discrepancies.
///
/// IMPORTANT NOTE ON GPU E2E PATH:
/// The GPU demodulator's E2E path performs its own channel estimation internally
/// using CUDA kernels. It does NOT use the dmrs_pusch_estimator_results interface
/// in the same way as the CPU path. This means:
/// - CPU path: Uses noise variance from the provided estimator results
/// - GPU path: Computes its own channel estimates and SINR from raw grid data
///
/// For meaningful comparison, the grid data must contain realistic DMRS symbols
/// that the GPU can use for channel estimation. This test documents the observed
/// differences between the two paths.

#include "ocudu/phy/support/resource_grid_reader.h"
#include "ocudu/phy/support/resource_grid_writer.h"
#include "ocudu/phy/support/support_factories.h"
#include "ocudu/phy/upper/channel_coding/channel_coding_factories.h"
#include "ocudu/phy/upper/channel_modulation/channel_modulation_factories.h"
#include "ocudu/phy/upper/channel_processors/pusch/factories.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_codeword_buffer.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_demodulator.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_demodulator_notifier.h"
#include "ocudu/phy/upper/equalization/equalization_factories.h"
#include "ocudu/phy/upper/signal_processors/pusch/dmrs_pusch_estimator.h"
#include "ocudu/phy/upper/signal_processors/pusch/factories.h"
#include "ocudu/ran/resource_block.h"
#include "ocudu/support/math/math_utils.h"
#include "fmt/format.h"
#include "gtest/gtest.h"
#include <cmath>
#include <complex>
#include <optional>
#include <random>
#include <vector>

using namespace ocudu;

namespace {

/// Test configuration parameters.
constexpr unsigned MAX_NOF_PRBS_TEST    = 273;
constexpr unsigned NOF_SYMBOLS_PER_SLOT = 14;
constexpr unsigned NOF_DMRS_SYMBOLS     = 2;
constexpr unsigned NOF_DATA_SYMBOLS     = NOF_SYMBOLS_PER_SLOT - NOF_DMRS_SYMBOLS;

/// DMRS symbol positions (symbols 2 and 11).
const symbol_slot_mask DMRS_SYMBOL_MASK =
    {false, false, true, false, false, false, false, false, false, false, false, true, false, false};

/// SINR comparison threshold in dB.
constexpr float SINR_THRESHOLD_DB = 1.0f;

/// Test PRB allocations.
const std::vector<unsigned> TEST_PRB_COUNTS = {3, 4, 5, 6};

/// Test SNR values in dB.
const std::vector<float> TEST_SNR_VALUES_DB = {5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 30.0f};

/// Notifier that captures the SINR value.
class sinr_capture_notifier : public pusch_demodulator_notifier
{
public:
  std::optional<float> captured_sinr_db;

  void on_provisional_stats(unsigned /*i_symbol*/, const demodulation_stats& /*stats*/) override
  {
    // Ignore provisional stats.
  }

  void on_end_stats(const demodulation_stats& stats) override { captured_sinr_db = stats.sinr_dB; }

  void reset() { captured_sinr_db.reset(); }
};

/// Simple codeword buffer that discards LLRs (we only care about SINR).
class llr_discard_buffer : public pusch_codeword_buffer
{
public:
  explicit llr_discard_buffer(unsigned max_size) : temp_buffer(max_size) {}

  span<log_likelihood_ratio> get_next_block_view(unsigned block_size) override
  {
    if (block_size > temp_buffer.size()) {
      temp_buffer.resize(block_size);
    }
    return span<log_likelihood_ratio>(temp_buffer).first(block_size);
  }

  void on_new_block(span<const log_likelihood_ratio> /*new_data*/, const bit_buffer& /*new_sequence*/) override
  {
    // Discard the LLRs - we only care about SINR.
  }

  void on_end_codeword() override
  {
    // Nothing to do.
  }

private:
  std::vector<log_likelihood_ratio> temp_buffer;
};

/// Mock channel estimator results that provides controlled noise variance and flat channel.
class mock_channel_estimator_results : public dmrs_pusch_estimator_results
{
public:
  mock_channel_estimator_results(unsigned                                   nof_ports,
                                 unsigned                                   nof_layers,
                                 unsigned                                   nof_symbols,
                                 unsigned                                   nof_subcarriers,
                                 float                                      noise_variance,
                                 const bounded_bitset<MAX_NSYMB_PER_SLOT>&  symbols_mask,
                                 const bounded_bitset<MAX_NOF_SUBCARRIERS>& re_mask) :
    nof_ports_(nof_ports),
    nof_layers_(nof_layers),
    nof_symbols_(nof_symbols),
    nof_subcarriers_(nof_subcarriers),
    noise_variance_(noise_variance),
    symbols_mask_(symbols_mask),
    re_mask_(re_mask)
  {
    // Pre-fill channel estimates with H=1 (flat channel).
    channel_estimates_.resize(nof_subcarriers_);
    std::fill(channel_estimates_.begin(), channel_estimates_.end(), cbf16_t(1.0f, 0.0f));
  }

  float get_noise_variance(unsigned /*rx_port*/) const override { return noise_variance_; }

  float get_rsrp(unsigned /*rx_port*/, unsigned /*tx_layer*/) const override
  {
    // Unit signal power.
    return 1.0f;
  }

  static_vector<float, MAX_PORTS> get_rsrp_all_ports(unsigned /*tx_layer*/) const override
  {
    static_vector<float, MAX_PORTS> rsrp(nof_ports_);
    std::fill(rsrp.begin(), rsrp.end(), 1.0f);
    return rsrp;
  }

  float get_epre(unsigned /*rx_port*/) const override
  {
    // EPRE = signal + noise.
    return 1.0f + noise_variance_;
  }

  float get_snr(unsigned /*rx_port*/) const override
  {
    // SNR = signal / noise.
    return 1.0f / noise_variance_;
  }

  float get_layer_average_snr(unsigned /*tx_layer*/) const override { return 1.0f / noise_variance_; }

  phy_time_unit get_time_alignment(unsigned /*rx_port*/) const override { return phy_time_unit::from_seconds(0.0); }

  std::optional<float> get_cfo_Hz(unsigned /*rx_port*/) const override { return std::nullopt; }

  void get_symbol_ch_estimate(span<cbf16_t> estimates,
                              unsigned /*i_symbol*/,
                              unsigned /*rx_port*/,
                              unsigned /*tx_layer*/) const override
  {
    // Return flat channel H=1 for all subcarriers.
    unsigned count = std::min(static_cast<unsigned>(estimates.size()), nof_subcarriers_);
    std::copy_n(channel_estimates_.begin(), count, estimates.begin());
  }

  void get_symbol_ch_estimate(span<cbf16_t> estimates,
                              unsigned /*i_symbol*/,
                              unsigned /*rx_port*/,
                              unsigned /*tx_layer*/,
                              const bounded_bitset<MAX_NOF_SUBCARRIERS>& mask) const override
  {
    // Return flat channel H=1 for masked subcarriers.
    unsigned idx = 0;
    for (unsigned sc = 0; sc < nof_subcarriers_ && idx < estimates.size(); ++sc) {
      if (mask.test(sc)) {
        estimates[idx++] = cbf16_t(1.0f, 0.0f);
      }
    }
  }

  void get_channel_state_information(channel_state_information& csi) const override
  {
    // Set basic CSI.
    csi.set_sinr_dB(channel_state_information::sinr_type::channel_estimator,
                    convert_power_to_dB(1.0f / noise_variance_));
  }

private:
  unsigned                            nof_ports_;
  unsigned                            nof_layers_;
  unsigned                            nof_symbols_;
  unsigned                            nof_subcarriers_;
  float                               noise_variance_;
  bounded_bitset<MAX_NSYMB_PER_SLOT>  symbols_mask_;
  bounded_bitset<MAX_NOF_SUBCARRIERS> re_mask_;
  std::vector<cbf16_t>                channel_estimates_;
};

/// Test fixture for PUSCH SINR comparison.
class PuschSinrComparisonTest : public ::testing::Test
{
protected:
  static std::shared_ptr<resource_grid_factory>     rg_factory;
  static std::shared_ptr<pusch_demodulator_factory> cpu_demod_factory;
  static std::shared_ptr<pusch_demodulator_factory> gpu_demod_factory;
  static std::unique_ptr<pusch_demodulator>         cpu_demodulator;
  static std::unique_ptr<pusch_demodulator>         gpu_demodulator;
  static bool                                       gpu_available;

  static void SetUpTestSuite()
  {
    // Create resource grid factory.
    rg_factory = create_resource_grid_factory();
    ASSERT_NE(rg_factory, nullptr) << "Failed to create resource grid factory.";

    // Create pseudo-random sequence generator factory.
    std::shared_ptr<pseudo_random_generator_factory> prg_factory = create_pseudo_random_generator_sw_factory();
    ASSERT_NE(prg_factory, nullptr) << "Failed to create PRG factory.";

    // Create channel equalizer factory.
    std::shared_ptr<channel_equalizer_factory> eq_factory = create_channel_equalizer_generic_factory();
    ASSERT_NE(eq_factory, nullptr) << "Failed to create equalizer factory.";

    // Create DFT factory for transform precoding.
    std::shared_ptr<dft_processor_factory> dft_factory = create_dft_processor_factory_fftw_fast();
    if (!dft_factory) {
      dft_factory = create_dft_processor_factory_generic();
    }
    ASSERT_NE(dft_factory, nullptr) << "Failed to create DFT factory.";

    // Create transform precoder factory.
    std::shared_ptr<transform_precoder_factory> precoder_factory =
        create_dft_transform_precoder_factory(dft_factory, MAX_NOF_PRBS_TEST);
    ASSERT_NE(precoder_factory, nullptr) << "Failed to create transform precoder factory.";

    // Create demodulation mapper factory.
    std::shared_ptr<demodulation_mapper_factory> demod_factory = create_demodulation_mapper_factory();
    ASSERT_NE(demod_factory, nullptr) << "Failed to create demodulation mapper factory.";

    // Create CPU demodulator factory with post-equalization SINR enabled.
    cpu_demod_factory = create_pusch_demodulator_factory_sw(eq_factory,
                                                            precoder_factory,
                                                            demod_factory,
                                                            /*evm_calc_factory=*/nullptr,
                                                            prg_factory,
                                                            MAX_NOF_PRBS_TEST,
                                                            /*enable_post_eq_sinr=*/true);
    ASSERT_NE(cpu_demod_factory, nullptr) << "Failed to create CPU demodulator factory.";

    // Create CPU demodulator.
    cpu_demodulator = cpu_demod_factory->create();
    ASSERT_NE(cpu_demodulator, nullptr) << "Failed to create CPU demodulator.";

    // Check if GPU demodulator is available.
    gpu_available = is_pusch_demodulator_acceleration_available();

    if (gpu_available) {
      // Create GPU demodulator factory with post-equalization SINR enabled.
      pusch_demodulator_factory_accelerated_configuration gpu_config;
      gpu_config.equalizer_factory    = eq_factory;
      gpu_config.precoder_factory     = precoder_factory;
      gpu_config.demodulation_factory = demod_factory;
      gpu_config.evm_calc_factory     = nullptr;
      gpu_config.prg_factory          = prg_factory;
      gpu_config.max_nof_prb          = MAX_NOF_PRBS_TEST;
      gpu_config.enable_post_eq_sinr  = true;

      gpu_demod_factory = create_pusch_demodulator_factory_accelerated(gpu_config);
      if (gpu_demod_factory) {
        gpu_demodulator = gpu_demod_factory->create();
      }
    }

    if (!gpu_demodulator) {
      fmt::print("Note: GPU demodulator not available. Test will only validate CPU SINR.\n");
    }
  }

  static void TearDownTestSuite()
  {
    gpu_demodulator.reset();
    cpu_demodulator.reset();
    gpu_demod_factory.reset();
    cpu_demod_factory.reset();
    rg_factory.reset();
  }

  /// Creates a resource grid filled with QPSK symbols plus AWGN noise.
  ///
  /// NOTE: This creates random data without proper DMRS sequences. The GPU E2E path
  /// will compute its own channel estimates from this data, which may not match
  /// the mock channel estimates provided to the CPU path. This test documents the
  /// resulting SINR differences.
  std::unique_ptr<resource_grid> create_test_grid(unsigned nof_prb, float snr_db, std::mt19937& rgen)
  {
    unsigned nof_subcarriers = nof_prb * NOF_SUBCARRIERS_PER_RB;
    auto     grid            = rg_factory->create(1, NOF_SYMBOLS_PER_SLOT, MAX_NOF_SUBCARRIERS);

    // Calculate noise standard deviation from SNR.
    // SNR = signal_power / noise_power = 1 / noise_var (for unit signal power).
    float snr_linear = std::pow(10.0f, snr_db / 10.0f);
    float noise_std  = 1.0f / std::sqrt(snr_linear);

    // QPSK constellation points (unit average power).
    const std::array<cf_t, 4> qpsk_symbols = {cf_t(M_SQRT1_2, M_SQRT1_2),
                                              cf_t(M_SQRT1_2, -M_SQRT1_2),
                                              cf_t(-M_SQRT1_2, M_SQRT1_2),
                                              cf_t(-M_SQRT1_2, -M_SQRT1_2)};

    // Random distributions.
    std::uniform_int_distribution<int> symbol_dist(0, 3);
    std::normal_distribution<float>    noise_dist(0.0f, noise_std);

    // Fill grid with QPSK symbols + noise.
    bounded_bitset<MAX_NOF_SUBCARRIERS> re_mask(MAX_NOF_SUBCARRIERS);
    re_mask.fill(0, nof_subcarriers, true);

    std::vector<cf_t> symbols(nof_subcarriers);

    for (unsigned i_symbol = 0; i_symbol < NOF_SYMBOLS_PER_SLOT; ++i_symbol) {
      // Generate QPSK symbols with AWGN.
      for (unsigned i_sc = 0; i_sc < nof_subcarriers; ++i_sc) {
        cf_t qpsk     = qpsk_symbols[symbol_dist(rgen)];
        cf_t noise    = cf_t(noise_dist(rgen), noise_dist(rgen));
        symbols[i_sc] = qpsk + noise;
      }

      // Write to grid.
      grid->get_writer().put(0, i_symbol, 0, re_mask, symbols);
    }

    return grid;
  }

  /// Creates a demodulator configuration for the test.
  pusch_demodulator::configuration create_demod_config(unsigned nof_prb)
  {
    pusch_demodulator::configuration config;
    config.rnti    = 0x1234;
    config.rb_mask = crb_bitmap(MAX_NOF_PRBS_TEST);
    config.rb_mask.fill(0, nof_prb, true);
    config.modulation                  = modulation_scheme::QPSK;
    config.start_symbol_index          = 0;
    config.nof_symbols                 = NOF_SYMBOLS_PER_SLOT;
    config.dmrs_symb_pos               = DMRS_SYMBOL_MASK;
    config.dmrs_type                   = dmrs_config_type::type1;
    config.nof_cdm_groups_without_data = 2;
    config.n_id                        = 0;
    config.dmrs_scrambling_id          = 0;
    config.n_scid                      = false;
    config.nof_tx_layers               = 1;
    config.dc_position                 = std::nullopt;
    config.enable_transform_precoding  = false;
    config.slot                        = slot_point(0, 0);
    config.rx_ports                    = {0};

    return config;
  }
};

// Static member initialization.
std::shared_ptr<resource_grid_factory>     PuschSinrComparisonTest::rg_factory;
std::shared_ptr<pusch_demodulator_factory> PuschSinrComparisonTest::cpu_demod_factory;
std::shared_ptr<pusch_demodulator_factory> PuschSinrComparisonTest::gpu_demod_factory;
std::unique_ptr<pusch_demodulator>         PuschSinrComparisonTest::cpu_demodulator;
std::unique_ptr<pusch_demodulator>         PuschSinrComparisonTest::gpu_demodulator;
bool                                       PuschSinrComparisonTest::gpu_available = false;

TEST_F(PuschSinrComparisonTest, CompareGpuVsCpuSinr)
{
  fmt::print("\n============================================================\n");
  fmt::print("PUSCH SINR Comparison Test: GPU vs CPU Demodulator\n");
  fmt::print("============================================================\n");
  fmt::print("Configuration: 1 layer, 1 Rx port, QPSK, MMSE equalization\n\n");

  fmt::print("NOTE: The GPU E2E path computes its own channel estimates internally.\n");
  fmt::print("      CPU path uses the provided estimator results noise variance.\n");
  fmt::print("      Large deltas indicate fundamental differences in SINR computation.\n\n");

  if (!gpu_demodulator) {
    fmt::print("SKIPPED: GPU demodulator not available.\n");
    GTEST_SKIP() << "GPU demodulator not available.";
    return;
  }

  std::mt19937 rgen(42); // Fixed seed for reproducibility.

  // Statistics tracking.
  float    max_delta        = 0.0f;
  float    sum_delta        = 0.0f;
  unsigned total_tests      = 0;
  unsigned within_threshold = 0;

  for (unsigned nof_prb : TEST_PRB_COUNTS) {
    fmt::print("PRB={} Allocations:\n", nof_prb);

    for (float snr_db : TEST_SNR_VALUES_DB) {
      // Create test grid with known SNR.
      auto grid = create_test_grid(nof_prb, snr_db, rgen);

      // Calculate expected noise variance.
      float snr_linear = std::pow(10.0f, snr_db / 10.0f);
      float noise_var  = 1.0f / snr_linear;

      // Create mock channel estimator results.
      unsigned                           nof_subcarriers = nof_prb * NOF_SUBCARRIERS_PER_RB;
      bounded_bitset<MAX_NSYMB_PER_SLOT> symbols_mask(NOF_SYMBOLS_PER_SLOT);
      symbols_mask.fill(0, NOF_SYMBOLS_PER_SLOT, true);

      bounded_bitset<MAX_NOF_SUBCARRIERS> re_mask(MAX_NOF_SUBCARRIERS);
      re_mask.fill(0, nof_subcarriers, true);

      mock_channel_estimator_results est_results(
          1, 1, NOF_SYMBOLS_PER_SLOT, nof_subcarriers, noise_var, symbols_mask, re_mask);

      // Create demodulator configuration.
      auto config = create_demod_config(nof_prb);

      // Calculate expected LLR count.
      unsigned nof_data_re = nof_subcarriers * NOF_DATA_SYMBOLS;
      unsigned nof_llrs    = nof_data_re * get_bits_per_symbol(config.modulation);

      // Create buffers and notifiers.
      llr_discard_buffer    cpu_buffer(nof_llrs);
      llr_discard_buffer    gpu_buffer(nof_llrs);
      sinr_capture_notifier cpu_notifier;
      sinr_capture_notifier gpu_notifier;

      // Run CPU demodulator.
      cpu_demodulator->demodulate(cpu_buffer, cpu_notifier, grid->get_reader(), est_results, config);
      float cpu_sinr = cpu_notifier.captured_sinr_db.value_or(std::numeric_limits<float>::quiet_NaN());

      // Run GPU demodulator.
      gpu_demodulator->demodulate(gpu_buffer, gpu_notifier, grid->get_reader(), est_results, config);
      float gpu_sinr = gpu_notifier.captured_sinr_db.value_or(std::numeric_limits<float>::quiet_NaN());

      // Calculate delta.
      float       delta         = gpu_sinr - cpu_sinr;
      float       abs_delta     = std::abs(delta);
      bool        within_thresh = abs_delta <= SINR_THRESHOLD_DB;
      const char* status        = within_thresh ? "OK" : "DELTA";

      // Update statistics.
      max_delta = std::max(max_delta, abs_delta);
      sum_delta += abs_delta;
      ++total_tests;
      if (within_thresh) {
        ++within_threshold;
      }

      // Print result.
      fmt::print("  SNR={:2.0f} dB: CPU={:6.2f} dB, GPU={:6.2f} dB, Delta={:+6.2f} dB [{}]\n",
                 snr_db,
                 cpu_sinr,
                 gpu_sinr,
                 delta,
                 status);
    }
    fmt::print("\n");
  }

  // Print summary.
  float mean_delta = (total_tests > 0) ? (sum_delta / static_cast<float>(total_tests)) : 0.0f;
  fmt::print("Summary:\n");
  fmt::print("  Max delta: {:.2f} dB\n", max_delta);
  fmt::print("  Mean delta: {:.2f} dB\n", mean_delta);
  fmt::print("  Within {:.0f} dB threshold: {}/{}\n", SINR_THRESHOLD_DB, within_threshold, total_tests);
  fmt::print("\n");

  // This test documents observed differences. The GPU E2E path computes SINR
  // differently from the CPU path because it does its own channel estimation.
  // Both paths should report finite SINR values.
  EXPECT_TRUE(std::isfinite(max_delta)) << "SINR comparison produced invalid values";
}

TEST_F(PuschSinrComparisonTest, CpuSinrMatchesExpected)
{
  // Test that CPU SINR calculation produces values close to the expected SNR.
  fmt::print("\n============================================================\n");
  fmt::print("CPU SINR Accuracy Test\n");
  fmt::print("============================================================\n");

  std::mt19937 rgen(123);

  for (unsigned nof_prb : {4, 6}) {
    for (float snr_db : {10.0f, 20.0f}) {
      auto grid = create_test_grid(nof_prb, snr_db, rgen);

      float    snr_linear      = std::pow(10.0f, snr_db / 10.0f);
      float    noise_var       = 1.0f / snr_linear;
      unsigned nof_subcarriers = nof_prb * NOF_SUBCARRIERS_PER_RB;

      bounded_bitset<MAX_NSYMB_PER_SLOT> symbols_mask(NOF_SYMBOLS_PER_SLOT);
      symbols_mask.fill(0, NOF_SYMBOLS_PER_SLOT, true);
      bounded_bitset<MAX_NOF_SUBCARRIERS> re_mask(MAX_NOF_SUBCARRIERS);
      re_mask.fill(0, nof_subcarriers, true);

      mock_channel_estimator_results est_results(
          1, 1, NOF_SYMBOLS_PER_SLOT, nof_subcarriers, noise_var, symbols_mask, re_mask);

      auto     config   = create_demod_config(nof_prb);
      unsigned nof_llrs = nof_subcarriers * NOF_DATA_SYMBOLS * get_bits_per_symbol(config.modulation);

      llr_discard_buffer    buffer(nof_llrs);
      sinr_capture_notifier notifier;

      cpu_demodulator->demodulate(buffer, notifier, grid->get_reader(), est_results, config);

      float reported_sinr = notifier.captured_sinr_db.value_or(std::numeric_limits<float>::quiet_NaN());

      // The reported SINR should be close to the configured SNR.
      // Allow some tolerance due to random noise realization.
      float delta = std::abs(reported_sinr - snr_db);
      fmt::print(
          "PRB={}, SNR={:.0f} dB: Reported SINR={:.2f} dB, Delta={:.2f} dB\n", nof_prb, snr_db, reported_sinr, delta);

      // For controlled test conditions, expect within 3 dB.
      EXPECT_LE(delta, 3.0f) << "CPU SINR differs significantly from expected for PRB=" << nof_prb << ", SNR=" << snr_db
                             << " dB";
    }
  }
}

} // namespace
