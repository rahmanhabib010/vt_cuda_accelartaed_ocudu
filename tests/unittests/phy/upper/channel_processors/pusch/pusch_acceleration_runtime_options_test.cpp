// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief Unit tests for PUSCH acceleration runtime decision helpers.

#include "pusch_acceleration_runtime_options.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <optional>
#include <string>

using namespace ocudu;

namespace {

class scoped_env_var
{
public:
  explicit scoped_env_var(const char* name_) : name(name_)
  {
    const char* value = std::getenv(name.c_str());
    if (value != nullptr) {
      original_value = value;
    }
  }

  ~scoped_env_var()
  {
    if (original_value.has_value()) {
      (void)setenv(name.c_str(), original_value->c_str(), 1);
    } else {
      (void)unsetenv(name.c_str());
    }
  }

  bool set(const char* value) { return setenv(name.c_str(), value, 1) == 0; }

  bool unset() { return unsetenv(name.c_str()) == 0; }

private:
  std::string                name;
  std::optional<std::string> original_value;
};

} // namespace

TEST(pusch_acceleration_runtime_options_test, sch_grant_thresholds_are_disabled_by_zero_values)
{
  pusch_acceleration_thresholds thresholds = {};

  EXPECT_FALSE(pusch_acceleration_prefers_cpu_for_sch_grant(true, 1, 1, thresholds));
  EXPECT_FALSE(pusch_acceleration_prefers_cpu_for_sch_grant(true, 273, 100000, thresholds));
}

TEST(pusch_acceleration_runtime_options_test, grants_without_sch_never_prefer_cpu_for_sch_thresholds)
{
  pusch_acceleration_thresholds thresholds = {.min_prb = 100, .min_ulsch_bits = 100000};

  EXPECT_FALSE(pusch_acceleration_prefers_cpu_for_sch_grant(false, 1, 1, thresholds));
}

TEST(pusch_acceleration_runtime_options_test, sch_grant_prefers_cpu_below_prb_threshold)
{
  pusch_acceleration_thresholds thresholds = {.min_prb = 24, .min_ulsch_bits = 0};

  EXPECT_TRUE(pusch_acceleration_prefers_cpu_for_sch_grant(true, 23, 100000, thresholds));
  EXPECT_FALSE(pusch_acceleration_prefers_cpu_for_sch_grant(true, 24, 1, thresholds));
}

TEST(pusch_acceleration_runtime_options_test, sch_grant_prefers_cpu_below_ulsch_bits_threshold)
{
  pusch_acceleration_thresholds thresholds = {.min_prb = 0, .min_ulsch_bits = 4096};

  EXPECT_TRUE(pusch_acceleration_prefers_cpu_for_sch_grant(true, 273, 4095, thresholds));
  EXPECT_FALSE(pusch_acceleration_prefers_cpu_for_sch_grant(true, 1, 4096, thresholds));
}

TEST(pusch_acceleration_runtime_options_test, demodulator_threshold_uses_prb_only)
{
  EXPECT_FALSE(pusch_acceleration_prefers_cpu_for_demodulator_grant(1, 0));
  EXPECT_TRUE(pusch_acceleration_prefers_cpu_for_demodulator_grant(5, 6));
  EXPECT_FALSE(pusch_acceleration_prefers_cpu_for_demodulator_grant(6, 6));
}

TEST(pusch_acceleration_runtime_options_test, reads_thresholds_from_environment)
{
  scoped_env_var min_prb("OCUDU_PUSCH_ACCELERATION_MIN_RB");
  scoped_env_var min_bits("OCUDU_PUSCH_ACCELERATION_MIN_ULSCH_BITS");

  ASSERT_TRUE(min_prb.set("17"));
  ASSERT_TRUE(min_bits.set("8192"));
  pusch_acceleration_thresholds thresholds = pusch_acceleration_read_thresholds();
  EXPECT_EQ(thresholds.min_prb, 17U);
  EXPECT_EQ(thresholds.min_ulsch_bits, 8192U);

  ASSERT_TRUE(min_prb.set("-1"));
  ASSERT_TRUE(min_bits.set("invalid"));
  thresholds = pusch_acceleration_read_thresholds();
  EXPECT_EQ(thresholds.min_prb, 0U);
  EXPECT_EQ(thresholds.min_ulsch_bits, 0U);
}

TEST(pusch_acceleration_runtime_options_test, device_uci_feature_flag_defaults_off_and_accepts_true_values)
{
  scoped_env_var device_uci("OCUDU_PUSCH_ENABLE_ACCELERATED_UCI");

  ASSERT_TRUE(device_uci.unset());
  EXPECT_FALSE(pusch_acceleration_device_uci_enabled());

  ASSERT_TRUE(device_uci.set("yes"));
  EXPECT_TRUE(pusch_acceleration_device_uci_enabled());

  ASSERT_TRUE(device_uci.set("off"));
  EXPECT_FALSE(pusch_acceleration_device_uci_enabled());
}
