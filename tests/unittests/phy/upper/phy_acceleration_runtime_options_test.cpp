// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief Unit tests for CUDA PHY acceleration runtime option parsing.

#include "phy_acceleration_runtime_options.h"
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

TEST(phy_acceleration_runtime_options_test, boolean_flag_accepts_expected_true_values)
{
  scoped_env_var env("OCUDU_TEST_ACCELERATION_FLAG");

  for (const char* value : {"1", "true", "True", "TRUE", "on", "ON", "yes", "YES"}) {
    ASSERT_TRUE(env.set(value));
    EXPECT_TRUE(phy_acceleration_env_flag_enabled("OCUDU_TEST_ACCELERATION_FLAG", false)) << value;
  }
}

TEST(phy_acceleration_runtime_options_test, boolean_flag_accepts_expected_false_values)
{
  scoped_env_var env("OCUDU_TEST_ACCELERATION_FLAG");

  for (const char* value : {"0", "false", "False", "FALSE", "off", "OFF", "no", "NO"}) {
    ASSERT_TRUE(env.set(value));
    EXPECT_FALSE(phy_acceleration_env_flag_enabled("OCUDU_TEST_ACCELERATION_FLAG", true)) << value;
  }
}

TEST(phy_acceleration_runtime_options_test, boolean_flag_uses_default_for_unset_empty_and_unknown_values)
{
  scoped_env_var env("OCUDU_TEST_ACCELERATION_FLAG");

  ASSERT_TRUE(env.unset());
  EXPECT_TRUE(phy_acceleration_env_flag_enabled("OCUDU_TEST_ACCELERATION_FLAG", true));
  EXPECT_FALSE(phy_acceleration_env_flag_enabled("OCUDU_TEST_ACCELERATION_FLAG", false));

  ASSERT_TRUE(env.set(""));
  EXPECT_TRUE(phy_acceleration_env_flag_enabled("OCUDU_TEST_ACCELERATION_FLAG", true));
  EXPECT_FALSE(phy_acceleration_env_flag_enabled("OCUDU_TEST_ACCELERATION_FLAG", false));

  ASSERT_TRUE(env.set("maybe"));
  EXPECT_TRUE(phy_acceleration_env_flag_enabled("OCUDU_TEST_ACCELERATION_FLAG", true));
  EXPECT_FALSE(phy_acceleration_env_flag_enabled("OCUDU_TEST_ACCELERATION_FLAG", false));
}

TEST(phy_acceleration_runtime_options_test, unsigned_option_parses_decimal_values)
{
  scoped_env_var env("OCUDU_TEST_ACCELERATION_UNSIGNED");

  ASSERT_TRUE(env.set("0"));
  EXPECT_EQ(phy_acceleration_env_unsigned("OCUDU_TEST_ACCELERATION_UNSIGNED", 77), 0U);

  ASSERT_TRUE(env.set("42"));
  EXPECT_EQ(phy_acceleration_env_unsigned("OCUDU_TEST_ACCELERATION_UNSIGNED", 77), 42U);
}

TEST(phy_acceleration_runtime_options_test, unsigned_option_uses_default_for_invalid_values)
{
  scoped_env_var env("OCUDU_TEST_ACCELERATION_UNSIGNED");

  ASSERT_TRUE(env.unset());
  EXPECT_EQ(phy_acceleration_env_unsigned("OCUDU_TEST_ACCELERATION_UNSIGNED", 77), 77U);

  ASSERT_TRUE(env.set(""));
  EXPECT_EQ(phy_acceleration_env_unsigned("OCUDU_TEST_ACCELERATION_UNSIGNED", 77), 77U);

  ASSERT_TRUE(env.set("12x"));
  EXPECT_EQ(phy_acceleration_env_unsigned("OCUDU_TEST_ACCELERATION_UNSIGNED", 77), 77U);

  ASSERT_TRUE(env.set("-1"));
  EXPECT_EQ(phy_acceleration_env_unsigned("OCUDU_TEST_ACCELERATION_UNSIGNED", 77), 77U);

  ASSERT_TRUE(env.set("4294967296"));
  EXPECT_EQ(phy_acceleration_env_unsigned("OCUDU_TEST_ACCELERATION_UNSIGNED", 77), 77U);
}

TEST(phy_acceleration_runtime_options_test, cuda_visible_grid_mode_prefers_direction_specific_override)
{
  scoped_env_var generic_env("OCUDU_CUDA_VISIBLE_GRID");
  scoped_env_var uplink_env("OCUDU_UL_CUDA_VISIBLE_GRID");

  ASSERT_TRUE(generic_env.set("managed"));
  ASSERT_TRUE(uplink_env.unset());
  EXPECT_STREQ(phy_acceleration_cuda_visible_grid_mode("OCUDU_UL_CUDA_VISIBLE_GRID"), "managed");
  EXPECT_TRUE(phy_acceleration_cuda_visible_grid_managed_requested("OCUDU_UL_CUDA_VISIBLE_GRID"));

  ASSERT_TRUE(uplink_env.set("pinned"));
  EXPECT_STREQ(phy_acceleration_cuda_visible_grid_mode("OCUDU_UL_CUDA_VISIBLE_GRID"), "pinned");
  EXPECT_FALSE(phy_acceleration_cuda_visible_grid_managed_requested("OCUDU_UL_CUDA_VISIBLE_GRID"));
}

TEST(phy_acceleration_runtime_options_test, cuda_visible_grid_mode_reports_absent_and_exact_managed_mode)
{
  scoped_env_var generic_env("OCUDU_CUDA_VISIBLE_GRID");
  scoped_env_var downlink_env("OCUDU_DL_CUDA_VISIBLE_GRID");

  ASSERT_TRUE(generic_env.unset());
  ASSERT_TRUE(downlink_env.unset());
  EXPECT_EQ(phy_acceleration_cuda_visible_grid_mode("OCUDU_DL_CUDA_VISIBLE_GRID"), nullptr);
  EXPECT_FALSE(phy_acceleration_cuda_visible_grid_managed_requested("OCUDU_DL_CUDA_VISIBLE_GRID"));

  ASSERT_TRUE(generic_env.set("Managed"));
  EXPECT_FALSE(phy_acceleration_cuda_visible_grid_managed_requested());

  ASSERT_TRUE(generic_env.set("managed"));
  EXPECT_TRUE(phy_acceleration_cuda_visible_grid_managed_requested());
}

TEST(phy_acceleration_runtime_options_test, pdsch_auto_enable_discrete_override_reports_configured_state)
{
  scoped_env_var env("OCUDU_PDSCH_AUTO_ENABLE_DISCRETE");

  ASSERT_TRUE(env.unset());
  EXPECT_FALSE(phy_acceleration_pdsch_auto_enable_discrete_configured());
  EXPECT_FALSE(phy_acceleration_pdsch_auto_enable_discrete_requested());

  ASSERT_TRUE(env.set(""));
  EXPECT_FALSE(phy_acceleration_pdsch_auto_enable_discrete_configured());
  EXPECT_FALSE(phy_acceleration_pdsch_auto_enable_discrete_requested());

  ASSERT_TRUE(env.set("1"));
  EXPECT_TRUE(phy_acceleration_pdsch_auto_enable_discrete_configured());
  EXPECT_TRUE(phy_acceleration_pdsch_auto_enable_discrete_requested());

  ASSERT_TRUE(env.set("0"));
  EXPECT_TRUE(phy_acceleration_pdsch_auto_enable_discrete_configured());
  EXPECT_FALSE(phy_acceleration_pdsch_auto_enable_discrete_requested());
}
