// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief Unit tests for PDSCH CUDA grid output strategy selection.

#include "../support/resource_grid_test_doubles.h"
#include "pdsch_grid_output_strategy.h"
#include "resource_grid_cuda_visible_impl.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/phy/support/support_factories.h"
#include <cstdlib>
#include <cuda_runtime.h>
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

testing::AssertionResult cuda_runtime_available()
{
  int         device_count = 0;
  cudaError_t status       = cudaGetDeviceCount(&device_count);
  if ((status != cudaSuccess) || (device_count == 0)) {
    return testing::AssertionFailure() << "CUDA runtime device is not available.";
  }
  return testing::AssertionSuccess();
}

resource_grid_context make_context()
{
  return {.slot = slot_point(0, 0), .sector = 0};
}

ocudulog::basic_logger& test_logger()
{
  ocudulog::basic_logger& logger = ocudulog::fetch_basic_logger("TEST", false);
  logger.set_level(ocudulog::basic_levels::none);
  return logger;
}

} // namespace

TEST(pdsch_grid_output_strategy_test, returns_host_writer_when_device_grid_writer_is_disabled)
{
  std::unique_ptr<resource_grid> grid = create_resource_grid_factory()->create(1, 14, 24);
  ASSERT_NE(grid, nullptr);

  shared_resource_grid_spy shared_grid_spy(*grid);
  shared_resource_grid     shared_grid = shared_grid_spy.get_grid();
  resource_grid_writer*    host_writer = &shared_grid.get_writer();

  std::unique_ptr<pdsch_grid_output_strategy> strategy = create_pdsch_grid_output_strategy(test_logger(), false);
  ASSERT_NE(strategy, nullptr);
  strategy->configure(make_context(), shared_grid);

  EXPECT_EQ(&strategy->get_pdsch_writer(shared_grid), host_writer);
  strategy->before_send_grid();
}

TEST(pdsch_grid_output_strategy_test, uses_sidecar_device_writer_for_host_resource_grid)
{
  testing::AssertionResult cuda_available = cuda_runtime_available();
  if (!cuda_available) {
    GTEST_SKIP() << cuda_available.message();
  }

  std::unique_ptr<resource_grid> grid = create_resource_grid_factory()->create(1, 14, 24);
  ASSERT_NE(grid, nullptr);

  shared_resource_grid_spy shared_grid_spy(*grid);
  shared_resource_grid     shared_grid = shared_grid_spy.get_grid();
  resource_grid_writer*    host_writer = &shared_grid.get_writer();

  std::unique_ptr<pdsch_grid_output_strategy> strategy = create_pdsch_grid_output_strategy(test_logger(), true);
  ASSERT_NE(strategy, nullptr);
  strategy->configure(make_context(), shared_grid);

  resource_grid_writer& selected_writer = strategy->get_pdsch_writer(shared_grid);
  EXPECT_NE(&selected_writer, host_writer);
  EXPECT_TRUE(selected_writer.supports_device_grid_mapping());
  strategy->before_send_grid();
}

TEST(pdsch_grid_output_strategy_test, uses_direct_writer_for_cuda_visible_grid_by_default)
{
  testing::AssertionResult cuda_available = cuda_runtime_available();
  if (!cuda_available) {
    GTEST_SKIP() << cuda_available.message();
  }

  scoped_env_var direct_grid("OCUDU_PDSCH_DIRECT_DEVICE_GRID");
  ASSERT_TRUE(direct_grid.unset());

  cuda_visible_resource_grid grid(1, 14, 24);
  if (!grid.is_valid()) {
    GTEST_SKIP() << "CUDA-visible resource grid is not available.";
  }

  shared_resource_grid_spy shared_grid_spy(grid);
  shared_resource_grid     shared_grid = shared_grid_spy.get_grid();
  resource_grid_writer*    host_writer = &shared_grid.get_writer();
  ASSERT_TRUE(host_writer->supports_device_grid_mapping());

  std::unique_ptr<pdsch_grid_output_strategy> strategy = create_pdsch_grid_output_strategy(test_logger(), true);
  ASSERT_NE(strategy, nullptr);
  strategy->configure(make_context(), shared_grid);

  EXPECT_EQ(&strategy->get_pdsch_writer(shared_grid), host_writer);
  strategy->before_send_grid();
}

TEST(pdsch_grid_output_strategy_test, direct_cuda_visible_grid_path_can_be_disabled)
{
  testing::AssertionResult cuda_available = cuda_runtime_available();
  if (!cuda_available) {
    GTEST_SKIP() << cuda_available.message();
  }

  scoped_env_var direct_grid("OCUDU_PDSCH_DIRECT_DEVICE_GRID");
  ASSERT_TRUE(direct_grid.set("0"));

  cuda_visible_resource_grid grid(1, 14, 24);
  if (!grid.is_valid()) {
    GTEST_SKIP() << "CUDA-visible resource grid is not available.";
  }

  shared_resource_grid_spy shared_grid_spy(grid);
  shared_resource_grid     shared_grid = shared_grid_spy.get_grid();
  resource_grid_writer*    host_writer = &shared_grid.get_writer();
  ASSERT_TRUE(host_writer->supports_device_grid_mapping());

  std::unique_ptr<pdsch_grid_output_strategy> strategy = create_pdsch_grid_output_strategy(test_logger(), true);
  ASSERT_NE(strategy, nullptr);
  strategy->configure(make_context(), shared_grid);

  resource_grid_writer& selected_writer = strategy->get_pdsch_writer(shared_grid);
  EXPECT_NE(&selected_writer, host_writer);
  EXPECT_TRUE(selected_writer.supports_device_grid_mapping());
  strategy->before_send_grid();
}
