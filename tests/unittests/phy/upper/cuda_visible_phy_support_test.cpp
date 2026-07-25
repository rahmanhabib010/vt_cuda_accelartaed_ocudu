// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief Unit tests for CUDA-visible PHY support buffers.

#include "phy_acceleration_prach_buffer_factory.h"
#include "phy_acceleration_resource_grid_factory.h"
#include "ocudu/phy/support/resource_grid.h"
#include "ocudu/phy/support/resource_grid_reader.h"
#include "ocudu/phy/support/resource_grid_writer.h"
#include "ocudu/phy/support/support_factories.h"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>

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

class cuda_stream_guard
{
public:
  bool create() { return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess; }

  ~cuda_stream_guard()
  {
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
  }

  cudaStream_t get() const { return stream; }

private:
  cudaStream_t stream = nullptr;
};

testing::AssertionResult cuda_managed_memory_available()
{
  int         device_count = 0;
  cudaError_t status       = cudaGetDeviceCount(&device_count);
  if ((status != cudaSuccess) || (device_count == 0)) {
    return testing::AssertionFailure() << "CUDA runtime device is not available.";
  }

  int device_id = 0;
  status        = cudaGetDevice(&device_id);
  if (status != cudaSuccess) {
    return testing::AssertionFailure() << "Failed to query active CUDA device: " << cudaGetErrorString(status);
  }

  int managed_memory = 0;
  status             = cudaDeviceGetAttribute(&managed_memory, cudaDevAttrManagedMemory, device_id);
  if (status != cudaSuccess) {
    return testing::AssertionFailure() << "Failed to query CUDA managed-memory support: " << cudaGetErrorString(status);
  }
  if (managed_memory == 0) {
    return testing::AssertionFailure() << "CUDA managed memory is not supported by the active device.";
  }

  return testing::AssertionSuccess();
}

std::vector<cbf16_t> generate_grid_payload(unsigned nof_ports, unsigned nof_symbols, unsigned nof_subc)
{
  std::vector<cbf16_t> payload(static_cast<size_t>(nof_ports) * nof_symbols * nof_subc);
  for (size_t i = 0; i != payload.size(); ++i) {
    payload[i] = cbf16_t(0.003F * static_cast<float>(i + 1), -0.002F * static_cast<float>(i + 3));
  }
  return payload;
}

std::vector<cbf16_t> generate_symbol_payload(unsigned nof_samples, float scale)
{
  std::vector<cbf16_t> payload(nof_samples);
  for (unsigned i = 0; i != nof_samples; ++i) {
    payload[i] = cbf16_t(scale * static_cast<float>(i + 1), -scale * static_cast<float>(i + 7));
  }
  return payload;
}

void expect_equal(span<const cbf16_t> actual, span<const cbf16_t> expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (unsigned i = 0; i != actual.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "sample=" << i;
  }
}

} // namespace

TEST(cuda_visible_phy_support_test, resource_grid_factory_can_force_managed_cuda_visible_grid)
{
  testing::AssertionResult cuda_available = cuda_managed_memory_available();
  if (!cuda_available) {
    GTEST_SKIP() << cuda_available.message();
  }

  scoped_env_var grid_mode("OCUDU_UL_CUDA_VISIBLE_GRID");
  ASSERT_TRUE(grid_mode.set("managed"));

  std::shared_ptr<resource_grid_factory> fallback_factory = create_resource_grid_factory();
  std::shared_ptr<resource_grid_factory> factory          = create_phy_acceleration_resource_grid_factory(
      fallback_factory, phy_acceleration_resource_grid_direction::uplink, true);
  ASSERT_NE(factory, nullptr);

  std::unique_ptr<resource_grid> grid = factory->create(2, 4, 24);
  ASSERT_NE(grid, nullptr);

  resource_grid_writer&       writer = grid->get_writer();
  const resource_grid_reader& reader = grid->get_reader();
  EXPECT_TRUE(writer.supports_device_grid_mapping());
  EXPECT_TRUE(writer.device_grid_mapping_aliases_host_grid());
  EXPECT_NE(writer.get_device_grid_bf16(), nullptr);
  EXPECT_TRUE(reader.supports_device_grid_reading());
  EXPECT_EQ(reader.get_device_grid_cbf16(), writer.get_device_grid_bf16());
}

TEST(cuda_visible_phy_support_test, resource_grid_factory_uses_fallback_when_accelerated_grid_is_disabled)
{
  scoped_env_var grid_mode("OCUDU_UL_CUDA_VISIBLE_GRID");
  ASSERT_TRUE(grid_mode.set("managed"));

  std::shared_ptr<resource_grid_factory> fallback_factory = create_resource_grid_factory();
  std::shared_ptr<resource_grid_factory> factory          = create_phy_acceleration_resource_grid_factory(
      fallback_factory, phy_acceleration_resource_grid_direction::uplink, false);
  ASSERT_EQ(factory, fallback_factory);

  std::unique_ptr<resource_grid> grid = factory->create(2, 4, 24);
  ASSERT_NE(grid, nullptr);
  EXPECT_FALSE(grid->get_writer().supports_device_grid_mapping());
  EXPECT_FALSE(grid->get_reader().supports_device_grid_reading());
}

TEST(cuda_visible_phy_support_test, resource_grid_factory_uses_fallback_when_pinned_mode_is_requested)
{
  scoped_env_var grid_mode("OCUDU_UL_CUDA_VISIBLE_GRID");
  ASSERT_TRUE(grid_mode.set("pinned"));

  std::shared_ptr<resource_grid_factory> factory = create_phy_acceleration_resource_grid_factory(
      create_resource_grid_factory(), phy_acceleration_resource_grid_direction::uplink, true);
  ASSERT_NE(factory, nullptr);

  std::unique_ptr<resource_grid> grid = factory->create(2, 4, 24);
  ASSERT_NE(grid, nullptr);
  EXPECT_FALSE(grid->get_writer().supports_device_grid_mapping());
  EXPECT_FALSE(grid->get_reader().supports_device_grid_reading());
}

TEST(cuda_visible_phy_support_test, resource_grid_device_mapping_is_visible_to_host_reader)
{
  testing::AssertionResult cuda_available = cuda_managed_memory_available();
  if (!cuda_available) {
    GTEST_SKIP() << cuda_available.message();
  }

  scoped_env_var grid_mode("OCUDU_UL_CUDA_VISIBLE_GRID");
  ASSERT_TRUE(grid_mode.set("managed"));

  static constexpr unsigned nof_ports   = 2;
  static constexpr unsigned nof_symbols = 4;
  static constexpr unsigned nof_subc    = 24;
  static constexpr unsigned port        = 1;
  static constexpr unsigned symbol      = 2;

  auto factory = create_phy_acceleration_resource_grid_factory(
      create_resource_grid_factory(), phy_acceleration_resource_grid_direction::uplink, true);
  std::unique_ptr<resource_grid> grid = factory->create(nof_ports, nof_symbols, nof_subc);
  ASSERT_NE(grid, nullptr);

  resource_grid_writer&       writer = grid->get_writer();
  const resource_grid_reader& reader = grid->get_reader();
  ASSERT_TRUE(writer.supports_device_grid_mapping());

  cuda_stream_guard stream;
  ASSERT_TRUE(stream.create());

  std::vector<cbf16_t> payload = generate_grid_payload(nof_ports, nof_symbols, nof_subc);
  ASSERT_TRUE(writer.prepare_device_grid_mapping(stream.get()));
  ASSERT_EQ(cudaMemcpyAsync(writer.get_device_grid_bf16(),
                            payload.data(),
                            payload.size() * sizeof(cbf16_t),
                            cudaMemcpyHostToDevice,
                            stream.get()),
            cudaSuccess);
  ASSERT_TRUE(writer.on_device_grid_mapping_enqueued(stream.get()));

  const size_t offset = (static_cast<size_t>(port) * nof_symbols + symbol) * nof_subc;
  expect_equal(reader.get_view(port, symbol), span<const cbf16_t>(payload.data() + offset, nof_subc));

  grid->set_all_zero();
  ASSERT_TRUE(reader.is_empty());
  span<const cbf16_t> cleared = reader.get_view(port, symbol);
  for (cbf16_t value : cleared) {
    EXPECT_EQ(value, cbf16_t());
  }
}

TEST(cuda_visible_phy_support_test, resource_grid_host_writer_is_visible_to_device_reader)
{
  testing::AssertionResult cuda_available = cuda_managed_memory_available();
  if (!cuda_available) {
    GTEST_SKIP() << cuda_available.message();
  }

  scoped_env_var grid_mode("OCUDU_UL_CUDA_VISIBLE_GRID");
  ASSERT_TRUE(grid_mode.set("managed"));

  static constexpr unsigned nof_ports   = 2;
  static constexpr unsigned nof_symbols = 4;
  static constexpr unsigned nof_subc    = 24;
  static constexpr unsigned port        = 0;
  static constexpr unsigned symbol      = 3;

  auto factory = create_phy_acceleration_resource_grid_factory(
      create_resource_grid_factory(), phy_acceleration_resource_grid_direction::uplink, true);
  std::unique_ptr<resource_grid> grid = factory->create(nof_ports, nof_symbols, nof_subc);
  ASSERT_NE(grid, nullptr);

  resource_grid_writer&       writer = grid->get_writer();
  const resource_grid_reader& reader = grid->get_reader();
  ASSERT_TRUE(reader.supports_device_grid_reading());

  std::vector<cbf16_t> expected = generate_symbol_payload(nof_subc, 0.005F);
  writer.put(port, symbol, 0, 1, span<const cbf16_t>(expected));

  cuda_stream_guard stream;
  ASSERT_TRUE(stream.create());

  ASSERT_TRUE(reader.prepare_device_grid_reading(stream.get()));
  std::vector<cbf16_t> copied(expected.size());
  const cbf16_t*       device_grid = static_cast<const cbf16_t*>(reader.get_device_grid_cbf16());
  ASSERT_NE(device_grid, nullptr);

  const size_t offset = (static_cast<size_t>(port) * nof_symbols + symbol) * nof_subc;
  ASSERT_EQ(
      cudaMemcpyAsync(
          copied.data(), device_grid + offset, copied.size() * sizeof(cbf16_t), cudaMemcpyDeviceToHost, stream.get()),
      cudaSuccess);
  ASSERT_TRUE(reader.on_device_grid_reading_enqueued(stream.get()));
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  ASSERT_TRUE(reader.synchronize_device_grid_reading());

  EXPECT_EQ(copied, expected);
}

TEST(cuda_visible_phy_support_test, resource_grid_repeated_device_mapping_and_reading_cycles_are_ordered)
{
  testing::AssertionResult cuda_available = cuda_managed_memory_available();
  if (!cuda_available) {
    GTEST_SKIP() << cuda_available.message();
  }

  scoped_env_var grid_mode("OCUDU_UL_CUDA_VISIBLE_GRID");
  ASSERT_TRUE(grid_mode.set("managed"));

  static constexpr unsigned nof_ports   = 2;
  static constexpr unsigned nof_symbols = 4;
  static constexpr unsigned nof_subc    = 24;

  auto factory = create_phy_acceleration_resource_grid_factory(
      create_resource_grid_factory(), phy_acceleration_resource_grid_direction::uplink, true);
  std::unique_ptr<resource_grid> grid = factory->create(nof_ports, nof_symbols, nof_subc);
  ASSERT_NE(grid, nullptr);

  resource_grid_writer&       writer = grid->get_writer();
  const resource_grid_reader& reader = grid->get_reader();
  ASSERT_TRUE(writer.supports_device_grid_mapping());
  ASSERT_TRUE(reader.supports_device_grid_reading());

  cuda_stream_guard stream;
  ASSERT_TRUE(stream.create());

  for (unsigned iteration = 0; iteration != 12; ++iteration) {
    SCOPED_TRACE("iteration=" + std::to_string(iteration));

    std::vector<cbf16_t> payload = generate_grid_payload(nof_ports, nof_symbols, nof_subc);
    for (cbf16_t& sample : payload) {
      cf_t value = to_cf(sample);
      sample     = cbf16_t(value.real() + 0.001F * static_cast<float>(iteration),
                       value.imag() - 0.001F * static_cast<float>(iteration));
    }

    ASSERT_TRUE(writer.prepare_device_grid_mapping(stream.get()));
    ASSERT_EQ(cudaMemcpyAsync(writer.get_device_grid_bf16(),
                              payload.data(),
                              payload.size() * sizeof(cbf16_t),
                              cudaMemcpyHostToDevice,
                              stream.get()),
              cudaSuccess);
    ASSERT_TRUE(writer.on_device_grid_mapping_enqueued(stream.get()));

    const unsigned port   = iteration % nof_ports;
    const unsigned symbol = iteration % nof_symbols;
    const size_t   offset = (static_cast<size_t>(port) * nof_symbols + symbol) * nof_subc;
    expect_equal(reader.get_view(port, symbol), span<const cbf16_t>(payload.data() + offset, nof_subc));

    ASSERT_TRUE(reader.prepare_device_grid_reading(stream.get()));
    std::vector<cbf16_t> copied(nof_subc);
    const cbf16_t*       device_grid = static_cast<const cbf16_t*>(reader.get_device_grid_cbf16());
    ASSERT_NE(device_grid, nullptr);
    ASSERT_EQ(
        cudaMemcpyAsync(
            copied.data(), device_grid + offset, copied.size() * sizeof(cbf16_t), cudaMemcpyDeviceToHost, stream.get()),
        cudaSuccess);
    ASSERT_TRUE(reader.on_device_grid_reading_enqueued(stream.get()));
    ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
    ASSERT_TRUE(reader.synchronize_device_grid_reading());
    EXPECT_EQ(copied, std::vector<cbf16_t>(payload.begin() + offset, payload.begin() + offset + nof_subc));

    ASSERT_TRUE(writer.prepare_device_grid_mapping(stream.get()));
    ASSERT_TRUE(writer.cancel_device_grid_mapping());
  }
}

TEST(cuda_visible_phy_support_test, prach_buffer_factory_uses_fallback_when_disabled_or_pinned)
{
  scoped_env_var buffer_mode("OCUDU_UL_CUDA_VISIBLE_PRACH_BUFFER");

  ASSERT_TRUE(buffer_mode.set("managed"));
  std::unique_ptr<prach_buffer> disabled_buffer = create_phy_acceleration_prach_buffer_short(2, 2, 2, false);
  ASSERT_NE(disabled_buffer, nullptr);
  EXPECT_FALSE(disabled_buffer->supports_device_prach_buffer_mapping());
  EXPECT_FALSE(disabled_buffer->supports_device_prach_buffer_reading());

  ASSERT_TRUE(buffer_mode.set("pinned"));
  std::unique_ptr<prach_buffer> pinned_buffer = create_phy_acceleration_prach_buffer_short(2, 2, 2, true);
  ASSERT_NE(pinned_buffer, nullptr);
  EXPECT_FALSE(pinned_buffer->supports_device_prach_buffer_mapping());
  EXPECT_FALSE(pinned_buffer->supports_device_prach_buffer_reading());
}

TEST(cuda_visible_phy_support_test, prach_buffer_device_mapping_is_visible_to_host_reader)
{
  testing::AssertionResult cuda_available = cuda_managed_memory_available();
  if (!cuda_available) {
    GTEST_SKIP() << cuda_available.message();
  }

  scoped_env_var buffer_mode("OCUDU_UL_CUDA_VISIBLE_PRACH_BUFFER");
  ASSERT_TRUE(buffer_mode.set("managed"));

  std::unique_ptr<prach_buffer> buffer = create_phy_acceleration_prach_buffer_short(2, 2, 2, true);
  ASSERT_NE(buffer, nullptr);
  ASSERT_TRUE(buffer->supports_device_prach_buffer_mapping());
  ASSERT_TRUE(buffer->supports_device_prach_buffer_reading());
  ASSERT_NE(buffer->get_device_prach_buffer_cbf16(), nullptr);

  static constexpr unsigned port        = 1;
  static constexpr unsigned td_occasion = 1;
  static constexpr unsigned fd_occasion = 0;
  static constexpr unsigned symbol      = 3;

  cuda_stream_guard stream;
  ASSERT_TRUE(stream.create());

  std::vector<cbf16_t> expected      = generate_symbol_payload(buffer->get_sequence_length(), 0.011F);
  const unsigned       offset        = buffer->get_device_prach_symbol_offset(port, td_occasion, fd_occasion, symbol);
  cbf16_t*             device_buffer = static_cast<cbf16_t*>(buffer->get_device_prach_buffer_cbf16());

  ASSERT_TRUE(buffer->prepare_device_prach_buffer_mapping(stream.get()));
  ASSERT_EQ(cudaMemcpyAsync(device_buffer + offset,
                            expected.data(),
                            expected.size() * sizeof(cbf16_t),
                            cudaMemcpyHostToDevice,
                            stream.get()),
            cudaSuccess);
  ASSERT_TRUE(buffer->on_device_prach_buffer_mapping_enqueued(stream.get()));

  expect_equal(buffer->get_symbol(port, td_occasion, fd_occasion, symbol), span<const cbf16_t>(expected));
}

TEST(cuda_visible_phy_support_test, prach_buffer_host_writer_is_visible_to_device_reader)
{
  testing::AssertionResult cuda_available = cuda_managed_memory_available();
  if (!cuda_available) {
    GTEST_SKIP() << cuda_available.message();
  }

  scoped_env_var buffer_mode("OCUDU_UL_CUDA_VISIBLE_PRACH_BUFFER");
  ASSERT_TRUE(buffer_mode.set("managed"));

  std::unique_ptr<prach_buffer> buffer = create_phy_acceleration_prach_buffer_short(2, 2, 2, true);
  ASSERT_NE(buffer, nullptr);
  ASSERT_TRUE(buffer->supports_device_prach_buffer_reading());

  static constexpr unsigned port        = 0;
  static constexpr unsigned td_occasion = 1;
  static constexpr unsigned fd_occasion = 1;
  static constexpr unsigned symbol      = 4;

  std::vector<cbf16_t> expected = generate_symbol_payload(buffer->get_sequence_length(), 0.013F);
  std::copy(expected.begin(), expected.end(), buffer->get_symbol(port, td_occasion, fd_occasion, symbol).begin());

  cuda_stream_guard stream;
  ASSERT_TRUE(stream.create());

  ASSERT_TRUE(buffer->prepare_device_prach_buffer_reading(stream.get()));
  std::vector<cbf16_t> copied(expected.size());
  const cbf16_t*       device_buffer = static_cast<const cbf16_t*>(buffer->get_device_prach_buffer_cbf16());
  const unsigned       offset        = buffer->get_device_prach_symbol_offset(port, td_occasion, fd_occasion, symbol);

  ASSERT_EQ(
      cudaMemcpyAsync(
          copied.data(), device_buffer + offset, copied.size() * sizeof(cbf16_t), cudaMemcpyDeviceToHost, stream.get()),
      cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  EXPECT_EQ(copied, expected);
}

TEST(cuda_visible_phy_support_test, prach_buffer_repeated_device_mapping_and_reading_cycles_are_ordered)
{
  testing::AssertionResult cuda_available = cuda_managed_memory_available();
  if (!cuda_available) {
    GTEST_SKIP() << cuda_available.message();
  }

  scoped_env_var buffer_mode("OCUDU_UL_CUDA_VISIBLE_PRACH_BUFFER");
  ASSERT_TRUE(buffer_mode.set("managed"));

  std::unique_ptr<prach_buffer> buffer = create_phy_acceleration_prach_buffer_short(2, 2, 2, true);
  ASSERT_NE(buffer, nullptr);
  ASSERT_TRUE(buffer->supports_device_prach_buffer_mapping());
  ASSERT_TRUE(buffer->supports_device_prach_buffer_reading());

  cuda_stream_guard stream;
  ASSERT_TRUE(stream.create());

  for (unsigned iteration = 0; iteration != 12; ++iteration) {
    SCOPED_TRACE("iteration=" + std::to_string(iteration));

    const unsigned port        = iteration % buffer->get_max_nof_ports();
    const unsigned td_occasion = (iteration / 2) % buffer->get_max_nof_td_occasions();
    const unsigned fd_occasion = (iteration / 3) % buffer->get_max_nof_fd_occasions();
    const unsigned symbol      = iteration % buffer->get_max_nof_symbols();

    std::vector<cbf16_t> expected =
        generate_symbol_payload(buffer->get_sequence_length(), 0.003F + 0.001F * static_cast<float>(iteration));
    const unsigned offset        = buffer->get_device_prach_symbol_offset(port, td_occasion, fd_occasion, symbol);
    cbf16_t*       device_buffer = static_cast<cbf16_t*>(buffer->get_device_prach_buffer_cbf16());
    ASSERT_NE(device_buffer, nullptr);

    ASSERT_TRUE(buffer->prepare_device_prach_buffer_mapping(stream.get()));
    ASSERT_EQ(cudaMemcpyAsync(device_buffer + offset,
                              expected.data(),
                              expected.size() * sizeof(cbf16_t),
                              cudaMemcpyHostToDevice,
                              stream.get()),
              cudaSuccess);
    ASSERT_TRUE(buffer->on_device_prach_buffer_mapping_enqueued(stream.get()));
    expect_equal(buffer->get_symbol(port, td_occasion, fd_occasion, symbol), span<const cbf16_t>(expected));

    ASSERT_TRUE(buffer->prepare_device_prach_buffer_reading(stream.get()));
    std::vector<cbf16_t> copied(expected.size());
    ASSERT_EQ(cudaMemcpyAsync(copied.data(),
                              device_buffer + offset,
                              copied.size() * sizeof(cbf16_t),
                              cudaMemcpyDeviceToHost,
                              stream.get()),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
    EXPECT_EQ(copied, expected);

    ASSERT_TRUE(buffer->prepare_device_prach_buffer_mapping(stream.get()));
    ASSERT_TRUE(buffer->cancel_device_prach_buffer_mapping());
  }
}
