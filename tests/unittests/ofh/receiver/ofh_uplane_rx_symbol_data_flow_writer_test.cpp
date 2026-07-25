// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "../../../../lib/ofh/receiver/ofh_uplane_rx_symbol_data_flow_writer.h"
#include "helpers.h"
#include "ocudu/ofh/serdes/ofh_message_decoder_properties.h"
#include "ocudu/ofh/serdes/ofh_uplane_message_decoder.h"
#include <gtest/gtest.h>

using namespace ocudu;
using namespace ofh;
using namespace ofh::testing;

namespace {

class resource_grid_device_writer_spy : public resource_grid_writer_bool_spy
{
public:
  explicit resource_grid_device_writer_spy(unsigned nof_prbs) : resource_grid_writer_bool_spy(nof_prbs) {}

  bool supports_device_grid_mapping() const override { return true; }

  void* get_device_grid_bf16() override { return this; }

  bool prepare_device_grid_mapping(void*) override
  {
    ++prepare_count;
    return true;
  }

  bool on_device_grid_mapping_enqueued(void*) override
  {
    ++enqueue_count;
    return true;
  }

  unsigned prepare_count = 0;
  unsigned enqueue_count = 0;
};

class uplane_message_decoder_device_spy : public uplane_message_decoder
{
public:
  bool decode(uplane_message_decoder_results& results, span<const uint8_t> message) override
  {
    (void)results;
    (void)message;
    return false;
  }

  bool decompress_to_resource_grid(resource_grid_writer&        grid,
                                   unsigned                     port_,
                                   unsigned                     symbol_,
                                   unsigned                     start_prb_,
                                   unsigned                     nof_prbs_,
                                   const uplane_section_params& section_) override
  {
    (void)grid;
    called    = true;
    port      = port_;
    symbol    = symbol_;
    start_prb = start_prb_;
    nof_prbs  = nof_prbs_;
    section   = &section_;
    if (success) {
      success = grid.prepare_device_grid_mapping(nullptr) && grid.on_device_grid_mapping_enqueued(nullptr);
    }
    return success;
  }

  bool                         success   = true;
  bool                         called    = false;
  unsigned                     port      = 0;
  unsigned                     symbol    = 0;
  unsigned                     start_prb = 0;
  unsigned                     nof_prbs  = 0;
  const uplane_section_params* section   = nullptr;
};

} // namespace

class ofh_uplane_rx_symbol_data_flow_writer_fixture : public ::testing::Test
{
protected:
  static constexpr ofdm_symbol_range                            symbol_range = {0, 14};
  static constexpr unsigned                                     nof_ports    = 1;
  static constexpr unsigned                                     nof_prb      = 51;
  static constexpr unsigned                                     sector       = 0;
  static constexpr unsigned                                     symbol_id    = 0;
  static constexpr std::array<unsigned, MAX_NOF_SUPPORTED_EAXC> eaxc         = {0, 1, 2, 3};
  static constexpr slot_point                                   slot         = {0, 0, 1};
  resource_grid_writer_bool_spy                                 rg_writer;
  resource_grid_reader_spy                                      rg_reader;
  resource_grid_spy                                             grid;
  shared_resource_grid_spy                                      shared_grid;
  std::shared_ptr<uplink_context_repository>                    repo = std::make_shared<uplink_context_repository>(1);
  uplane_message_decoder_results                                results;
  uplane_rx_symbol_data_flow_writer                             writer;

public:
  ofh_uplane_rx_symbol_data_flow_writer_fixture() :
    rg_writer(nof_prb),
    rg_reader(nof_ports, symbol_range.stop(), nof_prb),
    grid(rg_reader, rg_writer),
    shared_grid(grid),
    writer(eaxc, 0, ocudulog::fetch_basic_logger("TEST"), repo)
  {
    results.params.slot      = slot;
    results.params.symbol_id = symbol_id;
    results.sections.emplace_back();
  }
};

TEST_F(ofh_uplane_rx_symbol_data_flow_writer_fixture, empty_context_does_not_write)
{
  writer.write_to_resource_grid(eaxc[0], results);

  ASSERT_TRUE(repo->get(results.params.slot, results.params.symbol_id).empty());
  ASSERT_FALSE(rg_writer.has_grid_been_written());
}

#ifdef ASSERTS_ENABLED
TEST_F(ofh_uplane_rx_symbol_data_flow_writer_fixture, death_test_no_eaxc_found)
{
  unsigned invalid_eaxc = 4;

  repo->add({results.params.slot, sector}, shared_grid.get_grid(), symbol_range, ocudulog::fetch_basic_logger("TEST"));
  repo->process_pending_contexts();
  ASSERT_FALSE(repo->get(results.params.slot, results.params.symbol_id).empty());

  ASSERT_DEATH(writer.write_to_resource_grid(invalid_eaxc, results), "");
}
#endif

TEST_F(ofh_uplane_rx_symbol_data_flow_writer_fixture, decoded_prbs_outside_grid_prbs_do_not_write)
{
  auto& section     = results.sections.back();
  section.nof_prbs  = 50;
  section.start_prb = nof_prb;
  section.iq_samples.resize(section.nof_prbs * NOF_SUBCARRIERS_PER_RB);

  repo->add({results.params.slot, sector}, shared_grid.get_grid(), symbol_range, ocudulog::fetch_basic_logger("TEST"));
  repo->process_pending_contexts();
  writer.write_to_resource_grid(eaxc[0], results);

  ASSERT_FALSE(repo->get(results.params.slot, results.params.symbol_id).empty());
  ASSERT_FALSE(rg_writer.has_grid_been_written());

  const uplink_context& context  = repo->get(slot, symbol_id);
  const auto&           sym_data = context.get_re_written_mask();
  ASSERT_TRUE(std::all_of(sym_data.begin(), sym_data.end(), [](const auto& port) { return port.none(); }));
}

TEST_F(ofh_uplane_rx_symbol_data_flow_writer_fixture, decoded_prbs_match_grid_prbs_write)
{
  auto& section     = results.sections.back();
  section.nof_prbs  = 51;
  section.start_prb = 0;
  section.iq_samples.resize(section.nof_prbs * NOF_SUBCARRIERS_PER_RB);

  repo->add({results.params.slot, sector}, shared_grid.get_grid(), symbol_range, ocudulog::fetch_basic_logger("TEST"));
  repo->process_pending_contexts();
  writer.write_to_resource_grid(eaxc[0], results);

  ASSERT_FALSE(repo->get(results.params.slot, results.params.symbol_id).empty());
  ASSERT_TRUE(rg_writer.has_grid_been_written());
  ASSERT_EQ(section.nof_prbs, rg_writer.get_nof_prbs_written());

  {
    const uplink_context& context  = repo->get(slot, symbol_id);
    const auto&           sym_data = context.get_re_written_mask();
    ASSERT_TRUE(std::all_of(sym_data.begin(), sym_data.end(), [](const auto& port) { return port.all(); }));
  }

  repo->clear();
}

TEST_F(ofh_uplane_rx_symbol_data_flow_writer_fixture, decoded_prbs_bigger_than_grid_prbs_write)
{
  auto& section     = results.sections.back();
  section.nof_prbs  = 273;
  section.start_prb = 0;
  section.iq_samples.resize(section.nof_prbs * NOF_SUBCARRIERS_PER_RB);

  repo->add({results.params.slot, sector}, shared_grid.get_grid(), symbol_range, ocudulog::fetch_basic_logger("TEST"));
  repo->process_pending_contexts();
  writer.write_to_resource_grid(eaxc[0], results);

  ASSERT_FALSE(repo->get(results.params.slot, results.params.symbol_id).empty());
  ASSERT_TRUE(rg_writer.has_grid_been_written());
  ASSERT_EQ(rg_writer.get_nof_subc() / NOF_SUBCARRIERS_PER_RB, rg_writer.get_nof_prbs_written());

  const uplink_context& context  = repo->get(slot, symbol_id);
  const auto&           sym_data = context.get_re_written_mask();
  ASSERT_TRUE(std::all_of(sym_data.begin(), sym_data.end(), [](const auto& port) { return port.all(); }));
}

TEST_F(ofh_uplane_rx_symbol_data_flow_writer_fixture, segmented_prbs_inside_the_grid_write)
{
  auto& section     = results.sections.back();
  section.nof_prbs  = 10;
  section.start_prb = 0;
  section.iq_samples.resize(section.nof_prbs * NOF_SUBCARRIERS_PER_RB);

  repo->add({results.params.slot, sector}, shared_grid.get_grid(), symbol_range, ocudulog::fetch_basic_logger("TEST"));
  repo->process_pending_contexts();
  writer.write_to_resource_grid(eaxc[0], results);

  ASSERT_FALSE(repo->get(results.params.slot, results.params.symbol_id).empty());
  ASSERT_TRUE(rg_writer.has_grid_been_written());
  ASSERT_EQ(section.nof_prbs, rg_writer.get_nof_prbs_written());

  const uplink_context& context  = repo->get(slot, symbol_id);
  const auto&           sym_data = context.get_re_written_mask();
  ASSERT_TRUE(std::all_of(sym_data.begin(), sym_data.end(), [&section](const auto& port) {
    return port.all(0, (section.nof_prbs - 1) * NOF_SUBCARRIERS_PER_RB);
  }));
}

TEST_F(ofh_uplane_rx_symbol_data_flow_writer_fixture, segmented_prbs_write_the_prbs_overlapped_with_grid)
{
  auto& section     = results.sections.back();
  section.nof_prbs  = 60;
  section.start_prb = 40;
  section.iq_samples.resize(section.nof_prbs * NOF_SUBCARRIERS_PER_RB);

  repo->add({results.params.slot, sector}, shared_grid.get_grid(), symbol_range, ocudulog::fetch_basic_logger("TEST"));
  repo->process_pending_contexts();
  writer.write_to_resource_grid(eaxc[0], results);

  ASSERT_FALSE(repo->get(results.params.slot, results.params.symbol_id).empty());
  ASSERT_TRUE(rg_writer.has_grid_been_written());
  ASSERT_EQ(11, rg_writer.get_nof_prbs_written());

  const uplink_context& context  = repo->get(slot, symbol_id);
  const auto&           sym_data = context.get_re_written_mask();
  ASSERT_TRUE(std::all_of(sym_data.begin(), sym_data.end(), [](const auto& port) {
    return port.all(40 * NOF_SUBCARRIERS_PER_RB, 50 * NOF_SUBCARRIERS_PER_RB);
  }));
}

TEST_F(ofh_uplane_rx_symbol_data_flow_writer_fixture, segmented_prbs_fill_the_grid)
{
  auto& section     = results.sections.back();
  section.nof_prbs  = 50;
  section.start_prb = 0;
  section.iq_samples.resize(section.nof_prbs * NOF_SUBCARRIERS_PER_RB);

  repo->add({results.params.slot, sector}, shared_grid.get_grid(), symbol_range, ocudulog::fetch_basic_logger("TEST"));
  repo->process_pending_contexts();
  writer.write_to_resource_grid(eaxc[0], results);
  ASSERT_EQ(section.nof_prbs, rg_writer.get_nof_prbs_written());
  {
    const uplink_context& context  = repo->get(slot, symbol_id);
    const auto&           sym_data = context.get_re_written_mask();
    ASSERT_TRUE(std::all_of(sym_data.begin(), sym_data.end(), [&section](const auto& port) {
      return port.all(0, (section.nof_prbs - 1) * NOF_SUBCARRIERS_PER_RB);
    }));
  }

  unsigned nof_prbs = section.nof_prbs;

  section.nof_prbs  = 1;
  section.start_prb = 50;
  section.iq_samples.resize(section.nof_prbs * NOF_SUBCARRIERS_PER_RB);
  nof_prbs += section.nof_prbs;

  writer.write_to_resource_grid(eaxc[0], results);

  ASSERT_FALSE(repo->get(results.params.slot, results.params.symbol_id).empty());
  ASSERT_TRUE(rg_writer.has_grid_been_written());
  ASSERT_EQ(nof_prbs, rg_writer.get_nof_prbs_written());
  const uplink_context& context  = repo->get(slot, symbol_id);
  const auto&           sym_data = context.get_re_written_mask();
  ASSERT_TRUE(std::all_of(sym_data.begin(), sym_data.end(), [](const auto& port) { return port.all(); }));
}

TEST_F(ofh_uplane_rx_symbol_data_flow_writer_fixture, compressed_prbs_write_through_device_grid_path)
{
  resource_grid_device_writer_spy   device_writer(nof_prb);
  resource_grid_spy                 device_grid(rg_reader, device_writer);
  shared_resource_grid_spy          device_shared_grid(device_grid);
  std::array<uint8_t, 128>          compressed_payload = {};
  uplane_message_decoder_device_spy decoder;

  auto& section              = results.sections.back();
  section.nof_prbs           = nof_prb;
  section.start_prb          = 0;
  section.compressed_iq_data = compressed_payload;

  repo->add(
      {results.params.slot, sector}, device_shared_grid.get_grid(), symbol_range, ocudulog::fetch_basic_logger("TEST"));
  repo->process_pending_contexts();

  bool direct_written  = writer.write_compressed_to_resource_grid(eaxc[0], results, decoder);
  bool all_res_written = false;
  {
    const uplink_context& context  = repo->get(slot, symbol_id);
    const auto&           sym_data = context.get_re_written_mask();
    all_res_written = std::all_of(sym_data.begin(), sym_data.end(), [](const auto& port) { return port.all(); });
  }
  repo->clear();

  ASSERT_TRUE(direct_written);
  ASSERT_TRUE(decoder.called);
  ASSERT_EQ(0, decoder.port);
  ASSERT_EQ(symbol_id, decoder.symbol);
  ASSERT_EQ(section.start_prb, decoder.start_prb);
  ASSERT_EQ(section.nof_prbs, decoder.nof_prbs);
  ASSERT_EQ(&section, decoder.section);
  ASSERT_EQ(1, device_writer.prepare_count);
  ASSERT_EQ(1, device_writer.enqueue_count);
  ASSERT_FALSE(device_writer.has_grid_been_written());
  ASSERT_TRUE(all_res_written);
}

TEST_F(ofh_uplane_rx_symbol_data_flow_writer_fixture, compressed_prbs_return_false_when_device_grid_unavailable)
{
  std::array<uint8_t, 128>          compressed_payload = {};
  uplane_message_decoder_device_spy decoder;

  auto& section              = results.sections.back();
  section.nof_prbs           = nof_prb;
  section.start_prb          = 0;
  section.compressed_iq_data = compressed_payload;

  repo->add({results.params.slot, sector}, shared_grid.get_grid(), symbol_range, ocudulog::fetch_basic_logger("TEST"));
  repo->process_pending_contexts();

  ASSERT_FALSE(writer.write_compressed_to_resource_grid(eaxc[0], results, decoder));
  ASSERT_FALSE(decoder.called);
  ASSERT_FALSE(rg_writer.has_grid_been_written());
}
