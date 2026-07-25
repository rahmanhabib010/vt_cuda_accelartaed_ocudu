// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "../../../../lib/ofh/compression/iq_compression_none_impl.h"
#include "../../../../lib/ofh/compression/iq_compressor_selector.h"
#include "../../../../lib/ofh/serdes/ofh_uplane_message_builder_static_compression_impl.h"
#include "../../phy/support/resource_grid_test_doubles.h"
#include "ocudu/adt/static_vector.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/ocuduvec/conversion.h"
#include "ocudu/ofh/compression/compression_properties.h"
#include <algorithm>
#include <array>
#include <memory>
#include <gtest/gtest.h>

using namespace ocudu;
using namespace ofh;

namespace {

/// Open Fronthaul User-Plane section header size in bytes.
constexpr unsigned OFH_UP_SECTION_HEADER_SIZE = 4;
/// Open Fronthaul User-Plane radio application header size in bytes.
constexpr unsigned OFH_UP_RADIO_APP_HEADER_SIZE = 4;

/// Returns offset in number of bytes to the IQ data from the start of the packet.
units::bytes get_iq_data_offset_static_config()
{
  unsigned header_size = OFH_UP_SECTION_HEADER_SIZE + OFH_UP_RADIO_APP_HEADER_SIZE;
  return units::bytes(header_size);
}

/// Structure describing a test case.
struct test_case_t {
  /// Original IQ data.
  std::vector<cf_t> iq_data;
  /// OFH U-Plane packet builder parameters.
  uplane_message_params params;
  /// Expected output packet.
  std::vector<uint8_t> packet;
};

static const std::vector<test_case_t> ofh_uplane_builder_test_data = {
    {{// PRB 0
      {0.0116, 0.0116},
      {0.0119, 0.0119},
      {0.0122, 0.0122},
      {0.0125, 0.0125},
      {0.0128, 0.0128},
      {0.0131, 0.0131},
      {0.0134, 0.0134},
      {0.0137, 0.0137},
      {0.0140, 0.0140},
      {0.0143, 0.0143},
      {0.0146, 0.0146},
      {0.0150, 0.0150},
      // PRB 1
      {0.0153, 0.0153},
      {0.0156, 0.0156},
      {0.0159, 0.0159},
      {0.0162, 0.0162},
      {0.0165, 0.0165},
      {0.0168, 0.0168},
      {0.0171, 0.0171},
      {0.0174, 0.0174},
      {0.0177, 0.0177},
      {0.0180, 0.0180},
      {0.0183, 0.0183},
      {0.0186, 0.0186},
      // PRB 2
      {0.0189, 0.0189},
      {0.0192, 0.0192},
      {0.0195, 0.0195},
      {0.0198, 0.0198},
      {0.0201, 0.0201},
      {0.0204, 0.0204},
      {0.0208, 0.0208},
      {0.0211, 0.0211},
      {0.0214, 0.0214},
      {0.0217, 0.0217},
      {0.0220, 0.0220},
      {0.0223, 0.0223}},
     // Params: direction downlink, Numerology 1, slot 2, filter index 0, startPRB 0, 3 PRBs, Symbol
     // 9
     {data_direction::downlink,
      {1, 2},
      filter_index_type::standard_channel_filter,
      0,
      3,
      9,
      section_type::type_1,
      {compression_type::none, 16}},
     // Packet
     {0x90, 0x00, 0x10, 0x09, 0x00, 0x00, 0x00, 0x03, 0x01, 0x7c, 0x01, 0x7c, 0x01, 0x86, 0x01, 0x86, 0x01,
      0x90, 0x01, 0x90, 0x01, 0x9a, 0x01, 0x9a, 0x01, 0xa4, 0x01, 0xa4, 0x01, 0xae, 0x01, 0xae, 0x01, 0xb8,
      0x01, 0xb8, 0x01, 0xc2, 0x01, 0xc2, 0x01, 0xcc, 0x01, 0xcc, 0x01, 0xd6, 0x01, 0xd6, 0x01, 0xe0, 0x01,
      0xe0, 0x01, 0xea, 0x01, 0xea, 0x01, 0xf4, 0x01, 0xf4, 0x01, 0xfe, 0x01, 0xfe, 0x02, 0x08, 0x02, 0x08,
      0x02, 0x12, 0x02, 0x12, 0x02, 0x1c, 0x02, 0x1c, 0x02, 0x26, 0x02, 0x26, 0x02, 0x30, 0x02, 0x30, 0x02,
      0x3a, 0x02, 0x3a, 0x02, 0x44, 0x02, 0x44, 0x02, 0x4e, 0x02, 0x4e, 0x02, 0x58, 0x02, 0x58, 0x02, 0x62,
      0x02, 0x62, 0x02, 0x6c, 0x02, 0x6c, 0x02, 0x76, 0x02, 0x76, 0x02, 0x80, 0x02, 0x80, 0x02, 0x8a, 0x02,
      0x8a, 0x02, 0x94, 0x02, 0x94, 0x02, 0x9e, 0x02, 0x9e, 0x02, 0xa8, 0x02, 0xa8, 0x02, 0xb2, 0x02, 0xb2,
      0x02, 0xbc, 0x02, 0xbc, 0x02, 0xc6, 0x02, 0xc6, 0x02, 0xd0, 0x02, 0xd0, 0x02, 0xda, 0x02, 0xda}}};

class direct_payload_compressor_spy : public iq_compressor
{
public:
  static uint8_t get_payload_byte(unsigned buffer_index, unsigned byte_index)
  {
    return static_cast<uint8_t>(0x40U + ((buffer_index * 17U + byte_index) & 0x3fU));
  }

  void compress(span<uint8_t> buffer, span<const cbf16_t>, const ru_compression_params&) override
  {
    ++host_compress_count;
    std::fill(buffer.begin(), buffer.end(), 0);
  }

  bool compress_device_symbols(span<uint8_t>,
                               unsigned,
                               const resource_grid_reader&,
                               unsigned,
                               unsigned,
                               unsigned,
                               unsigned,
                               unsigned,
                               const ru_compression_params&) override
  {
    ++device_batch_fallback_count;
    return false;
  }

  bool compress_device_symbols_to_buffers(span<span<uint8_t>> buffers,
                                          const resource_grid_reader&,
                                          unsigned                     first_port,
                                          unsigned                     nof_ports,
                                          unsigned                     symbol,
                                          unsigned                     start_prb,
                                          unsigned                     nof_prbs,
                                          const ru_compression_params& params) override
  {
    ++direct_port_buffer_count;
    EXPECT_EQ(first_port, 1U);
    EXPECT_EQ(nof_ports, buffers.size());
    EXPECT_EQ(symbol, 3U);
    EXPECT_EQ(start_prb, 1U);
    EXPECT_EQ(nof_prbs, 2U);
    EXPECT_EQ(params.type, compression_type::BFP);
    EXPECT_EQ(params.data_width, 9U);
    fill_buffers(buffers);
    return true;
  }

  bool compress_device_symbol_batch(span<uint8_t>,
                                    unsigned,
                                    unsigned,
                                    const resource_grid_reader&,
                                    unsigned,
                                    unsigned,
                                    unsigned,
                                    unsigned,
                                    unsigned,
                                    unsigned,
                                    const ru_compression_params&) override
  {
    ++device_symbol_batch_fallback_count;
    return false;
  }

  bool compress_device_symbol_batch_to_buffers(span<span<uint8_t>> buffers,
                                               const resource_grid_reader&,
                                               unsigned                     first_port,
                                               unsigned                     nof_ports,
                                               unsigned                     first_symbol,
                                               unsigned                     nof_symbols,
                                               unsigned                     start_prb,
                                               unsigned                     nof_prbs,
                                               const ru_compression_params& params) override
  {
    ++direct_symbol_buffer_count;
    EXPECT_EQ(first_port, 1U);
    EXPECT_EQ(nof_ports * nof_symbols, buffers.size());
    EXPECT_EQ(first_symbol, 4U);
    EXPECT_EQ(nof_symbols, 3U);
    EXPECT_EQ(start_prb, 1U);
    EXPECT_EQ(nof_prbs, 2U);
    EXPECT_EQ(params.type, compression_type::BFP);
    EXPECT_EQ(params.data_width, 9U);
    fill_buffers(buffers);
    return true;
  }

  unsigned host_compress_count                = 0;
  unsigned device_batch_fallback_count        = 0;
  unsigned device_symbol_batch_fallback_count = 0;
  unsigned direct_port_buffer_count           = 0;
  unsigned direct_symbol_buffer_count         = 0;

private:
  static void fill_buffers(span<span<uint8_t>> buffers)
  {
    for (unsigned i_buffer = 0, e = buffers.size(); i_buffer != e; ++i_buffer) {
      for (unsigned i_byte = 0, e_byte = buffers[i_buffer].size(); i_byte != e_byte; ++i_byte) {
        buffers[i_buffer][i_byte] = get_payload_byte(i_buffer, i_byte);
      }
    }
  }
};

static std::vector<span<uint8_t>> make_packet_spans(std::vector<std::vector<uint8_t>>& packets)
{
  std::vector<span<uint8_t>> spans(packets.size());
  for (unsigned i = 0, e = packets.size(); i != e; ++i) {
    spans[i] = span<uint8_t>(packets[i].data(), packets[i].size());
  }
  return spans;
}

static std::unique_ptr<iq_compressor_selector>
make_selector_with_direct_payload_spy(direct_payload_compressor_spy*& spy)
{
  ocudulog::basic_logger& logger = ocudulog::fetch_basic_logger("TEST");

  std::array<std::unique_ptr<iq_compressor>, NOF_COMPRESSION_TYPES_SUPPORTED> compressors;
  for (unsigned i = 0, e = compressors.size(); i != e; ++i) {
    compressors[i] = std::make_unique<iq_compression_none_impl>(logger);
  }

  auto spy_compressor = std::make_unique<direct_payload_compressor_spy>();
  spy                 = spy_compressor.get();
  compressors[static_cast<unsigned>(compression_type::BFP)] = std::move(spy_compressor);
  return std::make_unique<iq_compressor_selector>(std::move(compressors));
}

static void expect_direct_payload(span<const uint8_t> packet, unsigned header_size, unsigned buffer_index)
{
  for (unsigned i_byte = 0, e_byte = packet.size() - header_size; i_byte != e_byte; ++i_byte) {
    ASSERT_EQ(packet[header_size + i_byte], direct_payload_compressor_spy::get_payload_byte(buffer_index, i_byte));
  }
}

TEST(ofh_uplane_packet_builder_static_impl_test, non_compressed_packet_should_pass)
{
  ocudulog::basic_logger& logger = ocudulog::fetch_basic_logger("TEST", false);

  for (const test_case_t& test_case : ofh_uplane_builder_test_data) {
    // Create a compressor.
    iq_compression_none_impl compressor(logger);

    ofh_uplane_message_builder_static_compression_impl builder(ocudulog::fetch_basic_logger("TEST"), compressor);

    // Prepare output buffer and build packet.
    std::vector<uint8_t> result_packet(test_case.packet.size(), 0);
    std::vector<cbf16_t> iq_data_cbf16(test_case.iq_data.size());
    ocuduvec::convert(iq_data_cbf16, test_case.iq_data);

    builder.build_message(result_packet, iq_data_cbf16, test_case.params);

    // First make sure basic header matches the expected one.
    units::bytes header_size = get_iq_data_offset_static_config();
    ASSERT_TRUE(
        std::equal(result_packet.begin(), result_packet.begin() + header_size.value(), test_case.packet.begin()))
        << "Wrong header in the generated U-Plane packet";

    // Compare IQ data with an error tolerance in LSB bits, that can be introduced due to quantization.
    //
    // Due to initial float -> bf16 'to nearest even' rounding we may have numbers where 8 MSBs are increased by 1 and
    // LSBs just wrap around. To get correct difference below we convert each byte to signed type.
    ASSERT_TRUE(std::equal(result_packet.begin() + header_size.value(),
                           result_packet.end(),
                           test_case.packet.begin() + header_size.value(),
                           [](uint8_t a, uint8_t b) { return std::abs(int8_t(a) - int8_t(b)) <= 2; }))
        << "Wrong IQ data in the generated U-Plane packet.";
  }
}

TEST(ofh_uplane_packet_builder_static_impl_test, device_grid_port_batch_uses_direct_payload_buffers)
{
  direct_payload_compressor_spy                      compressor;
  ofh_uplane_message_builder_static_compression_impl builder(ocudulog::fetch_basic_logger("TEST"), compressor);

  uplane_message_params params = {data_direction::downlink,
                                  {1, 2},
                                  filter_index_type::standard_channel_filter,
                                  1,
                                  2,
                                  3,
                                  section_type::type_1,
                                  {compression_type::BFP, 9}};

  const unsigned header_size = builder.get_header_size(params.compression_params).value();
  const unsigned port_bytes  = (get_compressed_prb_size(params.compression_params) * params.nof_prb).value();
  const unsigned nof_ports   = 2U;

  resource_grid_reader_spy          grid(nof_ports, 14, 4);
  std::vector<std::vector<uint8_t>> packets(nof_ports, std::vector<uint8_t>(header_size + port_bytes, 0xa5));
  std::vector<span<uint8_t>>        packet_spans = make_packet_spans(packets);
  std::vector<unsigned>             bytes_written(nof_ports);
  span<span<uint8_t>>               packet_span_view(packet_spans.data(), packet_spans.size());

  ASSERT_TRUE(builder.build_messages(packet_span_view, grid, 1, params, bytes_written));

  EXPECT_EQ(compressor.direct_port_buffer_count, 1U);
  EXPECT_EQ(compressor.direct_symbol_buffer_count, 0U);
  EXPECT_EQ(compressor.host_compress_count, 0U);
  EXPECT_EQ(compressor.device_batch_fallback_count, 0U);
  EXPECT_EQ(compressor.device_symbol_batch_fallback_count, 0U);
  for (unsigned i_port = 0; i_port != nof_ports; ++i_port) {
    EXPECT_EQ(bytes_written[i_port], header_size + port_bytes);
    expect_direct_payload(span<const uint8_t>(packets[i_port].data(), packets[i_port].size()), header_size, i_port);
  }
}

TEST(ofh_uplane_packet_builder_static_impl_test, selector_forwards_device_grid_port_batch_direct_payload_buffers)
{
  direct_payload_compressor_spy* spy      = nullptr;
  auto                           selector = make_selector_with_direct_payload_spy(spy);
  ofh_uplane_message_builder_static_compression_impl builder(ocudulog::fetch_basic_logger("TEST"), *selector);

  uplane_message_params params = {data_direction::downlink,
                                  {1, 2},
                                  filter_index_type::standard_channel_filter,
                                  1,
                                  2,
                                  3,
                                  section_type::type_1,
                                  {compression_type::BFP, 9}};

  const unsigned header_size = builder.get_header_size(params.compression_params).value();
  const unsigned port_bytes  = (get_compressed_prb_size(params.compression_params) * params.nof_prb).value();
  const unsigned nof_ports   = 2U;

  resource_grid_reader_spy          grid(nof_ports, 14, 4);
  std::vector<std::vector<uint8_t>> packets(nof_ports, std::vector<uint8_t>(header_size + port_bytes, 0xa5));
  std::vector<span<uint8_t>>        packet_spans = make_packet_spans(packets);
  std::vector<unsigned>             bytes_written(nof_ports);
  span<span<uint8_t>>               packet_span_view(packet_spans.data(), packet_spans.size());

  ASSERT_TRUE(builder.build_messages(packet_span_view, grid, 1, params, bytes_written));

  ASSERT_NE(spy, nullptr);
  EXPECT_EQ(spy->direct_port_buffer_count, 1U);
  EXPECT_EQ(spy->direct_symbol_buffer_count, 0U);
  EXPECT_EQ(spy->host_compress_count, 0U);
  EXPECT_EQ(spy->device_batch_fallback_count, 0U);
  EXPECT_EQ(spy->device_symbol_batch_fallback_count, 0U);
  for (unsigned i_port = 0; i_port != nof_ports; ++i_port) {
    EXPECT_EQ(bytes_written[i_port], header_size + port_bytes);
    expect_direct_payload(span<const uint8_t>(packets[i_port].data(), packets[i_port].size()), header_size, i_port);
  }
}

TEST(ofh_uplane_packet_builder_static_impl_test, device_grid_symbol_batch_uses_direct_payload_buffers)
{
  direct_payload_compressor_spy                      compressor;
  ofh_uplane_message_builder_static_compression_impl builder(ocudulog::fetch_basic_logger("TEST"), compressor);

  uplane_message_params params = {data_direction::downlink,
                                  {1, 2},
                                  filter_index_type::standard_channel_filter,
                                  1,
                                  2,
                                  4,
                                  section_type::type_1,
                                  {compression_type::BFP, 9}};

  const unsigned header_size = builder.get_header_size(params.compression_params).value();
  const unsigned port_bytes  = (get_compressed_prb_size(params.compression_params) * params.nof_prb).value();
  const unsigned nof_ports   = 2U;
  const unsigned nof_symbols = 3U;
  const unsigned nof_buffers = nof_ports * nof_symbols;

  resource_grid_reader_spy          grid(nof_ports, 14, 4);
  std::vector<std::vector<uint8_t>> packets(nof_buffers, std::vector<uint8_t>(header_size + port_bytes, 0xa5));
  std::vector<span<uint8_t>>        packet_spans = make_packet_spans(packets);
  std::vector<unsigned>             bytes_written(nof_buffers);
  span<span<uint8_t>>               packet_span_view(packet_spans.data(), packet_spans.size());

  ASSERT_TRUE(builder.build_symbol_batch_messages(packet_span_view, grid, 1, 4, nof_symbols, params, bytes_written));

  EXPECT_EQ(compressor.direct_port_buffer_count, 0U);
  EXPECT_EQ(compressor.direct_symbol_buffer_count, 1U);
  EXPECT_EQ(compressor.host_compress_count, 0U);
  EXPECT_EQ(compressor.device_batch_fallback_count, 0U);
  EXPECT_EQ(compressor.device_symbol_batch_fallback_count, 0U);
  for (unsigned i_buffer = 0; i_buffer != nof_buffers; ++i_buffer) {
    EXPECT_EQ(bytes_written[i_buffer], header_size + port_bytes);
    expect_direct_payload(
        span<const uint8_t>(packets[i_buffer].data(), packets[i_buffer].size()), header_size, i_buffer);
  }
}

TEST(ofh_uplane_packet_builder_static_impl_test, selector_forwards_device_grid_symbol_batch_direct_payload_buffers)
{
  direct_payload_compressor_spy* spy      = nullptr;
  auto                           selector = make_selector_with_direct_payload_spy(spy);
  ofh_uplane_message_builder_static_compression_impl builder(ocudulog::fetch_basic_logger("TEST"), *selector);

  uplane_message_params params = {data_direction::downlink,
                                  {1, 2},
                                  filter_index_type::standard_channel_filter,
                                  1,
                                  2,
                                  4,
                                  section_type::type_1,
                                  {compression_type::BFP, 9}};

  const unsigned header_size = builder.get_header_size(params.compression_params).value();
  const unsigned port_bytes  = (get_compressed_prb_size(params.compression_params) * params.nof_prb).value();
  const unsigned nof_ports   = 2U;
  const unsigned nof_symbols = 3U;
  const unsigned nof_buffers = nof_ports * nof_symbols;

  resource_grid_reader_spy          grid(nof_ports, 14, 4);
  std::vector<std::vector<uint8_t>> packets(nof_buffers, std::vector<uint8_t>(header_size + port_bytes, 0xa5));
  std::vector<span<uint8_t>>        packet_spans = make_packet_spans(packets);
  std::vector<unsigned>             bytes_written(nof_buffers);
  span<span<uint8_t>>               packet_span_view(packet_spans.data(), packet_spans.size());

  ASSERT_TRUE(builder.build_symbol_batch_messages(packet_span_view, grid, 1, 4, nof_symbols, params, bytes_written));

  ASSERT_NE(spy, nullptr);
  EXPECT_EQ(spy->direct_port_buffer_count, 0U);
  EXPECT_EQ(spy->direct_symbol_buffer_count, 1U);
  EXPECT_EQ(spy->host_compress_count, 0U);
  EXPECT_EQ(spy->device_batch_fallback_count, 0U);
  EXPECT_EQ(spy->device_symbol_batch_fallback_count, 0U);
  for (unsigned i_buffer = 0; i_buffer != nof_buffers; ++i_buffer) {
    EXPECT_EQ(bytes_written[i_buffer], header_size + port_bytes);
    expect_direct_payload(
        span<const uint8_t>(packets[i_buffer].data(), packets[i_buffer].size()), header_size, i_buffer);
  }
}

} // namespace
