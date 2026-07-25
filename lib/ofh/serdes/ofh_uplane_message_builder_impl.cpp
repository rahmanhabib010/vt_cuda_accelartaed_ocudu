// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ofh_uplane_message_builder_impl.h"
#include "../serdes/ofh_cuplane_constants.h"
#include "../support/network_order_binary_serializer.h"
#include "ocudu/ofh/compression/compression_properties.h"
#include "ocudu/ofh/compression/iq_compressor.h"
#include "ocudu/phy/support/resource_grid_reader.h"
#include "ocudu/ran/resource_block.h"
#include <algorithm>
#include <vector>

using namespace ocudu;
using namespace ofh;

/// Encodes data direction, payload version and filter index.
static uint8_t encode_data_direction()
{
  uint8_t octet = 0;
  // Data direction (DL); offset: 7, 1 bit long.
  octet |= uint8_t(to_value(data_direction::downlink)) << 7u;
  // Payload version; offset: 4, 3 bits long.
  octet |= uint8_t(OFH_PAYLOAD_VERSION) << 4u;
  // Filter index is fixed to 0, skip it.

  return octet;
}

/// Encodes subframe index and MSB bits of slot index.
static uint8_t encode_subframe_and_slot(slot_point slot)
{
  uint8_t octet = 0;
  // Subframe index; offset: 4, 4 bits long.
  octet |= uint8_t(slot.subframe_index()) << 4u;
  // Four MSBs of the slot index within 1ms subframe; offset: 4, 6 bits long.
  octet |= uint8_t(slot.subframe_slot_index() >> 2u);

  return octet;
}

/// Encodes remaining LSB bits of the slot index and then symbol index.
static uint8_t encode_slot_lsb_and_symbol(const uplane_message_params& params)
{
  uint8_t octet = 0;
  octet |= uint8_t(params.slot.subframe_slot_index() & 0x3) << 6u;
  octet |= uint8_t(params.symbol_id);

  return octet;
}

/// Encodes and returns the 4 LSB bits section id, the rb bit, number of symbols and the 2 MSB bits of start PRB.
static uint8_t encode_sect_id_rb_symbols(const uplane_message_params& params)
{
  uint8_t octet = 0;
  octet |= uint8_t(rb_id_type::every_rb_used) << 3u;
  octet |= uint8_t(symbol_incr_type::current_symbol_number) << 2u;
  octet |= uint8_t(params.start_prb >> 8u) & 0x3;

  return octet;
}

/// Writes radio application header to the output buffer.
static void build_radio_app_header(network_order_binary_serializer& serializer, const uplane_message_params& params)
{
  // Data direction + payload version + filter index (1 Byte).
  serializer.write(encode_data_direction());

  // Write FrameId (1 Byte) - a counter for 10 ms frames (wrapping period 2.56 seconds), range [0, 256].
  serializer.write(uint8_t(params.slot.sfn()));

  // Write subframe and slot index (1 Byte).
  serializer.write(encode_subframe_and_slot(params.slot));

  // Write 2 LSBs of slot index and symbol index.
  serializer.write(encode_slot_lsb_and_symbol(params));
}

/// Writes section1 header to the output buffer.
static void build_section1_header(network_order_binary_serializer& serializer, const uplane_message_params& params)
{
  // Section ID is fixed to 0.
  serializer.write(uint8_t(0));

  // Write rb, symInc and 2 MSB bits of start PRB.
  serializer.write(encode_sect_id_rb_symbols(params));

  // Write remaining LSBs of start PRB.
  serializer.write(uint8_t(params.start_prb));

  // Write number of PRBs.
  serializer.write(uint8_t((params.nof_prb > std::numeric_limits<uint8_t>::max()) ? 0 : params.nof_prb));
}

void uplane_message_builder_impl::serialize_iq_data(network_order_binary_serializer& serializer,
                                                    span<const cbf16_t>              iq_data,
                                                    unsigned                         nof_prbs,
                                                    const ru_compression_params&     compr_params)
{
  if (OCUDU_UNLIKELY(logger.debug.enabled())) {
    logger.debug("Packing '{}' PRBs inside a User-Plane message using compression type '{}' and bitwidth '{}'",
                 nof_prbs,
                 to_string(compr_params.type),
                 compr_params.data_width);
  }

  // Serialize compression header.
  serialize_compression_header(serializer, compr_params);

  if (ud_comp_length_support) {
    // The udCompLen field shall only be present for the following compression methods:
    // "BFP + selective RE sending" or "Modulation compression + selective RE sending".
    if (compr_params.type == compression_type::bfp_selective || compr_params.type == compression_type::mod_selective) {
      units::bits prb_iq_data_size_bits(NOF_SUBCARRIERS_PER_RB * 2U * compr_params.data_width);
      uint16_t    udCompLen = prb_iq_data_size_bits.round_up_to_bytes().value();
      serializer.write(udCompLen);
    }
  }

  // Size in bytes of one compressed PRB using the given compression parameters.
  units::bytes prb_size           = get_compressed_prb_size(compr_params);
  units::bytes bytes_to_serialize = prb_size * nof_prbs;

  span<uint8_t> compr_prb_view = serializer.get_view_and_advance(bytes_to_serialize.value());
  compressor.compress(compr_prb_view, iq_data, compr_params);
}

bool uplane_message_builder_impl::serialize_iq_data(network_order_binary_serializer& serializer,
                                                    const resource_grid_reader&      grid,
                                                    unsigned                         port,
                                                    const uplane_message_params&     params)
{
  const ru_compression_params& compr_params = params.compression_params;

  if (OCUDU_UNLIKELY(logger.debug.enabled())) {
    logger.debug("Packing '{}' PRBs directly from device grid inside a User-Plane message using compression type '{}' "
                 "and bitwidth '{}'",
                 params.nof_prb,
                 to_string(compr_params.type),
                 compr_params.data_width);
  }

  serialize_compression_header(serializer, compr_params);

  if (ud_comp_length_support) {
    if (compr_params.type == compression_type::bfp_selective || compr_params.type == compression_type::mod_selective) {
      units::bits prb_iq_data_size_bits(NOF_SUBCARRIERS_PER_RB * 2U * compr_params.data_width);
      uint16_t    udCompLen = prb_iq_data_size_bits.round_up_to_bytes().value();
      serializer.write(udCompLen);
    }
  }

  units::bytes prb_size           = get_compressed_prb_size(compr_params);
  units::bytes bytes_to_serialize = prb_size * params.nof_prb;

  span<uint8_t> compr_prb_view = serializer.get_view_and_advance(bytes_to_serialize.value());
  return compressor.compress_device_symbol(
      compr_prb_view, grid, port, params.symbol_id, params.start_prb, params.nof_prb, compr_params);
}

unsigned uplane_message_builder_impl::build_message(span<uint8_t>                buffer,
                                                    span<const cbf16_t>          iq_data,
                                                    const uplane_message_params& params)
{
  ocudu_assert(params.sect_type == section_type::type_1, "Unsupported section type");
  ocudu_assert(iq_data.size() == params.nof_prb * NOF_SUBCARRIERS_PER_RB,
               "The number of PRBs derived from the IQ samples is '{}' and requested number of PRBs to pack is '{}'",
               iq_data.size() / NOF_SUBCARRIERS_PER_RB,
               params.nof_prb);

  network_order_binary_serializer serializer(buffer.data());

  build_radio_app_header(serializer, params);
  build_section1_header(serializer, params);
  serialize_iq_data(serializer, iq_data, params.nof_prb, params.compression_params);

  return serializer.get_offset();
}

bool uplane_message_builder_impl::build_message(span<uint8_t>                buffer,
                                                const resource_grid_reader&  grid,
                                                unsigned                     port,
                                                const uplane_message_params& params,
                                                unsigned&                    bytes_written)
{
  bytes_written = 0;
  ocudu_assert(params.sect_type == section_type::type_1, "Unsupported section type");
  if (params.start_prb + params.nof_prb > grid.get_nof_subc() / NOF_SUBCARRIERS_PER_RB) {
    return false;
  }

  network_order_binary_serializer serializer(buffer.data());

  build_radio_app_header(serializer, params);
  build_section1_header(serializer, params);
  if (!serialize_iq_data(serializer, grid, port, params)) {
    return false;
  }

  bytes_written = serializer.get_offset();
  return true;
}

bool uplane_message_builder_impl::build_messages(span<span<uint8_t>>          buffers,
                                                 const resource_grid_reader&  grid,
                                                 unsigned                     first_port,
                                                 const uplane_message_params& params,
                                                 span<unsigned>               bytes_written)
{
  for (unsigned& value : bytes_written) {
    value = 0;
  }

  ocudu_assert(params.sect_type == section_type::type_1, "Unsupported section type");
  if (buffers.empty() || (bytes_written.size() < buffers.size())) {
    return false;
  }
  if (params.start_prb + params.nof_prb > grid.get_nof_subc() / NOF_SUBCARRIERS_PER_RB) {
    return false;
  }

  const ru_compression_params& compr_params = params.compression_params;
  if (OCUDU_UNLIKELY(logger.debug.enabled())) {
    logger.debug("Packing '{}' PRBs directly from device grid for '{}' contiguous User-Plane ports using compression "
                 "type '{}' and bitwidth '{}'",
                 params.nof_prb,
                 buffers.size(),
                 to_string(compr_params.type),
                 compr_params.data_width);
  }

  units::bytes prb_size    = get_compressed_prb_size(compr_params);
  unsigned     port_bytes  = (prb_size * params.nof_prb).value();
  unsigned     port_stride = port_bytes;
  unsigned     nof_ports   = buffers.size();

  thread_local std::vector<span<uint8_t>> payload_views;
  payload_views.resize(nof_ports);
  for (unsigned i_port = 0; i_port != nof_ports; ++i_port) {
    network_order_binary_serializer serializer(buffers[i_port].data());

    build_radio_app_header(serializer, params);
    build_section1_header(serializer, params);
    serialize_compression_header(serializer, compr_params);

    if (ud_comp_length_support) {
      if (compr_params.type == compression_type::bfp_selective ||
          compr_params.type == compression_type::mod_selective) {
        units::bits prb_iq_data_size_bits(NOF_SUBCARRIERS_PER_RB * 2U * compr_params.data_width);
        uint16_t    udCompLen = prb_iq_data_size_bits.round_up_to_bytes().value();
        serializer.write(udCompLen);
      }
    }

    payload_views[i_port] = serializer.get_view_and_advance(port_bytes);
    bytes_written[i_port] = serializer.get_offset();
  }

  if (compressor.compress_device_symbols_to_buffers(span<span<uint8_t>>(payload_views.data(), nof_ports),
                                                    grid,
                                                    first_port,
                                                    nof_ports,
                                                    params.symbol_id,
                                                    params.start_prb,
                                                    params.nof_prb,
                                                    compr_params)) {
    return true;
  }

  size_t                            batch_nofbytes = static_cast<size_t>(nof_ports) * port_stride;
  thread_local std::vector<uint8_t> batch_payload;
  batch_payload.resize(batch_nofbytes);

  if (!compressor.compress_device_symbols(span<uint8_t>(batch_payload.data(), batch_payload.size()),
                                          port_stride,
                                          grid,
                                          first_port,
                                          nof_ports,
                                          params.symbol_id,
                                          params.start_prb,
                                          params.nof_prb,
                                          compr_params)) {
    for (unsigned& value : bytes_written) {
      value = 0;
    }
    return false;
  }

  for (unsigned i_port = 0; i_port != nof_ports; ++i_port) {
    std::copy_n(
        batch_payload.data() + static_cast<size_t>(i_port) * port_stride, port_bytes, payload_views[i_port].data());
  }

  return true;
}

bool uplane_message_builder_impl::build_symbol_batch_messages(span<span<uint8_t>>          buffers,
                                                              const resource_grid_reader&  grid,
                                                              unsigned                     first_port,
                                                              unsigned                     first_symbol,
                                                              unsigned                     nof_symbols,
                                                              const uplane_message_params& params,
                                                              span<unsigned>               bytes_written)
{
  for (unsigned& value : bytes_written) {
    value = 0;
  }

  ocudu_assert(params.sect_type == section_type::type_1, "Unsupported section type");
  if ((nof_symbols == 0) || buffers.empty() || (bytes_written.size() < buffers.size())) {
    return false;
  }
  if ((buffers.size() % nof_symbols) != 0) {
    return false;
  }
  if (params.start_prb + params.nof_prb > grid.get_nof_subc() / NOF_SUBCARRIERS_PER_RB) {
    return false;
  }

  const unsigned               nof_ports    = buffers.size() / nof_symbols;
  const ru_compression_params& compr_params = params.compression_params;
  if (OCUDU_UNLIKELY(logger.debug.enabled())) {
    logger.debug("Packing '{}' PRBs directly from device grid for '{}' symbols and '{}' contiguous User-Plane ports "
                 "using compression type '{}' and bitwidth '{}'",
                 params.nof_prb,
                 nof_symbols,
                 nof_ports,
                 to_string(compr_params.type),
                 compr_params.data_width);
  }

  units::bytes prb_size      = get_compressed_prb_size(compr_params);
  unsigned     port_bytes    = (prb_size * params.nof_prb).value();
  unsigned     port_stride   = port_bytes;
  unsigned     symbol_stride = nof_ports * port_stride;

  thread_local std::vector<span<uint8_t>> payload_views;
  payload_views.resize(buffers.size());
  for (unsigned i_symbol = 0; i_symbol != nof_symbols; ++i_symbol) {
    for (unsigned i_port = 0; i_port != nof_ports; ++i_port) {
      const unsigned        buffer_index  = i_symbol * nof_ports + i_port;
      uplane_message_params symbol_params = params;
      symbol_params.symbol_id             = first_symbol + i_symbol;

      network_order_binary_serializer serializer(buffers[buffer_index].data());

      build_radio_app_header(serializer, symbol_params);
      build_section1_header(serializer, symbol_params);
      serialize_compression_header(serializer, compr_params);

      if (ud_comp_length_support) {
        if (compr_params.type == compression_type::bfp_selective ||
            compr_params.type == compression_type::mod_selective) {
          units::bits prb_iq_data_size_bits(NOF_SUBCARRIERS_PER_RB * 2U * compr_params.data_width);
          uint16_t    udCompLen = prb_iq_data_size_bits.round_up_to_bytes().value();
          serializer.write(udCompLen);
        }
      }

      payload_views[buffer_index] = serializer.get_view_and_advance(port_bytes);
      bytes_written[buffer_index] = serializer.get_offset();
    }
  }

  if (compressor.compress_device_symbol_batch_to_buffers(
          span<span<uint8_t>>(payload_views.data(), payload_views.size()),
          grid,
          first_port,
          nof_ports,
          first_symbol,
          nof_symbols,
          params.start_prb,
          params.nof_prb,
          compr_params)) {
    return true;
  }

  size_t                            batch_nofbytes = static_cast<size_t>(nof_symbols) * symbol_stride;
  thread_local std::vector<uint8_t> batch_payload;
  batch_payload.resize(batch_nofbytes);

  if (!compressor.compress_device_symbol_batch(span<uint8_t>(batch_payload.data(), batch_payload.size()),
                                               symbol_stride,
                                               port_stride,
                                               grid,
                                               first_port,
                                               nof_ports,
                                               first_symbol,
                                               nof_symbols,
                                               params.start_prb,
                                               params.nof_prb,
                                               compr_params)) {
    for (unsigned& value : bytes_written) {
      value = 0;
    }
    return false;
  }

  for (unsigned i_symbol = 0; i_symbol != nof_symbols; ++i_symbol) {
    for (unsigned i_port = 0; i_port != nof_ports; ++i_port) {
      const unsigned buffer_index = i_symbol * nof_ports + i_port;
      const size_t   payload_offset =
          static_cast<size_t>(i_symbol) * symbol_stride + static_cast<size_t>(i_port) * port_stride;
      std::copy_n(batch_payload.data() + payload_offset, port_bytes, payload_views[buffer_index].data());
    }
  }

  return true;
}
