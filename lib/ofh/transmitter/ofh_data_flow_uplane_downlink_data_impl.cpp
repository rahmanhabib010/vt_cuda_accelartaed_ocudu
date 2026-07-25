// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ofh_data_flow_uplane_downlink_data_impl.h"
#include "ofh_uplane_fragment_size_calculator.h"
#include "ocudu/ocuduvec/conversion.h"
#include "ocudu/ofh/ethernet/ethernet_frame_pool.h"
#include "ocudu/ofh/timing/slot_symbol_point.h"
#include "ocudu/phy/support/resource_grid_context.h"
#include "ocudu/phy/support/resource_grid_reader.h"
#include "ocudu/phy/support/shared_resource_grid.h"
#include "ocudu/ran/resource_block.h"
#include <thread>
#include <vector>

using namespace ocudu;
using namespace ofh;

/// Generates and returns the downlink Open Fronthaul user parameters for the given context.
static uplane_message_params generate_dl_ofh_user_parameters(slot_point                   slot,
                                                             unsigned                     symbol_id,
                                                             unsigned                     start_prb,
                                                             unsigned                     nof_prb,
                                                             const ru_compression_params& comp)
{
  uplane_message_params params;
  params.direction                     = data_direction::downlink;
  params.slot                          = slot;
  params.filter_index                  = filter_index_type::standard_channel_filter;
  params.start_prb                     = start_prb;
  params.nof_prb                       = nof_prb;
  params.symbol_id                     = symbol_id;
  params.sect_type                     = section_type::type_1;
  params.compression_params.type       = comp.type;
  params.compression_params.data_width = comp.data_width;

  return params;
}

/// Generates and returns the eCPRI IQ data parameters.
static ecpri::iq_data_parameters generate_ecpri_data_parameters(uint16_t seq_id, uint16_t eaxc)
{
  ecpri::iq_data_parameters params;
  // Only supporting 1 Port, 1 band and 1 CC.
  params.pc_id  = eaxc;
  params.seq_id = 0;

  // Set seq_id.
  params.seq_id |= (seq_id & 0x00ff) << 8;
  // Set the E bit to 1 and subsequence ID to 0. E bit set to 1 indicates that there is no radio transport layer
  // fragmentation.
  params.seq_id |= uint16_t(1U) << 7;

  return params;
}

data_flow_uplane_downlink_data_impl::data_flow_uplane_downlink_data_impl(
    const data_flow_uplane_downlink_data_impl_config&  config,
    data_flow_uplane_downlink_data_impl_dependencies&& dependencies) :
  logger(*dependencies.logger),
  nof_symbols_per_slot(get_nsymb_per_slot(config.cp)),
  ru_nof_prbs(config.ru_nof_prbs),
  sector_id(config.sector),
  compr_params(config.compr_params),
  frame_pool(std::move(dependencies.frame_pool)),
  compressor_sel(std::move(dependencies.compressor_sel)),
  eth_builder(std::move(dependencies.eth_builder)),
  ecpri_builder(std::move(dependencies.ecpri_builder)),
  up_builder(std::move(dependencies.up_builder)),
  formatted_trace_names(config.dl_eaxc)
{
  ocudu_assert(eth_builder, "Invalid Ethernet VLAN packet builder");
  ocudu_assert(ecpri_builder, "Invalid eCPRI packet builder");
  ocudu_assert(compressor_sel, "Invalid compressor selector");
  ocudu_assert(up_builder, "Invalid User-Plane message builder");
  ocudu_assert(frame_pool, "Invalid frame pool");
}

void data_flow_uplane_downlink_data_impl::enqueue_section_type_1_message(
    const data_flow_uplane_resource_grid_context& context,
    const shared_resource_grid&                   grid)
{
  trace_point tp = ofh_tracer.now();
  enqueue_section_type_1_message_symbol_burst(context, grid);

  ofh_tracer << trace_event(formatted_trace_names[context.eaxc].c_str(), tp);
}

void data_flow_uplane_downlink_data_impl::enqueue_section_type_1_messages(
    const data_flow_uplane_resource_grid_context& base_context,
    unsigned                                      first_port,
    span<const unsigned>                          eaxcs,
    const shared_resource_grid&                   grid)
{
  if (eaxcs.empty()) {
    return;
  }

  trace_point tp = ofh_tracer.now();
  if (enqueue_section_type_1_messages_slot_burst(base_context, first_port, eaxcs, grid)) {
    for (unsigned eaxc : eaxcs) {
      ofh_tracer << trace_event(formatted_trace_names[eaxc].c_str(), tp);
    }
    return;
  }

  if (enqueue_section_type_1_messages_symbol_burst(base_context, first_port, eaxcs, grid)) {
    for (unsigned eaxc : eaxcs) {
      ofh_tracer << trace_event(formatted_trace_names[eaxc].c_str(), tp);
    }
    return;
  }

  for (unsigned i_port = 0, e = eaxcs.size(); i_port != e; ++i_port) {
    data_flow_uplane_resource_grid_context context = base_context;
    context.port                                    = first_port + i_port;
    context.eaxc                                    = eaxcs[i_port];
    enqueue_section_type_1_message(context, grid);
  }
}

void data_flow_uplane_downlink_data_impl::enqueue_section_type_1_message_symbol_burst(
    const data_flow_uplane_resource_grid_context& context,
    const shared_resource_grid&                   grid)
{
  const resource_grid_reader& reader = grid.get_reader();

  // Temporary buffer used to store IQ data when the RU operating bandwidth is not the same to the cell bandwidth.
  std::array<cbf16_t, MAX_NOF_SUBCARRIERS> temp_buffer;
  if (OCUDU_UNLIKELY(ru_nof_prbs * NOF_SUBCARRIERS_PER_RB != reader.get_nof_subc())) {
    // Zero out the elements that won't be filled after reading the resource grid.
    std::fill(temp_buffer.begin() + reader.get_nof_subc(), temp_buffer.end(), 0);
  }

  units::bytes headers_size = eth_builder->get_header_size() +
                              ecpri_builder->get_header_size(ecpri::message_type::iq_data) +
                              up_builder->get_header_size(compr_params);

  // Iterate over all the symbols.
  for (unsigned symbol_id = context.symbol_range.start(), symbol_end = context.symbol_range.length();
       symbol_id != symbol_end;
       ++symbol_id) {
    slot_symbol_point symbol_point(context.slot, symbol_id, nof_symbols_per_slot);

    const bool full_bandwidth_grid = (ru_nof_prbs * NOF_SUBCARRIERS_PER_RB == reader.get_nof_subc());
    span<const cbf16_t> iq_data;
    bool                host_iq_data_ready = false;
    auto get_host_iq_data = [&]() {
      if (!host_iq_data_ready) {
        if (OCUDU_LIKELY(full_bandwidth_grid)) {
          iq_data = reader.get_view(context.port, symbol_id);
        } else {
          span<cbf16_t> temp_iq_data(temp_buffer.data(), ru_nof_prbs * NOF_SUBCARRIERS_PER_RB);
          reader.get(temp_iq_data.first(reader.get_nof_subc()), context.port, symbol_id, 0);
          iq_data = temp_iq_data;
        }
        host_iq_data_ready = true;
      }
      return iq_data;
    };

    // Split the data into multiple messages when it does not fit into a single one.
    ofh_uplane_fragment_size_calculator prb_fragment_calculator(0, ru_nof_prbs, compr_params);
    bool                                is_last_fragment   = false;
    unsigned                            fragment_start_prb = 0U;
    unsigned                            fragment_nof_prbs  = 0U;
    do {
      trace_point pool_access_tp = ofh_tracer.now();
      auto        scoped_buffer  = frame_pool->reserve(symbol_point);
      ofh_tracer << trace_event("ofh_uplane_pool_access", pool_access_tp);

      if (OCUDU_UNLIKELY(!scoped_buffer)) {
        logger.warning(
            "Sector#{}: not enough space in the buffer pool to create a downlink User-Plane message for slot "
            "'{}' and eAxC '{}', symbol_id '{}'",
            sector_id,
            context.slot,
            context.eaxc,
            symbol_id);
        return;
      }
      span<uint8_t> data = scoped_buffer->get_buffer();

      is_last_fragment = prb_fragment_calculator.calculate_fragment_size(
          fragment_start_prb, fragment_nof_prbs, data.size() - headers_size.value());

      // Skip frame buffers so small that cannot carry one PRB.
      if (OCUDU_UNLIKELY(fragment_nof_prbs == 0)) {
        logger.warning("Sector#{}: skipped frame buffer as it cannot store data for a single PRB, required buffer size "
                       "is '{}' bytes",
                       sector_id,
                       data.size());

        continue;
      }

      ofh_tracer << instant_trace_event{"ofh_uplane_symbol", instant_trace_event::cpu_scope::thread};

      uplane_message_params up_params =
          generate_dl_ofh_user_parameters(context.slot, symbol_id, fragment_start_prb, fragment_nof_prbs, compr_params);

      unsigned used_size = 0;
      if (full_bandwidth_grid && reader.supports_device_grid_reading()) {
        used_size = enqueue_section_type_1_message_symbol(reader, up_params, context.port, context.eaxc, data);
      }
      if (used_size == 0) {
        span<const cbf16_t> host_iq_data = get_host_iq_data();
        used_size = enqueue_section_type_1_message_symbol(
            host_iq_data.subspan(fragment_start_prb * NOF_SUBCARRIERS_PER_RB,
                                 fragment_nof_prbs * NOF_SUBCARRIERS_PER_RB),
            up_params,
            context.eaxc,
            data);
      }
      scoped_buffer->set_size(used_size);
    } while (!is_last_fragment);
  }
}

bool data_flow_uplane_downlink_data_impl::enqueue_section_type_1_messages_symbol_burst(
    const data_flow_uplane_resource_grid_context& base_context,
    unsigned                                      first_port,
    span<const unsigned>                          eaxcs,
    const shared_resource_grid&                   grid)
{
  const resource_grid_reader& reader = grid.get_reader();
  if (!reader.supports_device_grid_reading()) {
    return false;
  }
  if (ru_nof_prbs * NOF_SUBCARRIERS_PER_RB != reader.get_nof_subc()) {
    return false;
  }
  if ((first_port + eaxcs.size()) > reader.get_nof_ports()) {
    return false;
  }

  units::bytes headers_size = eth_builder->get_header_size() +
                              ecpri_builder->get_header_size(ecpri::message_type::iq_data) +
                              up_builder->get_header_size(compr_params);
  units::bytes offset = eth_builder->get_header_size() + ecpri_builder->get_header_size(ecpri::message_type::iq_data);

  bool committed_buffers = false;

  // Iterate over all the symbols.
  for (unsigned symbol_id = base_context.symbol_range.start(), symbol_end = base_context.symbol_range.length();
       symbol_id != symbol_end;
       ++symbol_id) {
    slot_symbol_point symbol_point(base_context.slot, symbol_id, nof_symbols_per_slot);

    // Split the data into multiple messages when it does not fit into a single one.
    ofh_uplane_fragment_size_calculator prb_fragment_calculator(0, ru_nof_prbs, compr_params);
    bool                                is_last_fragment   = false;
    unsigned                            fragment_start_prb = 0U;
    unsigned                            fragment_nof_prbs  = 0U;
    do {
      static_vector<ether::scoped_frame_buffer, MAX_NOF_SUPPORTED_EAXC> scoped_buffers;
      std::array<span<uint8_t>, MAX_NOF_SUPPORTED_EAXC>                 ofh_buffers;
      std::array<span<uint8_t>, MAX_NOF_SUPPORTED_EAXC>                 data_buffers;
      std::array<unsigned, MAX_NOF_SUPPORTED_EAXC>                      up_bytes_written = {};

      trace_point pool_access_tp = ofh_tracer.now();
      auto        first_buffer   = frame_pool->reserve(symbol_point);
      ofh_tracer << trace_event("ofh_uplane_pool_access", pool_access_tp);

      if (OCUDU_UNLIKELY(!first_buffer)) {
        logger.warning(
            "Sector#{}: not enough space in the buffer pool to create downlink User-Plane message batch for slot "
            "'{}', symbol_id '{}'",
            sector_id,
            base_context.slot,
            symbol_id);
        return committed_buffers;
      }

      span<uint8_t> first_data = first_buffer->get_buffer();
      if (OCUDU_UNLIKELY(first_data.size() < headers_size.value())) {
        return committed_buffers;
      }
      is_last_fragment        = prb_fragment_calculator.calculate_fragment_size(
          fragment_start_prb, fragment_nof_prbs, first_data.size() - headers_size.value());

      // Skip frame buffers so small that cannot carry one PRB.
      if (OCUDU_UNLIKELY(fragment_nof_prbs == 0)) {
        logger.warning("Sector#{}: skipped frame buffer as it cannot store data for a single PRB, required buffer size "
                       "is '{}' bytes",
                       sector_id,
                       first_data.size());

        continue;
      }

      data_buffers[0] = first_data;
      ofh_buffers[0]  = span<uint8_t>(first_data).last(first_data.size() - offset.value());
      scoped_buffers.emplace_back(std::move(first_buffer));

      for (unsigned i_port = 1, e = eaxcs.size(); i_port != e; ++i_port) {
        trace_point extra_pool_access_tp = ofh_tracer.now();
        auto        scoped_buffer        = frame_pool->reserve(symbol_point);
        ofh_tracer << trace_event("ofh_uplane_pool_access", extra_pool_access_tp);

        if (OCUDU_UNLIKELY(!scoped_buffer)) {
          logger.warning(
              "Sector#{}: not enough space in the buffer pool to create downlink User-Plane message batch for slot "
              "'{}', symbol_id '{}'",
              sector_id,
              base_context.slot,
              symbol_id);
          return committed_buffers;
        }

        span<uint8_t> data = scoped_buffer->get_buffer();
        if (OCUDU_UNLIKELY(data.size() < headers_size.value())) {
          return committed_buffers;
        }
        data_buffers[i_port] = data;
        ofh_buffers[i_port]  = span<uint8_t>(data).last(data.size() - offset.value());
        scoped_buffers.emplace_back(std::move(scoped_buffer));
      }

      ofh_tracer << instant_trace_event{"ofh_uplane_symbol", instant_trace_event::cpu_scope::thread};

      uplane_message_params up_params = generate_dl_ofh_user_parameters(
          base_context.slot, symbol_id, fragment_start_prb, fragment_nof_prbs, compr_params);

      if (!up_builder->build_messages(span<span<uint8_t>>(ofh_buffers.data(), eaxcs.size()),
                                      reader,
                                      first_port,
                                      up_params,
                                      span<unsigned>(up_bytes_written.data(), eaxcs.size()))) {
        return committed_buffers;
      }

      for (unsigned i_port = 0, e = eaxcs.size(); i_port != e; ++i_port) {
        unsigned used_size =
            finish_section_type_1_message_symbol(up_bytes_written[i_port], up_params, eaxcs[i_port], data_buffers[i_port]);
        scoped_buffers[i_port]->set_size(used_size);
      }
      committed_buffers = true;
    } while (!is_last_fragment);
  }

  return true;
}

bool data_flow_uplane_downlink_data_impl::enqueue_section_type_1_messages_slot_burst(
    const data_flow_uplane_resource_grid_context& base_context,
    unsigned                                      first_port,
    span<const unsigned>                          eaxcs,
    const shared_resource_grid&                   grid)
{
  const resource_grid_reader& reader = grid.get_reader();
  if (!reader.supports_device_grid_reading()) {
    return false;
  }
  if (ru_nof_prbs * NOF_SUBCARRIERS_PER_RB != reader.get_nof_subc()) {
    return false;
  }
  if ((first_port + eaxcs.size()) > reader.get_nof_ports()) {
    return false;
  }

  const unsigned first_symbol = base_context.symbol_range.start();
  const unsigned symbol_end   = base_context.symbol_range.length();
  if (symbol_end <= first_symbol) {
    return false;
  }
  const unsigned nof_symbols = symbol_end - first_symbol;

  units::bytes headers_size = eth_builder->get_header_size() +
                              ecpri_builder->get_header_size(ecpri::message_type::iq_data) +
                              up_builder->get_header_size(compr_params);
  units::bytes offset = eth_builder->get_header_size() + ecpri_builder->get_header_size(ecpri::message_type::iq_data);

  bool committed_buffers = false;

  ofh_uplane_fragment_size_calculator prb_fragment_calculator(0, ru_nof_prbs, compr_params);
  bool                                is_last_fragment   = false;
  unsigned                            fragment_start_prb = 0U;
  unsigned                            fragment_nof_prbs  = 0U;
  do {
    const unsigned nof_messages = nof_symbols * eaxcs.size();
    std::vector<ether::scoped_frame_buffer> scoped_buffers;
    std::vector<span<uint8_t>>              ofh_buffers(nof_messages);
    std::vector<span<uint8_t>>              data_buffers(nof_messages);
    std::vector<unsigned>                   up_bytes_written(nof_messages, 0U);
    scoped_buffers.reserve(nof_messages);

    trace_point pool_access_tp = ofh_tracer.now();
    auto        first_buffer   = frame_pool->reserve(slot_symbol_point(base_context.slot, first_symbol, nof_symbols_per_slot));
    ofh_tracer << trace_event("ofh_uplane_pool_access", pool_access_tp);

    if (OCUDU_UNLIKELY(!first_buffer)) {
      logger.warning(
          "Sector#{}: not enough space in the buffer pool to create downlink User-Plane symbol batch for slot '{}'",
          sector_id,
          base_context.slot);
      return committed_buffers;
    }

    span<uint8_t> first_data = first_buffer->get_buffer();
    if (OCUDU_UNLIKELY(first_data.size() < headers_size.value())) {
      return committed_buffers;
    }
    is_last_fragment = prb_fragment_calculator.calculate_fragment_size(
        fragment_start_prb, fragment_nof_prbs, first_data.size() - headers_size.value());

    if (OCUDU_UNLIKELY(fragment_nof_prbs == 0)) {
      logger.warning("Sector#{}: skipped frame buffer as it cannot store data for a single PRB, required buffer size "
                     "is '{}' bytes",
                     sector_id,
                     first_data.size());

      continue;
    }

    data_buffers[0] = first_data;
    ofh_buffers[0]  = span<uint8_t>(first_data).last(first_data.size() - offset.value());
    scoped_buffers.emplace_back(std::move(first_buffer));

    for (unsigned i_symbol = 0; i_symbol != nof_symbols; ++i_symbol) {
      for (unsigned i_port = 0, e = eaxcs.size(); i_port != e; ++i_port) {
        const unsigned message_index = i_symbol * eaxcs.size() + i_port;
        if (message_index == 0) {
          continue;
        }

        trace_point extra_pool_access_tp = ofh_tracer.now();
        auto        scoped_buffer        = frame_pool->reserve(
            slot_symbol_point(base_context.slot, first_symbol + i_symbol, nof_symbols_per_slot));
        ofh_tracer << trace_event("ofh_uplane_pool_access", extra_pool_access_tp);

        if (OCUDU_UNLIKELY(!scoped_buffer)) {
          logger.warning(
              "Sector#{}: not enough space in the buffer pool to create downlink User-Plane symbol batch for slot "
              "'{}'",
              sector_id,
              base_context.slot);
          return committed_buffers;
        }

        span<uint8_t> data = scoped_buffer->get_buffer();
        if (OCUDU_UNLIKELY(data.size() < headers_size.value())) {
          return committed_buffers;
        }
        data_buffers[message_index] = data;
        ofh_buffers[message_index]  = span<uint8_t>(data).last(data.size() - offset.value());
        scoped_buffers.emplace_back(std::move(scoped_buffer));
      }
    }

    ofh_tracer << instant_trace_event{"ofh_uplane_symbol", instant_trace_event::cpu_scope::thread};

    uplane_message_params up_params = generate_dl_ofh_user_parameters(
        base_context.slot, first_symbol, fragment_start_prb, fragment_nof_prbs, compr_params);

    if (!up_builder->build_symbol_batch_messages(span<span<uint8_t>>(ofh_buffers.data(), ofh_buffers.size()),
                                                 reader,
                                                 first_port,
                                                 first_symbol,
                                                 nof_symbols,
                                                 up_params,
                                                 span<unsigned>(up_bytes_written.data(), up_bytes_written.size()))) {
      return committed_buffers;
    }

    for (unsigned i_symbol = 0; i_symbol != nof_symbols; ++i_symbol) {
      for (unsigned i_port = 0, e = eaxcs.size(); i_port != e; ++i_port) {
        const unsigned message_index = i_symbol * eaxcs.size() + i_port;
        uplane_message_params symbol_params = up_params;
        symbol_params.symbol_id             = first_symbol + i_symbol;
        unsigned used_size = finish_section_type_1_message_symbol(
            up_bytes_written[message_index], symbol_params, eaxcs[i_port], data_buffers[message_index]);
        scoped_buffers[message_index]->set_size(used_size);
      }
    }
    committed_buffers = true;
  } while (!is_last_fragment);

  return true;
}

unsigned data_flow_uplane_downlink_data_impl::enqueue_section_type_1_message_symbol(span<const cbf16_t> iq_symbol_data,
                                                                                    const uplane_message_params& params,
                                                                                    unsigned                     eaxc,
                                                                                    span<uint8_t>                buffer)
{
  // Build the Open Fronthaul data message. Only one port supported.
  units::bytes  ether_header_size = eth_builder->get_header_size();
  units::bytes  ecpri_hdr_size    = ecpri_builder->get_header_size(ecpri::message_type::iq_data);
  units::bytes  offset            = ether_header_size + ecpri_hdr_size;
  span<uint8_t> ofh_buffer        = span<uint8_t>(buffer).last(buffer.size() - offset.value());
  unsigned      bytes_written     = up_builder->build_message(ofh_buffer, iq_symbol_data, params);

  return finish_section_type_1_message_symbol(bytes_written, params, eaxc, buffer);
}

unsigned data_flow_uplane_downlink_data_impl::enqueue_section_type_1_message_symbol(
    const resource_grid_reader&      reader,
    const uplane_message_params&     params,
    unsigned                         port,
    unsigned                         eaxc,
    span<uint8_t>                    buffer)
{
  units::bytes  ether_header_size = eth_builder->get_header_size();
  units::bytes  ecpri_hdr_size    = ecpri_builder->get_header_size(ecpri::message_type::iq_data);
  units::bytes  offset            = ether_header_size + ecpri_hdr_size;
  span<uint8_t> ofh_buffer        = span<uint8_t>(buffer).last(buffer.size() - offset.value());
  unsigned      bytes_written     = 0;
  if (!up_builder->build_message(ofh_buffer, reader, port, params, bytes_written)) {
    return 0;
  }

  return finish_section_type_1_message_symbol(bytes_written, params, eaxc, buffer);
}

unsigned data_flow_uplane_downlink_data_impl::finish_section_type_1_message_symbol(
    unsigned                     bytes_written,
    const uplane_message_params& params,
    unsigned                     eaxc,
    span<uint8_t>                buffer)
{
  units::bytes ether_header_size = eth_builder->get_header_size();
  units::bytes ecpri_hdr_size    = ecpri_builder->get_header_size(ecpri::message_type::iq_data);

  // Add eCPRI header. Create a subspan with the payload that skips the Ethernet header.
  span<uint8_t> ecpri_buffer =
      span<uint8_t>(buffer).subspan(ether_header_size.value(), ecpri_hdr_size.value() + bytes_written);
  ecpri_builder->build_data_packet(ecpri_buffer, generate_ecpri_data_parameters(up_seq_gen.generate(eaxc), eaxc));

  // Update the number of bytes written.
  bytes_written += ecpri_hdr_size.value();

  // Add Ethernet header.
  span<uint8_t> eth_buffer = span<uint8_t>(buffer).first(ether_header_size.value() + bytes_written);
  eth_builder->build_frame(eth_buffer);

  if (OCUDU_UNLIKELY(logger.debug.enabled())) {
    logger.debug("Sector#{}: packing a downlink User-Plane message for slot '{}' and eAxC '{}', symbol_id '{}', PRB "
                 "range '{}:{}', size '{}' bytes",
                 sector_id,
                 params.slot,
                 eaxc,
                 params.symbol_id,
                 params.start_prb,
                 params.nof_prb,
                 eth_buffer.size());
  }

  return eth_buffer.size();
}

data_flow_message_encoding_metrics_collector* data_flow_uplane_downlink_data_impl::get_metrics_collector()
{
  return nullptr;
}
