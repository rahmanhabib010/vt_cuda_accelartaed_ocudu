// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief GPU-accelerated PDSCH Transport Block encoder using CUDA.
///
/// This component performs end-to-end GPU processing for PDSCH transport block encoding:
/// - CRC attachment (CRC24A/CRC16 for TB, CRC24B for CBs)
/// - Code block segmentation
/// - LDPC encoding
/// - Rate matching
/// - Scrambling
/// - Modulation
///
/// The entire chain is executed on GPU with only TB input and symbol output transfers.

#pragma once

#include "ocudu/adt/complex.h"
#include "ocudu/adt/span.h"
#include <cstdint>
#include <memory>

namespace ocudu {

/// \brief GPU-accelerated PDSCH Transport Block encoder.
///
/// Performs the complete PDSCH TX chain on GPU from raw transport block bytes
/// to modulated symbols.
class pdsch_tb_encoder_gpu
{
public:
  /// Configuration for TB encoding.
  struct config {
    unsigned tb_size_bits;       ///< A - Transport block size in bits.
    unsigned num_layers;         ///< Number of MIMO layers.
    unsigned modulation_order;   ///< Q_m - Bits per symbol (2,4,6,8).
    unsigned num_coded_bits;     ///< G - Total number of coded bits.
    unsigned rv;                 ///< Redundancy version (0-3).
    uint16_t n_rnti;             ///< RNTI for scrambling.
    uint16_t n_id;               ///< Scrambling identity (0-1023).
    uint8_t  cw_index;           ///< Codeword index (0 or 1).
    unsigned base_graph;         ///< LDPC base graph (1 or 2) from segmenter.
    unsigned nof_codeblocks;     ///< Number of codeblocks from segmenter.
    unsigned lifting_size;       ///< Z - LDPC lifting size from segmenter.
    unsigned nof_filler_bits;    ///< F - Number of filler bits from segmenter.
    unsigned nof_short_segments; ///< Number of CBs with E_short from segmenter.
    unsigned E_short;            ///< Rate-matched bits for short CBs from segmenter.
    unsigned E_long;             ///< Rate-matched bits for long CBs from segmenter.
  };

  /// GPU encoder timing from the previous completed encode call.
  struct timing_stats {
    bool     valid       = false;
    bool     enabled     = false;
    float    h2d_us      = 0.0F;
    float    encode_us   = 0.0F;
    float    d2h_us      = 0.0F;
    float    total_us    = 0.0F;
    unsigned nof_cbs     = 0;
    unsigned nof_symbols = 0;
  };

  virtual ~pdsch_tb_encoder_gpu() = default;

  /// \brief Encode transport block and modulate to symbols.
  ///
  /// Performs the complete PDSCH TX chain on GPU:
  /// 1. Upload TB bytes to GPU
  /// 2. Attach CRC (CRC24A or CRC16)
  /// 3. Segment into code blocks (with CRC24B if needed)
  /// 4. LDPC encode all code blocks
  /// 5. Rate match all code blocks
  /// 6. Scramble
  /// 7. Modulate to symbols
  ///
  /// \param[in] tb_bytes Raw transport block bytes.
  /// \param[in] cfg      Encoding configuration.
  /// \return Number of output symbols.
  virtual unsigned encode(span<const uint8_t> tb_bytes, const config& cfg) = 0;

  /// \brief Get modulated symbols from GPU as ci8_t (quantized).
  ///
  /// Copies symbols from GPU to host buffer with float->int8 conversion.
  /// Call after encode().
  ///
  /// \param[out] symbols Output buffer for ci8_t symbols.
  /// \param[in]  offset  Starting symbol offset.
  /// \param[in]  count   Number of symbols to copy.
  virtual void get_symbols(span<ci8_t> symbols, unsigned offset, unsigned count) = 0;

  /// \brief Get modulated symbols from GPU as cf_t (float).
  ///
  /// Copies symbols from GPU to host buffer without conversion.
  /// Call after encode().
  ///
  /// \param[out] symbols Output buffer for cf_t symbols.
  /// \param[in]  offset  Starting symbol offset.
  /// \param[in]  count   Number of symbols to copy.
  virtual void get_symbols_float(span<cf_t> symbols, unsigned offset, unsigned count) = 0;

  /// \brief Get device pointer to symbols for GPU-resident downstream processing.
  /// \return Device pointer to modulated symbols in the backend-native complex-float layout.
  virtual void* get_device_symbols() const = 0;

  /// \brief Get device pointer to INT8 symbols for GPU-resident downstream processing.
  ///
  /// The fast PDSCH encoder path produces interleaved INT8 IQ symbols, i.e. two
  /// int8 values per modulation symbol. This is the symbol layout consumed by the
  /// host PDSCH resource-grid mapper.
  ///
  /// \return Device pointer to interleaved INT8 IQ symbols.
  virtual const int8_t* get_device_symbols_int8() const = 0;

  /// \brief Enables or disables the eager full-symbol device-to-host copy after encode.
  ///
  /// GPU-resident downstream consumers should disable this copy before calling
  /// \ref encode. Host consumers can leave it enabled so the transfer overlaps
  /// subsequent CPU work.
  virtual void set_defer_symbol_download(bool defer) = 0;

  /// \brief Get the backend execution context used by the previous encode call.
  ///
  /// The return type is intentionally opaque to keep backend runtime types out
  /// of this interface.
  virtual void* get_execution_context() const = 0;

  /// \brief Synchronize the previous encode call and collect pending timing.
  virtual void synchronize() = 0;

  /// \brief Get number of symbols from last encode call.
  virtual unsigned get_num_symbols() const = 0;

  /// \brief Get timing stats from the previous completed encode call.
  virtual const timing_stats& get_last_timing_stats() const = 0;

  /// \brief Check if GPU is available.
  virtual bool is_gpu_available() const = 0;
};

/// \brief Create GPU-accelerated PDSCH TB encoder using CUDA.
/// \return Unique pointer to TB encoder, or nullptr if GPU not available.
std::unique_ptr<pdsch_tb_encoder_gpu> create_pdsch_tb_encoder_cuda();

/// \brief Check if GPU PDSCH TB encoding is available.
bool is_pdsch_tb_encoder_gpu_available();

} // namespace ocudu
