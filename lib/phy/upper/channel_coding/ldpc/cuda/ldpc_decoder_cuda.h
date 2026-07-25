// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief LDPC decoder using CUDA CUDA acceleration.

#pragma once

#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_decoder.h"
#include <cuda_runtime.h>
#include <ocudu_phy_cuda.h>
#include <memory>
#include <vector>

namespace ocudu {

/// \brief LDPC decoder implementation using CUDA CUDA acceleration.
///
/// This class wraps the CUDA LDPC decoder to provide CUDA-accelerated
/// decoding while conforming to the srsRAN ldpc_decoder interface.
class ldpc_decoder_cuda : public ldpc_decoder
{
public:
  /// \brief Constructor - creates CUDA resources.
  ldpc_decoder_cuda();

  /// \brief Destructor - releases CUDA resources.
  ~ldpc_decoder_cuda() override;

  // Disable copy/move to prevent resource sharing issues.
  ldpc_decoder_cuda(const ldpc_decoder_cuda&)            = delete;
  ldpc_decoder_cuda& operator=(const ldpc_decoder_cuda&) = delete;
  ldpc_decoder_cuda(ldpc_decoder_cuda&&)                 = delete;
  ldpc_decoder_cuda& operator=(ldpc_decoder_cuda&&)      = delete;

  /// \brief Decodes a codeblock using CUDA acceleration.
  ///
  /// \param[out] output  Reconstructed message of information bits.
  /// \param[in]  input   Log-likelihood ratios of the codeblock to be decoded.
  /// \param[in]  crc     Pointer to a CRC calculator for verification. Set to \c nullptr for no CRC check.
  /// \param[in]  cfg     Decoder configuration.
  /// \return If the decoding is successful, returns the number of LDPC iterations. Otherwise, empty.
  std::optional<unsigned>
  decode(bit_buffer& output, span<const log_likelihood_ratio> input, crc_calculator* crc, const configuration& cfg) override;

private:
  /// \brief Configures the CUDA decoder for the given codeblock metadata.
  /// \param[in] cfg Decoder configuration.
  void configure_decoder(const configuration& cfg);

  /// \brief Converts srsRAN log_likelihood_ratio (int8) to float for CUDA.
  /// \param[in] llrs Input LLRs.
  /// \param[in] count Number of LLRs to convert.
  void convert_llrs_to_float(span<const log_likelihood_ratio> llrs, unsigned count);

  /// \brief Converts decoded bits from CUDA format to srsRAN bit_buffer.
  /// \param[out] output Output bit buffer.
  /// \param[in]  nof_bits Number of bits to extract.
  void extract_decoded_bits(bit_buffer& output, unsigned nof_bits);

  /// CUDA decoder handle.
  ldpc_decoder_handle_t decoder_handle_ = nullptr;

  /// CUDA rate dematcher handle (for de-rate matching on GPU).
  rate_matcher_handle_t rate_dematcher_handle_ = nullptr;

  /// CUDA stream for asynchronous operations.
  cudaStream_t stream_ = nullptr;

  /// Device memory for input LLRs (float format for CUDA).
  float* d_llr_input_ = nullptr;

  /// Device memory for de-rate matched LLRs.
  float* d_llr_dematched_ = nullptr;

  /// Device memory for decoder output (packed bits).
  uint32_t* d_output_bits_ = nullptr;

  /// Host memory for output bits (for transfer back from GPU).
  std::vector<uint32_t> h_output_bits_;

  /// Host memory for input LLRs in float format.
  std::vector<float> h_llr_float_;

  /// Maximum allocated sizes for GPU buffers.
  static constexpr unsigned MAX_LLR_SIZE         = 26112;  // Max N for LDPC
  static constexpr unsigned MAX_OUTPUT_WORDS     = 264;    // Max K * Z / 32

  /// Current decoder configuration (cached to avoid reconfiguration).
  ldpc_base_graph_type cached_base_graph_    = ldpc_base_graph_type::BG1;
  unsigned             cached_lifting_size_  = 0;
  unsigned             cached_filler_bits_   = 0;
  unsigned             cached_nof_info_bits_ = 0;
};

} // namespace ocudu
