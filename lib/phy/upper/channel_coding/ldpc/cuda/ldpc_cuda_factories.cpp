// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ldpc_cuda_factories.h"
#include "crc_calculator_cuda.h"
#include "demodulation_mapper_cuda.h"
#include "ldpc_decoder_cuda.h"
#include "ldpc_encoder_cuda.h"
#include "modulation_mapper_cuda.h"
#include "pseudo_random_generator_cuda.h"
#include <cuda_runtime.h>
#include <ocudu_phy_cuda.h>

using namespace ocudu;

namespace {

/// Factory class for CUDA LDPC decoders.
class ldpc_decoder_factory_cuda_impl : public ldpc_decoder_factory
{
public:
  std::unique_ptr<ldpc_decoder> create() override
  {
    return std::make_unique<ldpc_decoder_cuda>();
  }
};

/// Factory class for CUDA pseudo-random generators with GPU-accelerated LLR descrambling.
class pseudo_random_generator_factory_cuda_impl : public pseudo_random_generator_factory
{
public:
  explicit pseudo_random_generator_factory_cuda_impl(
      std::shared_ptr<pseudo_random_generator_factory> fallback_factory_) :
    fallback_factory(std::move(fallback_factory_))
  {
  }

  std::unique_ptr<pseudo_random_generator> create() override
  {
    // Create the fallback CPU generator
    std::unique_ptr<pseudo_random_generator> fallback = fallback_factory->create();
    // Wrap it with the CUDA GPU-accelerated version
    return create_pseudo_random_generator_cuda(std::move(fallback));
  }

private:
  std::shared_ptr<pseudo_random_generator_factory> fallback_factory;
};

/// Factory class for CUDA CRC calculators with GPU acceleration.
class crc_calculator_factory_cuda_impl : public crc_calculator_factory
{
public:
  explicit crc_calculator_factory_cuda_impl(std::shared_ptr<crc_calculator_factory> fallback_factory_) :
    fallback_factory(std::move(fallback_factory_))
  {
  }

  std::unique_ptr<crc_calculator> create(crc_generator_poly poly) override
  {
    // Create the fallback CPU calculator.
    std::unique_ptr<crc_calculator> fallback = fallback_factory->create(poly);
    // Wrap it with the CUDA GPU-accelerated version.
    return std::make_unique<crc_calculator_cuda>(poly, std::move(fallback));
  }

private:
  std::shared_ptr<crc_calculator_factory> fallback_factory;
};

/// Factory class for CUDA demodulation mappers with GPU acceleration.
class demodulation_mapper_factory_cuda_impl : public demodulation_mapper_factory
{
public:
  explicit demodulation_mapper_factory_cuda_impl(std::shared_ptr<demodulation_mapper_factory> fallback_factory_) :
    fallback_factory(std::move(fallback_factory_))
  {
  }

  std::unique_ptr<demodulation_mapper> create() override
  {
    // Create the fallback CPU demodulator.
    std::unique_ptr<demodulation_mapper> fallback = fallback_factory->create();
    // Wrap it with the CUDA GPU-accelerated version.
    return std::make_unique<demodulation_mapper_cuda>(std::move(fallback));
  }

private:
  std::shared_ptr<demodulation_mapper_factory> fallback_factory;
};

} // namespace

std::shared_ptr<ldpc_decoder_factory> ocudu::create_ldpc_decoder_factory_cuda()
{
  return std::make_shared<ldpc_decoder_factory_cuda_impl>();
}

bool ocudu::is_cuda_available()
{
  // Check if CUDA runtime is available.
  (void)cudaGetLastError();
  int device_count = 0;
  cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess) {
    (void)cudaGetLastError();
    cuda_status = cudaGetDeviceCount(&device_count);
  }

  if (cuda_status != cudaSuccess || device_count == 0) {
    (void)cudaGetLastError();
    return false;
  }
  (void)cudaGetLastError();

  // Check if CUDA library can be initialized.
  // Note: We do NOT call ocudu_phy_cuda_cleanup() here because that would destroy
  // the CUDA context that will be used by subsequent decoder/dematcher instances.
  // ocudu_phy_cuda_init() is a lightweight availability check and is safe to call multiple times.
  nr_ldpc_status_t status = ocudu_phy_cuda_init();
  if (status != NR_LDPC_SUCCESS) {
    return false;
  }

  return true;
}

std::shared_ptr<pseudo_random_generator_factory>
ocudu::create_pseudo_random_generator_factory_cuda(std::shared_ptr<pseudo_random_generator_factory> fallback_factory)
{
  return std::make_shared<pseudo_random_generator_factory_cuda_impl>(std::move(fallback_factory));
}

std::shared_ptr<crc_calculator_factory>
ocudu::create_crc_calculator_factory_cuda(std::shared_ptr<crc_calculator_factory> fallback_factory)
{
  return std::make_shared<crc_calculator_factory_cuda_impl>(std::move(fallback_factory));
}

std::shared_ptr<demodulation_mapper_factory>
ocudu::create_demodulation_mapper_factory_cuda(std::shared_ptr<demodulation_mapper_factory> fallback_factory)
{
  return std::make_shared<demodulation_mapper_factory_cuda_impl>(std::move(fallback_factory));
}

// ============================================================================
// TX-side GPU acceleration factories
// ============================================================================

namespace {

/// Factory class for CUDA LDPC encoders with GPU acceleration.
class ldpc_encoder_factory_cuda_impl : public ldpc_encoder_factory
{
public:
  explicit ldpc_encoder_factory_cuda_impl(std::shared_ptr<ldpc_encoder_factory> fallback_factory_) :
    fallback_factory(std::move(fallback_factory_))
  {
  }

  std::unique_ptr<ldpc_encoder> create() override
  {
    // Create the fallback CPU encoder.
    std::unique_ptr<ldpc_encoder> fallback = fallback_factory->create();
    // Wrap it with the CUDA GPU-accelerated version.
    return create_ldpc_encoder_cuda(std::move(fallback));
  }

private:
  std::shared_ptr<ldpc_encoder_factory> fallback_factory;
};

/// Factory class for CUDA modulation mappers with GPU acceleration.
class modulation_mapper_factory_cuda_impl : public modulation_mapper_factory
{
public:
  explicit modulation_mapper_factory_cuda_impl(std::shared_ptr<modulation_mapper_factory> fallback_factory_) :
    fallback_factory(std::move(fallback_factory_))
  {
  }

  std::unique_ptr<modulation_mapper> create() override
  {
    // Create the fallback CPU modulator.
    std::unique_ptr<modulation_mapper> fallback = fallback_factory->create();
    // Wrap it with the CUDA GPU-accelerated version.
    return create_modulation_mapper_cuda(std::move(fallback));
  }

private:
  std::shared_ptr<modulation_mapper_factory> fallback_factory;
};

} // namespace

std::shared_ptr<ldpc_encoder_factory>
ocudu::create_ldpc_encoder_factory_cuda(std::shared_ptr<ldpc_encoder_factory> fallback_factory)
{
  return std::make_shared<ldpc_encoder_factory_cuda_impl>(std::move(fallback_factory));
}

std::shared_ptr<modulation_mapper_factory>
ocudu::create_modulation_mapper_factory_cuda(std::shared_ptr<modulation_mapper_factory> fallback_factory)
{
  return std::make_shared<modulation_mapper_factory_cuda_impl>(std::move(fallback_factory));
}
