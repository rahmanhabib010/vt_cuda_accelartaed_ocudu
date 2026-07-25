# OCUDU PHY CUDA

This directory contains the in-tree CUDA kernel library used by OCUDU PHY and
Open Fronthaul CUDA adapters. It is built as the `ocudu_phy_cuda` static
library when the parent OCUDU build enables CUDA support.

The public C API is exposed through `include/ocudu_phy_cuda.h`; OCUDU-facing C++
adapters remain in the normal PHY and OFH implementation directories. This keeps
CUDA kernel/runtime code in one place while allowing higher-level factories and
selectors to continue using the existing OCUDU abstractions.

## Build Integration

From the OCUDU repository root:

```bash
cmake -S . -B build-cuda \
  -DENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build-cuda --target gnb -j$(nproc)
```

The parent build adds this directory from `lib/CMakeLists.txt` before other
libraries that may consume CUDA acceleration. The CUDA library publishes the
existing `CUDA_ACCEL_FOUND`, `CUDA_ACCEL_INCLUDE_DIRS`, and
`CUDA_ACCEL_LIBRARIES` CMake variables for downstream targets.

When CUDA is unavailable, or when configured with `-DENABLE_CUDA=OFF`, the
parent build leaves `CUDA_ACCEL_FOUND=FALSE` and the CPU implementations remain
available.

## Standalone Development

The directory can also be configured directly while working on kernels:

```bash
cmake -S lib/phy/cuda -B build-phy-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-phy-cuda -j$(nproc)
```

Standalone builds enable the broader local CUDA validation inventory by default.
Parent OCUDU builds keep it disabled unless
`-DBUILD_OCUDU_PHY_CUDA_TESTS=ON` is set. A small set of focused utility targets,
such as `test_polar` and `low_phy_tx_smoke_test`, remains available as explicit
`EXCLUDE_FROM_ALL` targets in parent builds.

Exploratory trace and dumped-LLR reproducers are not part of the default
standalone test inventory. Enable them with
`-DBUILD_OCUDU_PHY_CUDA_DIAGNOSTIC_TESTS=ON` when debugging kernel internals.

## Accelerated Areas

- LDPC encoding and decoding for BG1/BG2 transport blocks.
- Rate matching, rate dematching, scrambling, modulation, and soft
  demodulation helpers.
- PUSCH receive processing, including channel estimation, equalization,
  descrambling, and FP16 LLR output.
- PDSCH transmit processing with fused transport-block to symbol paths.
- Lower-PHY PRACH, PUxCH RX, and TX processing helpers.
- SRS estimation kernels.
- Open Fronthaul IQ compression and decompression kernels.

## Runtime Notes

- VkFFT support is required for the FFT-backed low-PHY kernels and is provided
  from `third_party/vkfft` by default. Override with
  `-DOCUDU_PHY_CUDA_VKFFT_ROOT=/path/to/vkfft` when needed.
- `OCUDU_PHY_CUDA_PUSCH_TRANSFORM_DEPRECODER=0|1` overrides the PUSCH
  transform-deprecoder path for debugging.
- `OCUDU_PHY_CUDA_PUSCH_VKFFT_MIN_DFT_SIZE=<n>` controls the minimum DFT size
  that uses the VkFFT path.

## License

This in-tree component follows the OCUDU repository license.
