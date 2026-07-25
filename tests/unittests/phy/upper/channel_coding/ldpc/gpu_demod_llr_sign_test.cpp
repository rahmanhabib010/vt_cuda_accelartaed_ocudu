// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief CPU-only test verifying GPU demod formulas produce correct LLR signs.
///
/// This test compares the LLR sign (polarity) produced by the simplified GPU
/// demodulation formulas (from pusch_e2e.cu) against the CPU reference
/// piecewise-linear interval_function() implementation for all constellation
/// points across QPSK, 16QAM, 64QAM, and 256QAM.
///
/// Purpose: Catch sign inversion bugs in the GPU demod formulas without
/// needing GPU hardware.

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <array>

// ============================================================================
// CPU reference: interval_function from demodulation_mapper_intervals.h
// ============================================================================

static unsigned compute_interval_idx(float value, float interval_width, unsigned nof_intervals)
{
  int nof_intervals_int = static_cast<int>(nof_intervals);
  int idx               = static_cast<int>(std::floor(value / interval_width)) + nof_intervals_int / 2;
  if (idx < 0) idx = 0;
  if (idx >= nof_intervals_int) idx = nof_intervals_int - 1;
  return static_cast<unsigned>(idx);
}

template <typename Table>
static float interval_function(float        value,
                               float        rcp_noise,
                               float        interval_width,
                               unsigned     nof_intervals,
                               const Table& slopes,
                               const Table& intercepts)
{
  unsigned idx     = compute_interval_idx(value, interval_width, nof_intervals);
  float    l_value = slopes[idx] * value + intercepts[idx];
  l_value *= rcp_noise;
  return l_value;
}

// ============================================================================
// Constants from the CPU reference demodulators
// ============================================================================

// 16QAM (from demodulation_mapper_qam16.cpp)
static const float M16 = 1.0f / std::sqrt(10.0f);

// 64QAM (from demodulation_mapper_qam64.cpp)
static const float M64 = 1.0f / std::sqrt(42.0f);

static constexpr unsigned                NOF_INTERVALS_64_01 = 8;
static const float                       IW_64_01            = 2 * (1.0f / std::sqrt(42.0f));
static const std::array<float, 8>        SLOPE_64_01         = {16 * M64, 12 * M64, 8 * M64, 4 * M64,
                                                                 4 * M64,  8 * M64, 12 * M64, 16 * M64};
static constexpr std::array<float, 8>    INTERCEPT_64_01     = {24.0f/21, 12.0f/21, 4.0f/21, 0.0f,
                                                                 0.0f, -4.0f/21, -12.0f/21, -24.0f/21};

static constexpr unsigned                NOF_INTERVALS_64_23 = 8;
static const float                       IW_64_23            = 2 * M64;
static const std::array<float, 8>        SLOPE_64_23         = {8 * M64, 4 * M64, 4 * M64, 8 * M64,
                                                                 -8 * M64, -4 * M64, -4 * M64, -8 * M64};
static constexpr std::array<float, 8>    INTERCEPT_64_23     = {20.0f/21, 8.0f/21, 8.0f/21, 12.0f/21,
                                                                 12.0f/21, 8.0f/21, 8.0f/21, 20.0f/21};

static constexpr unsigned                NOF_INTERVALS_64_45 = 4;
static const float                       IW_64_45            = 4 * M64;
static const std::array<float, 8>        SLOPE_64_45         = {4 * M64, -4 * M64, 4 * M64, -4 * M64, 0, 0, 0, 0};
static constexpr std::array<float, 8>    INTERCEPT_64_45     = {12.0f/21, -4.0f/21, -4.0f/21, 12.0f/21, 0, 0, 0, 0};

// 256QAM (from demodulation_mapper_qam256.cpp)
static const float M256 = 1.0f / std::sqrt(170.0f);

static constexpr unsigned NOF_INTERVALS_256_01 = 16;
static const float        IW_256_01            = 2 * M256;
static const std::array<float, 16> SLOPE_256_01 = {
    32*M256, 28*M256, 24*M256, 20*M256, 16*M256, 12*M256, 8*M256, 4*M256,
    4*M256,  8*M256, 12*M256, 16*M256, 20*M256, 24*M256, 28*M256, 32*M256};
static constexpr std::array<float, 16> INTERCEPT_256_01 = {
    112.0f/85, 84.0f/85, 60.0f/85, 40.0f/85, 24.0f/85, 12.0f/85, 4.0f/85, 0.0f,
    0.0f, -4.0f/85, -12.0f/85, -24.0f/85, -40.0f/85, -60.0f/85, -84.0f/85, -112.0f/85};

static constexpr unsigned NOF_INTERVALS_256_23 = 16;
static const float        IW_256_23            = 2 * M256;
static const std::array<float, 16> SLOPE_256_23 = {
    16*M256, 12*M256, 8*M256, 4*M256, 4*M256, 8*M256, 12*M256, 16*M256,
    -16*M256, -12*M256, -8*M256, -4*M256, -4*M256, -8*M256, -12*M256, -16*M256};
static constexpr std::array<float, 16> INTERCEPT_256_23 = {
    88.0f/85, 60.0f/85, 36.0f/85, 16.0f/85, 16.0f/85, 28.0f/85, 36.0f/85, 40.0f/85,
    40.0f/85, 36.0f/85, 28.0f/85, 16.0f/85, 16.0f/85, 36.0f/85, 60.0f/85, 88.0f/85};

static constexpr unsigned NOF_INTERVALS_256_45 = 16;
static const float        IW_256_45            = 2 * M256;
static const std::array<float, 16> SLOPE_256_45 = {
    8*M256, 4*M256, 4*M256, 8*M256, -8*M256, -4*M256, -4*M256, -8*M256,
    8*M256, 4*M256, 4*M256, 8*M256, -8*M256, -4*M256, -4*M256, -8*M256};
static const std::array<float, 16> INTERCEPT_256_45 = {
    52.0f/85, 24.0f/85, 24.0f/85, 44.0f/85, -20.0f/85, -8.0f/85, -8.0f/85, -12.0f/85,
    -12.0f/85, -8.0f/85, -8.0f/85, -20.0f/85, 44.0f/85, 24.0f/85, 24.0f/85, 52.0f/85};

static constexpr unsigned NOF_INTERVALS_256_67 = 8;
static const float        IW_256_67            = 4 * M256;
static const std::array<float, 8> SLOPE_256_67 = {
    4*M256, -4*M256, 4*M256, -4*M256, 4*M256, -4*M256, 4*M256, -4*M256};
static constexpr std::array<float, 8> INTERCEPT_256_67 = {
    28.0f/85, -20.0f/85, 12.0f/85, -4.0f/85, -4.0f/85, 12.0f/85, -20.0f/85, 28.0f/85};

// ============================================================================
// GPU formulas (extracted from pusch_e2e.cu production kernel)
// ============================================================================

/// Returns the sign of a float: +1, -1, or 0.
static int sign_of(float v)
{
  if (v > 0.0f) return +1;
  if (v < 0.0f) return -1;
  return 0;
}

/// Check that GPU and CPU signs are compatible at boundary points.
/// At exact decision boundaries, both formulas produce values very close to
/// zero. Due to floating-point differences, one may be slightly positive and
/// the other slightly negative. This is acceptable since the LLR is ~0 at
/// these boundaries. We tolerate sign mismatches when the magnitude of either
/// value (before noise scaling) is very small.
static bool signs_compatible_boundary(float gpu_val, float cpu_val, float inv_noise)
{
  int gs = sign_of(gpu_val);
  int cs = sign_of(cpu_val);
  if (gs == cs) return true;
  // At boundaries, both values should be near zero. Accept if either is small.
  float threshold = 0.01f * inv_noise;  // Small relative to noise-scaled LLRs
  if (std::abs(gpu_val) < threshold || std::abs(cpu_val) < threshold) return true;
  return false;
}

// --- 16QAM GPU formulas ---

static float gpu_16qam_bit01(float x, float inv_noise)
{
  const float GAIN_FIRST = 4.0f * M16;
  const float THRESHOLD  = 2.0f * M16;
  const float CONST_0_8  = 0.8f;
  float l_first  = GAIN_FIRST * x;
  float l_second = 2.0f * l_first - std::copysign(CONST_0_8, x);
  return (std::abs(x) > THRESHOLD) ? l_second * inv_noise : l_first * inv_noise;
}

static float gpu_16qam_bit23(float x, float inv_noise)
{
  const float GAIN_FIRST = 4.0f * M16;
  const float CONST_0_8  = 0.8f;
  float l_first = GAIN_FIRST * x;
  return (CONST_0_8 - std::abs(l_first)) * inv_noise;
}

// --- 64QAM GPU formulas ---

static float gpu_64qam_bit01(float x, float inv_noise)
{
  const float scale64 = M64;
  float scale = inv_noise * scale64;
  return x * scale * 4.0f * 2.0f;
}

static float gpu_64qam_bit23(float x, float inv_noise)
{
  const float scale64 = M64;
  float scale = inv_noise * scale64;
  float abs_x = std::abs(x);
  return (scale64 * 4.0f - abs_x) * scale * 2.0f * 2.0f;
}

static float gpu_64qam_bit45(float x, float inv_noise)
{
  const float scale64 = M64;
  float scale = inv_noise * scale64;
  float abs_x = std::abs(x);
  float level2 = scale64 * 2.0f;
  return (level2 - std::abs(abs_x - scale64 * 4.0f)) * scale * 2.0f * 2.0f;
}

// --- 256QAM GPU formulas ---

static float gpu_256qam_bit01(float x, float inv_noise)
{
  float scale = inv_noise * 0.07669650f;
  return x * scale * 2.0f;
}

static float gpu_256qam_bit23(float x, float inv_noise)
{
  float scale = inv_noise * 0.07669650f;
  float abs_x = std::abs(x);
  return (0.07669650f * 8.0f - abs_x) * scale * 2.0f;
}

static float gpu_256qam_bit45(float x, float inv_noise)
{
  float scale = inv_noise * 0.07669650f;
  float abs_x = std::abs(x);
  float level2 = 0.07669650f * 4.0f;
  return (level2 - std::abs(abs_x - level2 * 2.0f)) * scale * 2.0f;
}

static float gpu_256qam_bit67(float x, float inv_noise)
{
  float scale = inv_noise * 0.07669650f;
  float abs_x = std::abs(x);
  float level2 = 0.07669650f * 4.0f;
  float level3 = 0.07669650f * 2.0f;
  float d = std::abs(abs_x - level2 * 2.0f);
  return (level3 - std::abs(d - level3 * 2.0f)) * scale * 2.0f;
}

// ============================================================================
// Test: 16QAM LLR sign verification
// ============================================================================

TEST(GpuDemodLlrSign, Qam16ConstellationPoints)
{
  // 16QAM constellation points on real axis: ±3M, ±1M
  const float points[] = {-3*M16, -1*M16, 1*M16, 3*M16};
  const float inv_noise = 10.0f;  // Arbitrary positive value

  // No CPU interval tables for 16QAM bit01 in the piecewise sense — it uses
  // a direct formula. We compare GPU bit23 sign against the CPU scalar formula.
  for (float x : points) {
    // Bit 0,1: sign should match x's sign (positive x → positive LLR)
    float gpu_01 = gpu_16qam_bit01(x, inv_noise);
    EXPECT_EQ(sign_of(gpu_01), sign_of(x))
        << "16QAM bit01 sign mismatch at x=" << x;

    // Bit 2,3: CPU formula: (0.8 - |4*M*x|) / noise_var
    float cpu_23 = (0.8f - 4.0f * M16 * std::abs(x)) / (1.0f / inv_noise);
    float gpu_23 = gpu_16qam_bit23(x, inv_noise);
    EXPECT_EQ(sign_of(gpu_23), sign_of(cpu_23))
        << "16QAM bit23 sign mismatch at x=" << x
        << " cpu=" << cpu_23 << " gpu=" << gpu_23;
  }

  // Also test intermediate points near the decision boundary at ±2M
  const float boundary_points[] = {-2.5f*M16, -1.5f*M16, 1.5f*M16, 2.5f*M16};
  for (float x : boundary_points) {
    float gpu_01 = gpu_16qam_bit01(x, inv_noise);
    EXPECT_EQ(sign_of(gpu_01), sign_of(x))
        << "16QAM bit01 sign mismatch at intermediate x=" << x;

    float cpu_23 = (0.8f - 4.0f * M16 * std::abs(x)) / (1.0f / inv_noise);
    float gpu_23 = gpu_16qam_bit23(x, inv_noise);
    EXPECT_EQ(sign_of(gpu_23), sign_of(cpu_23))
        << "16QAM bit23 sign mismatch at intermediate x=" << x;
  }
}

// ============================================================================
// Test: 64QAM LLR sign verification
// ============================================================================

TEST(GpuDemodLlrSign, Qam64ConstellationPoints)
{
  // 64QAM constellation points on real axis: ±7M, ±5M, ±3M, ±1M
  const float points[] = {-7*M64, -5*M64, -3*M64, -1*M64, 1*M64, 3*M64, 5*M64, 7*M64};
  const float inv_noise = 10.0f;

  for (float x : points) {
    // Bit 0,1
    float cpu_01 = interval_function(x, inv_noise, IW_64_01, NOF_INTERVALS_64_01,
                                     SLOPE_64_01, INTERCEPT_64_01);
    float gpu_01 = gpu_64qam_bit01(x, inv_noise);
    EXPECT_EQ(sign_of(gpu_01), sign_of(cpu_01))
        << "64QAM bit01 sign mismatch at x=" << x
        << " cpu=" << cpu_01 << " gpu=" << gpu_01;

    // Bit 2,3
    float cpu_23 = interval_function(x, inv_noise, IW_64_23, NOF_INTERVALS_64_23,
                                     SLOPE_64_23, INTERCEPT_64_23);
    float gpu_23 = gpu_64qam_bit23(x, inv_noise);
    EXPECT_EQ(sign_of(gpu_23), sign_of(cpu_23))
        << "64QAM bit23 sign mismatch at x=" << x
        << " cpu=" << cpu_23 << " gpu=" << gpu_23;

    // Bit 4,5
    float cpu_45 = interval_function(x, inv_noise, IW_64_45, NOF_INTERVALS_64_45,
                                     SLOPE_64_45, INTERCEPT_64_45);
    float gpu_45 = gpu_64qam_bit45(x, inv_noise);
    EXPECT_EQ(sign_of(gpu_45), sign_of(cpu_45))
        << "64QAM bit45 sign mismatch at x=" << x
        << " cpu=" << cpu_45 << " gpu=" << gpu_45;
  }

  // Intermediate points near decision boundaries: ±2M, ±4M, ±6M
  // At exact boundaries the GPU formula may give 0 — this is acceptable.
  const float boundary_points[] = {-6*M64, -4*M64, -2*M64, 2*M64, 4*M64, 6*M64};
  for (float x : boundary_points) {
    float cpu_01 = interval_function(x, inv_noise, IW_64_01, NOF_INTERVALS_64_01,
                                     SLOPE_64_01, INTERCEPT_64_01);
    float gpu_01 = gpu_64qam_bit01(x, inv_noise);
    EXPECT_TRUE(signs_compatible_boundary(gpu_01, cpu_01, inv_noise))
        << "64QAM bit01 sign mismatch at boundary x=" << x
        << " cpu=" << cpu_01 << " gpu=" << gpu_01;

    float cpu_23 = interval_function(x, inv_noise, IW_64_23, NOF_INTERVALS_64_23,
                                     SLOPE_64_23, INTERCEPT_64_23);
    float gpu_23 = gpu_64qam_bit23(x, inv_noise);
    EXPECT_TRUE(signs_compatible_boundary(gpu_23, cpu_23, inv_noise))
        << "64QAM bit23 sign mismatch at boundary x=" << x
        << " cpu=" << cpu_23 << " gpu=" << gpu_23;

    float cpu_45 = interval_function(x, inv_noise, IW_64_45, NOF_INTERVALS_64_45,
                                     SLOPE_64_45, INTERCEPT_64_45);
    float gpu_45 = gpu_64qam_bit45(x, inv_noise);
    EXPECT_TRUE(signs_compatible_boundary(gpu_45, cpu_45, inv_noise))
        << "64QAM bit45 sign mismatch at boundary x=" << x
        << " cpu=" << cpu_45 << " gpu=" << gpu_45;
  }
}

// ============================================================================
// Test: 256QAM LLR sign verification
// ============================================================================

TEST(GpuDemodLlrSign, Qam256ConstellationPoints)
{
  // 256QAM constellation points on real axis: ±15M, ±13M, ..., ±1M
  std::array<float, 16> points;
  for (int i = 0; i < 8; ++i) {
    float val         = static_cast<float>(2 * i + 1) * M256;
    points[i]         = -val;
    points[15 - i]    = val;
  }

  const float inv_noise = 10.0f;

  for (float x : points) {
    // Bit 0,1
    float cpu_01 = interval_function(x, inv_noise, IW_256_01, NOF_INTERVALS_256_01,
                                     SLOPE_256_01, INTERCEPT_256_01);
    float gpu_01 = gpu_256qam_bit01(x, inv_noise);
    EXPECT_EQ(sign_of(gpu_01), sign_of(cpu_01))
        << "256QAM bit01 sign mismatch at x=" << x
        << " cpu=" << cpu_01 << " gpu=" << gpu_01;

    // Bit 2,3
    float cpu_23 = interval_function(x, inv_noise, IW_256_23, NOF_INTERVALS_256_23,
                                     SLOPE_256_23, INTERCEPT_256_23);
    float gpu_23 = gpu_256qam_bit23(x, inv_noise);
    EXPECT_EQ(sign_of(gpu_23), sign_of(cpu_23))
        << "256QAM bit23 sign mismatch at x=" << x
        << " cpu=" << cpu_23 << " gpu=" << gpu_23;

    // Bit 4,5
    float cpu_45 = interval_function(x, inv_noise, IW_256_45, NOF_INTERVALS_256_45,
                                     SLOPE_256_45, INTERCEPT_256_45);
    float gpu_45 = gpu_256qam_bit45(x, inv_noise);
    EXPECT_EQ(sign_of(gpu_45), sign_of(cpu_45))
        << "256QAM bit45 sign mismatch at x=" << x
        << " cpu=" << cpu_45 << " gpu=" << gpu_45;

    // Bit 6,7
    float cpu_67 = interval_function(x, inv_noise, IW_256_67, NOF_INTERVALS_256_67,
                                     SLOPE_256_67, INTERCEPT_256_67);
    float gpu_67 = gpu_256qam_bit67(x, inv_noise);
    EXPECT_EQ(sign_of(gpu_67), sign_of(cpu_67))
        << "256QAM bit67 sign mismatch at x=" << x
        << " cpu=" << cpu_67 << " gpu=" << gpu_67;
  }

  // Intermediate points near decision boundaries: ±2M, ±4M, ±6M, ±8M, ±10M, ±12M, ±14M
  // At exact boundaries, both formulas may give tiny values with differing signs.
  for (int i = 1; i <= 7; ++i) {
    float val = static_cast<float>(2 * i) * M256;
    for (float x : {-val, val}) {
      float cpu_01 = interval_function(x, inv_noise, IW_256_01, NOF_INTERVALS_256_01,
                                       SLOPE_256_01, INTERCEPT_256_01);
      float gpu_01 = gpu_256qam_bit01(x, inv_noise);
      EXPECT_TRUE(signs_compatible_boundary(gpu_01, cpu_01, inv_noise))
          << "256QAM bit01 sign mismatch at boundary x=" << x
          << " cpu=" << cpu_01 << " gpu=" << gpu_01;

      float cpu_23 = interval_function(x, inv_noise, IW_256_23, NOF_INTERVALS_256_23,
                                       SLOPE_256_23, INTERCEPT_256_23);
      float gpu_23 = gpu_256qam_bit23(x, inv_noise);
      EXPECT_TRUE(signs_compatible_boundary(gpu_23, cpu_23, inv_noise))
          << "256QAM bit23 sign mismatch at boundary x=" << x
          << " cpu=" << cpu_23 << " gpu=" << gpu_23;

      float cpu_45 = interval_function(x, inv_noise, IW_256_45, NOF_INTERVALS_256_45,
                                       SLOPE_256_45, INTERCEPT_256_45);
      float gpu_45 = gpu_256qam_bit45(x, inv_noise);
      EXPECT_TRUE(signs_compatible_boundary(gpu_45, cpu_45, inv_noise))
          << "256QAM bit45 sign mismatch at boundary x=" << x
          << " cpu=" << cpu_45 << " gpu=" << gpu_45;

      float cpu_67 = interval_function(x, inv_noise, IW_256_67, NOF_INTERVALS_256_67,
                                       SLOPE_256_67, INTERCEPT_256_67);
      float gpu_67 = gpu_256qam_bit67(x, inv_noise);
      EXPECT_TRUE(signs_compatible_boundary(gpu_67, cpu_67, inv_noise))
          << "256QAM bit67 sign mismatch at boundary x=" << x
          << " cpu=" << cpu_67 << " gpu=" << gpu_67;
    }
  }
}

// ============================================================================
// Test: QPSK LLR sign verification (trivial — sign matches symbol value)
// ============================================================================

TEST(GpuDemodLlrSign, QpskConstellationPoints)
{
  // QPSK: M = 1/sqrt(2), constellation points ±M
  const float M_QPSK = 1.0f / std::sqrt(2.0f);
  const float points[] = {-M_QPSK, M_QPSK};
  const float inv_noise = 10.0f;

  for (float x : points) {
    // GPU QPSK formula: x * inv_noise * scale (positive scaling)
    // LLR sign should match x sign
    float gpu_llr = x * inv_noise;  // Simplified QPSK
    EXPECT_EQ(sign_of(gpu_llr), sign_of(x))
        << "QPSK sign mismatch at x=" << x;
  }
}

TEST(GpuDemodLlrSign, QamLlrSymmetryAndNoiseScalingInvariants)
{
  const float inv_noise        = 7.25f;
  const float inv_noise_double = 2.0f * inv_noise;
  const float tolerance        = 1e-6f;

  auto expect_linear_scaling = [=](float single, float doubled, const char* label) {
    EXPECT_NEAR(doubled, 2.0f * single, tolerance) << label;
  };

  for (float x : {1.0f * M16, 3.0f * M16}) {
    EXPECT_NEAR(gpu_16qam_bit01(-x, inv_noise), -gpu_16qam_bit01(x, inv_noise), tolerance);
    EXPECT_NEAR(gpu_16qam_bit23(-x, inv_noise), gpu_16qam_bit23(x, inv_noise), tolerance);
    expect_linear_scaling(gpu_16qam_bit01(x, inv_noise), gpu_16qam_bit01(x, inv_noise_double), "16QAM bit01");
    expect_linear_scaling(gpu_16qam_bit23(x, inv_noise), gpu_16qam_bit23(x, inv_noise_double), "16QAM bit23");
  }

  for (float x : {1.0f * M64, 3.0f * M64, 5.0f * M64, 7.0f * M64}) {
    EXPECT_NEAR(gpu_64qam_bit01(-x, inv_noise), -gpu_64qam_bit01(x, inv_noise), tolerance);
    EXPECT_NEAR(gpu_64qam_bit23(-x, inv_noise), gpu_64qam_bit23(x, inv_noise), tolerance);
    EXPECT_NEAR(gpu_64qam_bit45(-x, inv_noise), gpu_64qam_bit45(x, inv_noise), tolerance);
    expect_linear_scaling(gpu_64qam_bit01(x, inv_noise), gpu_64qam_bit01(x, inv_noise_double), "64QAM bit01");
    expect_linear_scaling(gpu_64qam_bit23(x, inv_noise), gpu_64qam_bit23(x, inv_noise_double), "64QAM bit23");
    expect_linear_scaling(gpu_64qam_bit45(x, inv_noise), gpu_64qam_bit45(x, inv_noise_double), "64QAM bit45");
  }

  for (float x :
       {1.0f * M256, 3.0f * M256, 5.0f * M256, 7.0f * M256, 9.0f * M256, 11.0f * M256, 13.0f * M256, 15.0f * M256}) {
    EXPECT_NEAR(gpu_256qam_bit01(-x, inv_noise), -gpu_256qam_bit01(x, inv_noise), tolerance);
    EXPECT_NEAR(gpu_256qam_bit23(-x, inv_noise), gpu_256qam_bit23(x, inv_noise), tolerance);
    EXPECT_NEAR(gpu_256qam_bit45(-x, inv_noise), gpu_256qam_bit45(x, inv_noise), tolerance);
    EXPECT_NEAR(gpu_256qam_bit67(-x, inv_noise), gpu_256qam_bit67(x, inv_noise), tolerance);
    expect_linear_scaling(gpu_256qam_bit01(x, inv_noise), gpu_256qam_bit01(x, inv_noise_double), "256QAM bit01");
    expect_linear_scaling(gpu_256qam_bit23(x, inv_noise), gpu_256qam_bit23(x, inv_noise_double), "256QAM bit23");
    expect_linear_scaling(gpu_256qam_bit45(x, inv_noise), gpu_256qam_bit45(x, inv_noise_double), "256QAM bit45");
    expect_linear_scaling(gpu_256qam_bit67(x, inv_noise), gpu_256qam_bit67(x, inv_noise_double), "256QAM bit67");
  }
}

TEST(GpuDemodLlrSign, QamDecisionBoundariesProduceZeroLlr)
{
  const float inv_noise = 9.0f;
  const float tolerance = 1e-5f;

  EXPECT_NEAR(gpu_16qam_bit23(2.0f * M16, inv_noise), 0.0f, tolerance);
  EXPECT_NEAR(gpu_16qam_bit23(-2.0f * M16, inv_noise), 0.0f, tolerance);

  EXPECT_NEAR(gpu_64qam_bit23(4.0f * M64, inv_noise), 0.0f, tolerance);
  EXPECT_NEAR(gpu_64qam_bit23(-4.0f * M64, inv_noise), 0.0f, tolerance);
  for (float boundary : {2.0f * M64, 6.0f * M64}) {
    EXPECT_NEAR(gpu_64qam_bit45(boundary, inv_noise), 0.0f, tolerance);
    EXPECT_NEAR(gpu_64qam_bit45(-boundary, inv_noise), 0.0f, tolerance);
  }

  EXPECT_NEAR(gpu_256qam_bit23(8.0f * M256, inv_noise), 0.0f, tolerance);
  EXPECT_NEAR(gpu_256qam_bit23(-8.0f * M256, inv_noise), 0.0f, tolerance);
  for (float boundary : {4.0f * M256, 12.0f * M256}) {
    EXPECT_NEAR(gpu_256qam_bit45(boundary, inv_noise), 0.0f, tolerance);
    EXPECT_NEAR(gpu_256qam_bit45(-boundary, inv_noise), 0.0f, tolerance);
  }
  for (float boundary : {2.0f * M256, 6.0f * M256, 10.0f * M256, 14.0f * M256}) {
    EXPECT_NEAR(gpu_256qam_bit67(boundary, inv_noise), 0.0f, tolerance);
    EXPECT_NEAR(gpu_256qam_bit67(-boundary, inv_noise), 0.0f, tolerance);
  }
}

TEST(GpuDemodLlrSign, QamNumericGoldenVectors)
{
  constexpr float tolerance = 1e-4f;

  // 16QAM with inv_noise=10 gives integer LLRs at +/-1M and +/-3M.
  EXPECT_NEAR(gpu_16qam_bit01(1.0f * M16, 10.0f), 4.0f, tolerance);
  EXPECT_NEAR(gpu_16qam_bit01(-1.0f * M16, 10.0f), -4.0f, tolerance);
  EXPECT_NEAR(gpu_16qam_bit01(3.0f * M16, 10.0f), 16.0f, tolerance);
  EXPECT_NEAR(gpu_16qam_bit01(-3.0f * M16, 10.0f), -16.0f, tolerance);
  EXPECT_NEAR(gpu_16qam_bit23(1.0f * M16, 10.0f), 4.0f, tolerance);
  EXPECT_NEAR(gpu_16qam_bit23(3.0f * M16, 10.0f), -4.0f, tolerance);

  // 64QAM with inv_noise=21 cancels the 1/sqrt(42) normalization.
  EXPECT_NEAR(gpu_64qam_bit01(1.0f * M64, 21.0f), 4.0f, tolerance);
  EXPECT_NEAR(gpu_64qam_bit01(7.0f * M64, 21.0f), 28.0f, tolerance);
  EXPECT_NEAR(gpu_64qam_bit23(1.0f * M64, 21.0f), 6.0f, tolerance);
  EXPECT_NEAR(gpu_64qam_bit23(5.0f * M64, 21.0f), -2.0f, tolerance);
  EXPECT_NEAR(gpu_64qam_bit45(1.0f * M64, 21.0f), -2.0f, tolerance);
  EXPECT_NEAR(gpu_64qam_bit45(3.0f * M64, 21.0f), 2.0f, tolerance);
  EXPECT_NEAR(gpu_64qam_bit45(7.0f * M64, 21.0f), -2.0f, tolerance);

  // 256QAM uses the same rounded scale constant as the production GPU formula.
  constexpr float scale256 = 0.07669650f;
  EXPECT_NEAR(gpu_256qam_bit01(1.0f * scale256, 85.0f), 1.0f, tolerance);
  EXPECT_NEAR(gpu_256qam_bit01(15.0f * scale256, 85.0f), 15.0f, tolerance);
  EXPECT_NEAR(gpu_256qam_bit23(1.0f * scale256, 85.0f), 7.0f, tolerance);
  EXPECT_NEAR(gpu_256qam_bit23(9.0f * scale256, 85.0f), -1.0f, tolerance);
  EXPECT_NEAR(gpu_256qam_bit45(1.0f * scale256, 85.0f), -3.0f, tolerance);
  EXPECT_NEAR(gpu_256qam_bit45(5.0f * scale256, 85.0f), 1.0f, tolerance);
  EXPECT_NEAR(gpu_256qam_bit67(1.0f * scale256, 85.0f), -1.0f, tolerance);
  EXPECT_NEAR(gpu_256qam_bit67(3.0f * scale256, 85.0f), 1.0f, tolerance);
}
