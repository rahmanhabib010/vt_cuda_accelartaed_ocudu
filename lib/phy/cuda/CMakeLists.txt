# OCUDU PHY CUDA - CUDA acceleration for OCUDU
# 5G NR LDPC Encoder/Decoder with CUDA Acceleration

cmake_minimum_required(VERSION 3.18)

# Check if being built as a subdirectory
if(NOT CMAKE_PROJECT_NAME OR CMAKE_PROJECT_NAME STREQUAL "ocudu_phy_cuda")
    # Standalone build
    project(ocudu_phy_cuda LANGUAGES CXX CUDA)
    set(OCUDU_PHY_CUDA_STANDALONE TRUE)
else()
    # Being built as a subdirectory - CUDA should already be enabled by parent
    set(OCUDU_PHY_CUDA_STANDALONE FALSE)
endif()

# C++ and CUDA standards (only set if not already set by parent)
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ============================================================================
# Automatic Tegra Platform Detection
# Detects NVIDIA Tegra platforms (Thor, Spark, Orin, Xavier, etc.)
# ============================================================================
if(NOT DEFINED TARGET_TEGRA)
    set(TARGET_TEGRA OFF)
    set(TEGRA_PLATFORM "")

    # Method 1: Check for Tegra release file (most reliable)
    if(EXISTS "/etc/nv_tegra_release")
        set(TARGET_TEGRA ON)
        file(READ "/etc/nv_tegra_release" TEGRA_RELEASE_CONTENT)
        message(STATUS "Detected Tegra platform via /etc/nv_tegra_release")
    endif()

    # Method 2: Check for L4T (Linux for Tegra) version file
    if(NOT TARGET_TEGRA AND EXISTS "/etc/nv_boot_control.conf")
        set(TARGET_TEGRA ON)
        message(STATUS "Detected Tegra platform via /etc/nv_boot_control.conf")
    endif()

    # Method 3: Check for Tegra-specific device tree
    if(NOT TARGET_TEGRA AND EXISTS "/proc/device-tree/compatible")
        file(READ "/proc/device-tree/compatible" DT_COMPATIBLE)
        if(DT_COMPATIBLE MATCHES "nvidia,tegra" OR
           DT_COMPATIBLE MATCHES "nvidia,thor" OR
           DT_COMPATIBLE MATCHES "nvidia,orin")
            set(TARGET_TEGRA ON)
            message(STATUS "Detected Tegra platform via device tree")
        endif()
    endif()

    # Method 4: Check for Tegra-specific sysfs entries
    if(NOT TARGET_TEGRA AND EXISTS "/sys/devices/platform/host1x")
        set(TARGET_TEGRA ON)
        message(STATUS "Detected Tegra platform via host1x sysfs")
    endif()

    # Method 5: Check processor info for Tegra/Carmel/Denver cores
    if(NOT TARGET_TEGRA AND EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" CPUINFO_CONTENT)
        if(CPUINFO_CONTENT MATCHES "NVIDIA" OR
           CPUINFO_CONTENT MATCHES "Carmel" OR
           CPUINFO_CONTENT MATCHES "Denver")
            set(TARGET_TEGRA ON)
            message(STATUS "Detected Tegra platform via CPU info")
        endif()
    endif()

    # Detect specific Tegra platform type
    if(TARGET_TEGRA)
        # Try to identify specific platform from device tree model
        if(EXISTS "/proc/device-tree/model")
            file(READ "/proc/device-tree/model" DT_MODEL)
            if(DT_MODEL MATCHES "Thor")
                set(TEGRA_PLATFORM "thor")
            elseif(DT_MODEL MATCHES "Orin")
                set(TEGRA_PLATFORM "orin")
            elseif(DT_MODEL MATCHES "Xavier")
                set(TEGRA_PLATFORM "xavier")
            elseif(DT_MODEL MATCHES "Spark")
                set(TEGRA_PLATFORM "spark")
            else()
                set(TEGRA_PLATFORM "unknown")
            endif()
        endif()
    endif()
endif()

# Report Tegra detection result
if(TARGET_TEGRA)
    message(STATUS "TARGET_TEGRA: ON (Platform: ${TEGRA_PLATFORM})")
    add_definitions(-DTARGET_TEGRA=1)
    if(TEGRA_PLATFORM)
        string(TOUPPER ${TEGRA_PLATFORM} TEGRA_PLATFORM_UPPER)
        add_definitions(-DTEGRA_PLATFORM_${TEGRA_PLATFORM_UPPER}=1)
    endif()
else()
    message(STATUS "TARGET_TEGRA: OFF (Server/Desktop GPU detected)")
endif()

# CUDA architecture detection or specification
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default to native architecture, fallback to recent architectures
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Find CUDA (only if not already found by parent)
if(NOT CUDAToolkit_FOUND)
    find_package(CUDAToolkit REQUIRED)
endif()

# Set include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Collect source files
set(LDPC_SOURCES
    src/ldpc_encoder_flexible.cu
    src/ldpc_decoder_flexible.cu
    src/modulation.cu
    src/rate_matching.cu
    src/crc.cu
    src/transport_block.cu
    src/nr_constants.cu
    src/scrambling.cu
    src/pusch_e2e.cu
    src/srs_estimator.cu
    src/pdsch_fused.cu
    src/polar.cu
    src/low_phy_prach_rx.cu
    src/low_phy_puxch_rx.cu
    src/low_phy_tx.cu
    src/ofh_compression.cu
    src/prach_detector.cu
)

# Create the LDPC library
add_library(ocudu_phy_cuda STATIC ${LDPC_SOURCES})

target_include_directories(ocudu_phy_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ocudu_phy_cuda PUBLIC
    CUDA::cudart
    CUDA::curand
)

set(OCUDU_PHY_CUDA_VKFFT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vkfft" CACHE PATH
    "Path to a VkFFT checkout or bundled copy")

if(NOT EXISTS "${OCUDU_PHY_CUDA_VKFFT_ROOT}/vkFFT/vkFFT.h")
    message(FATAL_ERROR "OCUDU_PHY_CUDA_VKFFT_ROOT must contain vkFFT/vkFFT.h")
endif()

target_compile_definitions(ocudu_phy_cuda PRIVATE
    OCUDU_PHY_CUDA_ENABLE_VKFFT=1
    VKFFT_BACKEND=1
)
target_include_directories(ocudu_phy_cuda SYSTEM PRIVATE
    "${OCUDU_PHY_CUDA_VKFFT_ROOT}/vkFFT"
)
target_link_libraries(ocudu_phy_cuda PUBLIC
    CUDA::cuda_driver
    CUDA::nvrtc
)

set(CUDA_ACCEL_FOUND TRUE CACHE BOOL "CUDA acceleration library found and built" FORCE)
set(CUDA_ACCEL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "CUDA acceleration include dirs" FORCE)
set(CUDA_ACCEL_LIBRARIES ocudu_phy_cuda CACHE STRING "CUDA acceleration library target" FORCE)

# Register usage controlled per-kernel via __launch_bounds__ instead of global cap.
# Removed --maxrregcount=56 which caused 232-byte stack spills in BG1 half2 kernel.

# Set CUDA-specific compile options
target_compile_options(ocudu_phy_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-Wall
    >
)

# Do NOT enable CUDA separable compilation - all device code is internal to the library
# This makes the library self-contained and linkable from pure C++ code
set_target_properties(ocudu_phy_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Test Executables (only build when standalone or explicitly requested)
# ============================================================================

# Option to build tests - defaults to ON for standalone, OFF for subdirectory
if(OCUDU_PHY_CUDA_STANDALONE)
    option(BUILD_OCUDU_PHY_CUDA_TESTS "Build OCUDU PHY CUDA test executables" ON)
else()
    option(BUILD_OCUDU_PHY_CUDA_TESTS "Build OCUDU PHY CUDA test executables" OFF)
endif()

option(BUILD_OCUDU_PHY_CUDA_DIAGNOSTIC_TESTS
       "Build exploratory OCUDU PHY CUDA diagnostic and trace test executables"
       OFF)

# Focused polar primitive validation targets. Keep these explicit so parent builds
# can build them without enabling the full standalone OCUDU PHY CUDA test inventory.
function(ocudu_phy_cuda_add_explicit_cuda_tool target source)
    add_executable(${target} EXCLUDE_FROM_ALL ${source})
    target_link_libraries(${target} PRIVATE ocudu_phy_cuda)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endfunction()

ocudu_phy_cuda_add_explicit_cuda_tool(test_polar tests/unittests/test_polar.cu)
ocudu_phy_cuda_add_explicit_cuda_tool(polar_benchmark tests/benchmarks/polar_benchmark.cu)
ocudu_phy_cuda_add_explicit_cuda_tool(polar_sensitivity tests/benchmarks/polar_sensitivity.cu)
ocudu_phy_cuda_add_explicit_cuda_tool(low_phy_tx_smoke_test tests/unittests/low_phy_tx_smoke_test.cu)
ocudu_phy_cuda_add_explicit_cuda_tool(low_phy_tx_benchmark tests/benchmarks/low_phy_tx_benchmark.cu)

if(BUILD_OCUDU_PHY_CUDA_TESTS)
    # Standalone OCUDU PHY CUDA validation inventory. Parent OCUDU builds keep this
    # disabled by default and instead build the focused explicit targets above.

    # ============================================================================
    # Google Test Setup
    # ============================================================================
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    # Enable RTTI for Google Test (parent project has -fno-rtti)
    set(CMAKE_CXX_FLAGS_SAVED "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-fno-rtti" "-frtti" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    FetchContent_MakeAvailable(googletest)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_SAVED}")

    # Enable testing
    if(OCUDU_PHY_CUDA_STANDALONE)
        enable_testing()
    endif()
    include(GoogleTest)

    # ============================================================================
    # E2E Integration Tests (active modules only)
    # ============================================================================

    # PDSCH TX chain loopback test
    add_executable(pdsch_tx_loopback_test tests/integrationtests/pdsch_tx_loopback_test.cu)
    target_link_libraries(pdsch_tx_loopback_test PRIVATE ocudu_phy_cuda)

    # PUSCH E2E loopback test (production RX path)
    add_executable(pusch_e2e_loopback_test tests/integrationtests/pusch_e2e_loopback_test.cu)
    target_link_libraries(pusch_e2e_loopback_test PRIVATE ocudu_phy_cuda)

    # Comprehensive E2E validation suite (CRC, PDSCH, PUSCH, BLER curves, stress tests)
    add_executable(comprehensive_e2e_validation tests/integrationtests/comprehensive_e2e_validation.cu)
    target_link_libraries(comprehensive_e2e_validation PRIVATE ocudu_phy_cuda)

    # ============================================================================
    # Performance Benchmarks
    # ============================================================================

    # MCS performance sweep (comprehensive 5G NR MCS Tables 1 & 2)
    add_executable(mcs_sweep_5gnr tests/benchmarks/mcs_sweep_5gnr.cu)
    target_link_libraries(mcs_sweep_5gnr PRIVATE ocudu_phy_cuda)

    # Comprehensive MCS BLER sensitivity test (all 3 MCS tables)
    add_executable(mcs_bler_sensitivity_test tests/benchmarks/mcs_bler_sensitivity_test.cu)
    target_link_libraries(mcs_bler_sensitivity_test PRIVATE ocudu_phy_cuda)

    # MCS BLER curves generator (CSV output for plotting)
    add_executable(mcs_bler_curves tests/benchmarks/mcs_bler_curves.cu)
    target_link_libraries(mcs_bler_curves PRIVATE ocudu_phy_cuda)

    # CPU baseline benchmark (pure C++, no CUDA, for CPU vs GPU comparison)
    add_executable(cpu_baseline_benchmark tests/benchmarks/cpu_baseline_benchmark.cpp)
    target_include_directories(cpu_baseline_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    # Link pthread for std::thread
    find_package(Threads REQUIRED)
    target_link_libraries(cpu_baseline_benchmark PRIVATE Threads::Threads)

    # Simple Z=18 decode test (no filler bits)
    add_executable(test_z18_simple tests/unittests/test_z18_simple.cu)
    target_link_libraries(test_z18_simple PRIVATE ocudu_phy_cuda)
    target_include_directories(test_z18_simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Direct decoder test
    add_executable(test_decoder_direct tests/unittests/test_decoder_direct.cu)
    target_link_libraries(test_decoder_direct PRIVATE ocudu_phy_cuda)
    target_include_directories(test_decoder_direct PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # No puncture decoder test
    add_executable(test_decoder_no_punct tests/unittests/test_decoder_no_punct.cu)
    target_link_libraries(test_decoder_no_punct PRIVATE ocudu_phy_cuda)
    target_include_directories(test_decoder_no_punct PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Pattern decoder test
    add_executable(test_decoder_patterns tests/unittests/test_decoder_patterns.cu)
    target_link_libraries(test_decoder_patterns PRIVATE ocudu_phy_cuda)
    target_include_directories(test_decoder_patterns PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Encoder parity test
    add_executable(test_encoder_parity tests/unittests/test_encoder_parity.cu)
    target_link_libraries(test_encoder_parity PRIVATE ocudu_phy_cuda)
    target_include_directories(test_encoder_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Bit ordering test
    add_executable(test_bit_ordering tests/unittests/test_bit_ordering.cu)
    target_link_libraries(test_bit_ordering PRIVATE ocudu_phy_cuda)
    target_include_directories(test_bit_ordering PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # GPU vs CPU encoder comparison test
    add_executable(test_gpu_vs_cpu_encoder tests/unittests/test_gpu_vs_cpu_encoder.cu)
    target_link_libraries(test_gpu_vs_cpu_encoder PRIVATE ocudu_phy_cuda)
    target_include_directories(test_gpu_vs_cpu_encoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Encode/decode with MSB-first ordering test
    add_executable(test_encode_decode_msb tests/unittests/test_encode_decode_msb.cu)
    target_link_libraries(test_encode_decode_msb PRIVATE ocudu_phy_cuda)
    target_include_directories(test_encode_decode_msb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Decoder with filler bits test (isolates filler LLR handling)
    add_executable(test_decoder_with_filler tests/unittests/test_decoder_with_filler.cu)
    target_link_libraries(test_decoder_with_filler PRIVATE ocudu_phy_cuda)
    target_include_directories(test_decoder_with_filler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Decoder with puncture test
    add_executable(test_decoder_with_puncture tests/unittests/test_decoder_with_puncture.cu)
    target_link_libraries(test_decoder_with_puncture PRIVATE ocudu_phy_cuda)
    target_include_directories(test_decoder_with_puncture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Decoder without puncture test
    add_executable(test_decoder_no_puncture tests/unittests/test_decoder_no_puncture.cu)
    target_link_libraries(test_decoder_no_puncture PRIVATE ocudu_phy_cuda)
    target_include_directories(test_decoder_no_puncture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Iteration threshold / CRC early termination test
    add_executable(test_iteration_threshold tests/unittests/test_iteration_threshold.cu)
    target_link_libraries(test_iteration_threshold PRIVATE ocudu_phy_cuda)
    target_include_directories(test_iteration_threshold PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # BG1 puncture test (all-zero, encoded, noisy, multiple Z values)
    add_executable(test_bg1_puncture tests/unittests/test_bg1_puncture.cu)
    target_link_libraries(test_bg1_puncture PRIVATE ocudu_phy_cuda)
    target_include_directories(test_bg1_puncture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # CRC24A parallel implementation test
    add_executable(test_crc tests/unittests/test_crc.cu)
    target_link_libraries(test_crc PRIVATE ocudu_phy_cuda)
    target_include_directories(test_crc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    if(BUILD_OCUDU_PHY_CUDA_DIAGNOSTIC_TESTS)
        # Exploratory decoder diagnostics and dumped-LLR reproducers. Keep these out
        # of the default standalone test inventory so release builds focus on
        # maintained correctness and benchmark targets.
        add_executable(test_z_boundary tests/unittests/test_z_boundary.cu)
        target_link_libraries(test_z_boundary PRIVATE ocudu_phy_cuda)
        target_include_directories(test_z_boundary PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

        add_executable(test_z_diagnostics tests/unittests/test_z_diagnostics.cu)
        target_link_libraries(test_z_diagnostics PRIVATE ocudu_phy_cuda)
        target_include_directories(test_z_diagnostics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

        add_executable(test_z_compare tests/unittests/test_z_compare.cu)
        target_link_libraries(test_z_compare PRIVATE ocudu_phy_cuda)
        target_include_directories(test_z_compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

        add_executable(test_encoder_diagnostics tests/unittests/test_encoder_diagnostics.cu)
        target_link_libraries(test_encoder_diagnostics PRIVATE ocudu_phy_cuda)
        target_include_directories(test_encoder_diagnostics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

        add_executable(test_single_bit_trace tests/unittests/test_single_bit_trace.cu)
        target_link_libraries(test_single_bit_trace PRIVATE ocudu_phy_cuda)
        target_include_directories(test_single_bit_trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

        add_executable(test_decoder_diagnostics tests/unittests/test_decoder_diagnostics.cu)
        target_link_libraries(test_decoder_diagnostics PRIVATE ocudu_phy_cuda)
        target_include_directories(test_decoder_diagnostics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

        add_executable(test_encoder_decoder_trace tests/unittests/test_encoder_decoder_trace.cu)
        target_link_libraries(test_encoder_decoder_trace PRIVATE ocudu_phy_cuda)
        target_include_directories(test_encoder_decoder_trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

        add_executable(test_encoder_word_trace tests/unittests/test_encoder_word_trace.cu)
        target_link_libraries(test_encoder_word_trace PRIVATE ocudu_phy_cuda)
        target_include_directories(test_encoder_word_trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

        add_executable(test_e2e_llrs tests/unittests/test_e2e_llrs.cu)
        target_link_libraries(test_e2e_llrs PRIVATE ocudu_phy_cuda)
        target_include_directories(test_e2e_llrs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

        add_executable(test_srsran_llrs tests/unittests/test_srsran_llrs.cu)
        target_link_libraries(test_srsran_llrs PRIVATE ocudu_phy_cuda)
        target_include_directories(test_srsran_llrs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

        add_executable(test_integration_llrs tests/unittests/test_integration_llrs.cu)
        target_link_libraries(test_integration_llrs PRIVATE ocudu_phy_cuda)
        target_include_directories(test_integration_llrs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endif()

endif(BUILD_OCUDU_PHY_CUDA_TESTS)

# ============================================================================
# Installation
# ============================================================================

install(TARGETS ocudu_phy_cuda
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ============================================================================
# Configuration Summary
# ============================================================================

if(OCUDU_PHY_CUDA_STANDALONE)
    message(STATUS "===========================================")
    message(STATUS "OCUDU PHY CUDA Configuration:")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
    message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
    if(TARGET_TEGRA)
        message(STATUS "  Platform: Tegra (${TEGRA_PLATFORM})")
    else()
        message(STATUS "  Platform: Server/Desktop GPU")
    endif()
    message(STATUS "")
    message(STATUS "  Available targets:")
    message(STATUS "    make test                    - Run unit tests")
    message(STATUS "===========================================")
else()
    message(STATUS "OCUDU PHY CUDA: CUDA-accelerated 5G NR LDPC library configured")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    if(TARGET_TEGRA)
        message(STATUS "  Platform: Tegra (${TEGRA_PLATFORM})")
    else()
        message(STATUS "  Platform: Server/Desktop")
    endif()
    message(STATUS "  Library target: ocudu_phy_cuda")
endif()
