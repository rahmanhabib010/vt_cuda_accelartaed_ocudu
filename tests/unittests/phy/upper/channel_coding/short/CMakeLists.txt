# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
# Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.


set_directory_properties(PROPERTIES LABELS "phy")

add_executable(short_block_chain_test short_block_chain_test.cpp)
target_link_libraries(short_block_chain_test ocudu_channel_coding ocudu_upper_phy_support ocudulog GTest::gtest GTest::gtest_main)
add_test(short_block_chain_test short_block_chain_test)

if(CUDA_ACCEL_FOUND)
    add_executable(short_block_detector_cuda_test short_block_detector_cuda_test.cu)
    target_link_libraries(short_block_detector_cuda_test
        ocudu_ldpc_cuda
        ocudu_channel_coding
        ocudu_upper_phy_support
        ocudulog
        CUDA::cudart
        GTest::gtest
        GTest::gtest_main)
    target_include_directories(short_block_detector_cuda_test PRIVATE
        ${PROJECT_SOURCE_DIR}/lib/phy/upper/channel_coding/ldpc)
    set_target_properties(short_block_detector_cuda_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    add_test(short_block_detector_cuda_test short_block_detector_cuda_test)
endif()
