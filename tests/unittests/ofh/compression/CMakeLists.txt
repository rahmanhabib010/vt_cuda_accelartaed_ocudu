# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI

if (CUDA_ACCEL_FOUND)
    add_executable(ofh_iq_compression_cuda_test ofh_iq_compression_cuda_test.cpp)
    target_link_libraries(ofh_iq_compression_cuda_test
            ocudu_ofh_compression
            ocudu_support
            GTest::gtest
            GTest::gtest_main)
    target_compile_definitions(ofh_iq_compression_cuda_test PRIVATE ENABLE_CUDA)
    gtest_discover_tests(ofh_iq_compression_cuda_test)
endif()
