# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
# Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

set_directory_properties(PROPERTIES LABELS "phy")

if(CUDA_ACCEL_FOUND)
    add_executable(prach_detector_cuda_test prach_detector_cuda_test.cpp)
    target_include_directories(prach_detector_cuda_test PRIVATE ${PROJECT_SOURCE_DIR}/lib/phy/upper)
    target_link_libraries(prach_detector_cuda_test
            ocudu_prach_detector
            ocudu_upper_phy
            ocudu_phy_support
            ocudu_dft
            ocudulog
            GTest::gtest
            GTest::gtest_main)
    add_test(prach_detector_cuda_test prach_detector_cuda_test)
endif()
