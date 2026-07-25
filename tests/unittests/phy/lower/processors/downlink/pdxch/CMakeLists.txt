# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
# Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.


add_executable(pdxch_processor_test pdxch_processor_test.cpp)
target_link_libraries(pdxch_processor_test ocudu_lower_phy_pdxch_processor ocudulog GTest::gtest GTest::gtest_main)
add_test(pdxch_processor_test pdxch_processor_test)

if(CUDA_ACCEL_FOUND)
    add_executable(pdxch_baseband_modulator_cuda_test pdxch_baseband_modulator_cuda_test.cpp)
    target_include_directories(pdxch_baseband_modulator_cuda_test PRIVATE
            ${PROJECT_SOURCE_DIR}/lib/phy/lower/processors/downlink/pdxch)
    target_link_libraries(pdxch_baseband_modulator_cuda_test
            ocudu_lower_phy_pdxch_processor
            ocudu_dft
            ocudulog
            GTest::gtest
            GTest::gtest_main)
    add_test(pdxch_baseband_modulator_cuda_test pdxch_baseband_modulator_cuda_test)
endif()
