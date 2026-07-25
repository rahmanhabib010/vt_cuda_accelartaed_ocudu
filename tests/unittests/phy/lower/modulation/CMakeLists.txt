# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
# Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.


set_directory_properties(PROPERTIES LABELS "phy")

add_executable(ofdm_modulator_unittest ofdm_modulator_unittest.cpp)
target_link_libraries(ofdm_modulator_unittest ocudulog ocudu_lower_phy_modulation)
add_test(ofdm_modulator_unittest ofdm_modulator_unittest)

add_executable(ofdm_demodulator_unittest ofdm_demodulator_unittest.cpp)
target_link_libraries(ofdm_demodulator_unittest ocudulog ocudu_lower_phy_modulation)
add_test(ofdm_demodulator_unittest ofdm_demodulator_unittest)

if(CUDA_ACCEL_FOUND)
    add_executable(ofdm_demodulator_cuda_test ofdm_demodulator_cuda_test.cpp)
    target_include_directories(ofdm_demodulator_cuda_test PRIVATE ${PROJECT_SOURCE_DIR}/lib/phy/upper)
    target_link_libraries(ofdm_demodulator_cuda_test
            ocudu_lower_phy_modulation
            ocudu_upper_phy
            ocudulog
            CUDA::cudart
            GTest::gtest
            GTest::gtest_main)
    add_test(ofdm_demodulator_cuda_test ofdm_demodulator_cuda_test)

    add_executable(ofdm_prach_demodulator_cuda_test ofdm_prach_demodulator_cuda_test.cpp)
    target_include_directories(ofdm_prach_demodulator_cuda_test PRIVATE ${PROJECT_SOURCE_DIR}/lib/phy/upper)
    target_link_libraries(ofdm_prach_demodulator_cuda_test
            ocudu_lower_phy_modulation
            ocudu_upper_phy
            ocudulog
            CUDA::cudart
            GTest::gtest
            GTest::gtest_main)
    add_test(ofdm_prach_demodulator_cuda_test ofdm_prach_demodulator_cuda_test)
endif()
