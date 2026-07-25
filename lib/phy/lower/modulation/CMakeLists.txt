# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
# Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.


add_library(ocudu_lower_phy_modulation STATIC
        modulation_factories.cpp
        ofdm_modulator_impl.cpp
        ofdm_demodulator_impl.cpp
        ofdm_prach_demodulator_impl.cpp)
target_link_libraries(ocudu_lower_phy_modulation ocuduvec ocudu_ran ocudu_support)
if(CUDA_ACCEL_FOUND)
    target_sources(ocudu_lower_phy_modulation PRIVATE
            ofdm_demodulator_cuda_impl.cpp
            ofdm_prach_demodulator_cuda_impl.cpp)
    target_compile_definitions(ocudu_lower_phy_modulation PUBLIC ENABLE_CUDA)
    target_include_directories(ocudu_lower_phy_modulation PRIVATE ${CUDA_ACCEL_INCLUDE_DIRS})
    target_link_libraries(ocudu_lower_phy_modulation ${CUDA_ACCEL_LIBRARIES} CUDA::cudart CUDA::cuda_driver)
endif()
