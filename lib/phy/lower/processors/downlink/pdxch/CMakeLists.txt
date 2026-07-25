# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
# Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.


add_library(ocudu_lower_phy_pdxch_processor STATIC
        pdxch_processor_factories.cpp
        pdxch_processor_impl.cpp)
target_link_libraries(ocudu_lower_phy_pdxch_processor
        ocudu_instrumentation
        ocudu_lower_phy_modulation)

if(CUDA_ACCEL_FOUND)
    target_sources(ocudu_lower_phy_pdxch_processor PRIVATE
            pdxch_baseband_modulator_cuda.cpp)
    target_include_directories(ocudu_lower_phy_pdxch_processor PRIVATE
            ${CUDA_ACCEL_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR})
    target_compile_definitions(ocudu_lower_phy_pdxch_processor PRIVATE ENABLE_CUDA)
    target_link_libraries(ocudu_lower_phy_pdxch_processor
            ${CUDA_ACCEL_LIBRARIES})
endif()
