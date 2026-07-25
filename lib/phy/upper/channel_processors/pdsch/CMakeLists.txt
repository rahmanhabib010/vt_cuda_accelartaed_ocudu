# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
# Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.


set(PDSCH_SOURCES
        factories.cpp
        pdsch_block_processor_impl.cpp
        pdsch_block_processor_hw_impl.cpp
        pdsch_encoder_hw_impl.cpp
        pdsch_encoder_impl.cpp
        pdsch_modulator_impl.cpp
        pdsch_processor_flexible_impl.cpp
        pdsch_processor_impl.cpp
        pdsch_processor_validator_impl.cpp)

# Add GPU implementation if CUDA is available
if(CUDA_ACCEL_FOUND)
    list(APPEND PDSCH_SOURCES pdsch_block_processor_gpu_impl.cpp)
endif()

add_library(ocudu_pdsch_processor STATIC ${PDSCH_SOURCES})

target_link_libraries(ocudu_pdsch_processor
        PUBLIC
        ocudu_channel_modulation
        ocudu_phy_support
        ocudu_ran
        ocudu_upper_phy_support)

# Link CUDA if available
if(CUDA_ACCEL_FOUND)
    target_include_directories(ocudu_pdsch_processor PRIVATE
        ${CMAKE_SOURCE_DIR}/lib/phy/upper/channel_coding/ldpc)
    target_link_libraries(ocudu_pdsch_processor PRIVATE ocudu_ldpc_cuda)
    target_compile_definitions(ocudu_pdsch_processor PRIVATE ENABLE_CUDA)
endif()
