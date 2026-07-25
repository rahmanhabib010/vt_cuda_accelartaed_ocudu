# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
# Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.


set(PUSCH_DEMOD_SOURCES
        pusch_demodulator_impl.cpp
        demodulator_factories.cpp)

# Add GPU demodulator if CUDA is available
if(CUDA_ACCEL_FOUND)
    list(APPEND PUSCH_DEMOD_SOURCES pusch_demodulator_gpu_impl.cpp)
endif()

add_library(ocudu_pusch_demodulator STATIC ${PUSCH_DEMOD_SOURCES})

# Link CUDA for GPU demodulator
if(CUDA_ACCEL_FOUND)
    target_include_directories(ocudu_pusch_demodulator PRIVATE
        ${CMAKE_SOURCE_DIR}/lib/phy/upper/channel_coding/ldpc)
    target_link_libraries(ocudu_pusch_demodulator PRIVATE ocudu_ldpc_cuda)
    target_compile_definitions(ocudu_pusch_demodulator PRIVATE ENABLE_CUDA)
endif()

add_library(ocudu_pusch_processor STATIC
        processor_factories.cpp
        pusch_codeblock_decoder.cpp
        pusch_decoder_empty_impl.cpp
        pusch_decoder_hw_impl.cpp
        pusch_decoder_impl.cpp
        pusch_processor_impl.cpp
        pusch_processor_validator_impl.cpp
        ulsch_demultiplex_impl.cpp)
set(OCUDU_PUSCH_LIBS ocudu_upper_phy_support ocudu_ran ocudu_pusch_demodulator)

target_link_libraries(ocudu_pusch_processor PUBLIC ${OCUDU_PUSCH_LIBS})

# Link CUDA for GPU processor components
if(CUDA_ACCEL_FOUND)
    target_include_directories(ocudu_pusch_processor PRIVATE
        ${CMAKE_SOURCE_DIR}/lib/phy/upper/channel_coding/ldpc)
    target_link_libraries(ocudu_pusch_processor PRIVATE ocudu_ldpc_cuda)
    target_compile_definitions(ocudu_pusch_processor PRIVATE ENABLE_CUDA)
endif()
