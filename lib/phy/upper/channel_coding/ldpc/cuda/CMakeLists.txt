# SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI

# CUDA accelerated LDPC adapter.
# Only built when CUDA is enabled and the in-tree CUDA kernels are available.

if(CUDA_ACCEL_FOUND)
    message(STATUS "Building CUDA LDPC adapter")

    # Create the CUDA LDPC adapter library.
    add_library(ocudu_ldpc_cuda STATIC
        # RX-side components
        crc_calculator_cuda.cpp
        demodulation_mapper_cuda.cpp
        ldpc_decoder_cuda.cpp
        pseudo_random_generator_cuda.cpp
        pusch_codeblock_decoder_cuda_batch.cpp
        pusch_device_grid_reader_cuda.cpp
        pusch_sch_llr_compactor.cu
        # TX-side components (PDSCH)
        ldpc_encoder_cuda.cpp
        ldpc_encoder_cuda_batch.cpp
        modulation_mapper_cuda.cpp
        pdsch_device_grid_writer_cuda.cpp
        pdsch_resource_grid_mapper_cuda.cu
        pdsch_tb_encoder_cuda.cu
        # Factory
        ldpc_cuda_factories.cpp
    )

    target_include_directories(ocudu_ldpc_cuda
        PUBLIC
            ${CUDA_ACCEL_INCLUDE_DIRS}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Link against CUDA, CUDA runtime, and CUDA driver API.
    target_link_libraries(ocudu_ldpc_cuda
        PUBLIC
            ${CUDA_ACCEL_LIBRARIES}
            CUDA::cudart
            CUDA::cuda_driver
        PRIVATE
            ocudu_support
            ocudu_phy_support
            ocudu_channel_modulation
    )

    # Set CUDA-specific compile options
    # Do NOT enable separable compilation - library is self-contained.
    # Use the top-level CUDA architecture selection so local GPU builds can
    # target architectures beyond Hopper, such as GB10 compute capability 12.1.
    set_target_properties(ocudu_ldpc_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )

    target_compile_definitions(ocudu_ldpc_cuda PUBLIC ENABLE_CUDA)

else()
    message(STATUS "CUDA not available - skipping CUDA LDPC adapter")
endif()
