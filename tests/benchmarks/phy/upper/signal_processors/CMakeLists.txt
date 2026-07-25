# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
# Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.


add_executable(dmrs_pdsch_processor_benchmark dmrs_pdsch_processor_benchmark.cpp)
target_link_libraries(dmrs_pdsch_processor_benchmark
        ocuduvec
        ocudulog
        ocudu_dmrs_pdsch
        ocudu_phy_support
        ocudu_sequence_generators
        ocudu_channel_precoder)
add_test(dmrs_pdsch_processor_benchmark dmrs_pdsch_processor_benchmark -D -s -R 1)


add_executable(srs_estimator_benchmark srs_estimator_benchmark.cpp)
target_link_libraries(srs_estimator_benchmark
        ocuduvec
        ocudulog
        ocudu_channel_precoder
        ocudu_srs_estimator
        ocudu_phy_support
        ocudu_dft
        ocudu_sequence_generators)
add_test(srs_estimator_benchmark srs_estimator_benchmark -D -s -R 1)

foreach(srs_gpu_benchmark
        srs_estimator_gpu_config_sweep
        srs_estimator_gpu_latency_benchmark
        srs_estimator_gpu_sensitivity_sweep
        srs_estimator_gpu_thread_safety_test)
    add_executable(${srs_gpu_benchmark} ${srs_gpu_benchmark}.cpp)
    target_link_libraries(${srs_gpu_benchmark}
            ocuduvec
            ocudulog
            ocudu_channel_precoder
            ocudu_srs_estimator
            ocudu_phy_support
            ocudu_dft
            ocudu_sequence_generators)
    if(CUDA_ACCEL_FOUND)
        target_include_directories(${srs_gpu_benchmark} PRIVATE
                ${CMAKE_SOURCE_DIR}/lib/phy/upper
                ${CMAKE_SOURCE_DIR}/lib/phy/upper/channel_coding/ldpc)
        target_link_libraries(${srs_gpu_benchmark} ocudu_ldpc_cuda CUDA::cudart)
        target_compile_definitions(${srs_gpu_benchmark} PRIVATE ENABLE_CUDA)
    else()
        target_compile_options(${srs_gpu_benchmark} PRIVATE
                -Wno-unused-function
                -Wno-unused-variable
                -Wno-error=unused-function
                -Wno-error=unused-variable)
    endif()
endforeach()

set(srs_estimator_gpu_quick_profiles
        baseline_4x4_n4
        n2_1x1_phys3
        n2_2x2_sparse_rx
        n4_4x2_shifted
        n4_2x4_interleaved)

foreach(srs_estimator_gpu_profile IN LISTS srs_estimator_gpu_quick_profiles)
    add_test(NAME srs_estimator_gpu_latency_${srs_estimator_gpu_profile}
            COMMAND srs_estimator_gpu_latency_benchmark -R 1 -W 0 -P ${srs_estimator_gpu_profile} -G host)
    add_test(NAME srs_estimator_gpu_sensitivity_${srs_estimator_gpu_profile}
            COMMAND srs_estimator_gpu_sensitivity_sweep -F 1 -P ${srs_estimator_gpu_profile} -A 20 -B 20 -D 5 -G host)
endforeach()

add_test(NAME srs_estimator_gpu_config_sweep
        COMMAND srs_estimator_gpu_config_sweep -q -N 64 -G host)
add_test(NAME srs_estimator_gpu_config_sweep_windowed
        COMMAND srs_estimator_gpu_config_sweep -w -q -N 64 -G host)
add_test(NAME srs_estimator_gpu_thread_safety_host
        COMMAND srs_estimator_gpu_thread_safety_test -T 4 -I 8 -P baseline_4x4_n4 -G host)
add_test(NAME srs_estimator_gpu_thread_safety_visible
        COMMAND srs_estimator_gpu_thread_safety_test -T 4 -I 8 -P baseline_4x4_n4 -G visible)
