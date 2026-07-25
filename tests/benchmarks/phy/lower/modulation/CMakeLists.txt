# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
# Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

set_directory_properties(PROPERTIES LABELS "phy")

if(CUDA_ACCEL_FOUND)
    add_executable(ofdm_demodulator_cuda_benchmark ofdm_demodulator_cuda_benchmark.cpp)
    target_link_libraries(ofdm_demodulator_cuda_benchmark
            ocudu_lower_phy_modulation
            ocudu_phy_support
            ocudu_dft
            ocuduvec
            ocudulog
            ${CUDA_ACCEL_LIBRARIES}
            CUDA::cudart)
endif()
