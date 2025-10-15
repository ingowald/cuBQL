# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==================================================================
# sample that performs gpu-side closest-point queries between sets of
# float3 points
# ==================================================================
add_executable(cuBQL_sample01_points_closestPoint_cpu
  closestPoint.cpp
  )
target_link_libraries(cuBQL_sample01_points_closestPoint_cpu
  # the cpu-side builders for float3 data
  cuBQL_cpu_float3
  # common samples stuff
  cuBQL_samples_common
  )

