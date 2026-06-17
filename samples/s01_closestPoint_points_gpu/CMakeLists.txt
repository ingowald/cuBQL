# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if (CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP)
  if (CUBQL_USE_HIP)
    set_source_files_properties(closestPoint.cu closestPoint_WideBVH.cu
      PROPERTIES LANGUAGE HIP)
  endif()
  add_executable(cuBQL_sample01_points_closestPoint_cuda
    closestPoint.cu
    )
  target_link_libraries(cuBQL_sample01_points_closestPoint_cuda
    # the cuda-side builders for float3 data
    cuBQL_cuda_float3
    # common samples stuff
    cuBQL_samples_common
    )


  add_executable(cuBQL_sample01_points_closestPoint_wideBVH_cuda
    closestPoint_WideBVH.cu
    )
  target_link_libraries(cuBQL_sample01_points_closestPoint_wideBVH_cuda
    # the cuda-side builders for float3 data
    cuBQL_cuda_float3
    # common samples stuff
    cuBQL_samples_common
    )
endif()
