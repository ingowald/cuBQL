# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(CUBQL_USER_DIM "" CACHE STRING "user-specified dim for tests w/ dim!=2,3,4")

add_library(cuBQL_samples_common
  Generator.h
  Generator.cpp
  tiny_obj_loader.h
  loadOBJ.h
  loadOBJ.cpp
  loadBinMesh.h
  loadBinMesh.cpp
  )
target_link_libraries(cuBQL_samples_common
  PUBLIC
  cuBQL
  )

# ==================================================================
# for testing, allow to specify a compile-time but user-defined
# arbitrary dimensionality. not all tests might respect that (eg,
# triangles only work for float3), but some will
# ==================================================================
if (NOT ("${CUBQL_USER_DIM}" STREQUAL ""))
  target_compile_definitions(cuBQL_samples_common
    PUBLIC
    CUBQL_USER_DIM=${CUBQL_USER_DIM})
endif()
  
