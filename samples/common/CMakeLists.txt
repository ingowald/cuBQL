# ======================================================================== #
# Copyright 2024-2024 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

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
  



# ==================================================================
# tool that runs a 2d generator, and writes the points to an svg file
# ==================================================================
if (CUBQL_HAVE_CUDA)
  add_executable(cuBQL_plotPoints
    plotPoints.cu
    )
  target_link_libraries(cuBQL_plotPoints
    PRIVATE
    cuBQL_samples_common
    )
endif()
