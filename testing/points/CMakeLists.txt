# ======================================================================== #
# Copyright 2023-2024 Ingo Wald                                            #
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

# =========================================================================
# 'closestPoint': both query points and data are regular points of
# type <typeN> (float2, int3, etc). For each query point, find the
# closest data point.
# =========================================================================
function(add_testing_target sampleName device suffix T D)
  # sampleName - name of sample
  # device - host or cuda
  # T - scalar type
  # D - dimension
    add_executable(testing_points_${sampleName}_${device}_${T}${D}
      ${sampleName}.h
      ${sampleName}.${suffix}
      )
    target_link_libraries(testing_points_${sampleName}_${device}_${T}${D}
      PUBLIC
      cuBQL_testing_common
      cuBQL_${device}_${T}${D}
      )
    target_compile_definitions(testing_points_${sampleName}_${device}_${T}${D}
      PUBLIC
      -DCUBQL_TEST_T=${T}
      -DCUBQL_TEST_D=${D}
      )
endfunction()

foreach(T IN ITEMS float int double longlong)
  foreach(D IN ITEMS 2 3 4)
    add_testing_target(closestPoint cuda cu ${T} ${D})
    add_testing_target(closestPoint cpu cpp ${T} ${D})

    add_testing_target(knn5 cuda cu ${T} ${D})
    add_testing_target(knn5 cpu cpp ${T} ${D})
  endforeach()
endforeach()
