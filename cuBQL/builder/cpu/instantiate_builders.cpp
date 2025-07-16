// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

/*! instantiates the GPU builder(s) */
#define CUBQL_CPU_BUILDER_IMPLEMENTATION 1

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cpu/spatialMedian.h"

#ifdef CUBQL_INSTANTIATE_T
// instantiate an explict type and dimension
CUBQL_CPU_INSTANTIATE_BINARY_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D)
CUBQL_CPU_INSTANTIATE_WIDE_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D,4)
CUBQL_CPU_INSTANTIATE_WIDE_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D,8)
#else
// default instantiation(s) for float3 only
CUBQL_CPU_INSTANTIATE_BINARY_BVH(float,3)
CUBQL_CPU_INSTANTIATE_WIDE_BVH(float,3,4)
CUBQL_CPU_INSTANTIATE_WIDE_BVH(float,3,8)
#endif
  
 
