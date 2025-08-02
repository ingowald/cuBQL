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
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/builder/cuda/radix.h"
#include "cuBQL/builder/cuda/rebinMortonBuilder.h"


#define CUBQL_INSTANTIATE_BINARY_BVH(T,D)                               \
  namespace cuBQL {                                                     \
  namespace radixBuilder_impl {                                         \
    template                                                            \
    void build(BinaryBVH<T,D>        &bvh,                              \
               const typename BuildState<T,D>::box_t       *boxes,      \
               uint32_t           numPrims,                             \
               BuildConfig        buildConfig,                          \
               cudaStream_t       s,                                    \
               GpuMemoryResource &memResource);                         \
  }                                                                     \
  template void gpuBuilder(BinaryBVH<T,D>    &bvh,                      \
                           const box_t<T,D>  *boxes,                    \
                           uint32_t           numBoxes,                 \
                           BuildConfig        buildConfig,              \
                           cudaStream_t       s,                        \
                           GpuMemoryResource &mem_resource);            \
  namespace cuda {                                                      \
    template                                                            \
    void radixBuilder<T,D>(BinaryBVH<T,D>    &bvh,                      \
                           const box_t<T,D>  *boxes,                    \
                           uint32_t           numBoxes,                 \
                           BuildConfig        buildConfig,              \
                           cudaStream_t       s,                        \
                           GpuMemoryResource &mem_resource);            \
    template                                                            \
    void rebinRadixBuilder<T,D>(BinaryBVH<T,D>    &bvh,                 \
                                const box_t<T,D>  *boxes,               \
                                uint32_t           numBoxes,            \
                                BuildConfig        buildConfig,         \
                                cudaStream_t       s,                   \
                                GpuMemoryResource &mem_resource);       \
    template                                                            \
    void sahBuilder<T,D>(BinaryBVH<T,D>    &bvh,                        \
                         const box_t<T,D>  *boxes,                      \
                         uint32_t           numBoxes,                   \
                         BuildConfig        buildConfig,                \
                         cudaStream_t       s,                          \
                         GpuMemoryResource &mem_resource);              \
    template                                                            \
    void free(BinaryBVH<T,D>    &bvh,                                   \
              cudaStream_t       s,                                     \
              GpuMemoryResource &mem_resource);                         \
  }                                                                     \
  }                                                                     \
  
#define CUBQL_INSTANTIATE_WIDE_BVH(T,D,N)                       \
  namespace cuBQL {                                             \
    template void gpuBuilder(WideBVH<T,D,N>    &bvh,            \
                             const box_t<T,D>  *boxes,          \
                             uint32_t           numBoxes,       \
                             BuildConfig        buildConfig,    \
                             cudaStream_t       s,              \
                             GpuMemoryResource &mem_resource);  \
    template void free(WideBVH<T,D,N>  &bvh,                    \
                       cudaStream_t s,                          \
                       GpuMemoryResource& mem_resource);        \
  }


// CUBQL_INSTANTIATE_BINARY_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D)

#ifdef CUBQL_INSTANTIATE_T
// instantiate an explict type and dimension
CUBQL_INSTANTIATE_BINARY_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D)
CUBQL_INSTANTIATE_WIDE_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D,4)
CUBQL_INSTANTIATE_WIDE_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D,8)
#else
// default instantiation(s) for float3 only
CUBQL_INSTANTIATE_BINARY_BVH(float,3)
CUBQL_INSTANTIATE_WIDE_BVH(float,3,4)
CUBQL_INSTANTIATE_WIDE_BVH(float,3,8)
#endif
  
 
