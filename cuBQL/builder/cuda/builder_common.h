// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

/* this file contains the entire builder; this should never be included directly */
#pragma once

#include "cuBQL/bvh.h"
// #include "cuBQL/builder/cuda.h"
#ifdef __HIPCC__
# include <hipcub/hipcub.hpp>
#else
# include <cub/cub.cuh>
#endif
#include <float.h>
#include <limits.h>

namespace cuBQL {
  namespace gpuBuilder_impl {

    template<typename T, typename count_t>
    inline void _ALLOC(T *&ptr, count_t count, cudaStream_t s,
                       GpuMemoryResource &mem_resource)
    { mem_resource.malloc((void**)&ptr,count*sizeof(T),s); }
    
    template<typename T>
    inline void _FREE(T *&ptr, cudaStream_t s, GpuMemoryResource &mem_resource)
    { mem_resource.free((void*)ptr,s); ptr = 0; }
    
    typedef enum : int8_t { OPEN_BRANCH, OPEN_NODE, DONE_NODE } NodeState;
    
    template<typename box_t>
    struct CUBQL_ALIGN(8) AtomicBox {
      inline __device__ bool is_empty() const { return lower[0] > upper[0]; }
      inline __device__ void  set_empty();
      // set_empty, in owl::common-style naming
      inline __device__ void  clear() { set_empty(); }
      inline __device__ float get_center(int dim) const;
      inline __device__ box_t make_box() const;

      inline __device__ float get_lower(int dim) const {
        if (box_t::numDims>4) 
          return decode(lower[dim]);
        else if (box_t::numDims==4) {
          return decode(dim>1
                        ?((dim>2)?lower[3]:lower[2])
                        :((dim  )?lower[1]:lower[0]));
        } else if (box_t::numDims==3) {
          return decode(dim>1
                        ?lower[2]
                        :((dim  )?lower[1]:lower[0]));
        } else
          return decode(lower[dim]);
      }
      inline __device__ float get_upper(int dim) const {
        if (box_t::numDims>4) 
          return decode(upper[dim]);
        else if (box_t::numDims==4) {
          return decode(dim>1
                        ?((dim>2)?upper[3]:upper[2])
                        :((dim  )?upper[1]:upper[0]));
        } else if (box_t::numDims==3)
          return decode(dim>1
                        ?upper[2]
                        :((dim  )?upper[1]:upper[0]));
        else
          return decode(upper[dim]);
      }

      int32_t lower[box_t::numDims];
      int32_t upper[box_t::numDims];

      inline static __device__ int32_t encode(float f);
      inline static __device__ float   decode(int32_t bits);
    };
    
    template<typename box_t>
    inline __device__ float AtomicBox<box_t>::get_center(int dim) const
    {
      return 0.5f*(get_lower(dim)+get_upper(dim));
      // return 0.5f*(decode(lower[dim])+decode(upper[dim]));
    }

    template<typename box_t>
    inline __device__ box_t AtomicBox<box_t>::make_box() const
    {
      box_t box;
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        box.lower[d] = decode(lower[d]);
        box.upper[d] = decode(upper[d]);
      }
      return box;
    }
    
    template<typename box_t>
    inline __device__ int32_t AtomicBox<box_t>::encode(float f)
    {
      const int32_t sign = 0x80000000;
      int32_t bits = __float_as_int(f);
      if (bits & sign) bits ^= 0x7fffffff;
      return bits;
    }
      
    template<typename box_t>
    inline __device__ float AtomicBox<box_t>::decode(int32_t bits)
    {
      const int32_t sign = 0x80000000;
      if (bits & sign) bits ^= 0x7fffffff;
      return __int_as_float(bits);
    }
    
    template<typename box_t>
    inline __device__ void AtomicBox<box_t>::set_empty()
    {
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        lower[d] = encode(+FLT_MAX);
        upper[d] = encode(-FLT_MAX);
      }
    }

    template<typename box_t> inline __device__
    void atomic_grow(AtomicBox<box_t> &abox, const typename box_t::vec_t &other)
    {
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        const int32_t enc = AtomicBox<box_t>::encode(other[d]);//get(other,d));
        if (enc < abox.lower[d])
          atomicMin(&abox.lower[d],enc);
        if (enc > abox.upper[d])
          atomicMax(&abox.upper[d],enc);
      }
    } 
    
    template<typename box_t>
    inline __device__ void atomic_grow(AtomicBox<box_t> &abox, const box_t &other)
    {
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        const int32_t enc_lower = AtomicBox<box_t>::encode(other.get_lower(d));
        const int32_t enc_upper = AtomicBox<box_t>::encode(other.get_upper(d));
        if (enc_lower < abox.lower[d]) atomicMin(&abox.lower[d],enc_lower);
        if (enc_upper > abox.upper[d]) atomicMax(&abox.upper[d],enc_upper);
      }
    }

    template<typename box_t>
    inline __device__ void atomic_grow(AtomicBox<box_t> &abox, const AtomicBox<box_t> &other)
    {
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        const int32_t enc_lower = other.lower[d];
        const int32_t enc_upper = other.upper[d];
        if (enc_lower < abox.lower[d]) atomicMin(&abox.lower[d],enc_lower);
        if (enc_upper > abox.upper[d]) atomicMax(&abox.upper[d],enc_upper);
      }
    }
    
    struct BuildState {
      uint32_t  numNodes;
    };

  } // ::cuBQL::gpuBuilder_impl
} // ::cuBQL

