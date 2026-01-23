// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/bvh.h"
#include <omp.h>

namespace cuBQL {
  namespace omp {
    
    struct Context {
      Context(int gpuID);

      template<typename T>
      void alloc_and_upload(T *&d_data, const T *h_data, size_t Nelements);
      
      template<typename T>
      void alloc_and_upload(T *&d_data, const std::vector<T> &h_vector);

      template<typename T>
      std::vector<T> download_vector(const T *d_data, size_t N);
      
      int gpuID;
      int hostID;
    };
    struct Kernel {
      inline int workIdx() const { return _workIdx; }
      int _workIdx;
    };



    // ##################################################################
    // IMPLEMENTATION SECTION
    // ##################################################################
    Context::Context(int gpuID)
      : gpuID(gpuID),
        hostID(omp_get_initial_device())
    {
      assert(gpuID < omp_get_num_devices());
      printf("#cuBQL:omp:Context(gpu=%i/%i,host=%i)\n",
             gpuID,omp_get_num_devices(),hostID);
    }
    
    template<typename T>
    void Context::alloc_and_upload(T *&d_data,
                                   const T *h_data,
                                   size_t N)
    {
      printf("target_alloc N %li gpu %i\n",N,gpuID);
      d_data = (T *)omp_target_alloc(N*sizeof(T),gpuID);
      printf("ptr %p\n",d_data);
      assert(d_data);
      omp_target_memcpy(d_data,h_data,N*sizeof(T),
                        0,0,gpuID,hostID);
    }
      
    template<typename T>
    void Context::alloc_and_upload(T *&d_data,
                                   const std::vector<T> &h_vector)
    { alloc_and_upload(d_data,h_vector.data(),h_vector.size()); }

    template<typename T>
    std::vector<T> Context::download_vector(const T *d_data, size_t N)
    {
      std::vector<T> out(N);
      omp_target_memcpy(out.data(),d_data,N*sizeof(T),
                        0,0,hostID,gpuID);
      return out;
    }
    
  } // ::cuBQL::omp
} // ::cuBQL
