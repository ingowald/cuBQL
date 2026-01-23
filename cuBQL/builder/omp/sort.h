// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/omp/common.h"

namespace cuBQL {
  namespace omp {
    namespace bitonic {
      
      template<typename key_t>
      void print_all(key_t *const d_values,
                     int numValues,
                     int barInterval, int commaInterval,
                     Context *omp)
      {
        std::vector<key_t> v = omp->download_vector(d_values,numValues);
        for (int i=0;i<numValues;i++) {
          std::cout << v[i] << " ";
          if (barInterval && (i+1)%barInterval==0)
            printf("| ");
          else if (commaInterval && (i+1)%commaInterval==0)
            printf(", ");
        }
        std::cout << std::endl;
      }
      

      template<typename key_t>
      void g_orderSegmentPairs(Kernel kernel,
                               int logSegLen,
                               key_t *const d_values,
                               int numValues)

      {
        uint32_t tid   = kernel.workIdx();
        
        // bool dbg = tid == 1 || tid == 0;
        
        uint32_t segLen    = 1<<logSegLen;
        uint32_t pairIdx   = tid>>logSegLen;
        uint32_t pairRank  = tid-(pairIdx<<logSegLen);
        uint32_t pairBegin = 2*(pairIdx<<logSegLen);


        uint32_t r = pairBegin+segLen+pairRank;
        uint32_t l = pairBegin+segLen-1-pairRank;
        // if (dbg) printf("(%i) pairIdx %i pairLen %i rank %i l %i r %i\n",
        //                 tid,pairIdx,pairLen,pairRank,l,r);
                        
                        
        if (r >= numValues) return;
        
        key_t lv = d_values[l];
        key_t rv = d_values[r];
        if (rv < lv) {
          d_values[r] = lv;
          d_values[l] = rv;
        }
      }
      
      template<typename key_t, typename value_t>
      void g_orderSegmentPairs(Kernel kernel,
                               int logSegLen,
                               key_t *const d_keys,
                               value_t *const d_values,
                               int numValues)

      {
        uint32_t tid   = kernel.workIdx();
        
        // bool dbg = tid == 1 || tid == 0;
        
        uint32_t segLen    = 1<<logSegLen;
        uint32_t pairIdx   = tid>>logSegLen;
        uint32_t pairRank  = tid-(pairIdx<<logSegLen);
        uint32_t pairBegin = 2*(pairIdx<<logSegLen);


        uint32_t r = pairBegin+segLen+pairRank;
        uint32_t l = pairBegin+segLen-1-pairRank;
        // if (dbg) printf("(%i) pairIdx %i pairLen %i rank %i l %i r %i\n",
        //                 tid,pairIdx,pairLen,pairRank,l,r);
                        
                        
        if (r >= numValues) return;
        
        key_t lk = d_keys[l];
        key_t rk = d_keys[r];
        key_t lv = d_values[l];
        key_t rv = d_values[r];
        if (rk < lk) {
          d_values[r] = lv;
          d_values[l] = rv;
          d_keys[r]   = lk;
          d_keys[l]   = rk;
        }
      }
      
      template<typename key_t>
      void orderSegmentPairs(int logSegLen,
                             key_t *const d_values,
                             int numValues,
                             Context *omp)

      {
#pragma omp target device(omp->gpuID)
#pragma omp teams distribute parallel for
        for (int i=0;i<numValues;i++)
          g_orderSegmentPairs(Kernel{i},logSegLen,
                              d_values,numValues);
      }
      
      template<typename key_t, typename value_t>
      void orderSegmentPairs(int logSegLen,
                             key_t *const d_keys,
                             value_t *const d_values,
                             int numValues,
                             Context *omp)

      {
#pragma omp target device(omp->gpuID)
#pragma omp teams distribute parallel for
        for (int i=0;i<numValues;i++)
          g_orderSegmentPairs(Kernel{i},logSegLen,
                              d_keys,d_values,numValues);
      }
      
      template<typename key_t>
      void sortSegments(int logSegLen,
                        key_t *const d_keys,
                        int numValues,
                        Context *omp)
      {
        if (logSegLen == 0) return;

        sortSegments(logSegLen-1,d_keys,numValues,omp);
        orderSegmentPairs(logSegLen-1,d_keys,numValues,omp);
        sortSegments(logSegLen-1,d_keys,numValues,omp);
      }
      
      template<typename key_t, typename value_t>
      void sortSegments(int logSegLen,
                        key_t *const d_keys,
                        value_t *const d_values,
                        int numValues,
                        Context *omp)
      {
        if (logSegLen == 0) return;

        sortSegments(logSegLen-1,d_keys,d_values,numValues,omp);
        orderSegmentPairs(logSegLen-1,d_keys,d_values,numValues,omp);
        sortSegments(logSegLen-1,d_keys,d_values,numValues,omp);
      }
      
      template<typename key_t>
      void sort(key_t *const d_values,
                int numValues,
                Context *omp)
      
      {
        uint32_t logSegLen = 0;
        while (1<<logSegLen < numValues)
          logSegLen++;
        sortSegments(logSegLen,d_values,numValues,omp);
      }
      
      template<typename key_t, typename value_t>
      void sort(key_t *const d_keys,
                value_t *const d_values,
                int numValues,
                Context *omp)
      
      {
        uint32_t logSegLen = 0;
        while (1<<logSegLen < numValues)
          logSegLen++;
        sortSegments(logSegLen,d_keys,d_values,numValues,omp);
      }
      
    }
  
    template<typename key_t>
    void sort(key_t *const d_values,
              size_t numValues,
              Context *omp)
    {
      assert(numValues < (1ull<<31));
      bitonic::sort(d_values,(int)numValues,omp);
    }

    template<typename key_t, typename value_t>
    void sort(key_t *const d_keys,
              value_t *const d_values,
              size_t numValues,
              Context *omp)
    {
      assert(numValues < (1ull<<31));
      bitonic::sort(d_keys,d_values,(int)numValues,omp);
    }
  }
}
