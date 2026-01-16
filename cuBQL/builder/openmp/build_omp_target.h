// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <omp.h>
#include <atomic>

namespace cuBQL {
  namespace omp {

    template<typename T, typename count_t>
    inline void _ALLOC(T *&ptr, count_t count, int gpuID)
    { ptr = (T*)omp_target_alloc(count*sizeof(T),gpuID); }
    
    template<typename T>
    inline void _FREE(T *&ptr, int gpuID)
    { omp_target_free(ptr,gpuID); ptr = 0; }
    
    typedef enum : int8_t { OPEN_BRANCH, OPEN_NODE, DONE_NODE } NodeState;
    
    // ==================================================================
    // atomicbox
    // ==================================================================
    template<typename box_t>
    struct CUBQL_ALIGN(8) AtomicBox {
      inline bool is_empty() const { return lower[0] > upper[0]; }
      inline void  set_empty();
      // set_empty, in owl::common-style naming
      inline void  clear() { set_empty(); }
      inline float get_center(int dim) const;
      inline box_t make_box() const;

      inline float get_lower(int dim) const {
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
      inline float get_upper(int dim) const {
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
      
      inline static int32_t encode(float f);
      inline static float   decode(int32_t bits);
    };

#ifdef __cplusplus > 202302L
    inline void atomicMin(int32_t *ptr, int32_t value)
    { if (value < *ptr) ((std::atomic<int> *)ptr)->fetch_min(value, std::memory_order::seq_cst); }
    inline void atomicMax(int32_t *ptr, int32_t value)
    { if (value > *ptr) ((std::atomic<int> *)ptr)->fetch_max(value, std::memory_order::seq_cst); }
#else
    inline void atomicMin(int32_t *ptr, int32_t value)
    { 
      int current = *(volatile int *)addr;
      while (current > value) {
        bool wasChanged
          = ((std::atomic<int>*)addr)->compare_exchange_weak((int&)current,(int&)value);
        if (wasChanged) break;
      }
    }
    
    inline void atomicMax(int32_t *ptr, int32_t value)
    { 
      int current = *(volatile int *)addr;
      while (current < value) {
        bool wasChanged
          = ((std::atomic<int>*)addr)->compare_exchange_weak((int&)current,(int&)value);
        if (wasChanged) break;
      }
    }
    
#endif
    
    template<typename box_t>
    inline float AtomicBox<box_t>::get_center(int dim) const
    {
      return 0.5f*(get_lower(dim)+get_upper(dim));
      // return 0.5f*(decode(lower[dim])+decode(upper[dim]));
    }

    template<typename box_t>
    inline box_t AtomicBox<box_t>::make_box() const
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
    inline int32_t AtomicBox<box_t>::encode(float f)
    {
      const int32_t sign = 0x80000000;
      int32_t bits = __float_as_int(f);
      if (bits & sign) bits ^= 0x7fffffff;
      return bits;
    }
      
    template<typename box_t>
    inline float AtomicBox<box_t>::decode(int32_t bits)
    {
      const int32_t sign = 0x80000000;
      if (bits & sign) bits ^= 0x7fffffff;
      return __int_as_float(bits);
    }
    
    template<typename box_t>
    inline void AtomicBox<box_t>::set_empty()
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
    inline void atomic_grow(AtomicBox<box_t> &abox, const box_t &other)
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
    inline void atomic_grow(AtomicBox<box_t> &abox, const AtomicBox<box_t> &other)
    {
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        const int32_t enc_lower = other.lower[d];
        const int32_t enc_upper = other.upper[d];
        if (enc_lower < abox.lower[d]) atomicMin(&abox.lower[d],enc_lower);
        if (enc_upper > abox.upper[d]) atomicMax(&abox.upper[d],enc_upper);
      }
    }
    

    // ==================================================================
    // internal states
    // ==================================================================
    struct BuildState {
      uint32_t  numNodes;
    };
    
    struct PrimState {
      union {
        /* careful with this order - this is intentionally chosen such
           that all item with nodeID==-1 will end up at the end of the
           list; and all others will be sorted by nodeID */
        struct {
          uint64_t primID:31; //!< prim we're talking about
          uint64_t done  : 1;
          uint64_t nodeID:32; //!< node the given prim is (currently) in.
        };
        uint64_t bits;
      };
    };

    template<typename T, int D>
    struct CUBQL_ALIGN(16) TempNode {
      using box_t = cuBQL::box_t<T,D>;
      union {
        struct {
          AtomicBox<box_t> centBounds;
          uint32_t         count;
          uint32_t         unused;
        } openBranch;
        struct {
          uint32_t offset;
          int      dim;
          uint32_t tieBreaker;
          float    pos;
        } openNode;
        struct {
          uint32_t offset;
          uint32_t count;
          uint32_t unused[2];
        } doneNode;
      };
    };

    template<typename T, int D>
    void initState(int tid,
                   BuildState      *buildState,
                   NodeState       *nodeStates,
                   TempNode<T,D> *nodes)
    {
      if (tid >= 1) return;
      
      buildState->numNodes = 2;
      
      nodeStates[0]             = OPEN_BRANCH;
      nodes[0].openBranch.count = 0;
      nodes[0].openBranch.centBounds.set_empty();

      nodeStates[1]            = DONE_NODE;
      nodes[1].doneNode.offset = 0;
      nodes[1].doneNode.count  = 0;
    }

    
    /*! openmp based builder with #pragma omp target directives. */
    template<typename T, int D>
    inline
    void build_omp_target_impl(BinaryBVH<T,D>   &bvh,
                               /*! array of bounding boxes to build BVH over,
                                 must be in target device memory (ie, must be
                                 accessible in the device(gpuID) that the
                                 'gpuID' parameter refers to */
                               const box_t<T,D> *d_boxes,
                               uint32_t          numPrims,
                               BuildConfig       buildConfig,
                               int               gpuID)
    {
      TempNode<T,D> *tempNodes = 0;
      NodeState     *nodeStates = 0;
      PrimState     *primStates = 0;
      BuildState    *buildState = 0;
      _ALLOC(tempNodes,2*numPrims,gpuID);
      _ALLOC(nodeStates,2*numPrims,gpuID);
      _ALLOC(primStates,numPrims,gpuID);
      _ALLOC(buildState,1,gpuID);
#pragma omp target device(gpuID) \
  is_device_ptr(buildState)      \
  is_device_ptr(nodeStates)      \
  is_device_ptr(tempNodes)
#pragma omp teams distribute parallel for
      for (int i=0;i<1;i++)
        initState(i,
                  buildState,
                  nodeStates,
                  tempNodes);
    
    }
  }
    
  
  /*! openmp based builder with #pragma omp target directives. */
  template<typename T, int D>
  inline
  void build_omp_target(BinaryBVH<T,D>   &bvh,
                        /*! array of bounding boxes to build BVH over,
                          must be in target device memory (ie, must be
                          accessible in the device(gpuID) that the
                          'gpuID' parameter refers to */
                        const box_t<T,D> *d_boxes,
                        uint32_t          numBoxes,
                        BuildConfig       buildConfig,
                        int               gpuID)
  {
    omp::build_omp_target_impl(bvh,d_boxes,numBoxes,buildConfig,gpuID);
  }

} // ::cuBQL


