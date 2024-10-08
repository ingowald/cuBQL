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

#pragma once

#include "cuBQL/math/box.h"
#include <cuda_runtime_api.h>

namespace cuBQL {

  /*! struct used to control how exactly the builder is supposed to
      build the tree; in particular, at which threshold to make a
      leaf */
  struct BuildConfig {
    inline BuildConfig &enableSAH() { buildMethod = SAH; return *this; }
    inline BuildConfig &enableELH() { buildMethod = ELH; return *this; }
    typedef enum
      {
       /*! simple 'adaptive spatial median' strategy. When splitting a
         subtree, this first computes the centroid of each input
         primitive in that subtree, then computes the bounding box of
         those centroids, then creates a split plane along the widest
         dimension of that centroid boundig box, right through the
         middle */
       SPATIAL_MEDIAN=0,
       /*! use good old surface area heurstic. In theory that only
         makes sense for BVHes that are used for tracing rays
         (theoretic motivation is a bit wobbly for other sorts of
         queries), but it seems to help even for other queries. Much
         more expensive to build, though */
       SAH,
       /*! edge-length heuristic - experimental */
       ELH
    } BuildMethod;
    
    /*! what leaf size the builder is _allowed_ to make; no matter
        what input is specified, the builder may never produce leaves
        larger than this value */
    int maxAllowedLeafSize = 1<<15;

    /*! threshold below which the builder should make a leaf, no
        matter what the prims in the subtree look like. A value of 0
        means "leave it to the builder" */
    int makeLeafThreshold = 0;

    BuildMethod buildMethod = SPATIAL_MEDIAN;
  };

  /*! the most basic type of BVH where each BVH::Node is either a leaf
      (and contains Node::count primitives), or is a inner node (and
      points to a pair of child nodes). Node 0 is the root node; node
      1 is always unused (so all other node pairs start on n even
      index) */
  template<typename _scalar_t, int _numDims>
  struct BinaryBVH {
    using scalar_t = _scalar_t;
    enum { numDims = _numDims };
    using vec_t = cuBQL::vec_t<scalar_t,numDims>;
    using box_t = cuBQL::box_t<scalar_t,numDims>;

    struct CUBQL_ALIGN(16) Node {
      enum { count_bits = 16, offset_bits = 64-count_bits };
      
      box_t    bounds;

      struct Admin {
      /*! For inner nodes, this points into the nodes[] array, with
        left child at nodes.offset+0, and right child at
        nodes.offset+1. For leaf nodes, this points into the
        primIDs[] array, which first prim beign primIDs[offset],
        next one primIDs[offset+1], etc. */
        union {
          struct {
            uint64_t offset : offset_bits;
            /* number of primitives in this leaf, if a leaf; 0 for inner
               nodes. */
            uint64_t count  : count_bits;
          };
          // the same as a single int64, so we can read/write with a
          // single op
          uint64_t offsetAndCountBits;
        };
      };
      Admin admin;
    };

    enum { maxLeafSize=((1<<Node::count_bits)-1) };
    
    using node_t       = Node;
    node_t   *nodes    = 0;
    uint32_t  numNodes = 0;
    uint32_t *primIDs  = 0;
    uint32_t  numPrims = 0;
  };

  /*! a 'wide' BVH in which each node has a fixed number of
    `BVH_WIDTH` children (some of those children can be un-used) */
  template<typename _scalar_t, int _numDims, int BVH_WIDTH>
  struct WideBVH {
    using scalar_t = _scalar_t;
    enum { numDims = _numDims };
    using vec_t = cuBQL::vec_t<scalar_t,numDims>;
    using box_t = cuBQL::box_t<scalar_t,numDims>;

    /*! a n-wide node of this BVH; note that unlike BinaryBVH::Node
      this is not a "single" node, but actually N nodes merged
      together */
    struct CUBQL_ALIGN(16) Node {
      struct Child {
        box_t    bounds;
        struct {
          uint64_t valid  :  1;
          uint64_t offset : 45;
          uint64_t count  : 16;
        };
      } children[BVH_WIDTH];
    };

    Node     *nodes    = 0;
    //! number of (multi-)nodes on this WideBVH
    uint32_t  numNodes = 0;
    uint32_t *primIDs  = 0;
    uint32_t  numPrims = 0;
  };

  // ------------------------------------------------------------------
  /*! defines a 'memory resource' that can be used for allocating gpu
      memory; this allows the user to switch between using
      cudaMallocAsync (where available) vs regular cudaMalloc (where
      not), or to use their own memory pool, to use managed memory,
      etc... All memory allocations done during construction will use
      the memory resource passed to the respective build function. */
  struct GpuMemoryResource {
    virtual ~GpuMemoryResource() = default;
    virtual cudaError_t malloc(void** ptr, size_t size, cudaStream_t s) = 0;
    virtual cudaError_t free(void* ptr, cudaStream_t s) = 0;
  };

  struct ManagedMemMemoryResource : public GpuMemoryResource {
    cudaError_t malloc(void** ptr, size_t size, cudaStream_t s) override
    {
      cudaStreamSynchronize(s);
      return cudaMallocManaged(ptr,size);
    }
    cudaError_t free(void* ptr, cudaStream_t s) override
    {
      cudaStreamSynchronize(s);
      return cudaFree(ptr);
    }
  };

  /* by default let's use cuda malloc async, which is much better and
     faster than regular malloc; but that's available on cuda 11, so
     let's add a fall back for older cuda's, too */
#if CUDART_VERSION >= 11020
  struct AsyncGpuMemoryResource final : GpuMemoryResource {
    AsyncGpuMemoryResource()
    {
      cudaMemPool_t mempool;
      cudaDeviceGetDefaultMemPool(&mempool, 0);
      uint64_t threshold = UINT64_MAX;
      cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
    }
    cudaError_t malloc(void** ptr, size_t size, cudaStream_t s) override {
      return cudaMallocAsync(ptr, size, s);
    }
    cudaError_t free(void* ptr, cudaStream_t s) override {
      return cudaFreeAsync(ptr, s);
    }
  };

  inline GpuMemoryResource &defaultGpuMemResource() {
    static AsyncGpuMemoryResource memResource;
    return memResource;
  }
#else
  inline GpuMemoryResource &defaultGpuMemResource() {
    static ManagedMemMemoryResource memResource;
    return memResource;
  }
#endif

  // ------------------------------------------------------------------
  
  /*! Builds a BinaryBVH over a given set of primitive bounding boxes.

    The builder runs on the GPU; boxes[] must be a device-readable array
    (managed or device mem); bvh arrays will be allocated in device mem.

    Input primitives may be marked as "inactive/invalid" by using a
    bounding box whose lower/upper coordinates are inverted; such
    primitives will be ignored, and will thus neither be visited
    during traversal nor mess up the tree in any way, shape, or form.
  */
  template<typename T, int D>
  void gpuBuilder(BinaryBVH<T,D>   &bvh,
                  /*! array of bounding boxes to build BVH over, must
                      be in device memory */
                  const box_t<T,D> *boxes,
                  uint32_t          numBoxes,
                  BuildConfig       buildConfig,
                  cudaStream_t      s=0,
                  GpuMemoryResource &memResource=defaultGpuMemResource());
  
  /*! Builds a WideBVH over the given set of boxes (using the given
      stream), using a simple adaptive spatial median builder (ie,
      each subtree will be split by first computing the bounding box
      of all its contained primitives' spatial centers, then choosing
      a split plane that splits this centroid bounds in the center,
      along the widest dimension. Leaves will be created once the size
      of a subtree get to or below buildConfig.makeLeafThreshold.
  */
  template<typename /*scalar type*/T, int /*dims*/D, int /*branching factor*/N>
  void gpuBuilder(WideBVH<T,D,N> &bvh,
                  /*! array of bounding boxes to build BVH over, must
                      be in device memory */
                  const box_t<T,D>  *boxes,
                  uint32_t          numBoxes,
                  BuildConfig       buildConfig,
                  cudaStream_t      s=0,
                  GpuMemoryResource& memResource=defaultGpuMemResource());


  
  // ------------------------------------------------------------------
  /*! fast radix/morton builder for float3 data */
  // ------------------------------------------------------------------
  template<typename T, int D>
  void mortonBuilder(BinaryBVH<T,D>   &bvh,
                     const box_t<T,D> *boxes,
                     int                   numPrims,
                     BuildConfig           buildConfig,
                     cudaStream_t          s=0,
                     GpuMemoryResource    &memResource=defaultGpuMemResource());
  
  // ------------------------------------------------------------------
  
  /*! Frees the bvh.nodes[] and bvh.primIDs[] memory allocated when
      building the BVH.
  */
  template<typename T, int D>
  void free(BinaryBVH<T,D> &bvh,
            cudaStream_t      s=0,
            GpuMemoryResource& memResource=defaultGpuMemResource());

  /*! Frees the bvh.nodes[] and bvh.primIDs[] memory allocated when
      building the BVH.
  */
  template<typename T, int D, int N>
  void free(WideBVH<T,D,N> &bvh,
            cudaStream_t      s=0,
            GpuMemoryResource& memResource=defaultGpuMemResource());

  template<typename T, int D>
  using bvh_t = BinaryBVH<T,D>;

  // easy short-hand - though cubql also supports other types of bvhs,
  // scalars, etc, this will likely be the most commonly used one.
  using bvh3f = BinaryBVH<float,3>;
} // ::cuBQL

#ifdef __CUDACC__
# if CUBQL_GPU_BUILDER_IMPLEMENTATION
#  include "cuBQL/impl/gpu_builder.h"  
#  include "cuBQL/impl/sm_builder.h"  
#  include "cuBQL/impl/sah_builder.h"  
#  include "cuBQL/impl/elh_builder.h"  
#  include "cuBQL/impl/morton.h"  
#  include "cuBQL/impl/wide_gpu_builder.h"  
# endif
#endif




  
