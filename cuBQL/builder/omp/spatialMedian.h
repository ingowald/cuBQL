// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/omp/refit.h"
#include "cuBQL/builder/omp/sort.h"

namespace cuBQL {
  namespace omp {

    template<typename box_t>
    struct AtomicBox : public box_t {
    };

    template<typename T>
    void atomic_min(T *ptr, T v);
    template<typename T>
    void atomic_max(T *ptr, T v);
    
    /*! iw - note: this implementation of atomic min/max via atomic
        compare-exchange (CAS); which is cetainly not optimal on any
        sort of modern GPU - but it works in any C++-21 compliant
        compiler, so it's what we do for now */
    void atomic_min(float *ptr, float value)
    {
      float current = *(volatile float *)ptr;
      while (current > value) {
        bool wasChanged
          = ((std::atomic<int>*)ptr)
          ->compare_exchange_weak((int&)current,(int&)value);
        if (wasChanged) break;
      }
    }
    
    /*! iw - note: this implementation of atomic min/max via atomic
        compare-exchange (CAS); which is cetainly not optimal on any
        sort of modern GPU - but it works in any C++-21 compliant
        compiler, so it's what we do for now */
    void atomic_max(float *ptr, float value)
    {
      float current = *(volatile float *)ptr;
      while (current > value) {
        bool wasChanged
          = ((std::atomic<int>*)ptr)
          ->compare_exchange_weak((int&)current,(int&)value);
        if (wasChanged) break;
      }
    }
    
    /*! iw - note: this implementation of atomic min/max via atomic
        compare-exchange (CAS); which is cetainly not optimal on any
        sort of modern GPU - but it works in any C++-21 compliant
        compiler, so it's what we do for now */
    void atomic_min(double *ptr, double value)
    {
      double current = *(volatile double *)ptr;
      while (current > value) {
        bool wasChanged
          = ((std::atomic<long long int>*)ptr)
          ->compare_exchange_weak((long long int&)current,
                                  (long long int&)value);
        if (wasChanged) break;
      }
    }
    
    /*! iw - note: this implementation of atomic min/max via atomic
        compare-exchange (CAS); which is cetainly not optimal on any
        sort of modern GPU - but it works in any C++-21 compliant
        compiler, so it's what we do for now */
    void atomic_max(double *ptr, double value)
    {
      double current = *(volatile double *)ptr;
      while (current > value) {
        bool wasChanged
          = ((std::atomic<long long int>*)ptr)
          ->compare_exchange_weak((long long int&)current,
                                  (long long int&)value);
        if (wasChanged) break;
      }
    }
    
    template<typename T, int D>
    void v_atomic_min(vec_t<T,D> *ptr, vec_t<T,D> v);
    template<typename T, int D>
    void v_atomic_max(vec_t<T,D> *ptr, vec_t<T,D> v);
    

    template<typename T>
    void v_atomic_min(vec_t<T,2> *ptr, vec_t<T,2> v)
    {
      atomic_min(&ptr->x,v.x); 
      atomic_min(&ptr->y,v.y);
    }
    
    template<typename T>
    void v_atomic_min(vec_t<T,3> *ptr, vec_t<T,3> v)
    {
      atomic_min(&ptr->x,v.x); 
      atomic_min(&ptr->y,v.y);
      atomic_min(&ptr->z,v.z);
    }
    
    template<typename T>
    void v_atomic_min(vec_t<T,4> *ptr, vec_t<T,4> v)
    {
      atomic_min(&ptr->x,v.x); 
      atomic_min(&ptr->y,v.y);
      atomic_min(&ptr->z,v.z);
      atomic_min(&ptr->w,v.w);
    }

    template<typename T>
    void v_atomic_max(vec_t<T,2> *ptr, vec_t<T,2> v)
    {
      atomic_max(&ptr->x,v.x); 
      atomic_max(&ptr->y,v.y);
    }
    
    template<typename T>
    void v_atomic_max(vec_t<T,3> *ptr, vec_t<T,3> v)
    {
      atomic_max(&ptr->x,v.x); 
      atomic_max(&ptr->y,v.y);
      atomic_max(&ptr->z,v.z);
    }
    
    template<typename T>
    void v_atomic_max(vec_t<T,4> *ptr, vec_t<T,4> v)
    {
      atomic_max(&ptr->x,v.x); 
      atomic_max(&ptr->y,v.y);
      atomic_max(&ptr->z,v.z);
      atomic_max(&ptr->w,v.w);
    }
    
    template<typename box_t>
    void atomic_grow(AtomicBox<box_t> &ab, typename box_t::vec_t P)
    {
      v_atomic_min(&ab.lower,P);
      v_atomic_max(&ab.upper,P);
    }
    
    template<typename box_t>
    void atomic_grow(AtomicBox<box_t> &ab, box_t B)
    {
      v_atomic_min(&ab.lower,B.lower);
      v_atomic_max(&ab.upper,B.upper);
    }
    
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

    typedef enum : int8_t { OPEN_BRANCH, OPEN_NODE, DONE_NODE } NodeState;
    
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
    void initState(uint32_t      *pNumNodes,
                   NodeState     *nodeStates,
                   TempNode<T,D> *nodes)
    {
      *pNumNodes = 2;
      
      nodeStates[0]             = OPEN_BRANCH;
      nodes[0].openBranch.count = 0;
      nodes[0].openBranch.centBounds.set_empty();

      nodeStates[1]            = DONE_NODE;
      nodes[1].doneNode.offset = 0;
      nodes[1].doneNode.count  = 0;
    }

    template<typename T, int D>
    void initPrims(Kernel kernel,
                   TempNode<T,D>    *nodes,
                   PrimState        *primState,
                   const box_t<T,D> *primBoxes,
                   uint32_t          numPrims)
    {
      const int primID = kernel.workIdx();
      if (primID >= numPrims) return;
      
      auto &me = primState[primID];
      me.primID = primID;
                                                    
      const box_t<T,D> box = primBoxes[primID];
      if (box.get_lower(0) <= box.get_upper(0)) {
        me.nodeID = 0;
        me.done   = false;
        // this could be made faster by block-reducing ...
        atomicAdd(&nodes[0].openBranch.count,1);
        atomic_grow(nodes[0].openBranch.centBounds,box.center());//centerOf(box));
      } else {
        me.nodeID = (uint32_t)-1;
        me.done   = true;
      }
    }

    template<typename T, int D>
    void selectSplits(Kernel kernel,
                      uint32_t      *pNumNodes,
                      NodeState     *nodeStates,
                      TempNode<T,D> *nodes,
                      uint32_t       numNodes,
                      BuildConfig    buildConfig)
    {
      const int nodeID = kernel.workIdx();//threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= numNodes) return;

      NodeState &nodeState = nodeStates[nodeID];
      if (nodeState == DONE_NODE)
        // this node was already closed before
        return;
      
      if (nodeState == OPEN_NODE) {
        // this node was open in the last pass, can close it.
        nodeState   = DONE_NODE;
        int offset  = nodes[nodeID].openNode.offset;
        auto &done  = nodes[nodeID].doneNode;
        done.count  = 0;
        done.offset = offset;
        return;
      }
      
      auto in = nodes[nodeID].openBranch;
      if (in.count <= buildConfig.makeLeafThreshold) {
        auto &done  = nodes[nodeID].doneNode;
        done.count  = in.count;
        // set this to max-value, so the prims can later do atomicMin
        // with their position ion the leaf list; this value is
        // greater than any prim position.
        done.offset = (uint32_t)-1;
        nodeState   = DONE_NODE;
      } else {
        float widestWidth = 0.f;
        int   widestDim   = -1;
        float widestLo, widestHi, widestCtr;
#pragma unroll
        for (int d=0;d<D;d++) {
          float lo = in.centBounds.get_lower(d);
          float hi = in.centBounds.get_upper(d);
          float width = hi - lo;
          if (width <= widestWidth)
            continue;
          float ctr = 0.5f*(hi+lo);
          
          widestWidth = width;
          widestDim   = d;
          widestLo = lo;
          widestHi = hi;
          widestCtr = ctr;
        }
      
        auto &open = nodes[nodeID].openNode;
        if (widestDim >= 0) {
          open.pos = widestCtr;
        }
        open.dim
          = (widestDim < 0 || widestCtr == widestLo || widestCtr == widestHi)
          ? -1
          : widestDim;
        
        open.offset = atomicAdd(pNumNodes,2);
#pragma unroll
        for (int side=0;side<2;side++) {
          const int childID = open.offset+side;
          auto &child = nodes[childID].openBranch;
          child.centBounds.set_empty();
          child.count         = 0;
          nodeStates[childID] = OPEN_BRANCH;
        }
        nodeState = OPEN_NODE;
      }
    }

    template<typename T, int D>
    void updatePrims(Kernel kernel,
                     NodeState       *nodeStates,
                     TempNode<T,D> *nodes,
                     PrimState       *primStates,
                     const box_t<T,D>     *primBoxes,
                     int numPrims)
    {
      const int primID = kernel.workIdx();
      if (primID >= numPrims) return;

      const auto me = primStates[primID];
      if (me.done) return;
      
      const auto ns = nodeStates[me.nodeID];
      if (ns == DONE_NODE) {
        // node became a leaf, we're done.
        primStates[primID].done = true;
        return;
      }
      
      auto &split = nodes[me.nodeID].openNode;
      const box_t<T,D> primBox = primBoxes[me.primID];
      int side = 0;
      if (split.dim == -1) {
        // could block-reduce this, but will likely not happen often, anyway
        side = (atomicAdd(&split.tieBreaker,1) & 1);
      } else {
        const float center = 0.5f*(primBox.get_lower(split.dim)+
                                   primBox.get_upper(split.dim));
        side = (center >= split.pos);
      }
      int newNodeID = split.offset+side;
      auto &myBranch = nodes[newNodeID].openBranch;
      atomicAdd(&myBranch.count,1);
      atomic_grow(myBranch.centBounds,primBox.center());
      primStates[primID].nodeID = newNodeID;
    }
    
    /* given a sorted list of {nodeID,primID} pairs, this kernel does
       two things: a) it extracts the 'primID's and puts them into the
       bvh's primIDs[] array; and b) it writes, for each leaf nod ein
       the nodes[] array, the node.offset value to point to the first
       of this nodes' items in that bvh.primIDs[] list. */
    template<typename T, int D>
    void writePrimsAndLeafOffsets(Kernel kernel,
                                  TempNode<T,D> *nodes,
                                  uint32_t        *bvhItemList,
                                  PrimState       *primStates,
                                  int              numPrims)
    {
      const int offset = kernel.workIdx();//threadIdx.x+blockIdx.x*blockDim.x;
      if (offset >= numPrims) return;

      auto &ps = primStates[offset];
      bvhItemList[offset] = ps.primID;
      
      if ((int)ps.nodeID < 0)
        /* invalid prim, just skip here */
        return;
      auto &node = nodes[ps.nodeID];
      atomicMin(&node.doneNode.offset,offset);
    }

    /* writes main phase's temp nodes into final bvh.nodes[]
       layout. actual bounds of that will NOT yet bewritten */
    template<typename T, int D>
    void writeNodes(Kernel kernel,
                    typename BinaryBVH<T,D>::Node *finalNodes,
                    TempNode<T,D>  *tempNodes,
                    int        numNodes)
    {
      const int nodeID = kernel.workIdx();
      if (nodeID >= numNodes) return;

      finalNodes[nodeID].admin.offset = tempNodes[nodeID].doneNode.offset;
      finalNodes[nodeID].admin.count  = tempNodes[nodeID].doneNode.count;
    }

    
    template<typename T, int D>
    void spatialMedian(BinaryBVH<T,D> &bvh,
                       /*! DEVICE array of boxes */
                       const box_t<T,D> *boxes,
                       uint32_t          numPrims,
                       BuildConfig       buildConfig,
                       Context          *ctx)
    {
      assert(sizeof(PrimState) == sizeof(uint64_t));
      
      // ==================================================================
      // do build on temp nodes
      // ==================================================================
      TempNode<T,D> *tempNodes = 0;
      NodeState     *nodeStates = 0;
      PrimState     *primStates = 0;
      uint32_t      *d_numNodes = 0;
      ctx->alloc(tempNodes,2*numPrims);
      ctx->alloc(nodeStates,2*numPrims);
      ctx->alloc(primStates,numPrims);
      ctx->alloc(d_numNodes,1);
#pragma omp target device(ctx->gpuID)
#pragma omp teams distribute parallel for
      for (int tid=0;tid<1;tid++)
        initState(d_numNodes,
                  nodeStates,
                  tempNodes);
#pragma omp target device(ctx->gpuID)
#pragma omp teams distribute parallel for
      for (int tid=0;tid<numPrims;tid++)
        initPrims(Kernel{tid},tempNodes,
                  primStates,boxes,numPrims);

      int numDone = 0;
      uint32_t numNodes;

      // ------------------------------------------------------------------      
      while (true) {
        ctx->download(numNodes,d_numNodes);
        if (numNodes == numDone)
          break;
#pragma omp target device(ctx->gpuID)
#pragma omp teams distribute parallel for
        for (int tid=0;tid<numNodes;tid++)
          selectSplits(Kernel{tid},
                       d_numNodes,
                       nodeStates,tempNodes,numNodes,
                       buildConfig);
        numDone = numNodes;

#pragma omp target device(ctx->gpuID)
#pragma omp teams distribute parallel for
        for (int tid=0;tid<numPrims;tid++)
          updatePrims(Kernel{tid},
                      nodeStates,tempNodes,
                      primStates,boxes,numPrims);
      }
      // ==================================================================
      // sort {item,nodeID} list
      // ==================================================================
      
      ::omp::omp_target_sort((uint64_t*)primStates,numPrims,ctx->gpuID);
      
      // ==================================================================
      // allocate and write BVH item list, and write offsets of leaf nodes
      // ==================================================================

      bvh.numPrims = numPrims;
      ctx->alloc(bvh.primIDs,numPrims);
#pragma omp target device(ctx->gpuID)
#pragma omp teams distribute parallel for
      for (int tid=0;tid<numPrims;tid++)
        writePrimsAndLeafOffsets(Kernel{tid},
                                 tempNodes,
                                 bvh.primIDs,primStates,numPrims);
      
      // ==================================================================
      // allocate and write final nodes
      // ==================================================================
      bvh.numNodes = numNodes;
      ctx->alloc(bvh.nodes,numNodes);
#pragma omp target device(ctx->gpuID)
#pragma omp teams distribute parallel for
      for (int tid=0;tid<numNodes;tid++)
        writeNodes(Kernel{tid},bvh.nodes,tempNodes,numNodes);
      ctx->free(tempNodes);
      ctx->free(nodeStates);
      ctx->free(primStates);
      ctx->free(d_numNodes);

      cuBQL::omp::refit(bvh,boxes,ctx);
    }
    
  }
  
  template<typename T, int D>
  void build_omp_target(BinaryBVH<T,D> &bvh,
                        const box_t<T,D> *d_boxes,
                        uint32_t          numBoxes,
                        BuildConfig       buildConfig,
                        int               gpuID)
  {
    omp::Context ctx(gpuID);
    omp::spatialMedian(bvh,d_boxes,numBoxes,buildConfig,&ctx);
  }
}

