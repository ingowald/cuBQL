// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/omp/refit.h"

namespace cuBQL {
  namespace omp {

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
    __global__
    void initState(BuildState      *buildState,
                   NodeState       *nodeStates,
                   TempNode<T,D> *nodes)
    {
      buildState->numNodes = 2;
      
      nodeStates[0]             = OPEN_BRANCH;
      nodes[0].openBranch.count = 0;
      nodes[0].openBranch.centBounds.set_empty();

      nodeStates[1]            = DONE_NODE;
      nodes[1].doneNode.offset = 0;
      nodes[1].doneNode.count  = 0;
    }

    template<typename T, int D>
    __global__ void initPrims(TempNode<T,D> *nodes,
                              PrimState       *primState,
                              const box_t<T,D>     *primBoxes,
                              uint32_t         numPrims)
    {
      const int primID = threadIdx.x+blockIdx.x*blockDim.x;
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
    __global__
    void selectSplits(BuildState    *buildState,
                      NodeState     *nodeStates,
                      TempNode<T,D> *nodes,
                      uint32_t       numNodes,
                      BuildConfig    buildConfig)
    {
#if 1
      __shared__ int l_newNodeOfs;
      if (threadIdx.x == 0)
        l_newNodeOfs = 0;
      __syncthreads();

      int *t_nodeOffsetToWrite = 0;
      int  t_localOffsetToAdd = 0;

      while (true) {
        const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
        if (nodeID >= numNodes)
          break;
        
        NodeState &nodeState = nodeStates[nodeID];
        if (nodeState == DONE_NODE)
          // this node was already closed before
          break;
        
        if (nodeState == OPEN_NODE) {
          // this node was open in the last pass, can close it.
          nodeState   = DONE_NODE;
          int offset  = nodes[nodeID].openNode.offset;
          auto &done  = nodes[nodeID].doneNode;
          done.count  = 0;
          done.offset = offset;
          break;
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
          
          // this will be epensive - could make this faster by block-reducing
          // open.offset = atomicAdd(&buildState->numNodes,2);
          t_nodeOffsetToWrite = (int*)&open.offset;
          t_localOffsetToAdd = atomicAdd(&l_newNodeOfs,2);
          nodeState = OPEN_NODE;
        }
        break;
      }
      __syncthreads();
      if (threadIdx.x == 0 && l_newNodeOfs > 0)
        l_newNodeOfs = atomicAdd(&buildState->numNodes,l_newNodeOfs);
      __syncthreads();
      if (t_nodeOffsetToWrite) {
        int openOffset = *t_nodeOffsetToWrite = l_newNodeOfs + t_localOffsetToAdd;
#pragma unroll
          for (int side=0;side<2;side++) {
            const int childID = openOffset+side;
            auto &child = nodes[childID].openBranch;
            child.centBounds.set_empty();
            child.count         = 0;
            nodeStates[childID] = OPEN_BRANCH;
          }
      }
#else
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
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
        
        // this will be epensive - could make this faster by block-reducing
        open.offset = atomicAdd(&buildState->numNodes,2);
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
#endif
    }

    template<typename T, int D>
    __global__
    void updatePrims(NodeState       *nodeStates,
                     TempNode<T,D> *nodes,
                     PrimState       *primStates,
                     const box_t<T,D>     *primBoxes,
                     int numPrims)
    {
      const int primID = threadIdx.x+blockIdx.x*blockDim.x;
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
    __global__
    void writePrimsAndLeafOffsets(TempNode<T,D> *nodes,
                                  uint32_t        *bvhItemList,
                                  PrimState       *primStates,
                                  int              numPrims)
    {
      const int offset = threadIdx.x+blockIdx.x*blockDim.x;
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
    __global__
    void writeNodes(typename BinaryBVH<T,D>::Node *finalNodes,
                    TempNode<T,D>  *tempNodes,
                    int        numNodes)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= numNodes) return;

      finalNodes[nodeID].admin.offset = tempNodes[nodeID].doneNode.offset;
      finalNodes[nodeID].admin.count  = tempNodes[nodeID].doneNode.count;
    }

    
    template<typename T, int D>
    void build(BinaryBVH<T,D>    &bvh,
               const box_t<T,D>  *boxes,
               int                numPrims,
               BuildConfig        buildConfig,
               cudaStream_t       s,
               GpuMemoryResource &memResource)
    {
      assert(sizeof(PrimState) == sizeof(uint64_t));
      
      // ==================================================================
      // do build on temp nodes
      // ==================================================================
      TempNode<T,D> *tempNodes = 0;
      NodeState     *nodeStates = 0;
      PrimState     *primStates = 0;
      BuildState    *buildState = 0;
      ctx->malloc(tempNodes,2*numPrims);
      ctx->malloc(nodeStates,2*numPrims);
      ctx->malloc(primStates,numPrims);
      ctx->malloc(buildState,1);
      initState<<<1,1,0,s>>>(buildState,
                             nodeStates,
                             tempNodes);
      initPrims<<<divRoundUp(numPrims,1024),1024,0,s>>>
        (tempNodes,
         primStates,boxes,numPrims);

      int numDone = 0;
      int numNodes;

      // ------------------------------------------------------------------      
      cudaEvent_t stateDownloadedEvent;
      CUBQL_CUDA_CALL(EventCreate(&stateDownloadedEvent));
      
      
      while (true) {
        CUBQL_CUDA_CALL(MemcpyAsync(&numNodes,&buildState->numNodes,
                                    sizeof(numNodes),cudaMemcpyDeviceToHost,s));
        CUBQL_CUDA_CALL(EventRecord(stateDownloadedEvent,s));
        CUBQL_CUDA_CALL(EventSynchronize(stateDownloadedEvent));
        if (numNodes == numDone)
          break;
        selectSplits<<<divRoundUp(numNodes,1024),1024,0,s>>>
          (buildState,
           nodeStates,tempNodes,numNodes,
           buildConfig);
        numDone = numNodes;

        updatePrims<<<divRoundUp(numPrims,1024),1024,0,s>>>
          (nodeStates,tempNodes,
           primStates,boxes,numPrims);
      }
      CUBQL_CUDA_CALL(EventDestroy(stateDownloadedEvent));
      // ==================================================================
      // sort {item,nodeID} list
      // ==================================================================
      
      // set up sorting of prims
      uint8_t   *d_temp_storage     = NULL;
      size_t     temp_storage_bytes = 0;
      PrimState *sortedPrimStates   = 0;
      ctx->malloc(sortedPrimStates,numPrims);
      auto rc =
        cub::DeviceRadixSort::SortKeys((void*&)d_temp_storage, temp_storage_bytes,
                                     (uint64_t*)primStates,
                                     (uint64_t*)sortedPrimStates,
                                     numPrims,32,64,s);
      ctx->malloc(d_temp_storage,temp_storage_bytes);
      rc =
        cub::DeviceRadixSort::SortKeys((void*&)d_temp_storage, temp_storage_bytes,
                                       (uint64_t*)primStates,
                                       (uint64_t*)sortedPrimStates,
                                       numPrims,32,64,s);
      rc = rc;
      ctx->free(d_temp_storage);
      // ==================================================================
      // allocate and write BVH item list, and write offsets of leaf nodes
      // ==================================================================

      bvh.numPrims = numPrims;
      ctx->malloc(bvh.primIDs,numPrims);
      writePrimsAndLeafOffsets<<<divRoundUp(numPrims,1024),1024,0,s>>>
        (tempNodes,bvh.primIDs,sortedPrimStates,numPrims);

      // ==================================================================
      // allocate and write final nodes
      // ==================================================================
      bvh.numNodes = numNodes;
      ctx->malloc(bvh.nodes,numNodes);
      writeNodes<<<divRoundUp(numNodes,1024),1024,0,s>>>
        (bvh.nodes,tempNodes,numNodes);
      ctx->free(sortedPrimStates);
      ctx->free(tempNodes);
      ctx->free(nodeStates);
      ctx->free(primStates);
      ctx->free(buildState);

      refit(bvh);
    }
    
  }
}

