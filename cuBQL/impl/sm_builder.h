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

#include "cuBQL/impl/builder_common.h"

namespace cuBQL {
  namespace gpuBuilder_impl {

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
#pragma unroll
        for (int d=0;d<D;d++) {
          float width = in.centBounds.get_upper(d) - in.centBounds.get_lower(d);
          if (width <= widestWidth)
            continue;
          widestWidth = width;
          widestDim   = d;
        }
      
        auto &open = nodes[nodeID].openNode;
        if (widestDim >= 0) {
          open.pos = in.centBounds.get_center(widestDim);
          if (open.pos == in.centBounds.get_lower(widestDim) ||
              open.pos == in.centBounds.get_upper(widestDim))
            widestDim = -1;
        }
        open.dim = widestDim;
        
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

      auto &me = primStates[primID];
      if (me.done) return;
      
      auto ns = nodeStates[me.nodeID];
      if (ns == DONE_NODE) {
        // node became a leaf, we're done.
        me.done = true;
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
      me.nodeID = newNodeID;
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
      _ALLOC(tempNodes,2*numPrims,s,memResource);
      _ALLOC(nodeStates,2*numPrims,s,memResource);
      _ALLOC(primStates,numPrims,s,memResource);
      _ALLOC(buildState,1,s,memResource);
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
      _ALLOC(sortedPrimStates,numPrims,s,memResource);
      cub::DeviceRadixSort::SortKeys((void*&)d_temp_storage, temp_storage_bytes,
                                     (uint64_t*)primStates,
                                     (uint64_t*)sortedPrimStates,
                                     numPrims,32,64,s);
      _ALLOC(d_temp_storage,temp_storage_bytes,s,memResource);
      cub::DeviceRadixSort::SortKeys((void*&)d_temp_storage, temp_storage_bytes,
                                     (uint64_t*)primStates,
                                     (uint64_t*)sortedPrimStates,
                                     numPrims,32,64,s);
      _FREE(d_temp_storage,s,memResource);
      // ==================================================================
      // allocate and write BVH item list, and write offsets of leaf nodes
      // ==================================================================

      bvh.numPrims = numPrims;
      _ALLOC(bvh.primIDs,numPrims,s,memResource);
      writePrimsAndLeafOffsets<<<divRoundUp(numPrims,1024),1024,0,s>>>
        (tempNodes,bvh.primIDs,sortedPrimStates,numPrims);

      // ==================================================================
      // allocate and write final nodes
      // ==================================================================
      bvh.numNodes = numNodes;
      _ALLOC(bvh.nodes,numNodes,s,memResource);
      writeNodes<<<divRoundUp(numNodes,1024),1024,0,s>>>
        (bvh.nodes,tempNodes,numNodes);
      _FREE(sortedPrimStates,s,memResource);
      _FREE(tempNodes,s,memResource);
      _FREE(nodeStates,s,memResource);
      _FREE(primStates,s,memResource);
      _FREE(buildState,s,memResource);
    }

    template<typename T, int D>
    __global__ void
    refit_init(const typename BinaryBVH<T,D>::Node *nodes,
               uint32_t              *refitData,
               int numNodes)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID == 1 || nodeID >= numNodes) return;
      if (nodeID < 2)
        refitData[0] = 0;
      const auto &node = nodes[nodeID];
      if (node.admin.count) return;

      refitData[node.admin.offset+0] = nodeID << 1;
      refitData[node.admin.offset+1] = nodeID << 1;
    }
    
    template<typename T, int D>
    __global__
    void refit_run(BinaryBVH<T,D> bvh,
                   uint32_t *refitData,
                   const box_t<T,D> *boxes)
    {
      int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID == 1 || nodeID >= bvh.numNodes) return;
      
      typename BinaryBVH<T,D>::Node *node = &bvh.nodes[nodeID];
      if (node->admin.count == 0)
        // this is a inner node - exit
        return;

      box_t<T,D> bounds; bounds.set_empty();
      for (int i=0;i<node->admin.count;i++) {
        const box_t<T,D> primBox = boxes[bvh.primIDs[node->admin.offset+i]];
        bounds.lower = min(bounds.lower,primBox.lower);
        bounds.upper = max(bounds.upper,primBox.upper);
      }

      int parentID = (refitData[nodeID] >> 1);
      while (true) {
        node->bounds = bounds;
        __threadfence();
        if (node == bvh.nodes)
          break;

        uint32_t refitBits = atomicAdd(&refitData[parentID],1u);
        if ((refitBits & 1) == 0)
          // we're the first one - let other one do it
          break;

        nodeID   = parentID;
        node     = &bvh.nodes[parentID];
        parentID = (refitBits >> 1);
        
        typename BinaryBVH<T,D>::Node l = bvh.nodes[node->admin.offset+0];
        typename BinaryBVH<T,D>::Node r = bvh.nodes[node->admin.offset+1];
        bounds.lower = min(l.bounds.lower,r.bounds.lower);
        bounds.upper = max(l.bounds.upper,r.bounds.upper);
      }
    }

    template<typename T, int D>
    void refit(BinaryBVH<T,D>    &bvh,
               const box_t<T,D>  *boxes,
               cudaStream_t       s=0,
               GpuMemoryResource &memResource=defaultGpuMemResource())
    {
      uint32_t *refitData = 0;
      memResource.malloc((void**)&refitData,bvh.numNodes*sizeof(int),s);
      
      int numNodes = bvh.numNodes;
      refit_init<T,D><<<divRoundUp(numNodes,1024),1024,0,s>>>
        (bvh.nodes,refitData,bvh.numNodes);
      refit_run<<<divRoundUp(numNodes,32),32,0,s>>>
        (bvh,refitData,boxes);
      memResource.free((void*)refitData,s);
      // we're not syncing here - let APP do that
    }
    
  } // ::cuBQL::gpuBuilder_impl
} // ::cuBQL

