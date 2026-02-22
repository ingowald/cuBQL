// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/cuda/builder_common.h"
#include "cuBQL/builder/cuda/refit.h"

namespace cuBQL {
  namespace cuda {

    // ------------------------------------------------------------------
    // INTERFACE
    // ------------------------------------------------------------------
    template<
      typename T,
      int D,
      typename AggregateNodeData
      // ,
      // typename AggregateFct
      >
    void refit_aggregate(BinaryBVH<T,D> bvh,
                         AggregateNodeData *d_aggregateNodeData,
                         void (*aggregateFct)(bvh3f,
                                              AggregateNodeData[],
                                              int),
                         // const AggregateFct &aggregateFct,
                         cudaStream_t       s
                         =0,
                         GpuMemoryResource &memResource
                         =defaultGpuMemResource());
    
    template<typename T, int D,
             typename AggregateNodeData
             // ,
             // typename AggregateFct
             >
    __global__
    void refit_aggregate_run(BinaryBVH<T,D> bvh,
                             AggregateNodeData *aggregateNodeData,
                         void (*aggregateFct)(bvh3f,
                                              AggregateNodeData[],
                                              int),
                             // const AggregateFct &aggregateFct,
                             uint32_t *refitData)
    {
      int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID == 1 || nodeID >= bvh.numNodes) return;
      
      typename BinaryBVH<T,D>::Node *node = &bvh.nodes[nodeID];
      if (node->admin.count == 0)
        // this is a inner node - exit
        return;
      
      // box_t<T,D> bounds; bounds.set_empty();
      // for (int i=0;i<node->admin.count;i++) {
      //   const box_t<T,D> primBox = boxes[bvh.primIDs[node->admin.offset+i]];
      //   bounds.lower = min(bounds.lower,primBox.lower);
      //   bounds.upper = max(bounds.upper,primBox.upper);
      // }
      
      int parentID = (refitData[nodeID] >> 1);
      while (true) {
        aggregateFct(bvh,aggregateNodeData,nodeID);
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
        
        // typename BinaryBVH<T,D>::Node l = bvh.nodes[node->admin.offset+0];
        // typename BinaryBVH<T,D>::Node r = bvh.nodes[node->admin.offset+1];
        // bounds.lower = min(l.bounds.lower,r.bounds.lower);
        // bounds.upper = max(l.bounds.upper,r.bounds.upper);
      }
    }

    
    
    // ------------------------------------------------------------------
    // IMPLEMENTATION
    // ------------------------------------------------------------------
    template<
      typename T,
      int D,
      typename AggregateNodeData
      // ,
      // typename AggregateFct
      >
    void refit_aggregate(BinaryBVH<T,D> bvh,
                         AggregateNodeData *d_aggregateNodeData,
                         // const AggregateFct &aggregateFct,
                         // __device__
                         void (*aggregateFct)(bvh3f,
                                              AggregateNodeData[],
                                              int),
                         cudaStream_t       s,
                         GpuMemoryResource &memResource)
    {
      int numNodes = bvh.numNodes;
      
      uint32_t *refitData = 0;
      memResource.malloc((void**)&refitData,numNodes*sizeof(*refitData),s);
  CUBQL_CUDA_SYNC_CHECK();
      refit_init<T,D><<<divRoundUp(numNodes,1024),1024,0,s>>>
        (bvh.nodes,refitData,numNodes);
  CUBQL_CUDA_SYNC_CHECK();
      refit_aggregate_run<<<divRoundUp(numNodes,32),32,0,s>>>
        (bvh,d_aggregateNodeData,aggregateFct,refitData);
  CUBQL_CUDA_SYNC_CHECK();
      memResource.free((void*)refitData,s);
  CUBQL_CUDA_SYNC_CHECK();
      // we're not syncing here - let APP do that
    }
  }
}
