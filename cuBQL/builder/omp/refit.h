// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/omp/common.h"

namespace cuBQL {
  namespace omp {
    
    struct Context {
      Context(int gpuID);
      int gpuID;
      int hostID;
    };
    struct Kernel {
      inline int workIdx() const { return _workIdx; }
      int _workIdx;
    };

    template<typename T, int D> inline
    void refit_init(Kernel kernel,
                    const typename BinaryBVH<T,D>::Node *nodes,
                    uint32_t              *refitData,
                    int numNodes)
    {
      const int nodeID = kernel.workIdx();
      if (nodeID == 1 || nodeID >= numNodes) return;
      if (nodeID < 2)
        refitData[0] = 0;
      const auto &node = nodes[nodeID];
      if (node.admin.count) return;

      refitData[node.admin.offset+0] = nodeID << 1;
      refitData[node.admin.offset+1] = nodeID << 1;
    }

    template<typename T, int D> inline 
    void refit_run(Kernel kernel,
                   BinaryBVH<T,D> bvh,
                   uint32_t *refitData,
                   const box_t<T,D> *boxes)
    {
      int nodeID = kernel.workIdx();
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
        // __threadfence();
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
               Context *ctx)
    {
      int numNodes = bvh.numNodes;
      uint32_t *refitData
        = (uint32_t*)ctx->malloc(numNodes*sizeof(int));
      
# pragma omp target device(context->gpuID)
# pragma omp teams distribute parallel for
      for (int i=0;i<numNodes;i++)
        refit_init<T,D>(Kernel{i},bvh.nodes,refitData,bvh.numNodes);
# pragma omp target device(context->gpuID)
# pragma omp teams distribute parallel for
      for (int i=0;i<numNodes;i++)
        refit_run(Kernel{i},bvh,refitData,boxes);
      ctx->free((void*)refitData);
    }
    
  }
}

