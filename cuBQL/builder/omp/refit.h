// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/omp/common.h"
#include "cuBQL/builder/omp/AtomicBox.h"

namespace cuBQL {
  namespace omp {
    
    template<typename T, int D> inline
    void refit_init_x(Kernel kernel,
                    typename BinaryBVH<T,D>::Node *nodes,
                    uint32_t              *refitData,
                    int numNodes)
    {
      const int nodeID = kernel.workIdx();
      if (nodeID == 1 || nodeID >= numNodes) return;
      if (nodeID < 2)
        refitData[0] = 0;
      auto &node = nodes[nodeID];
      node.bounds = box_t<T,D>();
      if (node.admin.count) return;

      if (node.admin.offset+1 >= numNodes)
        printf("refit_init_overflow\n");
      refitData[node.admin.offset+0] = nodeID << 1;
      refitData[node.admin.offset+1] = nodeID << 1;
    }
    
    template<typename T, int D> inline 
    void refit_run_x(Kernel kernel,
                     // BinaryBVH<T,D> bvh,
                     uint32_t *bvh_primIDs,
                     typename BinaryBVH<T,D>::Node *bvh_nodes,
                     uint32_t *refitData,
                     const box_t<T,D> *boxes,
                     int numNodes)
    {
      int nodeID = kernel.workIdx();
      if (nodeID == 1 || nodeID >= numNodes) return;

      typename BinaryBVH<T,D>::Node *node = bvh_nodes+nodeID;

      if (nodeID < 0 || nodeID >= 601202) {
        printf("BLA\n"); return;
      }
      if (node->admin.count == 0)
        // this is a inner node - exit
        return;

      box_t<T,D> bounds; bounds.set_empty();
      for (int i=0;i<node->admin.count;i++) {
        const box_t<T,D> primBox = boxes[bvh_primIDs[node->admin.offset+i]];
        bounds.lower = min(bounds.lower,primBox.lower);
        bounds.upper = max(bounds.upper,primBox.upper);
      }

      int parentID = (refitData[nodeID] >> 1);
      int its = 0;
      while (true) {
        if (parentID == 1) {
          printf("1 is parent!!!!\n");
                 return;
        }
        
        atomic_grow(*(AtomicBox<box_t<T,D>> *)&node->bounds,bounds);
        // node->bounds = bounds;
          
        if (node == bvh_nodes)
          break;

        int it = its++;
        // if (it >= 4) return;

        // if (it == 3) { printf("parentID %i\n",parentID); return; }
        
        if (parentID < 0 || parentID >= 601202) {
          printf("BLA\n"); return;
        }
        uint32_t refitBits = atomicAdd(&refitData[parentID],1u);
        if ((refitBits & 1) == 0)
          // we're the first one - let other one do it
          break;

        nodeID   = parentID;
        node     = &bvh_nodes[parentID];
        parentID = (refitBits >> 1);

        int ofs = node->admin.offset;
        if (ofs < 0 || ofs+1 >= 601202) {
          printf("BLAB\n"); return;
        }
        
        typename BinaryBVH<T,D>::Node l = bvh_nodes[ofs+0];
        typename BinaryBVH<T,D>::Node r = bvh_nodes[ofs+1];
        bounds.lower = min(l.bounds.lower,r.bounds.lower);
        bounds.upper = max(l.bounds.upper,r.bounds.upper);
      }
    }
    
    template<typename T, int D>
    void refit(BinaryBVH<T,D>    &bvh,
               const box_t<T,D>  *boxes,
               Context *ctx)
    {
      PING;
      assert(bvh.nodes);
      assert(bvh.primIDs);
      int numNodes = bvh.numNodes;
      uint32_t *refitData
        = (uint32_t*)ctx->alloc(numNodes*sizeof(uint32_t));
      auto bvh_nodes = bvh.nodes;
      auto bvh_primIDs = bvh.primIDs;
      PING;
      PRINT(numNodes);
      PRINT(bvh.numNodes);
      {
#pragma omp target device(ctx->gpuID) is_device_ptr(refitData) is_device_ptr(bvh_nodes) 
#pragma omp teams distribute parallel for
        for (int i=0;i<numNodes;i++)
          refit_init_x<T,D>(Kernel{i},bvh_nodes,refitData,numNodes);
      }
      PING;
      PING;
      {
#if 1
        int nb = 128;
#pragma omp target  teams num_teams(nb) device(ctx->gpuID) is_device_ptr(bvh_primIDs) is_device_ptr(bvh_nodes) is_device_ptr(refitData) is_device_ptr(boxes)
        {
          int team = omp_get_team_num();
          int nteams = omp_get_num_teams();
          int beg = ((team+0)*numNodes)/nteams;
          int end = ((team+1)*numNodes)/nteams;
          for (int i=beg;i<end;i++)
            refit_run_x(Kernel{i},//bvh,
                        bvh_primIDs,bvh_nodes,refitData,boxes,numNodes);
        }
      }
#else
#pragma omp target device(ctx->gpuID) is_device_ptr(bvh_primIDs) is_device_ptr(bvh_nodes) is_device_ptr(refitData) is_device_ptr(boxes) 
#pragma omp teams distribute parallel for 
      for (int i=0;i<numNodes;i++)
        refit_run_x(Kernel{i},//bvh,
                    bvh_primIDs,bvh_nodes,refitData,boxes,numNodes);
      }
#endif
      PING;
      ctx->free((void*)refitData);
    }
    
  }
}

