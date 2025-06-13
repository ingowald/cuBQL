// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"
#include "cuBQL/math/Ray.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace cuBQL {
  namespace fixedRayQuery {

    // ******************************************************************
    // INTERFACE
    // (which functions this header file provides)
    // ******************************************************************
    
    template<typename Lambda>
    inline __cubql_both
    void forEachLeaf(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     cuBQL::Ray   ray,
                     bool dbg=false);
    
    template<typename Lambda>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     cuBQL::Ray   ray,
                     bool dbg=false);

    /*! traverse BVH with given fixed-length, axis-aligned ray, and
      call lambda for each prim encounterd.

      Traversal is UNORDERED (meaning it will NOT try to traverse
      front-to-back) and FIXED-SHAPE (ray will not shrink during
      traversal).

      Lambda is expected to return CUBQL_{CONTINUE|TERMINATE}_TRAVERSAL 
    */
    template<int axis, int sign, typename Lambda>
    inline __cubql_both
    void forEachLeaf(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     AxisAlignedRay<axis,sign> ray,
                     bool dbg=false);
    
    /*! traverse BVH with given fixed-length, axis-aligned ray, and
      call lambda for each prim encounterd.

      Traversal is UNORDERED (meaning it will NOT try to traverse
      front-to-back) and FIXED-SHAPE (ray will not shrink during
      traversal).

      Lambda is expected to return CUBQL_{CONTINUE|TERMINATE}_TRAVERSAL 
    */
    template<int axis, int sign, typename Lambda>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     AxisAlignedRay<axis,sign> ray,
                     bool dbg=false);
  }
  
  // ******************************************************************
  // IMPLEMENTATION
  // ******************************************************************

  template<int axis, int sign, typename Lambda>
  inline __cubql_both
  void fixedRayQuery::forEachLeaf(const Lambda &lambdaToExecuteForEachCandidate,
                                  cuBQL::bvh3f bvh,
                                  AxisAlignedRay<axis,sign> ray,
                                  bool dbg)
  {
    /* for an axis-aligned ray, we can just convert that ray to a
       box, and traverse that instad */
    vec3f A = ray.origin + ray.tmin * ray.direction();
    vec3f B = ray.origin + ray.tmax * ray.direction();
    box3f rayAsBox { min(A,B), max(A,B) };
    // if (dbg) dout << "asbox " << rayAsBox << dout.endl;
    cuBQL::fixedBoxQuery::forEachLeaf(lambdaToExecuteForEachCandidate,bvh,rayAsBox,dbg);
  }

  /*! this query assumes lambads that return CUBQL_CONTINUE_TRAVERSAL
    or CUBQL_TERMINATE_TRAVERSAL */
  template<int axis, int sign, typename Lambda>
  inline __cubql_both
  void fixedRayQuery::forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                                  cuBQL::bvh3f bvh,
                                  AxisAlignedRay<axis,sign> ray,
                                  bool dbg)
  {
    /* the code we want to have executed for each leaf that may
       contain candidates. we loop over each prim in a given leaf,
       and return the minimum culling distance returned by any of
       the per-prim lambdas */
    auto leafCode
      = [lambdaToExecuteForEachCandidate,dbg](const uint32_t *leafPrims,
                                              size_t numPrims)->int
      {
        // if (dbg) dout << "fixedRayQuery::forEachPrim leaf " << numPrims << endl;
        for (int i=0;i<numPrims;i++) 
          if (lambdaToExecuteForEachCandidate(leafPrims[i])
              == CUBQL_TERMINATE_TRAVERSAL)
            return CUBQL_TERMINATE_TRAVERSAL;
        return CUBQL_CONTINUE_TRAVERSAL;
      };
    forEachLeaf(leafCode,bvh,ray,dbg);
  }



  template<typename Lambda>
  inline __cubql_both
  void fixedRayQuery::forEachLeaf(const Lambda &lambdaToCallOnEachLeaf,
                                  cuBQL::bvh3f bvh,
                                  Ray ray,
                                  bool dbg)
  {
      struct StackEntry {
        uint32_t idx;
      };
      bvh3f::node_t::Admin traversalStack[64], *stackPtr = traversalStack;
      bvh3f::node_t::Admin node = bvh.nodes[0].admin;
      // ------------------------------------------------------------------
      // traverse until there's nothing left to traverse:
      // ------------------------------------------------------------------
      // if (dbg) dout << "fixedBoxQuery::traverse" << endl;
      while (true) {

        // ------------------------------------------------------------------
        // traverse INNER nodes downward; breaking out if we either find
        // a leaf within the current search radius, or found a dead-end
        // at which we need to pop
        // ------------------------------------------------------------------
        while (true) {
          if (node.count != 0)
            // it's a boy! - seriously: this is not a inner node, step
            // out of down-travesal and let leaf code pop in.
            break;

          uint32_t n0Idx = (uint32_t)node.offset+0;
          uint32_t n1Idx = (uint32_t)node.offset+1;
          bvh3f::node_t n0 = bvh.nodes[n0Idx];
          bvh3f::node_t n1 = bvh.nodes[n1Idx];
          bool o0 = rayIntersectsBox(ray,n0.bounds);
          bool o1 = rayIntersectsBox(ray,n1.bounds);

          // if (dbg) {
          //   dout << "at node " << node.offset << endl;
          //   dout << "w/ query box " << queryBox << endl;
          //   dout << "  " << n0.bounds << " -> " << (int)o0 << endl;
          //   dout << "  " << n1.bounds << " -> " << (int)o1 << endl;
          // }
          
          if (o0) {
            if (o1) {
              *stackPtr++ = n1.admin;
            } else {
            }
            node = n0.admin;
          } else {
            if (o1) {
              node = n1.admin;
            } else {
              // both children are too far away; this is a dead end
              node.count = 0;
              break;
            }
          }
        }

        // if (dbg)
        //   dout << "at leaf ofs " << (int)node.offset << " cnt " << node.count << endl;
        if (node.count != 0) {
          // we're at a valid leaf: call the lambda and see if that gave
          // us a enw, closer cull radius
          int leafResult
            = lambdaToCallOnEachLeaf(bvh.primIDs+node.offset,node.count);
          // if (dbg)
          //   dout << "leaf returned " << leafResult << endl;
          if (leafResult == CUBQL_TERMINATE_TRAVERSAL)
            return;
        }
        // ------------------------------------------------------------------
        // pop next un-traversed node from stack, discarding any nodes
        // that are more distant than whatever query radius we now have
        // ------------------------------------------------------------------
        // if (dbg) dout << "rem stack depth " << (stackPtr-traversalStack) << endl;
        if (stackPtr == traversalStack)
          return;
        node = *--stackPtr;
      }
  }

  /*! this query assumes lambads that return CUBQL_CONTINUE_TRAVERSAL
    or CUBQL_TERMINATE_TRAVERSAL */
  template<typename Lambda>
  inline __cubql_both
  void fixedRayQuery::forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                                  cuBQL::bvh3f bvh,
                                  Ray ray,
                                  bool dbg)
  {
    /* the code we want to have executed for each leaf that may
       contain candidates. we loop over each prim in a given leaf,
       and return the minimum culling distance returned by any of
       the per-prim lambdas */
    auto leafCode
      = [lambdaToExecuteForEachCandidate,dbg](const uint32_t *leafPrims,
                                              size_t numPrims)->int
      {
        // if (dbg) dout << "fixedRayQuery::forEachPrim leaf " << numPrims << endl;
        for (int i=0;i<numPrims;i++) 
          if (lambdaToExecuteForEachCandidate(leafPrims[i])
              == CUBQL_TERMINATE_TRAVERSAL)
            return CUBQL_TERMINATE_TRAVERSAL;
        return CUBQL_CONTINUE_TRAVERSAL;
      };
    forEachLeaf(leafCode,bvh,ray,dbg);
  }
    
  
} // ::cuBQL
