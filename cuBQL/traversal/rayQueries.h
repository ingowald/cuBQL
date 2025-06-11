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
                                    bool dbg);
    
    template<typename Lambda>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     cuBQL::Ray   ray,
                                    bool dbg);

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
                                    bool dbg);
    
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
                                    bool dbg);
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
      vec3f A = ray.origin;
      vec3f B = ray.origin + ray.length * ray.direction();
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
    
  } // ::cuBQL
