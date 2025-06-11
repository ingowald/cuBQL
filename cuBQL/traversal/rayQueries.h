// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"
#include "cuBQL/math/Ray.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace cuBQL {
  namespace rayQuery {

    // ******************************************************************
    // INTERFACE
    // (which functions this header file provides)
    // ******************************************************************
    
    template<typename Lambda>
    inline __cubql_both
    void forEachLeaf(/*! lambda that gets called for each BVH leaf
                       that may may contain any new result(s) within
                       the current max query radius. if this lamdba
                       does find a new, better result than whatever
                       the query had before this lambda MUST return
                       the SQUARE of the new culling radius, returning
                       a culling radius < 0 will immediately terminate
                       any further traversal steps */
                     const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     cuBQL::Ray   ray);
    
    template<typename Lambda>
    inline __cubql_both
    void forEachPrim(/*! lambda that gets called for each candidate
                       primitive index that may contain any new result
                       within the current max query radius. if this
                       lamdba does find a new, better result than
                       whatever the query had before this lambda MUST
                       return the SQUARE of the new culling
                       radius. Returning a culling radius < 0 will
                       immediately terminate any future traversal
                       steps */
                     const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     cuBQL::Ray   ray);

    template<int axis, int direction, typename Lambda>
    inline __cubql_both
    void forEachLeaf(/*! lambda that gets called for each BVH leaf
                       that may may contain any new result(s) within
                       the current max query radius. if this lamdba
                       does find a new, better result than whatever
                       the query had before this lambda MUST return
                       the SQUARE of the new culling radius, returning
                       a culling radius < 0 will immediately terminate
                       any further traversal steps */
                     const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     AxisAlignedRay<axis,direction> ray);
    
    template<int axis, int direction, typename Lambda>
    inline __cubql_both
    void forEachPrim(/*! lambda that gets called for each candidate
                       primitive index that may contain any new result
                       within the current max query radius. if this
                       lamdba does find a new, better result than
                       whatever the query had before this lambda MUST
                       return the SQUARE of the new culling
                       radius. Returning a culling radius < 0 will
                       immediately terminate any future traversal
                       steps */
                     const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     AxisAlignedRay<axis,direction> ray);
    
    // ******************************************************************
    // IMPLEMENTATION
    // ******************************************************************
    
    
    template<typename Lambda>
    inline __cubql_both
    void forEachPrim(/*! lambda that gets called for each candidate
                       primitive index that may contain any new result
                       within the current max query radius. if this
                       lamdba does find a new, better result than
                       whatever the query had before this lambda MUST
                       return the SQUARE of the new culling
                       radius. Returning a culling radius < 0 will
                       immediately terminate any future traversal
                       steps */
                     const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     Ray          ray)
    {
      /* the code we want to have executed for each leaf that may
         contain candidates. we loop over each prim in a given leaf,
         and return the minimum culling distance returned by any of
         the per-prim lambdas */
      auto leafCode
        = [lambdaToExecuteForEachCandidate](const uint32_t *leafPrims,
                                            size_t numPrims)->float
        {
          float leafResult = CUBQL_INF;
          for (int i=0;i<numPrims;i++) {
            float primResult
              = lambdaToExecuteForEachCandidate(leafPrims[i]);
            leafResult = min(leafResult,primResult);
            if (leafResult < 0.f) break;
          }
          return leafResult;
        };
      forEachLeaf(leafCode,bvh,ray);
    }



    template<int axis, int direction, typename Lambda>
    inline __cubql_both
    void forEachLeaf(/*! lambda that gets called for each BVH leaf
                       that may may contain any new result(s) within
                       the current max query radius. if this lamdba
                       does find a new, better result th {an whatever
                       the query had before this lambda MUST return
                       the SQUARE of the new culling radius, returning
                       a culling radius < 0 will immediately terminate
                       any further traversal steps */
                     const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     AxisAlignedRay<axis,direction> ray)
    {
      box3f rayAsBox = { ray.origin,ray.origin };
      if (direction == +1) {
        rayAsBox.lower = rayAsBox.lower[axis] + ray.tmin;
        rayAsBox.upper = rayAsBox.upper[axis] + ray.tmax;
      } else {
        rayAsBox.upper = rayAsBox.upper[axis] - ray.tmin;
        rayAsBox.lower = rayAsBox.lower[axis] - ray.tmax;
      }
      cuBQL::fixedBoxQuery::forEachLeaf(lambdaToExecuteForEachCandidate,bvh,rayAsBox);
    }
    
    template<int axis, int direction, typename Lambda>
    inline __cubql_both
    void forEachPrim(/*! lambda that gets called for each candidate
                       primitive index that may contain any new result
                       within the current max query radius. if this
                       lamdba does find a new, better result than
                       whatever the query had before this lambda MUST
                       return the SQUARE of the new culling
                       radius. Returning a culling radius < 0 will
                       immediately terminate any future traversal
                       steps */
                     const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     AxisAlignedRay<axis,direction> ray)
    {
      /* the code we want to have executed for each leaf that may
         contain candidates. we loop over each prim in a given leaf,
         and return the minimum culling distance returned by any of
         the per-prim lambdas */
      auto leafCode
        = [lambdaToExecuteForEachCandidate](const uint32_t *leafPrims,
                                            size_t numPrims)->float
        {
          float leafResult = CUBQL_INF;
          for (int i=0;i<numPrims;i++) {
            float primResult
              = lambdaToExecuteForEachCandidate(leafPrims[i]);
            leafResult = min(leafResult,primResult);
            if (leafResult < 0.f) break;
          }
          return leafResult;
        };
      forEachLeaf(leafCode,bvh,ray);
    }
    
  } // ::cuBQL::rayQuery
} // ::cuBQL
