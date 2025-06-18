// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/bvh.h"
// the kind of model data we operate on
#include "cuBQL/queries/triangleData/closestPointOnAnyTriangle.h"
// the kind of traversal we need for this query
#include "cuBQL/traversal/fixedRadiusQuery.h"

namespace cuBQL {
  /*! \namespace triangles for any queries operating on triangle model data */
  namespace triangles {

    // =============================================================================
    // *** INTERFACE ***
    // =============================================================================

    /*! returns number of triangles within a given (non-squared) radius
        r around a point P. Will only look for up to
        'maxNumToLookFor'; as soon as that many are found this will
        return this value.
        
      getTriangle is lambda getTriangle(uint32_t triID)->Triangle
    */
    template<typename GetTriangleLambda>
    inline __cubql_both
    int numWithinRadius(bvh3f bvh,
                       GetTriangleLambda getTriangle,
                       vec3f queryBallCenter,
                       float queryBallRadius,
                       int maxNumToLookFor=INT_MAX);
    
    /*! checks if there are _any_ triangles within a given
        (non-squared) radius r of a point P */
    template<typename GetTriangleLambda>
    inline __cubql_both
    bool anyWithinRadius(bvh3f bvh,
                         GetTriangleLambda getTriangle,
                         vec3f queryBallCenter,
                         float queryBallRadius);
    
    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================


    template<typename GetTriangleLambda>
    inline __cubql_both
    int numWithinRadius(bvh3f bvh,
                        GetTriangleLambda getTriangle,
                        vec3f queryBallCenter,
                        float queryBallRadius,
                        int maxNumToLookFor)
    {
      int numFound = 0;
      auto perPrim
        = [&numFound,maxNumToLookFor,getTriangle,queryBallCenter,queryBallRadius]
        (uint32_t triID)
        {
          auto sqrDist = computeClosestPoint(queryBallCenter,getTriangle(triID)).sqrDist;
          if (sqrDist != CUBQL_INF && sqrDist <= squareOf(queryBallRadius)) 
            ++numFound;
          return numFound >= maxNumToLookFor
            ? CUBQL_TERMINATE_TRAVERSAL
            : CUBQL_CONTINUE_TRAVERSAL;
        };
      fixedRadiusQuery::forEachPrim(perPrim,bvh,
                                    queryBallCenter,
                                    /* traversal templates uses SQUARE of radius!! */
                                    squareOf(queryBallRadius));
      return numFound;
    }
    
    template<typename GetTriangleLambda>
    inline __cubql_both
    bool anyWithinRadius(bvh3f bvh,
                         GetTriangleLambda getTriangle,
                         vec3f queryBallCenter,
                         float queryBallRadius)
    { return numWithinRadius(bvh,getTriangle,
                             queryBallCenter,
                             queryBallRadius,/* max to look for for early exit */1) > 0; }
    
    
  } // ::cuBQL::triangles
} // ::cuBQL
