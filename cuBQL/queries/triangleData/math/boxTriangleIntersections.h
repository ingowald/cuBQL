// ======================================================================== //
// Copyright 2025++ Ingo Wald                                               //
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

#include "cuBQL/queries/triangles/Triangle.h"
#include "cuBQL/math/box.h"

namespace cuBQL {

  // =============================================================================
  // *** INTERFACE ***
  // =============================================================================

    
  // **********************************************************************
  /*! helper class that defines the possible result types of a "lava
    box query", as well as a function ther implements this
    query. Lava box query is defined as follows

    - Input is a set of boxes that are the bounding boxes of what
    is supposed to be a closed triangular surface, as well as a
    BVH over it.

    - query is executed for arbitrary other query boxes, and
    reutrns one of the following:

    a) "box is fully inside the surface

    or, b) box is fully outside that surface

    or, c) box intersects with at least one of the boundary boxes
  */
  struct LavaBoxQuery {
    typedef enum { BOX_IS_INSIDE, BOX_IS_OUTSIDE, BOX_OVERLAPS_PRIMBOX } result_t;

    static inline __device__
    result_t run(const box3f         queryBox,
                 const cuBQL::bvh3f  bvh,
                 const box3f        *primBoxes,
                 const vec3i        *indices,
                 const vec3f        *vertices);
  };
    

  // **********************************************************************
  //                             IMPLEMENTATION
  // **********************************************************************
    
  inline __device__
  bool horizontalLineIntersectsTriangle(const vec3f lineOrigin,
                                        vec3f a,
                                        vec3f b,
                                        vec3f c,
                                        bool dbg=false)
  {
    using cuBQL::dot;
    using cuBQL::cross;
      
    // transform triangle into space centered aorund line origin
    a = a - lineOrigin;
    b = b - lineOrigin;
    c = c - lineOrigin;
    // compute normal, for plane equation
    vec3f n = cross(b-a,c-a);

    // create horitonzal semi-infite "ray" from origin=0 alone x axis
    const vec3f org = vec3f(0.f);
    const vec3f dir = vec3f(1.f,0.f,0.f);
      
    bool orgOnFront = dot(vec3f(0.f)-a,n) > 0.f;
    bool normalPointingRight = n.x > 0.f;
    if (orgOnFront && normalPointingRight ||
        !orgOnFront && !normalPointingRight)
      return false;

    auto pluecker=[](vec3f a0, vec3f a1, vec3f b0, vec3f b1) 
    { return dot(a1-a0,cross(b1,b0))+dot(b1-b0,cross(a1,a0)); };

    // compute pluecker coordinates dot product of all edges wrt x
    // axis ray. since the ray is mostly 0es and 1es, this shold all
    // evaluate to some fairly simple expressions
    float sx = pluecker(org,org+dir,a,b);
    float sy = pluecker(org,org+dir,b,c);
    float sz = pluecker(org,org+dir,c,a);
      
    // for ray to be inside edges it must have all positive or all
    // negative pluecker winding order
    auto min3=[](float x, float y, float z)
    { return min(min(x,y),z); };
    auto max3=[](float x, float y, float z)
    { return max(max(x,y),z); };
    if (min3(sx,sy,sz) >= 0.f || max3(sx,sy,sz) <= 0.f)
      return true;
      
    return false;
  }
    
    
  inline __device__
  bool horizontalLineIntersectsTriangle(const vec3f         lineOrigin,
                                        const cuBQL::bvh3f  bvh,
                                        const vec3i        *indices,
                                        const vec3f        *vertices,
                                        bool dbg = false)
  {
    bool hasIntersection = false;
    box3f queryBox = { lineOrigin, lineOrigin };
    queryBox.upper.x = INFINITY;
    auto lambda = [&hasIntersection,lineOrigin,indices,vertices,dbg]
      (const uint32_t *primIDs, size_t numPrims) -> int
    {
      for (int i=0;i<numPrims;i++) {
        const int primID = primIDs[i];
        const vec3i triangle = indices[primID];
        const vec3f a = vertices[triangle.x];
        const vec3f b = vertices[triangle.y];
        const vec3f c = vertices[triangle.z];
        if (horizontalLineIntersectsTriangle(lineOrigin,a,b,c,dbg)) {
          hasIntersection = true;
          return CUBQL_TERMINATE_TRAVERSAL;
        }
      }
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    cuBQL::fixedBoxQuery_forEachLeaf(bvh,queryBox,lambda);
    return hasIntersection;
  }    

  /*! helper functoin that returns whether (or not) a given query
    box overlaps any of the primitmive boxes of the given BVH */
  inline __device__
  bool doesThisOverlapAnyPrimBoxes(const box3f         queryBox,
                                   const cuBQL::bvh3f  bvh,
                                   const box3f        *primBoxes)
  {
    bool foundAnOverlap = false;
    auto lambda = [&foundAnOverlap,queryBox,primBoxes]
      (const uint32_t *primIDs, size_t numPrims) -> int
    {
      for (int i=0;i<numPrims;i++) {
        const int primID = primIDs[i];
        const box3f primBox = primBoxes[primID];
        if (primBox.overlaps(queryBox)) {
          foundAnOverlap = true;
          return CUBQL_TERMINATE_TRAVERSAL;
        }
      }
      return CUBQL_TERMINATE_TRAVERSAL;
    };
    cuBQL::fixedBoxQuery_forEachLeaf(bvh,queryBox,lambda);
    return foundAnOverlap;
  }

  inline __device__
  int horizontalLineCrossingCount(const vec3f         lineOrigin,
                                  const cuBQL::bvh3f  bvh,
                                  const vec3i        *indices,
                                  const vec3f        *vertices,
                                  bool dbg = false)
  {
    int crossingCount = 0;
    box3f queryBox = { lineOrigin, lineOrigin };
    queryBox.upper.x = INFINITY;
    auto lambda = [&crossingCount,lineOrigin,indices,vertices,dbg]
      (const uint32_t *primIDs, size_t numPrims) -> int
    {
      for (int i=0;i<numPrims;i++) {
        const int primID = primIDs[i];
        const vec3i triangle = indices[primID];
        const vec3f a = vertices[triangle.x];
        const vec3f b = vertices[triangle.y];
        const vec3f c = vertices[triangle.z];
        if (dbg) {
          printf("testing triangle %i\n",primID);
          printf("  a %f %f %f\n",a.x,a.y,a.z);
          printf("  b %f %f %f\n",b.x,b.y,b.z);
          printf("  c %f %f %f\n",c.x,c.y,c.z);
          printf(" pt %f %f %f\n",lineOrigin.x,lineOrigin.y,lineOrigin.z);
        }
        if (horizontalLineIntersectsTriangle(lineOrigin,a,b,c,dbg)) 
          ++crossingCount;
      }
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    cuBQL::fixedBoxQuery_forEachLeaf(bvh,queryBox,lambda);
    return crossingCount;
  }

  inline __device__
  LavaBoxQuery::result_t LavaBoxQuery::run(const box3f         queryBox,
                                           const cuBQL::bvh3f  bvh,
                                           const box3f        *primBoxes,
                                           const vec3i        *indices,
                                           const vec3f        *vertices) 
  {
    if (doesThisOverlapAnyPrimBoxes(queryBox,bvh,primBoxes)) {
      overlap::LavaTriangleQuery triangleQuery(bvh,primBoxes,indices,vertices);
      int numTrisInThisQueryBox = 0;
      // create a lambda function callback that, in this case, just counts.
      auto countCallback=[&numTrisInThisQueryBox,triangleQuery](int triangleIndex)
      { ++numTrisInThisQueryBox; };
      triangleQuery.forEachTriangleInBox(queryBox,countCallback);
      if (numTrisInThisQueryBox > 0)
        return BOX_OVERLAPS_PRIMBOX;
    }
      
    const vec3f lineQueryOrigin = queryBox.center();
    int cc = horizontalLineCrossingCount(lineQueryOrigin,bvh,indices,vertices);
    if ((cc % 2) == 1) 
      /* crossing count is odd - we are entering once more than
         we're leaving, so are inside, and we need to check for triangle overlap */
      return BOX_IS_INSIDE;
      
    else
      /*! crossing count is even - we leave once for every time we
        enter, so we're out */
      return BOX_IS_OUTSIDE;
  }
    
} // ::cuBQL

  
