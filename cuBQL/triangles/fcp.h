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

#include "cuBQL/bvh.h"
// we're stealing the line seg test for edges
#include "cuBQL/lineSegs/LineSegs3f.h"
#include "cuBQL/queries/shrinkingRadiusQuery.h"

namespace cuBQL {
  namespace triangles {

    /*! a line segment, referring to two input points that define the
        two opposite corners of that line segment */
    struct Triangle {
      vec3f a, b, c;
    };

    /*! result of a fcp (find closest point) query */
    struct FCPResult {
      inline __device__ void clear(float maxDistSqr) { primID = -1; sqrDistance = maxDistSqr; }
      
      int   primID;
      // float u, v;
      float sqrDistance;
    };

    /*! fcp = find-closest-point on triangle mesh. Finds, for a given
        query point, the closest 3D point on any tirangles in the
        triangles[]/vertices[] triangle mesh, and stores that in
        result. result should have been cleared via
        reuslt.clear(maxSearchDistSquared) before calling this. Only
        results within the max query radius passed to result.clear()
        will be returned */
    inline __device__
    void fcp(FCPResult &result,
             const vec3f                     queryPoint,
             const bvh3f                     bvh,
             const vec3i *const __restrict__ triangles,
             const vec3f *const __restrict__ vertices);

    // ==================================================================
    // implementation
    // ==================================================================
    
    /*! result of a closest-point intersection operation */
    struct CPResult {
      vec3f point;
      
      /*! parameterized distance along the line triangle of where the
          hit is; u=0 being begin point, u=1 being end point */
      float u, v;

      float sqrDistance;
    };
    
    /*! compute point on 'triangle' that is closest to 'queryPoint',
      and return the square distance to that point. */
    inline __device__
    CPResult closestPoint(const vec3f queryPoint, const Triangle triangle);
    
    


    /*! compute point on 'triangle' that is closest to 'queryPoint',
      and return the square distance to that point. */
    inline __device__
    CPResult closestPoint(const vec3f q, const Triangle triangle)
    {
      vec3f a = triangle.a;
      vec3f b = triangle.b;
      vec3f c = triangle.c;
      vec3f N = cross(b-a,c-a);
      vec3f Nab = cross(b-a,N);
      vec3f Nbc = cross(c-b,N);
      vec3f Nca = cross(a-c,N);
      CPResult result;
      lineSegs::Segment edge;
      bool edgeTest = true;
      if (dot(q-a,Nab) >= 0.f) {
        edge = { a, b };
        // do edge test below
      } else if (dot(q-b,Nbc) >= 0.f) {
        edge = { b, c };
        // do edge test below
      } else if (dot(q-c,Nca) >= 0.f) {
        edge = { c, a };
        // do edge test below
      } else {
        // point must be inside.
        N = normalize(N);
        float dist = dot(q-a,N);
        result.point = q - dist*N;
        edgeTest = false;
      }
      if (edgeTest) {
        lineSegs::CPResult cp = lineSegs::closestPoint(q,edge);
        result.point = cp.point;
      }

      result.sqrDistance = sqrDistance(q,result.point);
      return result;
    }
    
    /*! find closest point (to query point) among a set of line
        triangles (given by triangles[] and vertices[], up to a maximum
        (square) query distance provided in result.sqrDistance. any
        line egments further away than result.sqrDistance will get
        rejected; at the end of the query result.maxSqrDistance will
        be the (square) distnace to the found triangle (if found), or
        will be left un-modified if no such triangle could be found
        within the initial query radius */
    inline __device__
    void fcp(FCPResult   &result,
             const vec3f  queryPoint,
             const bvh3f  bvh,
             const vec3i *const __restrict__ indices,
             const vec3f *const __restrict__ vertices)
    {
      /* as with all cubql traversal routines, we use a lambda that
         gets in a per-primitive ID, and that returns a square
         distance that the traversal routines can from now on use to
         cull (note if the traversal already has a closer culling
         distance it will keep on using this no matter what this
         function returns, so we don't need to be too sanguine about
         which value to return) */
      auto perPrim=[&result,queryPoint,indices,vertices](uint32_t primID) {
        vec3i index = indices[primID];
        Triangle triangle{vertices[index.x],vertices[index.y],vertices[index.z]};
        CPResult primResult = closestPoint(queryPoint,triangle);
        if (primResult.sqrDistance < result.sqrDistance) {
          result.primID = primID;
          result.sqrDistance = primResult.sqrDistance;
        }
        return result.sqrDistance;
      };
      cuBQL::shrinkingRadiusQuery_forEachPrim(bvh,queryPoint,result.sqrDistance,perPrim);
    } // fcp()
    
  } // ::cuBQL::triangles
} // ::cuBQL
