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
// the kind of model data we operate on
#include "cuBQL/queries/triangles/Triangle.h"
#include "cuBQL/queries/common/Ray.h"
// the kind of traversal we need for this query
#include "cuBQL/traversal/shrinkingRadiusQuery.h"

namespace cuBQL {
  /*! \namespace triangles for any queries operating on triangle model data */
  namespace triangles {
    
    /*! result of a cpat query (cpat =
      closest-point-on-any-triangle); if no result was found within
      the specified max seach distance, triangleIex will be returned
      as -1 */
    struct CPAT {

      /*! index of triangle that had closest hit; -1 means 'none found
        that was closer than cut-off distance */
      int   triangleIdx = -1;
      
      /*! actual 3D point on triangle that was closest. Undefined if
        triangleIdx==-1 */
      vec3f P;

      /*! (square) distance between query point and this->P; or square
        of query cut-off distance if triangleIdx == -1 */
      float sqrDist     = INFINITY;

      /*! performs one complete query, starting with an empty CPAT
        result, traversing the BVH for the givne mesh, and processing
        every triangle that needs consideration. Only intersections
        that are < maxQueryRadius will get accepted */
      inline __device__
      void runQuery(const cuBQL::vec3f       *mesh_vertices,
                    const cuBQL::vec3i       *mesh_indices,
                    const cuBQL::bvh3f        bvh,
                    const cuBQL::vec3f        queryPoint,
                    float maxQueryRadius = CUBQL_INF);
      
      /*! performs one complete query, starting with an empty CPAT
        result, traversing the BVH for the givne mesh, and processing
        every triangle that needs consideration. Only intersections
        that are < maxQueryRadius will get accepted */
      inline __device__
      void runQuery(const cuBQL::Triangle    *triangles,
                    const cuBQL::bvh3f        bvh,
                    const cuBQL::vec3f        queryPoint,
                    float maxQueryRadius = CUBQL_INF);
      
    };


#if CUBQL_TRIANGLE_CPAT_IMPLEMENTATION

    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================

    namespace cpat {
      /*! helper struct for a edge with double coordinates; mainly
        exists for the Edge::closestPoint test method */
      struct Edge {
        inline __host__ __device__
        Edge(vec3f a, vec3f b) : a(a), b(b) {}
        
        /*! compute point-to-distance for this triangle; returns true if the
          result struct was updated with a closer point than what it
          previously contained */
        inline __host__ __device__
        bool closestPoint(CPAT &result,
                          const vec3f &referencePointToComputeDistanceTo,
                          bool dbg=0) const;
        
        const vec3f a, b; 
      };
      
      
      /*! compute point-to-distance for this edge; returns true if the
        result struct was updated with a closer point than what it
        previously contained */
      inline __device__
      bool Edge::closestPoint(CPAT &result,
                              const vec3f &p,
                              bool dbg) const
      {
        float t = dot(p-a,b-a) / dot(b-a,b-a);
        t = clamp(t);
        vec3f cp = a + t * (b-a);
        float sqrDist = dot(cp-p,cp-p);
        if (sqrDist >= result.sqrDist) 
          return false;
        
        result.sqrDist = sqrDist;
        result.P       = cp;
        return true;
      }
      
      /*! computes the querypoint-triangle test for a given pair of
        triangle and query point; returns true if this _was_ closer
        than what 'this' stored before (and if so, 'this' was
        updated); if this returns false the computed distance was
        greater than the already stored distance, and 'this' was
        left unmodified */
      inline __device__
      bool computeOneIntersection(CPAT &result,
                                  const cuBQL::Triangle triangle,
                                  const cuBQL::vec3f    queryPoint)
      {
        const vec3f a = triangle.a;
        const vec3f b = triangle.b;
        const vec3f c = triangle.c;
        vec3f N = cross(b-a,c-a);
        bool projectsOutside
          =  (N == vec3f(0.f,0.f,0.f))
          || (dot(queryPoint-a,cross(b-a,N)) >= 0.f)
          || (dot(queryPoint-b,cross(c-b,N)) >= 0.f)
          || (dot(queryPoint-c,cross(a-c,N)) >= 0.f);
        if (projectsOutside) {
          return
            Edge(a,b).closestPoint(result,queryPoint) |
            Edge(b,c).closestPoint(result,queryPoint) |
            Edge(c,a).closestPoint(result,queryPoint);
        } else {
          N = normalize(N);
          float signed_dist = dot(queryPoint-a,N);
          float sqrDist = signed_dist*signed_dist;
          if (sqrDist >= result.sqrDist) return false;
          result.sqrDist = sqrDist;
          result.P       = queryPoint - signed_dist * N;
          return true;
        }
      }

    } // ::cuBQL::trianlges::cpat
      
    /*! performs one complete query, starting with an empty CPAT
      result, traversing the BVH for the givne mesh, and processing
      every triangle that needs consideration. Only intersections
      that are < maxQueryRadius will get accepted */
    inline __device__
    void CPAT::runQuery(const cuBQL::vec3f *triangle_vertices,
                        const cuBQL::vec3i *triangle_indices,
                        const cuBQL::bvh3f  bvh,
                        const cuBQL::vec3f  queryPoint,
                        float               maxQueryRadius)
    {
      triangleIdx = -1;
      sqrDist     = maxQueryRadius*maxQueryRadius;
      auto perPrimitiveCode
        = [bvh,triangle_vertices,triangle_indices,queryPoint,this]
        (uint32_t triangleIdx)->float
        {
          vec3i idx = triangle_indices[triangleIdx];
          const Triangle triangle = { triangle_vertices[idx.x],
                                      triangle_vertices[idx.y],
                                      triangle_vertices[idx.z] };
          if (cpat::computeOneIntersection(*this,triangle,queryPoint))
            this->triangleIdx = triangleIdx;
          /*! the (possibly new?) max cut-off radius (squared, as
              traversals operate on square distances!) */
          return this->sqrDist;
        };
      // careful: traversals operate on the SQUARE radii
      const float maxQueryRadiusDistance
        = maxQueryRadius * maxQueryRadius;
      cuBQL::shrinkingRadiusQuery::forEachPrim
        (/* what we want to execute for each candidate: */perPrimitiveCode,
         /* what we're querying into*/bvh,
         /* where we're querying */queryPoint,
         /* initial maximum search radius */maxQueryRadiusDistance
         );
    }
    

    /*! performs one complete query, starting with an empty CPAT
      result, traversing the BVH for the givne mesh, and processing
      every triangle that needs consideration. Only intersections
      that are < maxQueryRadius will get accepted */
    inline __device__
    void CPAT::runQuery(const cuBQL::Triangle *triangles,
                        const cuBQL::bvh3f     bvh,
                        const cuBQL::vec3f     queryPoint,
                        float                  maxQueryRadius)
    {
      triangleIdx = -1;
      sqrDist     = maxQueryRadius*maxQueryRadius;
      auto perPrimitiveCode
        = [bvh,triangles,queryPoint,this]
        (uint32_t triangleIdx)->float
        {
          const Triangle triangle = triangles[triangleIdx];
          if (cpat::computeOneIntersection(*this,triangle,queryPoint))
            this->triangleIdx = triangleIdx;
          /*! the (possibly new?) max cut-off radius (squared, as
              traversals operate on square distances!) */
          return this->sqrDist;
        };
      // careful: traversals operate on the SQUARE radii
      const float maxQueryRadiusDistance
        = maxQueryRadius * maxQueryRadius;
      cuBQL::shrinkingRadiusQuery::forEachPrim
        (/* what we want to execute for each candidate: */perPrimitiveCode,
         /* what we're querying into*/bvh,
         /* where we're querying */queryPoint,
         /* initial maximum search radius */maxQueryRadiusDistance
         );
    }
    
    // /*! result of a fcp (find closest point) query */
    // struct FCPResult {
    //   inline __device__ void clear(float maxDistSqr) { primID = -1; sqrDist = maxDistSqr; }
      
    //   int   primID;
    //   float sqrDist;
    //   vec3f P;
    // };

    // /*! fcp = find-closest-point on triangle mesh. Finds, for a given
    //     query point, the closest 3D point on any tirangles in the
    //     triangles[]/vertices[] triangle mesh, and stores that in
    //     result. result should have been cleared via
    //     reuslt.clear(maxSearchDistSquared) before calling this. Only
    //     results within the max query radius passed to result.clear()
    //     will be returned */
    // inline __device__
    // void fcp(FCPResult &result,
    //          const vec3f                     queryPoint,
    //          const bvh3f                     bvh,
    //          const vec3i *const __restrict__ triangles,
    //          const vec3f *const __restrict__ vertices);

    // // ==================================================================
    // // implementation
    // // ==================================================================
    
    // /*! helper struct for a edge with double coordinates; mainly
    //   exists for the Edge::closestPoint test method */
    // struct Edge {
    //   inline __cubql_both
    //   Edge(vec3f a, vec3f b) : a(a), b(b) {}
    
    //   /*! compute point-to-distance for this triangle; returns true if the
    //     result struct was updated with a closer point than what it
    //     previously contained */
    //   inline __cubql_both
    //   bool closestPoint(FCPResult &result,
    //                     const vec3f &referencePointToComputeDistanceTo) const;
    
    //   const vec3f a, b; 
    // };
    

    // /*! compute point-to-distance for this edge; returns true if the
    //   result struct was updated with a closer point than what it
    //   previously contained */
    // inline __device__
    // bool Edge::closestPoint(FCPResult &result,
    //                         const vec3f &p) const
    // {
    //   float t = dot(p-a,b-a) / dot(b-a,b-a);
    //   t = clamp(t);
    //   vec3f cp = a + t * (b-a);
    //   float sqrDist = dot(cp-p,cp-p);
    //   if (sqrDist >= result.sqrDist) 
    //     return false;

    //   result.sqrDist = sqrDist;
    //   result.P      = cp;
    //   return true;
    // }
  
    // /*! compute point-to-distance for this triangle; returns true if the
    //   result struct was updated with a closer point than what it
    //   previously contained */
    // inline __device__
    // bool closestPoint(FCPResult &result,
    //                   Triangle triangle,
    //                   vec3f qp) 
    // {
    //   vec3f a = triangle.a;
    //   vec3f b = triangle.b;
    //   vec3f c = triangle.c;
      
    //   vec3f N = cross(b-a,c-a);
    //   bool projectsOutside
    //     =  (N == vec3f(0.f,0.f,0.f))
    //     || (dot(qp-a,cross(b-a,N)) >= 0.f)
    //     || (dot(qp-b,cross(c-b,N)) >= 0.f)
    //     || (dot(qp-c,cross(a-c,N)) >= 0.f);
    //   if (projectsOutside) {
    //     return
    //       Edge(a,b).closestPoint(result,qp) |
    //       Edge(b,c).closestPoint(result,qp) |
    //       Edge(c,a).closestPoint(result,qp);
    //   } else {
    //     N = normalize(N);
    //     float signed_dist = dot(qp-a,N);
    //     float sqrDist = signed_dist*signed_dist;
    //     if (sqrDist >= result.sqrDist) return false;
    //     result.sqrDist = sqrDist;
    //     result.P       = qp - signed_dist * N;
    //     return true;
    //   }
    // }

    // /*! find closest point (to query point) among a set of line
    //     triangles (given by triangles[] and vertices[], up to a maximum
    //     (square) query dist provided in result.sqrDist. any
    //     line egments further away than result.sqrDist will get
    //     rejected; at the end of the query result.maxSqrDist will
    //     be the (square) distnace to the found triangle (if found), or
    //     will be left un-modified if no such triangle could be found
    //     within the initial query radius */
    // inline __device__
    // void findClosest(FCPResult      &result,
    //                  const bvh3f     bvh,
    //                  const Triangle *triangles,
    //                  const vec3f     queryPoint)
    // {
    //   /* as with all cubql traversal routines, we use a lambda that
    //      gets in a per-primitive ID, and that returns a square
    //      dist that the traversal routines can from now on use to
    //      cull (note if the traversal already has a closer culling
    //      dist it will keep on using this no matter what this
    //      function returns, so we don't need to be too sanguine about
    //      which value to return) */
    //   auto perPrim=[&result,queryPoint,triangles](uint32_t primID)->float {
    //     Triangle triangle = triangles[primID];
    //     if (closestPoint(result,triangle,queryPoint))
    //       result.primID = primID;
    //     return result.sqrDist;
    //   };
    //   cuBQL::shrinkingRadiusQuery::forEachPrim(perPrim,bvh,queryPoint,result.sqrDist);
    // } // fcp()
#endif
    
  } // ::cuBQL::triangles
} // ::cuBQL
