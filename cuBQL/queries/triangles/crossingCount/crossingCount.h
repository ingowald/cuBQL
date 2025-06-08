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

/*! \file queries/triangles/crossingCount Implement a "ray-triangle
    crossing count" query

    In this query, the data model is a triangle mesh (with a cuBQL BVH
    built over it, obviously), and the query is a list of ray segments
    (given by origin point and direction vector, respectively. The job
    of the query is to perform a 'crossing count', where each ray is
    traced against the triangles, and for every triangle it
    intersects, increases or decreses a given per-ray counter: -1 for
    crossing _into_ a surface (ie, the ray hits the triangle on its
    "front" side), and +1 for every crossing _out of_ the surface (if
    ray intersects triangle's back side).

 */


#pragma once

#include "queries/common/Ray.h"
#include "queries/triangles/TriangleMesh.h"

/*! \namespace cuq - *cu*BQL based geometric *q*ueries */
namespace cuq {
  namespace triangles {

    // =============================================================================
    // *** INTERFACE ***
    // =============================================================================

    /*! defines a crossing count kernel. The struct itself defines the
      return values computed for this query, the 'compute()' method
      provides a device-side implementation of that kernel for a given
      set of inputs */
    struct CrossingCount {
      // ====================== COMPUTED VALUES ======================
      
      /* sum of all ray-triangle crossings, using "-1" for crossing
         _into_ a surface, and "+1" for crossing _out of_ a surface. in
         theory for a closed and properly outside-oriented surface and
         infinite-length query rays a point not inside the object should
         have value 0, no point should ever have values < 0 (because
         that would require a ray to enter an object and never leave
         it), and points inside the object should hae a value of exactly
         1 (because it should cross out exactly once more than it
         crosses in). Caveat: for query rays hitting edges, vertices, or
         just numerically fancy configurations this theory will probably
         not match practice :-/ */
      int crossingCount = 0;
      
      /*! total number of ray-triangle intersections, no matter which
        sign. note this *may* count certain surfaces twice if the ray
        happens to hit on an edge or vertex */
      int totalCount = 0;
      
      // ====================== ACTUAL QUERIES ======================
      
      /*! runs one complete crossing-count query; will compute
          crossing count for every triangle whose bounding box
          intersects the given ray */
      inline __device__
      void runQuery(const cuq::TriangleMesh mesh,
                    const cuBQL::bvh3f      bvh,
                    const cuq::Ray          query);
      
      /*! performs one complete query, starting with an empty CPAT
        result, traversing the BVH for the givne mesh, and processing
        every triangle that needs consideration. This variant will do
        at most maxTOConsider ray-triangle tests, and if this number
        gets exceeded, will return true. Returns true if limit was
        exceeded; false if query finished witout reaching this
        limit */
      inline __device__
      bool runQueryWithLimit(const cuq::TriangleMesh mesh,
                             const cuBQL::bvh3f      bvh,
                             const cuq::Ray          query,
                             /*! maximum number of ray-triangle tests to
                               perform; allowing the user some control */
                             uint32_t maxTrianglesToIntersectWith);
    };

    /*! provides a host-side launch of N parallel crossing-count
      kernels (using one GPU thread per query).

      d_results and d_rays have to be *device* side arrays (and
      non-null), numrays must be greater than 0.
    */
    void crossingCount(/* ----------- output ----------- */
                       CrossingCount           *d_results,
                       /* ----------- inputs ----------- */
                       const cuq::TriangleMesh &mesh,
                       const cuBQL::bvh3f      &bvh,
                       const cuq::Ray          *d_rays,
                       int                      numRays,
                       /*! if non-null, the pointed-to cuda stream
                         will be used to launch that kernel into,
                         and _no_ sync will be issues. If null, we
                         will launch into the default stream, and
                         perform a sync-check after the launch to
                         wait for completion and check for errors */
                       cudaStream_t            *stream = nullptr);
    
    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================

    /* this ifdef allows an application to use this kernel in
       'header-only' form, without having to link the actual
       libcuqbl_triangles_crossingcount. To do so, simply define
       CUBQL_TRIANGLES_CROSSINGCOUNT_IMPLEMENTATION to 1 before
       including this header file (but make sure to do that in only
       one compilation unit */
#if CUBQL_TRIANGLES_CROSSINGCOUNT_IMPLEMENTATION
    inline __device__
    void CrossingCount::compute(const cuq::TriangleMesh mesh,
                                const cuBQL::bvh3f      bvh,
                                const cuq::Ray          query)
    {
    }
    
    __global__ void launch_crossingCount(/* ----------- output ----------- */
                                         CrossingCount    *d_results,
                                         /* ----------- inputs ----------- */
                                         cuq::TriangleMesh mesh,
                                         cuBQL::bvh3f      bvh,
                                         const cuq::Ray   *d_rays,
                                         int               numRays)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numRays) return;
      d_results[tid].compute(mesh,bvh,d_rays[tid]);
    }
    
    /*! provides a host-side launch of N parallel crossing-count
      kernels (using one GPU thread per query).

      d_results and d_rays have to be *device* side arrays
    */
    void crossingCount(/* ----------- output ----------- */
                       CrossingCount           *d_results,
                       /* ----------- inputs ----------- */
                       const cuq::TriangleMesh &mesh,
                       const cuBQL::bvh3f      &bvh,
                       const cuq::Ray          *d_rays,
                       int                      numRays,
                       /*! if non-null, the pointed-to cuda stream
                         will be used to launch that kernel into,
                         and _no_ sync will be issues. If null, we
                         will launch into the default stream, and
                         perform a sync-check after the launch to
                         wait for completion and check for errors */
                       cudaStream_t            *stream)
    {
      int bs = 128;
      int bn = divRoundUp(numRays,bs);
      launch_crossingCount<<<nb,bs,0,stream?*stream:0>>>
        (d_results,mesh,bvh,numRays);
      if (stream == nullptr)
        CUBQL_CUDA_SYNC_CHECK();
    }
#endif
  } // ::cuq::triangles
} // ::cuq
