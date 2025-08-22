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

namespace cuBQL {
  namespace points {

<<<<<<< HEAD:cuBQL/points/fcp.h
    /*! Find closest point, up to (and excluding) provided maximum
        query distance. Return value is index of point in data points
        array, or -1 if none was found. */
=======
    /*! find closest point, up to (and exluding) provided maximum
        query distance. return value is index of point in data points
        array, or -1 if none was found */
>>>>>>> devel:cuBQL/queries/points/old_fcp.h
    inline __device__
    int fcp(const vec3f  queryPoint,
            const bvh3f  bvh,
            const vec3f *dataPoints,
            float        sqrMaxQueryDist=INFINITY);

    /*! Same as regular fcp function, but explicitly _excluding_ the
        primitive index specified in the first parameter. */
    inline __device__
    int fcp_excluding(/*! primitive that will NOT be accepted as fcp
                          point, in case the query point itself is
                          part of the input data set */
                      int          primIDtoIgnore,
                      const vec3f  queryPoint,
                      const bvh3f  bvh,
                      const vec3f *dataPoints,
                      float        sqrMaxQueryDist=INFINITY);


    // ==================================================================
    // implementation
    // ==================================================================
    inline __device__
    int fcp_excluding(int primIDtoIgnore,
                      const vec3f  queryPoint,
                      const bvh3f  bvh,
                      const vec3f *dataPoints,
                      float        maxQueryDistSquare)
    {
      using node_t = typename bvh3f::Node;
      int result = -1;
      
      int2 stackBase[32], *stackPtr = stackBase;
      int nodeID = 0;
      int offset = 0;
      int count  = 0;
      while (true) {
        while (true) {
          offset = bvh.nodes[nodeID].admin.offset;
          count  = bvh.nodes[nodeID].admin.count;
          if (count>0)
            // leaf
            break;
          const node_t child0 = bvh.nodes[offset+0];
          const node_t child1 = bvh.nodes[offset+1];
          float dist0 = fSqrDistance(child0.bounds,queryPoint);
          float dist1 = fSqrDistance(child1.bounds,queryPoint);
          int closeChild = offset + ((dist0 > dist1) ? 1 : 0);
          if (dist1 < maxQueryDistSquare) {
            float dist = max(dist0,dist1);
            int distBits = __float_as_int(dist);
            *stackPtr++ = make_int2(closeChild^1,distBits);
          }
          if (min(dist0,dist1) > maxQueryDistSquare) {
            count = 0;
            break;
          }
          nodeID = closeChild;
        }
        for (int i=0;i<count;i++) {
          int primID = bvh.primIDs[offset+i];
          if (primID == primIDtoIgnore) continue;
          
          float dist2 = sqrDistance(queryPoint,dataPoints[primID]);
          if (dist2 >= maxQueryDistSquare) continue;
          maxQueryDistSquare = dist2;
          result             = primID;
        }
        while (true) {
          if (stackPtr == stackBase) 
            return result;
          --stackPtr;
          if (__int_as_float(stackPtr->y) > maxQueryDistSquare) continue;
          nodeID = stackPtr->x;
          break;
        }
      }
    }

    inline __device__
    int fcp(const vec3f  queryPoint,
            const bvh3f  bvh,
            const vec3f *dataPoints,
            float        maxQueryDistSquare)
    {
      return fcp_excluding(-1,queryPoint,bvh,dataPoints,maxQueryDistSquare);
    }
    
  } // ::cuBQL::points
} // ::cuBQL

