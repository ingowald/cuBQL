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
#include "cuBQL/math/Ray.h"

namespace cuBQL {

  // =============================================================================
  // *** INTERFACE ***
  // =============================================================================
  
  struct RayTriangleIntersection {
    vec3f N;
    float t,u,v;
    
    inline __cubql_both bool compute(Ray ray, Triangle tri);
  };

  
  // =============================================================================
  // *** IMPLEMENTATION ***
  // =============================================================================

  inline __cubql_both
  bool RayTriangleIntersection::compute(Ray ray, Triangle tri)
  {
    const vec3f v0 = tri.a;
    const vec3f v1 = tri.b;
    const vec3f v2 = tri.c;
    
    const vec3f e1 = v1-v0;
    const vec3f e2 = v2-v0;

    N = cross(e1,e2);
    if (N == vec3f(0.f)) return false;
    
    N = normalize(N);
    if (fabsf(dot(ray.direction,N)) < 1e-12f) return false;
    
    // P = o+td
    // dot(P-v0,N) = 0
    // dot(o+td-v0,N) = 0
    // dot(td,N)+dot(o-v0,N)=0
    // t*dot(d,N) = -dot(o-v0,N)
    // t = -dot(o-v0,N)/dot(d,N)
    t = -dot(ray.origin-v0,N)/dot(ray.direction,N);
    
    if (t < ray.tmin || t > ray.tmax) return false;
    
    vec3f P = (ray.origin - v0) + t*ray.direction;
    
    float e1u,e2u,Pu;
    float e1v,e2v,Pv;
    if (fabsf(N.x) >= max(fabsf(N.y),fabsf(N.z))) {
      e1u = e1.y; e2u = e2.y; Pu = P.y;
      e1v = e1.z; e2v = e2.z; Pv = P.z;
    } else if (fabsf(N.y) > fabsf(N.z)) {
      e1u = e1.x; e2u = e2.x; Pu = P.x;
      e1v = e1.z; e2v = e2.z; Pv = P.z;
    } else {
      e1u = e1.x; e2u = e2.x; Pu = P.x;
      e1v = e1.y; e2v = e2.y; Pv = P.y;
    }
    auto det = [](float a, float b, float c, float d) -> float
    { return a*d - c*b; };
    
    // P = v0 + u * e1 + v * e2 + h * N
    // (P-v0) = [e1,e2]*(u,v,h)
    if (det(e1u,e1v,e2u,e2v) == 0.f) return false;
    
    u = det(Pu,e2u,Pv,e2v)/det(e1u,e2u,e1v,e2v);
    v = det(e1u,Pu,e1v,Pv)/det(e1u,e2u,e1v,e2v);

    if ((u < 0.f) || (v < 0.f) || ((u+v) > 1.f)) return false;
    
    return true;
  }

}

