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
    
    // N = normalize(N);
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




  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both
  bool intersectsTriangle(AxisAlignedRay<axis,sign> ray,
                          Triangle triangle,
                          bool dbg=false)
  {
    using cuBQL::dot;
    using cuBQL::cross;

    const vec3f lineOrigin = ray.origin;
    vec3f a = triangle.a;
    vec3f b = triangle.b;
    vec3f c = triangle.c;
    
    // transform triangle into space centered aorund line origin
    a = a - lineOrigin;
    b = b - lineOrigin;
    c = c - lineOrigin;
    // compute normal, for plane equation
    vec3f n = triangle.normal();

    // create horitonzal semi-infite "ray" from origin=0 alone x axis
    const vec3f org = vec3f(0.f);
    const vec3f dir = ray.direction();
    const vec3f end = ray.length * ray.direction();

    bool planeEq_org = dot(org - a, n);
    bool planeEq_end = dot(end - a, n);

    bool bothOnSameSide = planeEq_org * planeEq_end > 0.f;
    if (!bothOnSameSide)
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
  
}

