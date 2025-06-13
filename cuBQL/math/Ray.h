// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"

namespace cuBQL {

  // =============================================================================
  // *** INTERFACE ***
  // =============================================================================
  
  struct Ray {
    vec3f origin;
    float tmin;
    vec3f direction;
    float tmax;
  };

  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  struct AxisAlignedRay {
    __cubql_both AxisAlignedRay(const vec3f origin);
    __cubql_both AxisAlignedRay(const vec3f origin, float tmin, float tmax);
    
    vec3f origin;
    float tmin=0.f, tmax=CUBQL_INF;

    inline __cubql_both vec3f direction() const;
    inline __cubql_both Ray   makeRay() const;
  };

  // =============================================================================
  // *** IMPLEMENTATION ***
  // =============================================================================

  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both
  AxisAlignedRay<axis,sign>::AxisAlignedRay(const vec3f origin,
                                            float tmin, float tmax)
    : origin(origin), tmin(tmin), tmax(tmax)
  {}
  
  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both
  AxisAlignedRay<axis,sign>::AxisAlignedRay(const vec3f origin)
    : origin(origin)
  {}
  
  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both vec3f AxisAlignedRay<axis,sign>::direction() const
  {
    return {
      (axis == 0) ? (sign > 0 ? +1.f : -1.f) : 0.f,
      (axis == 1) ? (sign > 0 ? +1.f : -1.f) : 0.f,
      (axis == 2) ? (sign > 0 ? +1.f : -1.f) : 0.f
    };
  }
  
  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both Ray AxisAlignedRay<axis,sign>::makeRay() const
  {
    return { origin, tmin, direction(), tmax };
  }

  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both dbgout operator<<(dbgout o, AxisAlignedRay<axis,sign> ray)
  {
    o << "AARay<"<<axis<<","<<sign<<">("<<ray.origin<<",["<<ray.tmin<<","<<ray.tmax<<"])";
    return o;
  }

  inline __cubql_both dbgout operator<<(dbgout o, Ray ray)
  {
    o << "Ray{"<<ray.origin<<"+["<<ray.tmin<<","<<ray.tmax<<"]*"<<ray.direction<<"}";
    return o;
  }

}



