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

  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */direction>
  struct AxisAlignedRay {
    vec3f origin;
    float length;

    inline __cubql_both Ray makeRay() const;
  };

  // =============================================================================
  // *** IMPLEMENTATION ***
  // =============================================================================

  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */direction>
  inline __cubql_both Ray AxisAlignedRay<axis,direction>::makeRay() const
  {
    vec3f D {
      (axis == 0) ? (axis > 0 ? +1.f : -1.f) : 0.f,
      (axis == 1) ? (axis > 0 ? +1.f : -1.f) : 0.f,
      (axis == 2) ? (axis > 0 ? +1.f : -1.f) : 0.f
    };
    return { origin, 0.f, origin + D, length };
  }
  
}
