// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"

namespace cuBQL {

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
  };
}
