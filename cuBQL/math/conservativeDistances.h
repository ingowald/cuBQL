// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

/*! \file cuBQL/math/conservativeDistances.h Implements various helper functions to compute distances with well defined conservative rounding modes.

  When traversing BVHes we often need to compute distances between,
  say, a query point and a given subtree's bounding box. Based on the
  type of data used, this can lead to all kind of numerical issues, in
  particular because of the squares used in the dot product used in L2
  distance computations. To mitigate this we define some helper
  functions (defined in this header file) that will always - no matter
  what input data type - compute distances in floats, and use well
  defined rounding modes that return conservative values. Ie, the
  `float fSqrDistance_rd(vec2i, box2i)` will return a float that is
  guaranteed to be _smaller or equal to_ than whatever the correct
  number would have been */
#pragma once

#include "cuBQL/math/box.h"

namespace cuBQL {

  /*! convert a _positive_ int64 to float, with round-down */
  inline __cubql_both float toFloat_pos_rd(int64_t v)
  {
#ifdef __CUDA_ARCH__
    return __ll2float_rd(v);
#else
    float f = (float)v;
    if ((int64_t)f >= v) f = nextafter(f,CUBQL_INF);
    return f;
#endif
  }
  
  /*! convert a _positive_ double to float, with round-down */
  inline __cubql_both float toFloat_pos_rd(double v)
  {
#ifdef __CUDA_ARCH__
    return __double2float_rd(v);
#else
    float f = (float)v;
    if ((double)f >= v) f = nextafter(f,CUBQL_INF);
    return f;
#endif
  }
  
  // ------------------------------------------------------------------
  inline __cubql_both float fSquare_rd(float v)
  { return v*v; }

  inline __cubql_both float fSquare_rd(double v)
  { return toFloat_pos_rd(v*v); }

  inline __cubql_both float fSquare_rd(int v)
  { return toFloat_pos_rd(v*(int64_t)v); }
  
  inline __cubql_both float fSquare_rd(int64_t v)
  { float f = toFloat_pos_rd(v < 0 ? -v : v); return f*f; }

  // ------------------------------------------------------------------
  template<typename T, int D>
  inline __cubql_both float fSqrLength_rd(vec_t<T,D> v)
  {
    float sum = 0.f;
#pragma unroll
    for (int i=0;i<D;i++)
      sum += fSquare_rd(v[i]);
    return sum;
  }

  template<typename T>
  inline __cubql_both float fSqrLength_rd(vec_t<T,2> v)
  { return fSquare_rd(v.x)+fSquare_rd(v.y); }
  
  template<typename T>
  inline __cubql_both float fSqrLength_rd(vec_t<T,3> v)
  { return fSquare_rd(v.x)+fSquare_rd(v.y)+fSquare_rd(v.z); }

  template<typename T>
  inline __cubql_both float fSqrLength_rd(vec_t<T,4> v)
  { return fSquare_rd(v.x)+fSquare_rd(v.y)+fSquare_rd(v.z)+fSquare_rd(v.w); }
  
  
  // ------------------------------------------------------------------
  template<typename T, int D>
  inline __cubql_both float fSqrDistance_rd(vec_t<T,D> a, vec_t<T,D> b)
  { return fSqrLength_rd(a-b); }

  /*! distance of point to box */
  template<typename T, int D>
  inline __cubql_both float fSqrDistance_rd(vec_t<T,D> a, box_t<T,D> b)
  { return fSqrDistance_rd(a,project(b,a)); }

  /*! distance of point to box */
  template<typename T, int D>
  inline __cubql_both float fSqrDistance_rd(box_t<T,D> b, vec_t<T,D> a)
  { return fSqrDistance_rd(a,project(b,a)); }

  /*! box-box distance */
  template<typename T, int D>
  inline __cubql_both float fSqrDistance_rd(box_t<T,D> a, box_t<T,D> b)
  {
    vec_t<T,D> lo = max(a.lower,b.lower);
    vec_t<T,D> hi = min(a.upper,b.upper);
    vec_t<T,D> diff = max(vec_t<T,D>((T)0),lo-hi);
    return fSqrLength_rd(diff);
  }
  
}
