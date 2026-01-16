// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <omp.h>

namespace cuBQL {

  /*! openmp based builder with #pragma omp target directives. */
  template<typename T, int D>
  void build_omp_target(BinaryBVH<T,D>   &bvh,
                        /*! array of bounding boxes to build BVH over,
                          must be in target device memory (ie, must be
                          accessible in the device(gpuID) that the
                          'gpuID' parameter refers to */
                        const box_t<T,D> *d_boxes,
                        uint32_t          numBoxes,
                        BuildConfig       buildConfig=BuildConfig(),
                        int               gpuID = 0);
}
#if CUBQL_OPENMP_BUILDER_IMPLEMENTATION
# include "openmp/build_omp_target.h"
#endif


