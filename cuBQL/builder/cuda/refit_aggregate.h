// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/cuda/builder_common.h"

namespace cuBQL {
  namespace cuda {

    template<
      typename T,
      int D,
      typename AggregateNodeData,
      typename AggregateFct>
    void refit_aggregate(bvh_t<T,D> bvh,
                         AggregateNodeData *d_aggregateNodeData,
                         const AggregateFct &aggregateFct);
   
  }
}
