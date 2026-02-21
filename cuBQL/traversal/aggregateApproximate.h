// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/bvh.h"
#include <cuBQL/math/vec.h>
#include <cuBQL/math/box.h>
#include <cuBQL/math/affine.h>
#include <cuBQL/math/conservativeDistances.h>

/* Defines and implements cuBQL "approximate/aggregate" style
   traversals that can, for example, be used for N-body style
   problems.

   The core idea of these types of queries is that the user provides
   three things:

   - one, some per-subtree 'aggregate data' (of the user's choosing,
     and computed, for example, via refit_aggregate()). For an n-body
     style problem this could, for example, be the sum of all
     planets/bodies/masses in a subtree.

   - second, a callback function that checks if a given query can be
     approximately fulfilled with the subtree's aggregate data; i.e.,
     _without_ having to traverse that subtree's children. If so, this
     helper function can accumulate this partial result (in whatever
     way it chooses - it's user code, after all), and returns 'true'
     to tell cuBQL that this subtree is 'done' and does not require
     further processing. Otherwise, it returns 'false' and cuBQL will
     process the children

   - third, a second callback function that operates on individual
     primitmives, and gets called by cuBQL if traversal reaches a leaf
     without ever having decided to approximate in any of that child
     dnoe's parent nodes

   Obviously both callback functions need additional data to do their
   job: the bvh to be traversed (eg to get a node's bounding box), the
   (tempalted) aggregate data (obviously), the (templated) query_t for
   which the query is performed, and some (templated) result_t in
   which both callbacks can accumulate their partial results (eg for
   an n-body style, this could be the sum of all forces)

   Note that "approximate/aggregate" refers to the two key concepts
   required to realize these kind of traversals: the idea to avoid a
   "full" tree traversal by "approimating" certain subtrees (instead
   of just traversing both children); and the idea that one needs some
   sort of "aggregate data" for a subtree to even decide whether
   that's possible or not.
*/ 
namespace cuBQL {
  namespace aggregateApproximate {

    /*! implements a approximate/aggregate traversal (see above for
        the core idea). Note this function is heavily templated, so to
        allow template matchign to do its magic the order of
        parameters is pretty important.

        agg
    */
    template
    </* T/D that describes the BVH data/dimensionality */
      typename T, int D,
      typename aggregateNodeData_t,
      typename primitive_t,
      typename result_t,
      typename query_t,
      typename approximateSubtreeFct_t,
      typename perPrimFct_t>
    inline __device__
    void traverse(bvh_t<T,D> bvh,
                  aggregateNodeData_t aggregateData[],
                  primitive_t primitives[],
                  result_t result,
                  query_t queryPoint,
                  const approximateSubtreeFct_t &approximateSubtreeFct,
                  const perPrimFct_t perPrimFct);
  }
}
  
