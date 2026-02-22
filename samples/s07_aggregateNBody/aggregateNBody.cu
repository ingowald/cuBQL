// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file aggregateNBodya.cu Provides a mini-sample for how to use the
    "aggregate refit" and "nbody-style traversal" concepts, in the
    example of a simplified n-body problem */

// cuBQL itself, and the BVH type(s) it defines
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
// some specialized query kernels for find-closest, on 'points' data
#include "cuBQL/queries/pointData/findClosest.h"
// helper class to generate various data distributions
#include "samples/common/Generator.h"

// pull in ability to refit aggregate data onto a BVH:
#include "cuBQL/builder/cuda/refit_aggregate.h"
// pull in traversal that can cull based on that aggregate data:
#include "cuBQL/traversal/aggregateApproximate.h"

using namespace cuBQL;

__global__
void computeBoxes(box3f *d_boxes, const vec3f *d_data, int numData)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numData) return;

  d_boxes[tid] = box3f().including(d_data[tid]);
}

namespace nBody {
  /*! the data we want cuBQL to store for us in each subtree. For this
      sample we simply store the total number of points in each
      subtree. A real n-body code might want to track more or other
      data (like sum of all masses, etc) - if so, just change this */
  struct AggregateNodeData {
    int numBodiesInSubtree;
  };

  /*! aggregation function that computes a node's aggregate data
    during aggragate_refit */
  // inline
  __device__
  void aggregate(bvh3f bvh,
                 AggregateNodeData nodeAggregates[],
                 int nodeID)
  {
    auto node = bvh.nodes[nodeID].admin;
    if (node.count != 0) {
      // this is a leaf - aggregate data is the leaf count itself
      nodeAggregates[nodeID].numBodiesInSubtree = node.count;
    } else {
      // this is a inner node - aggregate data is the sum of both
      // children. note that aggragate_refit() guarantees that both
      // children have alrady been aggregated before this gets called
      nodeAggregates[nodeID].numBodiesInSubtree
        = nodeAggregates[node.offset+0].numBodiesInSubtree
        + nodeAggregates[node.offset+1].numBodiesInSubtree;
    }
  }

  typedef void (*AggregateNodeFctPtr)(bvh3f, AggregateNodeData *, int);
  __global__ 
  void k_get_aggregate(AggregateNodeFctPtr *d_result)
  {
    if (threadIdx.x != 0) return;
    *d_result = aggregate;
  }
  
  AggregateNodeFctPtr  get_aggregate()
  {
    AggregateNodeFctPtr result = 0;
    AggregateNodeFctPtr *d_resultPtr = 0;
    CUBQL_CUDA_CALL(Malloc((void**)&d_resultPtr,
                           sizeof(AggregateNodeFctPtr)));
    k_get_aggregate<<<1,32>>>(d_resultPtr);
    CUBQL_CUDA_CALL(Memcpy((void*)&result,(void*)d_resultPtr,
                           sizeof(void*), cudaMemcpyDefault));
    return result;
  }
  
  /*! the final result type that results from iterating over the
      entire tree. FOr our simple n-body mock-up, it's simply a float
      to track sum_i{1/sqrDistance(queryPoint,dataPoint[i])} */
  struct ResultType {
    float sumOfForces;
  };

  /*! callback function that checks if a subtree can be approximated;
      if so it accumulates the approximated result and returns true
      ('yes, i could approximate this subtree); otherwise it returns
      false and let's cuBQL traverse to the children */
  inline __device__
  bool approximateSubtree(/* param #1: the actual BVH */
                          bvh3f bvh,
                          /* param #2: the pre-computed per-node
                             aggregate data */
                          AggregateNodeData nodeAggregates[],
                          /*! param #3: the subtree we're supposed to
                            evaluate (specified by its node ID) */
                          int nodeID,
                          /*! param #4: the user-supplied struct where
                              we can accumulate the approximated
                              subtree's partial result in (if we chose
                              to do so) */
                          ResultType &result,
                          /*! param #5 the actual query that this
                              traversal is run on */
                          const vec3f &queryPoint)
  {
    auto node = bvh.nodes[nodeID];
    // first, check if we can approximate this subtree
    float sqrDist = sqrDistance(node.bounds,queryPoint);
    float sqrDiag = sqrLength(node.bounds.size());
    const float approxThreshold = /* 1% of fource*/0.01f;
    bool canApproximate
      = sqrDist > 0.f
      && (sqrDiag / sqrDist <= approxThreshold);

    if (canApproximate) {
      result.sumOfForces
        += nodeAggregates[nodeID].numBodiesInSubtree
        *  (1.f/sqrDist);
      /* yes we DID approximate the subtree (cuBQL doesn't need to
         traverse any more*/
      return true;
    }

    return false;
  }

  inline __device__
  void processPrim(/*! the query's final result type */
                   ResultType &result,
                   /*! the query's actual query point */
                   const vec3f &queryPoint,
                   int primID,
                   const vec3f prims[])
  {
    float sqrDist = sqrLength(prims[primID]-queryPoint);
    if (sqrDist != 0.f)
      result.sumOfForces += 1.f/sqrDist;
  }
}


__global__
void runQueries(float *d_results,
                bvh3f bvh,
                nBody::AggregateNodeData nodeAggregates[],
                const vec3f *d_data,
                const vec3f *d_queries,
                int numQueries)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;
  
  vec3f queryPoint = d_queries[tid];
  nBody::ResultType result = { 0.f };
  /* run a cuBQL "approximate/aggregate" traversal, which requires two
     callbacks: one that tries to approximate a subtree (using
     pre-aggregated data), and one that processes individual
     primitives (if traversal went all the way to leaves */
  cuBQL::aggregateApproximate::traverse
    (/* all the per-tree input data: */
     bvh,nodeAggregates,d_data,
     /* all the per-query state data */
     result,queryPoint,
     /* and the two callbacks */
     nBody::approximateSubtree,
     nBody::processPrim
     );
  d_results[tid] = result.sumOfForces;
}


int main(int, char **)
{
  int numDataPoints = 100000;
  /*! generate 10,000 uniformly distributed data points */
  std::vector<vec3f> dataPoints
    = cuBQL::samples::convert<float>
    (cuBQL::samples::UniformPointGenerator<3>()
     .generate(numDataPoints,290374));
  std::cout << "#cubql: generated " << dataPoints.size()
            << " data points" << std::endl;

  vec3f *d_dataPoints = 0;
  box3f *d_primBounds = 0;
  CUBQL_CUDA_CALL(Malloc((void **)&d_dataPoints,
                         numDataPoints*sizeof(*d_dataPoints)));
  CUBQL_CUDA_CALL(Memcpy((void *)d_dataPoints,dataPoints.data(),
                         numDataPoints*sizeof(*d_dataPoints),
                         cudaMemcpyDefault));
  
  CUBQL_CUDA_CALL(Malloc((void **)&d_primBounds,
                         numDataPoints*sizeof(box3f)));
  computeBoxes<<<divRoundUp(numDataPoints,128),128>>>
    (d_primBounds,d_dataPoints,numDataPoints);

  // ------------------------------------------------------------------
  // generate initial cuBQL bvh over the data points
  // ------------------------------------------------------------------
  bvh3f bvh;
  cuBQL::gpuBuilder(bvh,d_primBounds,numDataPoints,BuildConfig(8));
  CUBQL_CUDA_SYNC_CHECK();
  
  // ------------------------------------------------------------------
  // re-fit kernel-specific aggregate data on top of the bvh
  // ------------------------------------------------------------------
  nBody::AggregateNodeData *d_nodeAggregates = 0;
  CUBQL_CUDA_CALL(Malloc((void **)&d_nodeAggregates,
                         bvh.numNodes*sizeof(*d_nodeAggregates)));
  CUBQL_CUDA_SYNC_CHECK();
  cuBQL::cuda::refit_aggregate(bvh,
                               d_nodeAggregates,
                               nBody::get_aggregate()
                               // nBody::aggregate
                               );
  CUBQL_CUDA_SYNC_CHECK();
  
  // ------------------------------------------------------------------
  // ready to run query
  // ------------------------------------------------------------------
  float *d_results = 0;
  int numQueryPoints = numDataPoints;
  // int numQueryPoints = std::min(numDataPoints,16*1024);
  CUBQL_CUDA_CALL(Malloc((void **)&d_results,
                         numQueryPoints*sizeof(*d_results)));

  auto d_queryPoints = d_dataPoints;
  runQueries<<<divRoundUp(numQueryPoints,128),128>>>
    (d_results,bvh,d_nodeAggregates,
     d_dataPoints,d_queryPoints,numQueryPoints);
  
  CUBQL_CUDA_SYNC_CHECK();
  return 0;
}
 
