// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file closestPointGPU.cu Implements a small demo-app that
  generates a set of data points, another set of query points, and
  then uses cuBQL to perform closest-point qeuries (ie, it finds,
  for each query point, the respectively closest data point */

// cuBQL itself, and the BVH type(s) it defines
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cpu.h"
// some specialized query kernels for find-closest, on 'points' data
#include "cuBQL/queries/pointData/findClosest.h"
// helper class to generate various data distributions
#include "samples/common/Generator.h"
#include <cstring>

using namespace cuBQL;

void computeBoxes(int tid,
                  box3f *d_boxes, const vec3f *d_data, int numData)
{
  d_boxes[tid] = box3f().including(d_data[tid]);
}

void runQueries(int tid,
                bvh3f bvh,
                const vec3f *d_data,
                const vec3f *d_queries,
                int numQueries)
{
  vec3f queryPoint = d_queries[tid];
  int closestID = cuBQL::points::findClosest
    (/* the cubql bvh we've built */
     bvh,
     /* data that this bvh was built over*/
     d_data,
     queryPoint);
  vec3f closestPoint = d_data[closestID];
  printf("[%i] closest point to (%f %f %f) is point #%i, at (%f %f %f)\n",
         tid,
         queryPoint.x,
         queryPoint.y,
         queryPoint.z,
         closestID,
         closestPoint.x,
         closestPoint.y,
         closestPoint.z);
}


int main(int, char **)
{
  int numDataPoints = 10000;
  int numQueryPoints = 20;
  /*! generate 10,000 uniformly distributed data points */
  std::vector<vec3f> dataPoints
    = cuBQL::samples::convert<float>
    (cuBQL::samples::UniformPointGenerator<3>().generate(numDataPoints,290374));
  std::cout << "#cubql: generated " << dataPoints.size() << " data points" << std::endl;
  std::vector<vec3f> queryPoints
    = cuBQL::samples::convert<float>
    (cuBQL::samples::UniformPointGenerator<3>().generate(numQueryPoints,/*seed*/1234567));
  std::cout << "#cubql: generated " << queryPoints.size() << " query points" << std::endl;

  vec3f *d_queryPoints = 0;
  vec3f *d_dataPoints = 0;
  box3f *d_primBounds = 0;
  (void *&)d_queryPoints = malloc(queryPoints.size()*sizeof(vec3f));
  memcpy(d_queryPoints,queryPoints.data(),
         queryPoints.size()*sizeof(queryPoints[0]));
  (void *&)d_dataPoints = malloc(dataPoints.size()*sizeof(vec3f));
  memcpy(d_dataPoints,dataPoints.data(),
         dataPoints.size()*sizeof(dataPoints[0]));
  (void *&)d_primBounds = malloc(dataPoints.size()*sizeof(box3f));
  for (int tid=0;tid<numDataPoints;tid++)
    computeBoxes//<<<divRoundUp(numDataPoints,128),128>>>
      (tid,d_primBounds,d_dataPoints,numDataPoints);
  
  // generate cuBQL bvh
  bvh3f bvh;
  cuBQL::cpuBuilder(bvh,d_primBounds,numDataPoints,BuildConfig());
  for (int tid=0;tid<numQueryPoints;tid++)
                   runQueries //<<<divRoundUp(numQueryPoints,128),128>>>
                   (tid,bvh,d_dataPoints,d_queryPoints,numQueryPoints);
  
  return 0;
}
 
