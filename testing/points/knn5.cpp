// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "knn5.h"
#include "cuBQL/builder/cpu.h"

namespace testing {

  void computeBoxes(box_t *d_boxes,
                    const point_t *d_data,
                    int numData)
  {
    for (int tid=0; tid<numData; tid++)
      d_boxes[tid] = box_t().including(d_data[tid]);
  }
      
  bvh_t computeBVH(const box_t *d_boxes, int numBoxes)
  {
    bvh_t bvh;
    cuBQL::cpu::spatialMedian(bvh,d_boxes,numBoxes,BuildConfig());
    return bvh;
  }
      
  void launchQueries(bvh_t bvh,
                     const point_t  *d_data,
                     float      *d_results,
                     const point_t *d_queries,
                     int            numQueries)
  {
    for (int tid=0;tid<numQueries;tid++)
      d_results[tid] = runQuery(bvh,d_data,d_queries[tid]);
  }

  void free(bvh_t bvh)
  { cuBQL::cpu::freeBVH(bvh); }

} // ::testing

int main(int ac, char **av)
{
  cuBQL::testRig::HostDevice device;
  testing::main(ac,av,device);
  return 0;
}
