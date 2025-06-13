// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "buildBench.h"
#include "cuBQL/builder/cpu.h"

namespace testing {

  void computeBoxes(box_t *d_boxes,
                    const point_t *d_data,
                    int numData)
  {
    for (int tid=0; tid<numData; tid++)
      d_boxes[tid] = box_t().including(d_data[tid]);
  }
      
  bvh_t computeBVH(const box_t *d_boxes,
                   int numBoxes,
                   BuildType buildType)
  {
    if (buildType != BUILDTYPE_DEFAULT)
      std::cout << "#warning: host builder doesn't support this build type, falling back to default" << std::endl;
    bvh_t bvh;
    cuBQL::cpu::spatialMedian(bvh,d_boxes,numBoxes,BuildConfig());
    return bvh;
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
