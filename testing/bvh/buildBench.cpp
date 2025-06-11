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
