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

#include "closestPoint.h"
#include "cuBQL/builder/cuda.h"

namespace cuBQL {
  namespace points {
    namespace closestPoint {

      __global__
      void computeBox(box_t *d_boxes, const data_t *d_data, int numData)
      {
        int tid = threadIdx.x+blockIdx.x*blockDim.x;
        if (tid >= numData) return;

        d_boxes[tid] = box_t().including(d_data[tid]);
      }
      
      void computeBoxes(box_t *d_boxes,
                        const data_t *d_data,
                        int numData)
      {
        computeBox<<<divRoundUp(numData,128),128>>>(d_boxes,d_data,numData);
        CUBQL_CUDA_SYNC_CHECK();
      }
      
      bvh_t computeBVH(const box_t *d_boxes,
                       int numBoxes)
      {
        bvh_t bvh;
        cuBQL::gpuBuilder(bvh,d_boxes,numBoxes,BuildConfig());
        return bvh;
      }

      void launchQueries(bvh_t bvh,
                         const data_t  *d_data,
                         result_t      *d_results,
                         const query_t *d_queries,
                         int            numQueries)
      {
        int tid = threadIdx.x+blockIdx.x*blockDim.x;
        if (tid >= numQueries) return;

        d_results[tid] = runQuery(bvh,d_data,d_queries[tid]);
      }

      void free(bvh_t bvh)
      { cuBQL::cuda::free(bvh); }
    }
  }
}
int main(int ac, char **av)
{
  cuBQL::testRig::CUDADevice device;
  cuBQL::points::closestPoint::main(ac,av,device);
  return 0;
}