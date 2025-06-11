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

/*! \file check.cu builds all BVH variants we can, and runs some
    sanity and quality checks on the result */

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/builder/cpu.h"
#include "samples/common/CmdLine.h"
#include "samples/common/IO.h"
#include "testing/common/testRig.h"
#include "samples/common/Generator.h"
#include <set>
#include "cuBQL/traversal/shrinkingRadiusQuery.h"
#include <cuda.h>

namespace testing {

  using namespace cuBQL;
  using namespace cuBQL::samples;


  template<typename T>
  inline double costEstimate(box_t<T,2> b)
  {
    vec_t<double,2> dim = vec_t<double,2>(b.upper - b.lower);
    return dim.x + dim.y;
  }
  
  template<typename T>
  inline double costEstimate(box_t<T,3> b)
  {
    vec_t<double,3> dim = vec_t<double,3>(b.upper - b.lower);
    return dim.x*dim.y + dim.x*dim.z + dim.y*dim.z;
  }
  template<typename T>
  inline double costEstimate(box_t<T,4> b)
  {
    vec_t<double,4> dim = vec_t<double,4>(b.upper - b.lower);
    return
      dim.x*dim.y + dim.x*dim.z + dim.x*dim.w
      + dim.y*dim.z + dim.z*dim.w
      + dim.z*dim.w;
  }

    template<typename T, int D> bool isFloat3() { return false; };
    template<> bool isFloat3<float,3>() { return true; }
    

  template<typename T, int D>
  struct TreeChecker {
    using box_t   = cuBQL::box_t<T,D>;
    using vec_t   = cuBQL::vec_t<T,D>;
    using bvh_t   = cuBQL::bvh_t<T,D>;
    using node_t  = typename bvh_t::Node;

    TreeChecker(const std::vector<node_t> &nodes,
                const std::vector<int>    &primIDs,
                const std::vector<box_t>  &boxes)
      : nodes(nodes), primIDs(primIDs), boxes(boxes)
    {
      traverse(0);
      if (primIDsFound.size() != primIDs.size())
        throw std::runtime_error("some prims not found!?");
    }

    box_t traverse(int nodeID)
    {
      node_t node = nodes[nodeID];
      box_t bounds;
      if (node.admin.count) {
        for (int i=0;i<node.admin.count;i++) {
          int primID = primIDs[node.admin.offset+i];
          primIDsFound.insert(primID);
          bounds.extend(boxes[primID]);
        }
      } else {
        box_t lBounds = traverse(node.admin.offset+0);
        box_t rBounds = traverse(node.admin.offset+1);
        bounds.extend(lBounds);
        bounds.extend(rBounds);
      }
      if (node.bounds != bounds) {
        throw std::runtime_error("boxes do not match");
      }
      return bounds;
    }
    
    int numNodesFound = 0;
    std::set<int> primIDsFound;
    const std::vector<node_t> &nodes;
    const std::vector<int> &primIDs;
    const std::vector<box_t>    &boxes;
  };

  template<typename T, int D>
  __global__ void runQueries(uint64_t *d_numNodesVisited,
                             uint64_t *d_numPrimsVisited,
                             BinaryBVH<T,D> bvh,
                             const box_t<T,D> *boxes,
                             int numPoints)
  {
    using box_t   = cuBQL::box_t<T,D>;
    using vec_t   = cuBQL::vec_t<T,D>;
    using bvh_t   = cuBQL::bvh_t<T,D>;
    using node_t  = typename bvh_t::Node;
    
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPoints) return;

    vec_t queryPoint = boxes[tid].lower;
    uint64_t numNodesVisited = 0;
    uint64_t numPrimsVisited = 0;
    auto nodeDist = [&](const node_t &node) -> float
    {
      numNodesVisited++;
      return fSqrDistance_rd(queryPoint,node.bounds);
    };
    auto primCode = [&](uint32_t primID) {
      numPrimsVisited++;
      vec_t point = boxes[primID].lower;
      return fSqrDistance_rd(queryPoint,point);
    };
    shrinkingRadiusQuery::forEachPrim(primCode,nodeDist,bvh);
    atomicAdd((unsigned long long int *)d_numNodesVisited,
              (unsigned long long int)numNodesVisited);
    atomicAdd((unsigned long long int *)d_numPrimsVisited,
              (unsigned long long int)numPrimsVisited);
  }
                             
  template<typename T, int D>
  struct Checker {
    using vecND   = cuBQL::vec_t<double,D>;
    using box_t   = cuBQL::box_t<T,D>;
    using vec_t   = cuBQL::vec_t<T,D>;
    using bvh_t   = cuBQL::bvh_t<T,D>;
    using node_t  = typename bvh_t::Node;

    Checker(const std::vector<vecND> &doublePoints)
      : points(convert<T,D>(doublePoints))
    {
      //srand48(290374);
      box_t bbox;
      for (auto point : points)
        bbox.grow(point);
      float halfBoxScale = reduce_max(bbox.size()) * .5f / powf(doublePoints.size(),1./D);
       
      for (auto point : points) {
        vec_t halfBoxSize = halfBoxScale;
//        for (int i=0;i<D;i++)
          //halfBoxSize[i] *= drand48();
        boxes.push_back({point-halfBoxSize,point+halfBoxSize});
      }
      CUBQL_CUDA_CALL(Malloc((void **)&d_boxes,boxes.size()*sizeof(boxes[0])));
      CUBQL_CUDA_CALL(Memcpy((void*)d_boxes,boxes.data(),boxes.size()*sizeof(boxes[0]),
                             cudaMemcpyDefault));
    }
    
    ~Checker()
    {
      CUBQL_CUDA_CALL(Free(d_boxes));
      d_boxes = 0;
    }

    double computeSAH_rec(const std::vector<node_t> &nodes,
                          const std::vector<int>    &primIDs,
                          int nodeID)
    {
      auto node = nodes[nodeID];
      double sum
        = costEstimate(node.bounds)*(1+node.admin.count);
      // double sum
      //   = node.admin.count
      //   ? costEstimate(node.bounds)*(1+node.admin.count)
      //   : 0.;
      if (node.admin.count == 0) {
        sum += computeSAH_rec(nodes,primIDs,node.admin.offset+0);
        sum += computeSAH_rec(nodes,primIDs,node.admin.offset+1);
      }
      return sum;
    }
    inline double computeSAH(const std::vector<node_t> &nodes,
                             const std::vector<int>    &primIDs)
    {
      return computeSAH_rec(nodes,primIDs,0) / costEstimate(nodes[0].bounds);
    }

    void runQuery(const std::string &description)
    {
      uint64_t *p_numNodesVisited = 0;
      uint64_t *p_numPrimsVisited = 0;
      CUBQL_CUDA_CALL(Malloc((void **)&p_numNodesVisited,sizeof(uint64_t)));
      CUBQL_CUDA_CALL(Malloc((void **)&p_numPrimsVisited,sizeof(uint64_t)));
      CUBQL_CUDA_CALL(Memset(p_numNodesVisited,0,sizeof(uint64_t)));
      CUBQL_CUDA_CALL(Memset(p_numPrimsVisited,0,sizeof(uint64_t)));
      int numPoints = (int)points.size();
      runQueries<T,D>
        <<<divRoundUp(numPoints,128),128>>>
        (p_numNodesVisited,
         p_numPrimsVisited,
         bvh,d_boxes,numPoints);
      uint64_t numNodesVisited;
      uint64_t numPrimsVisited;
      CUBQL_CUDA_CALL(Memcpy(&numNodesVisited,p_numNodesVisited,sizeof(numNodesVisited),
                             cudaMemcpyDefault));
      CUBQL_CUDA_CALL(Memcpy(&numPrimsVisited,p_numPrimsVisited,sizeof(numPrimsVisited),
                             cudaMemcpyDefault));
      printf("  --> num visits %20s : \tnodes %8s \tprims %8s\n",
             description.c_str(),
             prettyNumber(numNodesVisited).c_str(),
             prettyNumber(numPrimsVisited).c_str());
      // std::cout << " --> num NODES visited " << numNodesVisited << std::endl;
      // std::cout << " --> num PRIMS visited " << numPrimsVisited << std::endl;
      CUBQL_CUDA_CALL(Free(p_numNodesVisited));
      CUBQL_CUDA_CALL(Free(p_numPrimsVisited));
    }
                  
    template<
      typename runBuilderT,
      typename freeT,
      typename downloadT>
    void check(const runBuilderT &runBuilder,
               const freeT       &freeBVH,
               const downloadT   &download,
               const std::string &description,
               bool runQueryAsWell=true)
    {
      // std::cout << "# ----------------------- " << description << " ----------------------------"
      //           << std::endl;
      // std::cout << "# ...building '" << description << "'" << std::endl;
      runBuilder();

      if (runQueryAsWell) {
        runQuery(description);
      }
      
      // std::cout << "# ...downloading nodes" << std::endl;
      std::vector<typename bvh_t::Node> nodes;
      std::vector<int> primIDs;
      download(nodes,primIDs);
      // std::cout << "# ...freeing BVH" << std::endl;
      freeBVH();
      // std::cout << "# ...computing SAH cost\t\t\t" << std::flush;
      TreeChecker<T,D> check(nodes,primIDs,boxes);
      std::cout << "SAH(" << description << "): " << computeSAH(nodes,primIDs) << std::endl;
    }
    
    void checkHost()
    {
      auto freeBVH
        = [&]()
        {
          cuBQL::cpu::freeBVH(bvh);
          bvh = bvh_t{};
        };
      auto download
        = [&](std::vector<typename bvh_t::Node> &nodes,
              std::vector<int>                  &primIDs)
        {
          nodes.resize(bvh.numNodes);
          memcpy(nodes.data(),bvh.nodes,bvh.numNodes*sizeof(nodes[0]));
          primIDs.resize(bvh.numPrims);
          memcpy(primIDs.data(),bvh.primIDs,bvh.numPrims*sizeof(primIDs[0]));
        };
      check([&](){cuBQL::cpu::spatialMedian(bvh,boxes.data(),boxes.size(),BuildConfig());},
            freeBVH,
            download,
            "cpu::spatialMedian",false);
    }

    void checkDev()
    {
      auto freeBVH
        = [&]()
        {
          cuBQL::cuda::free(bvh);
          bvh = bvh_t{};
        };
      auto download
        = [&](std::vector<typename bvh_t::Node> &nodes,
              std::vector<int>                  &primIDs)
        {
          nodes.resize(bvh.numNodes);
          CUBQL_CUDA_CALL(Memcpy(nodes.data(),bvh.nodes,bvh.numNodes*sizeof(nodes[0]),
                                 cudaMemcpyDefault));
          primIDs.resize(bvh.numPrims);
          CUBQL_CUDA_CALL(Memcpy(primIDs.data(),bvh.primIDs,bvh.numPrims*sizeof(primIDs[0]),
                                 cudaMemcpyDefault));
        };
      check([&](){cuBQL::gpuBuilder(bvh,d_boxes,boxes.size(),BuildConfig());},
            freeBVH,
            download,
            "cuda::gpuBuilder");
      check([&](){cuBQL::cuda::radixBuilder(bvh,d_boxes,boxes.size(),BuildConfig());},
            freeBVH,
            download,
            "cuda::radixBuilder");
      check([&](){cuBQL::cuda::rebinRadixBuilder(bvh,d_boxes,boxes.size(),BuildConfig());},
            freeBVH,
            download,
            "cuda::rebinRadixBuilder");
      if (isFloat3<T,D>()) {
        check([&](){cuBQL::cuda::sahBuilder(bvh,d_boxes,boxes.size(),BuildConfig().enableSAH());},
              freeBVH,
              download,
              "cuda::sahBuilder");
      }
    }
    
    void run()
    {
      std::cout << "== " << cuBQL::vec_t<T,D>::typeName() << " ==" << std::endl;
      checkHost();
      checkDev();
    }

    std::vector<vec_t> points;
    std::vector<box_t> boxes;
    box_t             *d_boxes = 0;
    bvh_t bvh;
  };

  template<int D>
  void checkD(const std::string &generator, size_t numPoints)
  {
    std::cout << "############### numDims = " << D
              << "  generator = " << generator  << " numPoints = " << prettyNumber(numPoints)
              << " ############### " << std::endl;
    using vecND   = cuBQL::vec_t<double,D>;
    
    std::vector<vecND> points
      = PointGenerator<D>::createFromString(generator)
      ->generate(numPoints,290374);

    Checker<float,D>    (points).run();
    Checker<double,D>   (points).run();
    Checker<int,D>      (points).run();
    Checker<longlong,D> (points).run();
  }
  
  void usage(const std::string &error = "")
  {
    if (!error.empty())
      std::cout << "Error: " << error << "\n\n";
    std::cout << "Usage: ./cuBQL...cuBQL_checkBuilders [no options]" << std::endl;
    exit(error.empty()?0:1);
  }
      
} // ::testing

int main(int ac, char **av)
{
  std::vector<std::string> generatorStrings
    = {
    "uniform",
    "clustered",
    // "mixture .1 remap [ -10000000 ] [ 10000000 ] uniform remap [ 1000 1000 ] [ 10000000 1020 ] clustered",
    "mixture .1 remap [ -1000000 ] [ 1000000 ] uniform remap [ 100000 ] [ 100010 ] uniform",
  };
  // const std::string generatorString = "uniform";
  std::vector<int> numPointsToTest = { 100,10000,10000000 };
  for (auto numPoints: numPointsToTest) {
    for (auto generatorString : generatorStrings) {
      
      testing::checkD<2>(generatorString,numPoints);
      testing::checkD<3>(generatorString,numPoints);
      testing::checkD<4>(generatorString,numPoints);
    }      
  }

  return 0;
}


 
