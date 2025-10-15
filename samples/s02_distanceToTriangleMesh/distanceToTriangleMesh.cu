// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file samples/closestPointOnTrianglesSurface Simple example of
    building bvhes over, and quering closest points on, sets of 3D
    triangles

    This example will, in successive steps:

    1) load a cmdline-specified OBJ file of triangles

    2) build BVH over those triangles

    3) run some sample find-closst-point queries: generate a grid of
    512x512x512 cells (stretched over the bounding box of the model),
    then for each cell center, perform a bvh fcp closest-point query
    on those line segmetns.
*/

// cuBQL:
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#define CUBQL_TRIANGLE_CPAT_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/queries/triangleData/closestPointOnAnyTriangle.h"
#include "samples/common/loadOBJ.h"

// std:
#include <random>
#include <fstream>

using cuBQL::vec3i;
using cuBQL::vec3f;
using cuBQL::box3f;
using cuBQL::bvh3f;
using cuBQL::divRoundUp;
using cuBQL::prettyNumber;
using cuBQL::prettyDouble;
using cuBQL::getCurrentTime;
using cuBQL::Triangle;

/*! helper function that allocates managed memory, and cheks for errors */
template<typename T>
T *allocManaged(int N)
{
  T *ptr = 0;
  CUBQL_CUDA_CALL(MallocManaged((void **)&ptr,N*sizeof(T)));
  return ptr;
}

/*! generate boxes (required for bvh builder) from prim type 'index line triangles' */
__global__ void generateBoxes(box3f *boxForBuilder,
                              const Triangle *triangles,
                              int numTriangles)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numTriangles) return;
  
  auto triangle = triangles[tid];
  boxForBuilder[tid] = triangle.bounds();
}


/*! the actual sample query: generates points in a gridDim^3 grid of points, then for each such grid point perform a query */
__global__
void runQueries(bvh3f           trianglesBVH,
                const Triangle *triangles,
                box3f           worldBounds,
                int             numQueries)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  // compute a point on the diagonal of the world bounding box
  float t = tid / (numQueries-1.f);
  t = -.2f + t * 1.4f;
  
  vec3f queryPoint = worldBounds.lerp(vec3f(t));
  
  cuBQL::triangles::CPAT cpat;

  cpat.runQuery(triangles,
                trianglesBVH,
                queryPoint);

  printf("[%i] closest surface point to point (%f %f %f) is on triangle %i, at (%f %f %f), and %f units away\n",
         tid,
         queryPoint.x,
         queryPoint.y,
         queryPoint.z,
         cpat.triangleIdx,
         cpat.P.x,
         cpat.P.y,
         cpat.P.z,
         sqrtf(cpat.sqrDist));
}


int main(int ac, const char **av)
{
  const char *inFileName = "../samples/bunny.obj";
  if (ac != 1)
    inFileName = av[1];
  
  // ------------------------------------------------------------------
  // step 1: load triangle mesh
  // ------------------------------------------------------------------
  std::cout << "loading triangles from " << inFileName << std::endl;
  std::vector<Triangle> h_triangles
    = cuBQL::samples::loadOBJ(inFileName);
  int numTriangles = (int)h_triangles.size();
  std::cout << "loaded OBJ file, got " << prettyNumber(numTriangles)
            << " triangles" << std::endl;
  box3f worldBounds;
  for (auto tri : h_triangles)
    worldBounds.extend(tri.bounds());
  std::cout << "world bounding box of triangles is " << worldBounds
            << std::endl;
  
  // upload to the device:
  Triangle *d_triangles = 0;
  CUBQL_CUDA_CALL(Malloc((void**)&d_triangles,numTriangles*sizeof(Triangle)));
  CUBQL_CUDA_CALL(Memcpy(d_triangles,h_triangles.data(),
                         numTriangles*sizeof(Triangle),cudaMemcpyDefault));
  
  // ------------------------------------------------------------------
  // step 2) build BVH over those triangles, so we can do queries on
  // them
  // ------------------------------------------------------------------
  
  bvh3f trianglesBVH;
  {
    
    // allocate memory for bounding boxes (to build BVH over)
    box3f *d_boxes = 0;
    CUBQL_CUDA_CALL(Malloc((void**)&d_boxes,numTriangles*sizeof(box3f)));
    
    // run cuda kernel that generates a bounding box for each point 
    generateBoxes<<<divRoundUp(numTriangles,1024),1024>>>
      (d_boxes,d_triangles,numTriangles);
    
    // ... aaaand build the BVH
    cuBQL::gpuBuilder(trianglesBVH,d_boxes,numTriangles,cuBQL::BuildConfig());
    // free the boxes - we could actually re-use that memory below, but
    // let's just do this cleanly here.
    CUBQL_CUDA_CALL(Free(d_boxes));
    std::cout << "done building BVH over " << prettyNumber(numTriangles)
              << " triangles" << std::endl;
  }

  // ------------------------------------------------------------------
  // step 3: run some sample query - this query will generate query
  // points on the diagonal, and just print the results on the
  // terminal
  // ------------------------------------------------------------------
  
  int numQueries = 16;
  float *sqrDist = allocManaged<float>(numQueries);
  runQueries<<<divRoundUp(numQueries,128),128>>>
    (trianglesBVH,d_triangles,worldBounds,numQueries);
  CUBQL_CUDA_SYNC_CHECK();

  CUBQL_CUDA_CALL(Free(d_triangles));
  cuBQL::cuda::free(trianglesBVH);

  return 0;
}
