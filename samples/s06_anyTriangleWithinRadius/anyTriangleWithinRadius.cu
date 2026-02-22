// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuBQL/queries/triangleData/anyWithinRadius.h"
#include "cuBQL/builder/cuda.h"
#include <fstream>
#include "../common/loadOBJ.h"

using cuBQL::Triangle;
using cuBQL::vec3i;
using cuBQL::vec3f;
using cuBQL::box3f;
using cuBQL::bvh3f;
using cuBQL::divRoundUp;

__global__
void d_computeVolume(float   *d_result,
                     vec3i    dims,
                     vec3i   *d_indices,
                     vec3f   *d_vertices,
                     box3f    worldBounds,
                     bvh3f    bvh)
{
  int ix = threadIdx.x+blockIdx.x*blockDim.x; if (ix >= dims.x) return;
  int iy = threadIdx.y+blockIdx.y*blockDim.y; if (iy >= dims.y) return;
  int iz = threadIdx.z+blockIdx.z*blockDim.z; if (iz >= dims.z) return;
  
  vec3f f = (vec3f(ix,iy,iz)+vec3f(.5f)) / vec3f(dims);
  vec3f queryPoint = worldBounds.lerp(f);
  auto getTriangle = [d_indices,d_vertices](uint32_t primID)
  {
    vec3i idx = d_indices[primID];
    return Triangle{d_vertices[idx.x],d_vertices[idx.y],d_vertices[idx.z]};
  };

  /* as radius, in this example use 1% of scene diameter */
  float queryRadius = length(worldBounds.size()) * .05f;

  bool dbg = vec3f(ix,iy,iz) == vec3f(16);
  
  using namespace cuBQL::triangles;
  bool closeToSurface
    = cuBQL::triangles::anyWithinRadius(// the model we're querying
                                       bvh,getTriangle,
                                       // the point and radius we're querying with
                                       queryPoint,queryRadius,dbg);
  d_result[ix+iy*dims.x+iz*dims.x*dims.y] = closeToSurface?1.f:0.f;
}

template<typename T>
T *upload(const std::vector<T> &vec)
{
  T *d_vec = 0;
  cudaMalloc((void**)&d_vec,vec.size()*sizeof(T));
  cudaMemcpy(d_vec,vec.data(),vec.size()*sizeof(T),cudaMemcpyDefault);
  return d_vec;
}
 
__global__ void fillBounds(box3f *d_bounds,
                           int numTriangles,
                           const vec3i *d_indices,
                           const vec3f *d_vertices)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numTriangles) return;
  vec3i idx = d_indices[tid];
  d_bounds[tid] = box3f()
    .extend(d_vertices[idx.x])
    .extend(d_vertices[idx.y])
    .extend(d_vertices[idx.z]);
}

cuBQL::bvh3f buildBVH(int numTriangles,
                      const vec3i *d_indices,
                      const vec3f *d_vertices)
{
  box3f *d_boxes;
  cudaMalloc((void**)&d_boxes,numTriangles*sizeof(box3f));
  fillBounds<<<divRoundUp(numTriangles,1024),1024>>>
    (d_boxes,numTriangles,d_indices,d_vertices);

  std::cout << "building bvh" << std::endl;
  bvh3f bvh;
  ::cuBQL::gpuBuilder(bvh,d_boxes,numTriangles);
  std::cout << " ... done." << std::endl;
  cudaFree(d_boxes);
  return bvh;
}
  
std::vector<float> computeVolume(const std::vector<vec3i> &indices,
                                 const std::vector<vec3f> &vertices,
                                 vec3i dims,
                                 box3f worldBounds)
{
  int numCells = dims.x*dims.y*dims.z;
  std::vector<float> result(numCells);
  float *d_result = 0;
  cudaMalloc((void **)&d_result,numCells*sizeof(float));

  vec3f *d_vertices = upload(vertices);
  vec3i *d_indices  = upload(indices);
  
  bvh3f bvh = buildBVH(indices.size(),d_indices,d_vertices);

  vec3i bs(8);
  vec3i nb = divRoundUp(dims,bs);
  d_computeVolume<<<(dim3)nb,(dim3)bs>>>(d_result,dims,d_indices,d_vertices,
                             worldBounds,
                             bvh);
  
  cuBQL::cuda::free(bvh);
  
  cudaMemcpy(result.data(),d_result,numCells*sizeof(float),cudaMemcpyDefault);
  cudaFree(d_result);
  cudaFree(d_indices);
  cudaFree(d_vertices);
  return result;
}

void usage(const std::string &error)
{
  std::cerr << "Error : " << error << "\n\n";
  std::cout << "Usage: ./insideOutside inFile.obj -o outFilePrefix [-n maxRes]" << std::endl;
  exit(0);
}

int main(int ac, char **av)
{
  std::string inFileName = "";
  std::string outFilePrefix = "";
  int n = 256;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg[0] != '-')
      inFileName = arg;
    else if (arg == "-o")
      outFilePrefix = av[++i];
    else if (arg == "-n") 
      n = std::stoi(av[++i]);
    else
      usage("unknown cmdline arg '"+arg+"'");
  }

  if (inFileName.empty()) usage("no input obj file name specified");
  if (outFilePrefix.empty()) usage("no output volume file prefix specified");

  std::vector<vec3f> vertices;
  std::vector<vec3i> indices;
  std::cout << "loading obj file " << inFileName << std::endl;
  cuBQL::samples::loadOBJ(indices,vertices,inFileName);
  std::cout << "done, got " << indices.size() << " triangles" << std::endl;
  for (auto &v : vertices) v = v * 1000.f;
  box3f bb;
  for (auto v : vertices)
    bb.extend(v);
  PRINT(bb);
  vec3f size = bb.size();
  float max_size = reduce_max(size);
  vec3i dims = min(vec3i(n),vec3i(size/max_size*vec3f(n)+1.f));
  std::cout << "using volume dims of " << dims << std::endl;

  std::vector<float> result
    = computeVolume(indices,vertices,dims,bb);
  const std::string outFileName =
    outFilePrefix
    +"_"+std::to_string(dims.x)
    +"x"+std::to_string(dims.y)
    +"x"+std::to_string(dims.z)
    +"_float.raw";
  std::ofstream out(outFileName,
                    std::ios::binary);
  out.write((const char *)result.data(),
            dims.x*dims.y*dims.z*sizeof(float));
  std::cout << "done. volume saved to " << outFileName << std::endl;
}
