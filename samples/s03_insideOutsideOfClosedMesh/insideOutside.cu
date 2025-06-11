// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/queries/triangles/crossingCount/cc.h"
#include <fstream>
#include "../common/loadOBJ.h"

using cuBQL::Triangle;
using cuBQL::vec3i;
using cuBQL::vec3f;
using cuBQL::box3f;
using cuBQL::bvh3f;
using cuBQL::divRoundUp;

template<int axis, int direction>
inline __device__
/*! returns {signedCount,totalCount} */
int computeCrossingCount(vec3f  P,
                         vec3i *d_indices,
                         vec3f *d_vertices,
                         bvh3f  bvh,
                         bool   useTotalCount)
{
  auto getTriangle = [d_vertices,d_indices](uint32_t primID)
  {
    vec3i idx = d_indices[primID];
    return Triangle{d_vertices[idx.x],d_vertices[idx.y],d_vertices[idx.z]};
  };

  cuBQL::triangles::CrossingCount cc;
  cuBQL::AxisAlignedRay<axis,direction> queryRay { P, CUBQL_INF };
  cc.runQuery(getTriangle,bvh,queryRay);

  return useTotalCount ? cc.totalCount : cc.crossingCount;
}

__global__ void d_computeVolume(float   *d_result,
                                float   *d_crossingCount,
                                vec3i    dims,
                                vec3i   *d_indices,
                                vec3f   *d_vertices,
                                box3f    worldBounds,
                                bvh3f    bvh,
                                bool     useTotalCount)
{
  int ix = threadIdx.x+blockIdx.x*blockDim.x; if (ix >= dims.x) return;
  int iy = threadIdx.y+blockIdx.y*blockDim.x; if (iy >= dims.y) return;
  int iz = threadIdx.z+blockIdx.z*blockDim.x; if (iz >= dims.z) return;

  vec3f f = (vec3f(ix,iy,iz)+.5f) / vec3f(dims);
  vec3f P = (1.f-f)*worldBounds.lower + f*worldBounds.upper;

  /*! we trace 6 rays - one per principle axis - using the
      AxisAlignedRay rayquery. In theory, if the mesh is closed then
      these 6 calls should all agree; but in practice there's always
      some holes or double counting when rays going right through
      vertices or edges, so we just trace one ray in each direction
      and take a majority vote. */
  int numIn = 0;
  numIn += computeCrossingCount<0,-1>(P,d_indices,d_vertices,bvh,useTotalCount);
  numIn += computeCrossingCount<0,+1>(P,d_indices,d_vertices,bvh,useTotalCount);
  numIn += computeCrossingCount<1,-1>(P,d_indices,d_vertices,bvh,useTotalCount);
  numIn += computeCrossingCount<1,+1>(P,d_indices,d_vertices,bvh,useTotalCount);
  numIn += computeCrossingCount<2,-1>(P,d_indices,d_vertices,bvh,useTotalCount);
  numIn += computeCrossingCount<2,+1>(P,d_indices,d_vertices,bvh,useTotalCount);

  d_result[ix+iy*dims.x+iz*dims.x*dims.y] = numIn;
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

  bvh3f bvh;
  ::cuBQL::gpuBuilder(bvh,d_boxes,numTriangles);
  cudaFree(d_boxes);
  return bvh;
}
  
std::vector<float> computeVolume(const std::vector<vec3i> &indices,
                                 const std::vector<vec3f> &vertices,
                                 vec3i dims,
                                 bool  usedTotalCount)
{
  int numCells = dims.x*dims.y*dims.z;
  std::vector<float> result(numCells);
  float *d_result = 0;
  cudaMalloc((void **)&d_result,numCells*sizeof(float));

  vec3f *d_vertices = upload(vertices);
  vec3i *d_indices  = upload(indices);
  
  bvh3f bvh = buildBVH(indices.size(),d_indices,d_vertices);

  cuBQL::cuda::free(bvh);
  
  cudaMemcpy(result.data(),d_result,numCells*sizeof(float),cudaMemcpyDefault);
  cudaFree(d_result);
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
  std::string outFileName = "";
  bool useTotalCount = false;
  int n = 256;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg[0] != '-')
      inFileName = arg;
    else if (arg == "-o")
      outFileName = av[++i];
    else if (arg == "-tc") 
      useTotalCount = true;
    else if (arg == "-n") 
      n = std::stoi(av[++i]);
    else
      usage("unknown cmdline arg '"+arg+"'");
  }

  if (inFileName.empty()) usage("no input obj file name specified");
  if (outFileName.empty()) usage("no output volume file prefix specified");

  std::vector<vec3f> vertices;
  std::vector<vec3i> indices;
  cuBQL::samples::loadOBJ(indices,vertices,inFileName);
  box3f bb;
  for (auto v : vertices)
    bb.extend(v);
  vec3f size = bb.size();
  float width = arg_max(size);
  vec3i dims = min(vec3i(n),vec3i(size/width*vec3f(n)+1.f));
  std::cout << "using volume dims of " << dims << std::endl;

  std::vector<float> result
    = computeVolume(indices,vertices,dims,useTotalCount);
  std::ofstream out(outFileName
                    +"_"+std::to_string(dims.x)
                    +"x"+std::to_string(dims.y)
                    +"x"+std::to_string(dims.z)
                    +"_float.raw",
                    std::ios::binary);
  out.write((const char *)result.data(),
            dims.x*dims.y*dims.z*sizeof(float));
}
