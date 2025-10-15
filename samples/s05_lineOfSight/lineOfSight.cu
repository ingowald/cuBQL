// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// for 'pointsMutuallyVisible()'
#include "cuBQL/queries/triangleData/lineOfSight.h"
#include "cuBQL/builder/cuda.h"
#include <fstream>
#include "../common/loadOBJ.h"
#define STB_IMAGE_IMPLEMENTATION 1
#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

using cuBQL::Triangle;
using cuBQL::vec2i;
using cuBQL::vec2f;
using cuBQL::vec3i;
using cuBQL::vec3f;
using cuBQL::box3f;
using cuBQL::bvh3f;
using cuBQL::divRoundUp;

__global__
void d_computeImage(uint32_t *d_result,
                    vec2i     dims,
                    vec3i    *d_indices,
                    vec3f    *d_vertices,
                    box3f     worldBounds,
                    bvh3f     bvh)
{
  int ix = threadIdx.x+blockIdx.x*blockDim.x; if (ix >= dims.x) return;
  int iy = threadIdx.y+blockIdx.y*blockDim.y; if (iy >= dims.y) return;

  vec3f up(0.f,1.f,0.f);
  vec3f diag = worldBounds.size();
  vec3f du = length(diag)*normalize(cross(diag,up));
  vec3f dv = length(diag)*normalize(cross(du,diag));
  
  vec2f f = (vec2f(ix,iy)) / vec2f(dims) - vec2f(.5f);

  vec3f A = worldBounds.center() - .5f*diag + f.x * du + f.y * dv;
  vec3f B = A + diag;
  
  auto getTriangle = [d_indices,d_vertices](uint32_t primID)
  {
    vec3i idx = d_indices[primID];
    return Triangle{d_vertices[idx.x],d_vertices[idx.y],d_vertices[idx.z]};
  };

  using namespace cuBQL::triangles;
  bool visible = cuBQL::triangles::pointsMutuallyVisible(bvh,getTriangle,A,B);
  d_result[ix+iy*dims.x]
    = visible
    ? 0xff000000
    : 0xffffffff;
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
  
std::vector<uint32_t> computeImage(const std::vector<vec3i> &indices,
                                   const std::vector<vec3f> &vertices,
                                   vec2i dims,
                                   box3f worldBounds)
{
  int numCells = dims.x*dims.y;
  std::vector<uint32_t> result(numCells);
  uint32_t *d_result = 0;
  cudaMalloc((void **)&d_result,numCells*sizeof(uint32_t));

  vec3f *d_vertices = upload(vertices);
  vec3i *d_indices  = upload(indices);
  
  bvh3f bvh = buildBVH(indices.size(),d_indices,d_vertices);

  vec2i bs(8);
  vec2i nb = divRoundUp(dims,bs);
  d_computeImage<<<(dim3)nb,(dim3)bs>>>(d_result,dims,d_indices,d_vertices,
                                         worldBounds,bvh);
  
  cuBQL::cuda::free(bvh);
  
  cudaMemcpy(result.data(),d_result,numCells*sizeof(uint32_t),cudaMemcpyDefault);
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
  std::string outFileName = "";
  vec2i dims(1024,1024);
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg[0] != '-')
      inFileName = arg;
    else if (arg == "-o")
      outFileName = av[++i];
    else if (arg == "-n") {
      dims.x = std::stoi(av[++i]);
      dims.y = std::stoi(av[++i]);
    } else
      usage("unknown cmdline arg '"+arg+"'");
  }
  
  if (inFileName.empty()) usage("no input obj file name specified");
  if (outFileName.empty()) usage("no output image file prefix specified");
  
  std::vector<vec3f> vertices;
  std::vector<vec3i> indices;
  std::cout << "loading obj file " << inFileName << std::endl;
  cuBQL::samples::loadOBJ(indices,vertices,inFileName);
  std::cout << "done, got " << indices.size() << " triangles" << std::endl;
  box3f bb;
  for (auto v : vertices)
    bb.extend(v);
  std::vector<uint32_t> result
    = computeImage(indices,vertices,dims,bb);
  stbi_flip_vertically_on_write(true);
  stbi_write_png(outFileName.c_str(),dims.x,dims.y,4,
                 result.data(),dims.x*sizeof(uint32_t));
  std::cout << "done. image saved to " << outFileName << std::endl;
}
