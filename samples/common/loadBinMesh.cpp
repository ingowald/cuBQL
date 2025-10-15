// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <samples/common/loadBinMesh.h>
#include <fstream>

namespace cuBQL {
  namespace samples {
    
    void loadBinMesh(std::vector<vec3i> &indices,
                     std::vector<vec3f> &vertices,
                     const std::string &inFileName)
    {
      std::ifstream in(inFileName.c_str(),std::ios::binary);
      size_t numVertices;
      size_t numTriangles;

      in.read((char*)&numVertices,sizeof(numVertices));
      vertices.resize(numVertices);
      in.read((char*)vertices.data(),numVertices*sizeof(vec3f));

      in.read((char*)&numTriangles,sizeof(numTriangles));
      indices.resize(numTriangles);
      in.read((char*)indices.data(),numTriangles*sizeof(vec3i));
    }
    
    std::vector<Triangle> loadBinMesh(const std::string &fileName)
    {
      std::vector<vec3i> indices;
      std::vector<vec3f> vertices;
      loadBinMesh(indices,vertices,fileName);
      std::vector<Triangle> res;
      for (auto idx : indices)
        res.push_back({vertices[idx.x],vertices[idx.y],vertices[idx.z]});
      return res;
    }
    
  }
}

