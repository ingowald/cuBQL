// ======================================================================== //
// Copyright 2025-2025 Ingo Wald                                            //
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

