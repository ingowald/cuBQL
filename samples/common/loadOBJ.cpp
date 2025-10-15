// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "loadOBJ.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace std {
  inline bool operator<(const tinyobj::index_t &a,
                        const tinyobj::index_t &b)
  {
    if (a.vertex_index < b.vertex_index) return true;
    if (a.vertex_index > b.vertex_index) return false;
    
    if (a.normal_index < b.normal_index) return true;
    if (a.normal_index > b.normal_index) return false;
    
    if (a.texcoord_index < b.texcoord_index) return true;
    if (a.texcoord_index > b.texcoord_index) return false;
    
    return false;
  }
}

namespace cuBQL {
  namespace samples {

    std::vector<Triangle> loadOBJ(const std::string &objFile)
    {
      std::string modelDir = "";
      tinyobj::attrib_t attributes;
      std::vector<tinyobj::shape_t> shapes;
      std::vector<tinyobj::material_t> materials;
    
      std::string err = "";
      bool readOK
        = tinyobj::LoadObj(&attributes,
                           &shapes,
                           &materials,
                           &err,
                           &err,
                           objFile.c_str(),
                           modelDir.c_str(),
                           /* triangulate */true);
      if (!readOK) 
        throw std::runtime_error("Could not read OBJ model from "+objFile+" : "+err);

      std::vector<Triangle> triangles;
      const vec3f *vertex_array   = (const vec3f*)attributes.vertices.data();
      for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
        tinyobj::shape_t &shape = shapes[shapeID];
        for (size_t faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
          tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
          tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
          tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
        
          vec3f a = vertex_array[idx0.vertex_index];
          vec3f b = vertex_array[idx1.vertex_index];
          vec3f c = vertex_array[idx2.vertex_index];
          triangles.push_back({a,b,c});
        }
      }
      return triangles;
    }

    void loadOBJ(std::vector<vec3i> &indices,
                 std::vector<vec3f> &vertices,
                 const std::string &objFile)
    {
      std::string modelDir = "";
      tinyobj::attrib_t attributes;
      std::vector<tinyobj::shape_t> shapes;
      std::vector<tinyobj::material_t> materials;
    
      std::string err = "";
      bool readOK
        = tinyobj::LoadObj(&attributes,
                           &shapes,
                           &materials,
                           &err,
                           &err,
                           objFile.c_str(),
                           modelDir.c_str(),
                           /* triangulate */true);
      if (!readOK) 
        throw std::runtime_error("Could not read OBJ model from "+objFile+" : "+err);

      std::vector<Triangle> triangles;
      const vec3f *vertex_array   = (const vec3f*)attributes.vertices.data();
      int maxUsedVertex = 0;
      for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
        tinyobj::shape_t &shape = shapes[shapeID];
        for (size_t faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
          tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
          tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
          tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];

          int a = idx0.vertex_index;
          int b = idx1.vertex_index;
          int c = idx2.vertex_index;
          maxUsedVertex = std::max(maxUsedVertex,a);
          maxUsedVertex = std::max(maxUsedVertex,b);
          maxUsedVertex = std::max(maxUsedVertex,c);
          // vec3f a = vertex_array[idx0.vertex_index];
          // vec3f b = vertex_array[idx1.vertex_index];
          // vec3f c = vertex_array[idx2.vertex_index];
          if (a >= 0 && b >=0 && c >= 0 && a != b && a != c && b != c)
            indices.push_back({a,b,c});
        }
      }
      vertices.resize(maxUsedVertex+1);
      std::copy(vertex_array,vertex_array+maxUsedVertex+1,vertices.data());
    }
    
    void saveOBJ(const std::vector<Triangle> &triangles,
                 const std::string &outFileName)
    {
      std::ofstream out(outFileName.c_str());
      for (auto tri : triangles) {
        out << "v " << tri.a.x << " " << tri.a.y << " " << tri.a.z << std::endl;
        out << "v " << tri.b.x << " " << tri.b.y << " " << tri.b.z << std::endl;
        out << "v " << tri.c.x << " " << tri.c.y << " " << tri.c.z << std::endl;
        out << "f -1 -2 -3" << std::endl;
      }
    }
  
    std::vector<Triangle> triangulate(const std::vector<box3f> &boxes)
    {
      std::vector<Triangle> triangles;
      int indices[] = {0,1,3, 2,3,0,
                       5,7,6, 5,6,4,
                       0,4,5, 0,5,1,
                       2,3,7, 2,7,6,
                       1,5,7, 1,7,3,
                       4,0,2, 4,2,6};
      Triangle tri;
      for (auto box : boxes) {
        vec3f vertices[8], *vtx = vertices;
        for (int iz=0;iz<2;iz++)
          for (int iy=0;iy<2;iy++)
            for (int ix=0;ix<2;ix++) {
              vtx->x = (ix?box.lower:box.upper).x;
              vtx->y = (iy?box.lower:box.upper).y;
              vtx->z = (iz?box.lower:box.upper).z;
              vtx++;
            }
        for (int i=0;i<12;i++) {
          tri.a = vertices[indices[3*i+0]];
          tri.b = vertices[indices[3*i+1]];
          tri.c = vertices[indices[3*i+2]];
          triangles.push_back(tri);
        }
      }
      return triangles;
    }
  
  } // ::cuBQL::samples
} // ::cuBQL

