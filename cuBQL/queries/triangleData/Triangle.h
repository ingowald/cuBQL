// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

/*! \file cuBQL/triangles/Triangle.h Defines a generic triangle type and
  some operations thereon, that various queries can then build on */

#pragma once

#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"

namespace cuBQL {

  // =============================================================================
  // *** INTERFACE ***
  // =============================================================================
  
  /*! a simple triangle consisting of three vertices. In order to not
    overload this class with too many functions the actual
    operations on triangles - such as intersectin with a ray,
    computing distance to a point, etc - will be defined in the
    respective queries */
  struct Triangle {
    /*! returns an axis aligned bounding box enclosing this triangle */
    inline __cubql_both box3f bounds() const;
    inline __cubql_both vec3f sample(float u, float v) const;
    inline __cubql_both vec3f normal() const;
    
    vec3f a, b, c;
  };

  /*! a typical triangle mesh, with array of vertices and
      indices. This class will NOT do any allocation/deallocation, not
      use smart pointers - it's just a 'view' on what whoever else
      might own and manage, and may thus be used exactly the same on
      device as well as on host. */
  struct TriangleMesh {
    inline __cubql_both Triangle getTriangle(int i) const;
    
    /*! pointer to array of vertices; must be in same memory space as
        the operations performed on it (eg, if passed to a gpu builder
        it has to be gepu memory */
    vec3f *vertices;
    
    /*! pointer to array of vertices; must be in same memory space as
        the operations performed on it (eg, if passed to a gpu builder
        it has to be gepu memory */
    vec3i *indices;

    int numVertices;
    int numIndices;
  };

  // =============================================================================
  // *** IMPLEMENTATION ***
  // =============================================================================

  // ---------------------- TriangleMesh ----------------------
  inline __cubql_both Triangle TriangleMesh::getTriangle(int i) const
  {
    vec3i index = indices[i];
    return { vertices[index.x],vertices[index.y],vertices[index.z] };
  }
  
  // ---------------------- Triangle ----------------------
  inline __cubql_both vec3f Triangle::normal() const
  { return cross(b-a,c-a); }
  
  inline __cubql_both box3f Triangle::bounds() const
  { return box3f().including(a).including(b).including(c); }

  inline __cubql_both float area(Triangle tri)
  { return length(cross(tri.b-tri.a,tri.c-tri.a)); }

  inline __cubql_both vec3f Triangle::sample(float u, float v) const
  {
    if (u+v >= 1.f) { u = 1.f-u; v = 1.f-v; }
    return (1.f-u-v)*a + u * b + v * c;
  }

  inline __cubql_both dbgout operator<<(dbgout o, const Triangle &triangle)
  { o << "{" << triangle.a << "," << triangle.b << "," << triangle.c << "}"; return o; }
  
  
} // ::cuBQL

