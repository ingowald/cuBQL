// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"
#include "cuBQL/math/Ray.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace cuBQL {
  namespace fixedRayQuery {

    // ******************************************************************
    // INTERFACE
    // (which functions this header file provides)
    // ******************************************************************
    
    template<typename Lambda>
    inline __cubql_both
    void forEachLeaf(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     cuBQL::Ray   ray,
                     bool dbg=false);
    
    template<typename Lambda>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     cuBQL::Ray   ray,
                     bool dbg=false);

    /*! traverse BVH with given fixed-length, axis-aligned ray, and
      call lambda for each prim encounterd.

      Traversal is UNORDERED (meaning it will NOT try to traverse
      front-to-back) and FIXED-SHAPE (ray will not shrink during
      traversal).

      Lambda is expected to return CUBQL_{CONTINUE|TERMINATE}_TRAVERSAL 
    */
    template<int axis, int sign, typename Lambda>
    inline __cubql_both
    void forEachLeaf(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     AxisAlignedRay<axis,sign> ray,
                     bool dbg=false);
    
    /*! traverse BVH with given fixed-length, axis-aligned ray, and
      call lambda for each prim encounterd.

      Traversal is UNORDERED (meaning it will NOT try to traverse
      front-to-back) and FIXED-SHAPE (ray will not shrink during
      traversal).

      Lambda is expected to return CUBQL_{CONTINUE|TERMINATE}_TRAVERSAL 
    */
    template<int axis, int sign, typename Lambda>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     AxisAlignedRay<axis,sign> ray,
                     bool dbg=false);
  }
  
  // ******************************************************************
  // IMPLEMENTATION
  // ******************************************************************

  template<int axis, int sign, typename Lambda>
  inline __cubql_both
  void fixedRayQuery::forEachLeaf(const Lambda &lambdaToExecuteForEachCandidate,
                                  cuBQL::bvh3f bvh,
                                  AxisAlignedRay<axis,sign> ray,
                                  bool dbg)
  {
    /* for an axis-aligned ray, we can just convert that ray to a
       box, and traverse that instad */
    vec3f A = ray.origin + ray.tmin * ray.direction();
    vec3f B = ray.origin + ray.tmax * ray.direction();
    box3f rayAsBox { min(A,B), max(A,B) };
    // if (dbg) dout << "asbox " << rayAsBox << dout.endl;
    cuBQL::fixedBoxQuery::forEachLeaf(lambdaToExecuteForEachCandidate,bvh,rayAsBox,dbg);
  }

  /*! this query assumes lambads that return CUBQL_CONTINUE_TRAVERSAL
    or CUBQL_TERMINATE_TRAVERSAL */
  template<int axis, int sign, typename Lambda>
  inline __cubql_both
  void fixedRayQuery::forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                                  cuBQL::bvh3f bvh,
                                  AxisAlignedRay<axis,sign> ray,
                                  bool dbg)
  {
    /* the code we want to have executed for each leaf that may
       contain candidates. we loop over each prim in a given leaf,
       and return the minimum culling distance returned by any of
       the per-prim lambdas */
    auto leafCode
      = [lambdaToExecuteForEachCandidate,dbg](const uint32_t *leafPrims,
                                              size_t numPrims)->int
      {
        // if (dbg) dout << "fixedRayQuery::forEachPrim leaf " << numPrims << endl;
        for (int i=0;i<numPrims;i++) 
          if (lambdaToExecuteForEachCandidate(leafPrims[i])
              == CUBQL_TERMINATE_TRAVERSAL)
            return CUBQL_TERMINATE_TRAVERSAL;
        return CUBQL_CONTINUE_TRAVERSAL;
      };
    forEachLeaf(leafCode,bvh,ray,dbg);
  }



  template<typename Lambda>
  inline __cubql_both
  void fixedRayQuery::forEachLeaf(const Lambda &lambdaToCallOnEachLeaf,
                                  cuBQL::bvh3f bvh,
                                  Ray ray,
                                  bool dbg)
  {
      struct StackEntry {
        uint32_t idx;
      };
      bvh3f::node_t::Admin traversalStack[64], *stackPtr = traversalStack;
      bvh3f::node_t::Admin node = bvh.nodes[0].admin;
      // ------------------------------------------------------------------
      // traverse until there's nothing left to traverse:
      // ------------------------------------------------------------------
      // if (dbg) dout << "fixedBoxQuery::traverse" << endl;
      while (true) {

        // ------------------------------------------------------------------
        // traverse INNER nodes downward; breaking out if we either find
        // a leaf within the current search radius, or found a dead-end
        // at which we need to pop
        // ------------------------------------------------------------------
        while (true) {
          if (node.count != 0)
            // it's a boy! - seriously: this is not a inner node, step
            // out of down-travesal and let leaf code pop in.
            break;

          uint32_t n0Idx = (uint32_t)node.offset+0;
          uint32_t n1Idx = (uint32_t)node.offset+1;
          bvh3f::node_t n0 = bvh.nodes[n0Idx];
          bvh3f::node_t n1 = bvh.nodes[n1Idx];
          bool o0 = rayIntersectsBox(ray,n0.bounds);
          bool o1 = rayIntersectsBox(ray,n1.bounds);

          // if (dbg) {
          //   dout << "at node " << node.offset << endl;
          //   dout << "w/ query box " << queryBox << endl;
          //   dout << "  " << n0.bounds << " -> " << (int)o0 << endl;
          //   dout << "  " << n1.bounds << " -> " << (int)o1 << endl;
          // }
          
          if (o0) {
            if (o1) {
              *stackPtr++ = n1.admin;
            } else {
            }
            node = n0.admin;
          } else {
            if (o1) {
              node = n1.admin;
            } else {
              // both children are too far away; this is a dead end
              node.count = 0;
              break;
            }
          }
        }

        // if (dbg)
        //   dout << "at leaf ofs " << (int)node.offset << " cnt " << node.count << endl;
        if (node.count != 0) {
          // we're at a valid leaf: call the lambda and see if that gave
          // us a enw, closer cull radius
          int leafResult
            = lambdaToCallOnEachLeaf(bvh.primIDs+node.offset,node.count);
          // if (dbg)
          //   dout << "leaf returned " << leafResult << endl;
          if (leafResult == CUBQL_TERMINATE_TRAVERSAL)
            return;
        }
        // ------------------------------------------------------------------
        // pop next un-traversed node from stack, discarding any nodes
        // that are more distant than whatever query radius we now have
        // ------------------------------------------------------------------
        // if (dbg) dout << "rem stack depth " << (stackPtr-traversalStack) << endl;
        if (stackPtr == traversalStack)
          return;
        node = *--stackPtr;
      }
  }

  /*! this query assumes lambads that return CUBQL_CONTINUE_TRAVERSAL
    or CUBQL_TERMINATE_TRAVERSAL */
  template<typename Lambda>
  inline __cubql_both
  void fixedRayQuery::forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                                  cuBQL::bvh3f bvh,
                                  Ray ray,
                                  bool dbg)
  {
    /* the code we want to have executed for each leaf that may
       contain candidates. we loop over each prim in a given leaf,
       and return the minimum culling distance returned by any of
       the per-prim lambdas */
    auto leafCode
      = [lambdaToExecuteForEachCandidate,dbg](const uint32_t *leafPrims,
                                              size_t numPrims)->int
      {
        // if (dbg) dout << "fixedRayQuery::forEachPrim leaf " << numPrims << endl;
        for (int i=0;i<numPrims;i++) 
          if (lambdaToExecuteForEachCandidate(leafPrims[i])
              == CUBQL_TERMINATE_TRAVERSAL)
            return CUBQL_TERMINATE_TRAVERSAL;
        return CUBQL_CONTINUE_TRAVERSAL;
      };
    forEachLeaf(leafCode,bvh,ray,dbg);
  }
    
  
  namespace cuBQL {
  
    template<typename T>
    struct Ray {
      using scalar_t = T;
      using vec3 = vec_t<T,3>;
    
      vec3 origin;
      vec3 direction;
      T tmin;
      T tmax;
    };

    struct Triangle {
      vec3f a;
      vec3f b;
      vec3f c;
    };
  
    template<typename T>
    struct RayTriangleIntersection {
      T t=0,u=0,v=0;
    
      inline __device__ bool compute(const Ray<T> &ray, const Triangle &tri, bool dbg=false);
    };
  
  
    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================

    template<typename T>
    inline __cubql_both
    bool rayIntersectsBox(Ray<T> ray, box_t<T,3> box)
    {
      using vec3 = vec_t<T,3>;
      vec3 inv = rcp(ray.direction);
      vec3 lo = (box.lower - ray.origin) * inv;
      vec3 hi = (box.upper - ray.origin) * inv;
      vec3 nr = min(lo,hi);
      vec3 fr = max(lo,hi);
      T tin  = max(ray.tmin,reduce_max(nr));
      T tout = min(ray.tmax,reduce_min(fr));
      return tin <= tout;
    }

    template<typename T>
    inline __cubql_both
    bool rayIntersectsBox(Ray<T> ray, vec_t<T,3> rcp_dir, box_t<T,3> box)
    {
      using vec3 = vec_t<T,3>;
      vec3 lo = (box.lower - ray.origin) * rcp_dir;
      vec3 hi = (box.upper - ray.origin) * rcp_dir;
      vec3 nr = min(lo,hi);
      vec3 fr = max(lo,hi);
      T tin  = max(ray.tmin,reduce_max(nr));
      T tout = min(ray.tmax,reduce_min(fr));
      return tin <= tout;
    }

    inline __cubql_both float abst(float f)   { return (f < 0.f) ? -f : f; }
    inline __cubql_both double abst(double f) { return (f < 0. ) ? -f : f; }


    template<typename T>
    inline __device__
    bool RayTriangleIntersection<T>::compute(const Ray<T> &ray,
                                             const Triangle &tri,
                                             bool dbg)
    {
      using vec3 = vec_t<T,3>;
      const vec3 v0(tri.a);
      const vec3 v1(tri.b);
      const vec3 v2(tri.c);
    
      const vec3 e1 = v1-v0;
      const vec3 e2 = v2-v0;

      // if (dbg) printf("e2 %lf %lf %lf\n",e2.x,e2.y,e2.z);
      vec3 N = cross(e1,e2);
      // if (dbg) printf("N %lf %lf %lf\n",N.x,N.y,N.z);
      if (N == vec3(T(0)))
        return false;

      if (abst(dot(ray.direction,N)) < T(1e-12)) return false;
    
      // P = o+td
      // dot(P-v0,N) = 0
      // dot(o+td-v0,N) = 0
      // dot(td,N)+dot(o-v0,N)=0
      // t*dot(d,N) = -dot(o-v0,N)
      // t = -dot(o-v0,N)/dot(d,N)
      t = -dot(ray.origin-v0,N)/dot(ray.direction,N);
      if (t < ray.tmin || t > ray.tmax) return false;
    
      vec3 P = (ray.origin - v0) + t*ray.direction;
    
      T e1u,e2u,Pu;
      T e1v,e2v,Pv;
      if (abst(N.x) >= ::max(abst(N.y),abst(N.z))) {
        e1u = e1.y; e2u = e2.y; Pu = P.y;
        e1v = e1.z; e2v = e2.z; Pv = P.z;
      } else if (abst(N.y) > abst(N.z)) {
        e1u = e1.x; e2u = e2.x; Pu = P.x;
        e1v = e1.z; e2v = e2.z; Pv = P.z;
      } else {
        e1u = e1.x; e2u = e2.x; Pu = P.x;
        e1v = e1.y; e2v = e2.y; Pv = P.y;
      }
      auto det = [](T a, T b, T c, T d) -> T
      { return a*d - c*b; };
    
      // P = v0 + u * e1 + v * e2 + h * N
      // (P-v0) = [e1,e2]*(u,v,h)
      if (det(e1u,e1v,e2u,e2v) == T(0)) return false;

#if 1
      T den = det(e1u,e2u,e1v,e2v);
      T sign = den < T(0) ? T(-1):T(1);
      den *= sign;
      T den_u = sign*det(Pu,e2u,Pv,e2v);
      if (den_u < T(0)) return false;
      T den_v = sign*det(e1u,Pu,e1v,Pv);
      if (den_v < T(0)) return false;
      if (den_u + den_v > den) return false;
      T rcp_den = rcp(den);
      u = den_u * rcp_den;
      v = den_v * rcp_den;
#else
      u = det(Pu,e2u,Pv,e2v)/det(e1u,e2u,e1v,e2v);
      v = det(e1u,Pu,e1v,Pv)/det(e1u,e2u,e1v,e2v);

      if ((u < T(0)) || (v < T(0)) || ((u+v) > T(1))) return false;
#endif
      return true;
    }
  }
  
  namespace shrinkingRayQuery {
    
    template<typename Lambda, typename bvh_t, typename ray_t>
    inline __cubql_both
    float forEachLeaf(const Lambda &lambdaToCallOnEachLeaf,
                      bvh_t bvh,
                      ray_t ray,
                      bool dbg=false)
    {
      using node_t = typename bvh_t::node_t;
      using T = bvh_t::scalar_t;
      struct StackEntry {
        uint32_t idx;
      };
      typename node_t::Admin traversalStack[64], *stackPtr = traversalStack;
      typename node_t::Admin node = bvh.nodes[0].admin;

      if (ray.direction.x == 0.f) ray.direction.x = T(1e-20);
      if (ray.direction.y == 0.f) ray.direction.y = T(1e-20);
      if (ray.direction.z == 0.f) ray.direction.z = T(1e-20);
      vec_t<T,3> rcp_dir = rcp(ray.direction);
      
      // ------------------------------------------------------------------
      // traverse until there's nothing left to traverse:
      // ------------------------------------------------------------------
      // if (dbg) dout << "fixedBoxQuery::traverse" << endl;
      while (true) {

        // ------------------------------------------------------------------
        // traverse INNER nodes downward; breaking out if we either find
        // a leaf within the current search radius, or found a dead-end
        // at which we need to pop
        // ------------------------------------------------------------------
        while (true) {
          // if (dbg) printf("node %i.%i\n",(int)node.offset,(int)node.count);
          if (node.count != 0)
            // it's a boy! - seriously: this is not a inner node, step
            // out of down-travesal and let leaf code pop in.
            break;

          uint32_t n0Idx = (uint32_t)node.offset+0;
          uint32_t n1Idx = (uint32_t)node.offset+1;
          node_t n0 = bvh.nodes[n0Idx];
          node_t n1 = bvh.nodes[n1Idx];
          bool o0 = rayIntersectsBox(ray,rcp_dir,n0.bounds);
          bool o1 = rayIntersectsBox(ray,rcp_dir,n1.bounds);

          // if (dbg) {
          //   dout << "at node " << node.offset << endl;
          //   dout << "w/ query box " << queryBox << endl;
          //   dout << "  " << n0.bounds << " -> " << (int)o0 << endl;
          //   dout << "  " << n1.bounds << " -> " << (int)o1 << endl;
          // }
          
          if (o0) {
            if (o1) {
              *stackPtr++ = n1.admin;
            } else {
            }
            node = n0.admin;
          } else {
            if (o1) {
              node = n1.admin;
            } else {
              // both children are too far away; this is a dead end
              node.count = 0;
              break;
            }
          }
        }

        // if (dbg)
        //   dout << "at leaf ofs " << (int)node.offset << " cnt " << node.count << endl;
        if (node.count != 0) {
          // we're at a valid leaf: call the lambda and see if that gave
          // us a enw, closer cull radius
          ray.tmax
            = lambdaToCallOnEachLeaf(bvh.primIDs+node.offset,node.count);
        }
        // ------------------------------------------------------------------
        // pop next un-traversed node from stack, discarding any nodes
        // that are more distant than whatever query radius we now have
        // ------------------------------------------------------------------
        // if (dbg) dout << "rem stack depth " << (stackPtr-traversalStack) << endl;
        if (stackPtr == traversalStack)
          return ray.tmax;
        node = *--stackPtr;
      }
    }

    template<typename Lambda, typename bvh_t, typename ray_t>
      inline __cubql_both
    void forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                     bvh_t bvh,
                     ray_t &ray,
                     bool dbg=false)
    {
      auto perLeaf = [dbg,bvh,&ray,lambdaToExecuteForEachCandidate]
        (const uint32_t *leaf, int count) {
        for (int i=0;i<count;i++)
          ray.tmax = lambdaToExecuteForEachCandidate(leaf[i]);
        return ray.tmax;
      };
      shrinkingRayQuery::forEachLeaf(perLeaf,bvh,ray,dbg);
    }
    
  } // ::cuBQL::shrinkingRayQuery
} // ::cuBQL
