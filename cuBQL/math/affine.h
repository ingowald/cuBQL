// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "linear.h"

namespace cuBQL {

#define VectorT typename L::vector_t
#define ScalarT typename L::vector_t::scalar_t

  ////////////////////////////////////////////////////////////////////////////////
  // Affine Space
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L>
  struct AffineSpaceT
    {
      L l;           /*< linear part of affine space */
      VectorT p;     /*< affine part of affine space */

      ////////////////////////////////////////////////////////////////////////////////
      // Constructors, Assignment, Cast, Copy Operations
      ////////////////////////////////////////////////////////////////////////////////

      // inline AffineSpaceT           ( ) = default;
      // #ifdef __CUDA_ARCH__
      inline __cubql_both
      AffineSpaceT           ( )
      : l(OneTy()),
      p(ZeroTy())
      {}
      // #else
      //        inline __cubql_both AffineSpaceT           ( ) : l(one), p(zero) {}
      // #endif

      inline// __cubql_both
      AffineSpaceT           ( const AffineSpaceT& other ) = default;
      inline __cubql_both AffineSpaceT           ( const L           & other ) { l = other  ; p = VectorT(ZeroTy()); }
      inline __cubql_both AffineSpaceT& operator=( const AffineSpaceT& other ) { l = other.l; p = other.p; return *this; }

      inline __cubql_both AffineSpaceT( const VectorT& vx, const VectorT& vy, const VectorT& vz, const VectorT& p ) : l(vx,vy,vz), p(p) {}
      inline __cubql_both AffineSpaceT( const L& l, const VectorT& p ) : l(l), p(p) {}

      template<typename L1> inline __cubql_both AffineSpaceT( const AffineSpaceT<L1>& s ) : l(s.l), p(s.p) {}

      ////////////////////////////////////////////////////////////////////////////////
      // Constants
      ////////////////////////////////////////////////////////////////////////////////

      inline AffineSpaceT( ZeroTy ) : l(ZeroTy()), p(ZeroTy()) {}
      inline AffineSpaceT( OneTy )  : l(OneTy()),  p(ZeroTy()) {}

      /*! return matrix for scaling */
      static inline AffineSpaceT scale(const VectorT& s) { return L::scale(s); }

      /*! return matrix for translation */
      static inline AffineSpaceT translate(const VectorT& p) { return AffineSpaceT(OneTy(),p); }

      /*! return matrix for rotation, only in 2D */
      static inline AffineSpaceT rotate(const ScalarT& r) { return L::rotate(r); }

      /*! return matrix for rotation around arbitrary point (2D) or axis (3D) */
      static inline AffineSpaceT rotate(const VectorT& u, const ScalarT& r) { return L::rotate(u,r); }

      /*! return matrix for rotation around arbitrary axis and point, only in 3D */
      static inline AffineSpaceT rotate(const VectorT& p, const VectorT& u, const ScalarT& r) { return translate(+p) * rotate(u,r) * translate(-p);  }

      /*! return matrix for looking at given point, only in 3D; right-handed coordinate system */
      static inline AffineSpaceT lookat(const VectorT& eye, const VectorT& point, const VectorT& up) {
        VectorT Z = normalize(point-eye);
        VectorT U = normalize(cross(Z,up));
        VectorT V = cross(U,Z);
        return AffineSpaceT(L(U,V,Z),eye);
      }

    };

  ////////////////////////////////////////////////////////////////////////////////
  // Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline AffineSpaceT<L> operator -( const AffineSpaceT<L>& a ) { return AffineSpaceT<L>(-a.l,-a.p); }
  template<typename L> inline AffineSpaceT<L> operator +( const AffineSpaceT<L>& a ) { return AffineSpaceT<L>(+a.l,+a.p); }
  template<typename L>
  inline __cubql_both
  AffineSpaceT<L> rcp( const AffineSpaceT<L>& a ) {
    L il = rcp(a.l);
    return AffineSpaceT<L>(il,-(il*a.p));
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline AffineSpaceT<L> operator +( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l+b.l,a.p+b.p); }
  template<typename L> inline AffineSpaceT<L> operator -( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l-b.l,a.p-b.p); }

  template<typename L> inline __cubql_both AffineSpaceT<L> operator *( const ScalarT        & a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a*b.l,a*b.p); }
  template<typename L> inline __cubql_both AffineSpaceT<L> operator *( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l*b.l,a.l*b.p+a.p); }
  template<typename L> inline AffineSpaceT<L> operator /( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a * rcp(b); }
  template<typename L> inline AffineSpaceT<L> operator /( const AffineSpaceT<L>& a, const ScalarT        & b ) { return a * rcp(b); }

  template<typename L> inline AffineSpaceT<L>& operator *=( AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a = a * b; }
  template<typename L> inline AffineSpaceT<L>& operator *=( AffineSpaceT<L>& a, const ScalarT        & b ) { return a = a * b; }
  template<typename L> inline AffineSpaceT<L>& operator /=( AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a = a / b; }
  template<typename L> inline AffineSpaceT<L>& operator /=( AffineSpaceT<L>& a, const ScalarT        & b ) { return a = a / b; }

  template<typename L> inline __cubql_both const VectorT xfmPoint (const AffineSpaceT<L>& m, const VectorT& p) { return madd(VectorT(p.x),m.l.vx,madd(VectorT(p.y),m.l.vy,madd(VectorT(p.z),m.l.vz,m.p))); }
  template<typename L> inline __cubql_both const VectorT xfmVector(const AffineSpaceT<L>& m, const VectorT& v) { return xfmVector(m.l,v); }
  template<typename L> inline __cubql_both const VectorT xfmNormal(const AffineSpaceT<L>& m, const VectorT& n) { return xfmNormal(m.l,n); }


  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline bool operator ==( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a.l == b.l && a.p == b.p; }
  template<typename L> inline bool operator !=( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a.l != b.l || a.p != b.p; }

  ////////////////////////////////////////////////////////////////////////////////
  // Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline std::ostream& operator<<(std::ostream& cout, const AffineSpaceT<L>& m) {
    return cout << "{ l = " << m.l << ", p = " << m.p << " }";
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Type Aliases
  ////////////////////////////////////////////////////////////////////////////////

  using AffineSpace2f      = AffineSpaceT<LinearSpace2f>;
  using AffineSpace3f      = AffineSpaceT<LinearSpace3f>;
  // using AffineSpace3fa     = AffineSpaceT<LinearSpace3fa>;
  using OrthonormalSpace3f = AffineSpaceT<Quaternion3f >;

  using affine2f = AffineSpace2f;
  using affine3f = AffineSpace3f;

  ////////////////////////////////////////////////////////////////////////////////
  /*! Template Specialization for 2D: return matrix for rotation around point (rotation around arbitrarty vector is not meaningful in 2D) */
  template<> inline AffineSpace2f AffineSpace2f::rotate(const vec2f& p, const float& r)
  { return translate(+p) * AffineSpace2f(LinearSpace2f::rotate(r)) * translate(-p); }

#undef VectorT
#undef ScalarT


  // inline __cubql_both box3f xfmBounds(const affine3f &xfm,
  //                                 const box3f &box)
  // {
  //   box3f dst;
  //   const vec3f lo = box.lower;
  //   const vec3f hi = box.upper;
  //   dst.extend(xfmPoint(xfm,vec3f(lo.x,lo.y,lo.z)));
  //   dst.extend(xfmPoint(xfm,vec3f(lo.x,lo.y,hi.z)));
  //   dst.extend(xfmPoint(xfm,vec3f(lo.x,hi.y,lo.z)));
  //   dst.extend(xfmPoint(xfm,vec3f(lo.x,hi.y,hi.z)));
  //   dst.extend(xfmPoint(xfm,vec3f(hi.x,lo.y,lo.z)));
  //   dst.extend(xfmPoint(xfm,vec3f(hi.x,lo.y,hi.z)));
  //   dst.extend(xfmPoint(xfm,vec3f(hi.x,hi.y,lo.z)));
  //   dst.extend(xfmPoint(xfm,vec3f(hi.x,hi.y,hi.z)));
  //   return dst;
  // }
    
} // ::cuBQL
