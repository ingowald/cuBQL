// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/queries/triangleData/Triangle.h"
#include "samples/common/IO.h"

namespace cuBQL {
  namespace samples {

    inline double defaultDomainSize() { return 100000.; }
    
    // ==================================================================
    /*! a 'point generator' is a class that implements a procedure to
      create a randomized set of points (of given type and
      dimensoins). In particular, this library allows for describing
      various point generators through a string, such as "uniform"
      (uniformly distributed points), "nrooks" (a n-rooks style
      distribution, see below), "clustered", etc. */
    template<int D=3>
    struct PointGenerator {
      typedef std::shared_ptr<PointGenerator> SP;
      /*! create a set of requested number of elements with given
          generator seed*/
      virtual std::vector<vec_t<double,D>> generate(int numRequested, int seed) = 0;
      
      // helper stuff to parse itself from cmd-line descriptor string
      static SP createFromString(const std::string &wholeString);
      
      static SP createAndParse(const char *&curr);
      virtual void parse(const char *&currentParsePos);
    
    };
  
    // ==================================================================
    template<int D=3>
    struct BoxGenerator {
      typedef std::shared_ptr<BoxGenerator<D>> SP;

      /*! create a set of requested number of elements with given
          generator seed*/
      virtual std::vector<box_t<double,D>> generate(int numRequested, int seed) = 0;
      
      static SP createFromString(const std::string &wholeString);
    
      static SP createAndParse(const char *&curr);
      virtual void parse(const char *&currentParsePos);
    };




    // ==================================================================
    template<int D=3>
    struct UniformPointGenerator : public PointGenerator<D>
    {
      UniformPointGenerator(vec_t<double,D> lower = -defaultDomainSize(),
                            vec_t<double,D> upper = +defaultDomainSize())
        : lower(lower), upper(upper)
      {}
                            
      std::vector<vec_t<double,D>> generate(int numRequested, int seed) override;
      vec_t<double,D> lower, upper;
    };

    template<int D=3>
    struct UniformBoxGenerator : public BoxGenerator<D>
    {
      std::vector<box_t<double,D>> generate(int numRequested, int seed) override;
    };
  
    /*! re-maps points from the 'default domain' to the domain specified
      by [lower,upper]. Ie, for float the defulat domain is [0,1]^N,
      so a poitn with coordinate x=1 would be re-mapped to
      x=lower. Note this does not require the input points to *be*
      inside that default domain - if they are outside the domain the
      generated points will simply be outside the target domain,
      too 

      To create via generator string, use the string "remap x0 y0
      ... x1 y1 ... <source>", where x0,x1 etc are the lower coordinates of
      the target domain; x1, y1 etc are the upper bounds of the target
      domain, and <source> is another generator that produces the input
      points. E.g., assuimng we'd be dealing with <int,2> data, the string
      "remap 2 2 4 4 nrooks" would first generate points with the "nrooks"
      generator, then re-map those to [(2,2),(4,4)].
    */
    template<int D=3>
    struct RemapPointGenerator : public PointGenerator<D>
    {
      RemapPointGenerator();
    
      std::vector<vec_t<double,D>> generate(int numRequested, int seed) override;
    
      virtual void parse(const char *&currentParsePos);

      vec_t<double,D> lower, upper;
      typename PointGenerator<D>::SP source;
    };
    template<int D=3>
    struct RemapBoxGenerator : public BoxGenerator<D>
    {
      RemapBoxGenerator();
    
      std::vector<box_t<double,D>> generate(int numRequested, int seed) override;
    
      virtual void parse(const char *&currentParsePos);

      vec_t<double,D> lower, upper;
      typename BoxGenerator<D>::SP source;
    };



    // ==================================================================
    template<int D=3>
    struct ClusteredPointGenerator : public PointGenerator<D>
    {
      std::vector<vec_t<double,D>> generate(int numRequested, int seed) override;
    };
  
    template<int D=3>
    struct ClusteredBoxGenerator : public BoxGenerator<D>
    {
      void parse(const char *&currentParsePos) override;
      std::vector<box_t<double,D>> generate(int numRequested, int seed) override;
      
      struct {
        float mean = -1.f, sigma = 0.f, scale = 1.f;
      } gaussianSize;
      struct {
        float min = -1.f, max = -1.f;
      } uniformSize;
    };

    // ==================================================================
    /*! "nrooks": generate ~sqrt(N) N clusters of around sqrt(N)
        points each, and arrange thsoe in a n-rooks patterns */
    template<int D=3>
    struct NRooksPointGenerator : public PointGenerator<D>
    {
      std::vector<vec_t<double,D>> generate(int numRequested, int seed) override;
    };

    // ==================================================================
    /*! "nrooks": same as n-rooks point generator (for the box centers),
      then surrounds each of these points with a box whose size can be
      controlled through various distributions */
    template<int D=3>
    struct NRooksBoxGenerator : public BoxGenerator<D>
    {
      std::vector<box_t<double,D>> generate(int numRequested, int seed) override;
      void parse(const char *&currentParsePos) override;
      struct {
        float mean = -1.f, sigma = 0.f, scale = 1.f;
      } gaussianSize;
      struct {
        float min = -1.f, max = -1.f;
      } uniformSize;
    };

    // ==================================================================
    /*! takes a file of triangles, and creates one box per
      triangle. will ignore the number of requested samples, and just
      return the boxes. Will only work for float3 data, and error-exit
      for all other cases T,D configurations. 

      *must* be created with a a generator string that specifies a
      file (and format) to read those triangles from; this is
      specified through two strings: one for the format ('obj' for
      .obj files), and a second with a file name. E.g., to read
      triangles from bunny.obj, just the generator string "triangles
      obj bunny.obj"
    */
    template<int D=3>
    struct TrianglesBoxGenerator : public BoxGenerator<D>
    {
      std::vector<box_t<double,D>> generate(int numRequested, int seed) override;
    
      void parse(const char *&currentParsePos) override;
    
      std::vector<cuBQL::Triangle> triangles;
    };

    // ==================================================================
    /*! takes a file of triangles, then generates points by sampling
        these proportional to their surface area

      *must* be created with a a generator string that specifies a
      file (and format) to read those triangles from; this is
      specified through two strings: one for the format ('obj' for
      .obj files), and a second with a file name. E.g., to read
      triangles from bunny.obj, just the generator string "triangles
      obj bunny.obj"
    */
    template<int D=3>
    struct TrianglesPointGenerator : public PointGenerator<D>
    {
      std::vector<vec_t<double,D>> generate(int numRequested, int seed) override;
    
      void parse(const char *&currentParsePos) override;
    
      std::vector<cuBQL::Triangle> triangles;
    };

    // ==================================================================
    
    /*! "mixture" generator - generates a new distributoin based by
      randomly picking between two input distributions */
    template<int D=3>
    struct MixturePointGenerator : public PointGenerator<D> {
      std::vector<vec_t<double,D>> generate(int numRequested, int seed) override;
    
      void parse(const char *&currentParsePos) override;
    
      typename PointGenerator<D>::SP gen_a;
      typename PointGenerator<D>::SP gen_b;
      float prob_a;
    };

    /*! "mixture" generator - generates a new distributoin based by
      randomly picking between two input distributions */
    template<int D=3>
    struct MixtureBoxGenerator : public BoxGenerator<D> {
      std::vector<box_t<double,D>> generate(int numRequested, int seed) override;
    
      void parse(const char *&currentParsePos) override;
    
      typename BoxGenerator<D>::SP gen_a;
      typename BoxGenerator<D>::SP gen_b;
      float prob_a;
    };

  } // ::cuBQL::samples
} // ::cuBQL


