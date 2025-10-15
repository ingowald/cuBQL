// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "samples/common/Generator.h"
#include "cuBQL/math/random.h"
#include <exception>
#include <string>
#include <cstring>

namespace cuBQL {
  namespace samples {

    namespace tokenizer {
      std::string findFirst(const char *curr, const char *&endOfFound)
      {
        if (curr == 0)
          return "";
      
        while (*curr && strchr(" \t\r\n",*curr)) {
          ++curr;
        }
        if (*curr == 0) {
          endOfFound = curr;
          return "";
        }
      
        std::stringstream ss;
        if (strchr("[]{}():,",*curr)) {
          ss << *curr++;
        } else if (isalnum(*curr) || *curr && strchr("+-.",*curr)) {
          while (isalnum(*curr) || *curr && strchr("+-.",*curr)) {
            ss << *curr;
            curr++;
          }
        }
        else
          throw std::runtime_error("unable to parse ... '"+std::string(curr)+"'");
      
        endOfFound = curr;
        return ss.str();
      }
    };

    template<typename T> inline T to_scalar(const std::string &s);
    template<> inline double to_scalar<double>(const std::string &s)
    { return std::stof(s); }
    template<> inline int to_scalar<int>(const std::string &s)
    { return std::stoi(s); }

    template<int D>
    vec_t<double,D> parseVector(const char *&curr)
    {
      const char *next = 0;
      std::string tok = tokenizer::findFirst(curr,next);
      assert(tok != "");
      curr = next;
      if (tok == "[") {
        std::vector<double> values;
        while(1) {
          tok = tokenizer::findFirst(curr,next);
          assert(tok != "");
          curr = next;
          if (tok == "]") break;
          values.push_back(to_scalar<double>(tok));
          std::string tok = tokenizer::findFirst(curr,next);
        }
        assert(!values.empty());
        vec_t<double,D> ret;
        for (int i=0;i<D;i++)
          ret[i] = values[i % values.size()];
        return ret;
      } else
        return vec_t<double,D>(std::stof(tok));
    }
    
    // ==================================================================
    // point generator base
    // ==================================================================
    template<int D>
    typename PointGenerator<D>::SP
    PointGenerator<D>::createAndParse(const char *&curr)
    {
      const char *next = 0;
      std::string type = tokenizer::findFirst(curr,next);
      if (type == "") throw std::runtime_error("could not parse generator type");

      typename PointGenerator<D>::SP gen;
      if (type == "uniform")
        gen = std::make_shared<UniformPointGenerator<D>>();
      else if (type == "clustered")
        gen = std::make_shared<ClusteredPointGenerator<D>>();
#if 0
      else if (type == "nrooks")
        gen = std::make_shared<NRooksPointGenerator<D>>();
#endif
      else if (type == "mixture")
        gen = std::make_shared<MixturePointGenerator<D>>();
      else if (type == "remap")
        gen = std::make_shared<RemapPointGenerator<D>>();
      else
        throw std::runtime_error("un-recognized point generator type '"+type+"'");
      curr = next;
      gen->parse(curr);
      return gen;
    }


    template<int D>
    typename PointGenerator<D>::SP
    PointGenerator<D>::createFromString(const std::string &wholeString)
    {
      const char *curr = wholeString.c_str(), *next = 0;
      SP generator = createAndParse(curr);
      std::string trailing = tokenizer::findFirst(curr,next);
      if (!trailing.empty())
        throw std::runtime_error("un-recognized trailing stuff '"
                                 +std::string(curr)
                                 +"' at end of point generator string");
      return generator;
    }
  
    template<int D>
    void PointGenerator<D>::parse(const char *&currentParsePos)
    {}

    // ==================================================================



  
    // ==================================================================
    // box generator base
    // ==================================================================
    template<int D>
    typename BoxGenerator<D>::SP
    BoxGenerator<D>::createAndParse(const char *&curr)
    {
      const char *next = 0;
      std::string type = tokenizer::findFirst(curr,next);
      if (type == "") throw std::runtime_error("could not parse generator type");

      typename BoxGenerator<D>::SP gen;
      if (type == "uniform")
        gen = std::make_shared<UniformBoxGenerator<D>>();
      else if (type == "clustered")
        gen = std::make_shared<ClusteredBoxGenerator<D>>();
      // else if (type == "nrooks")
      //   gen = std::make_shared<NRooksBoxGenerator<D>>();
      else if (type == "remap")
        gen = std::make_shared<RemapBoxGenerator<D>>();
      else if (type == "mixture")
        gen = std::make_shared<MixtureBoxGenerator<D>>();
      else
        throw std::runtime_error("un-recognized box generator type '"+type+"'");
      curr = next;
      gen->parse(curr);
      return gen;
    }


    template<int D>
    typename BoxGenerator<D>::SP
    BoxGenerator<D>::createFromString(const std::string &wholeString)
    {
      const char *curr = wholeString.c_str(), *next = 0;
      SP generator = createAndParse(curr);
      std::string trailing = tokenizer::findFirst(curr,next);
      if (!trailing.empty())
        throw std::runtime_error("un-recognized trailing stuff '"
                                 +std::string(curr)
                                 +"' at end of box generator string");
      return generator;
    }
  
    template<int D>
    void BoxGenerator<D>::parse(const char *&currentParsePos)
    {}

    template struct BoxGenerator<2>;
    template struct BoxGenerator<3>;
    template struct BoxGenerator<4>;
#if CUBQL_TEST_N
    template struct BoxGenerator<CUBQL_TEST_N>;
#endif
    // ==================================================================
  
  
    template<int D>
    inline 
    void uniformPointGenerator(std::vector<vec_t<double,D>> &points,
                               vec_t<double,D> lower,
                               vec_t<double,D> upper,
                               int seed)
    {
      for (int tid=0;tid<(int)points.size();tid++) {
        LCG<8> rng(seed,tid);
        auto &mine = points[tid];
        
        // double lo = lower[othis->lower;//- defaultDomainSize();
        // double hi = this->upper;//+ defaultDomainSize();
        
        for (int i=0;i<D;i++)
          mine[i] = lower[i] + rng() * (upper[i] - lower[i]);
      }
    }

    template<int D>
    std::vector<vec_t<double,D>>
    UniformPointGenerator<D>::generate(int count, int seed)
    {
      if (count <= 0)
        throw std::runtime_error("UniformPointGenerator<D>::generate(): invalid count...");
      std::vector<vec_t<double,D>> res(count);
      uniformPointGenerator<D>(res,lower,upper,seed);
      return res;
    }

    // ------------------------------------------------------------------
    template<int D>
    void uniformBoxGenerator(std::vector<box_t<double,D>> &boxes,
                             int seed, double size)
    {
      for (int tid=0;tid<int(boxes.size());tid++) {
        box_t<double,D> &mine = boxes[tid];
        
        LCG<8> rng(seed,tid);
        
        double lo = - defaultDomainSize();
        double hi = + defaultDomainSize();
        
        vec_t<double,D> center;
        for (int i=0;i<D;i++)
          center[i] = lo + rng() * (hi-lo);
        
        for (int i=0;i<D;i++) {
          double scalarSize = (hi - lo) * size;
          mine.lower[i] = center[i] - scalarSize/2;
          mine.upper[i] = mine.lower[i] + scalarSize;
        }
      }
    }
    
    template<int D>
    std::vector<box_t<double,D>>
    UniformBoxGenerator<D>::generate(int count, int seed)
    {
      if (count <= 0)
        throw std::runtime_error("UniformBoxGenerator<D>::generate(): invalid count...");
      std::vector<box_t<double,D>> res(count);
      double size = 0.5 / pow((double)count,double(1.f/D));
      uniformBoxGenerator<D>(res,seed,size);
      return res;
    }

    // ==================================================================
    // re-mapping
    // ==================================================================
  
    template<int D>
    RemapPointGenerator<D>::RemapPointGenerator()
    {
      for (int d=0;d<D;d++) {
        double lo = - defaultDomainSize();
        double hi = + defaultDomainSize();
        
        lower[d] = lo;
        upper[d] = hi;
      }
    }

    template<int D>
    std::vector<vec_t<double,D>>
    RemapPointGenerator<D>::generate(int count, int seed)
    {
      if (!source)
        throw std::runtime_error("RemapPointGenerator: no source defined");
      std::vector<vec_t<double,D>> pts = source->generate(count,seed);

      box_t<double,D> bbox;
      for (auto pt : pts)
        bbox.extend(pt);
      
      for (auto &point : pts) {
        for (int d=0;d<D;d++) {
          double v = point[d];
          v = v - bbox.lower[d];
          if (bbox.lower[d] != bbox.upper[d])
            v = v / (double(bbox.upper[d] - bbox.lower[d]));

          v = v * (double)(upper[d]-lower[d]) + (double)lower[d];
          point[d] = v;
        }
      }
      return pts;
    }

    template<int D>
    void RemapPointGenerator<D>::parse(const char *&currentParsePos)
    {
      // const char *next = 0;
      // for (int d=0;d<D;d++) {
      //   std::string tok = tokenizer::findFirst(currentParsePos,next);
      //   assert(tok != "");
      //   lower[d] = to_scalar<double>(tok);
      //   currentParsePos = next;
      // }
      // for (int d=0;d<D;d++) {
      //   std::string tok = tokenizer::findFirst(currentParsePos,next);
      //   assert(tok != "");
      //   upper[d] = to_scalar<double>(tok);
      //   currentParsePos = next;
      // }
      lower = parseVector<D>(currentParsePos);
      upper = parseVector<D>(currentParsePos);
      source = PointGenerator<D>::createAndParse(currentParsePos);
    }

    // ------------------------------------------------------------------

    template<int D>
    RemapBoxGenerator<D>::RemapBoxGenerator()
    {
      for (int d=0;d<D;d++) {
        lower[d] = - defaultDomainSize();
        upper[d] = + defaultDomainSize();
      }
    }

    template<int D>
    std::vector<box_t<double,D>>
    RemapBoxGenerator<D>::generate(int count, int seed)
    {
      if (!source)
        throw std::runtime_error("RemapBoxGenerator: no source defined");
      std::vector<box_t<double,D>> boxes
        = source->generate(count,seed);

      for (auto &box : boxes) {
        for (int d=0;d<D;d++)
          box.lower[d]
            = lower[d]
            + box.lower[d]
            * (typename dot_result_t<double>::type)(upper[d]-lower[d])
            / (defaultDomainSize() - -defaultDomainSize());
        for (int d=0;d<D;d++)
          box.upper[d]
            = lower[d]
            + box.upper[d]
            * (typename dot_result_t<double>::type)(upper[d]-lower[d])
            / (defaultDomainSize() - -defaultDomainSize());
      }
      return boxes;
    }

    template<int D>
    void RemapBoxGenerator<D>::parse(const char *&currentParsePos)
    {
      // const char *next = 0;
      lower = parseVector<D>(currentParsePos);
      upper = parseVector<D>(currentParsePos);
      // for (int d=0;d<D;d++) {
      //   std::string tok = tokenizer::findFirst(currentParsePos,next);
      //   assert(tok != "");
      //   lower[d] = to_scalar<double>(tok);
      //   currentParsePos = next;
      // }
      // for (int d=0;d<D;d++) {
      //   std::string tok = tokenizer::findFirst(currentParsePos,next);
      //   assert(tok != "");
      //   upper[d] = to_scalar<double>(tok);
      //   currentParsePos = next;
      // }
      source = BoxGenerator<D>::createAndParse(currentParsePos);
    }


  
    // ==================================================================
    // clustered points
    // ==================================================================

    template<int D>
    std::vector<vec_t<double,D>>
    ClusteredPointGenerator<D>::generate(int count, int seed)
    {
      std::default_random_engine rng;
      rng.seed(seed);

      double lo = - defaultDomainSize();
      double hi = + defaultDomainSize();
        
      double width = hi - lo;
        
      std::uniform_real_distribution<double> uniform(lo,hi);

      int numClusters = int(1+pow((double)count,(D-1.)/D));
      // = int(1+powf(count/50.f);
      // = int(1+sqrtf(count));
      // = this->numClusters
      // ? this->numClusters
      // : int(1+sqrtf(count));
      std::vector<vec_t<double,D>> clusterCenters;
      for (int cc=0;cc<numClusters;cc++) {
        vec_t<double,D> c;
        for (int i=0;i<D;i++)
          c[i] = uniform(rng);
        clusterCenters.push_back(c);
      }
      
      double sigma = (hi-lo)/numClusters;
      std::normal_distribution<double> gaussian(0.f,sigma);
      std::uniform_int_distribution<int> uniform_clusterID(0,numClusters-1);
      std::vector<vec_t<double,D>> points;
      for (int sID=0;sID<count;sID++) {
        int clusterID = uniform_clusterID(rng);
        vec_t<double,D> pt;
        for (int i=0;i<D;i++)
          pt[i] = gaussian(rng) + clusterCenters[clusterID][i];
        points.push_back(pt);
      }

      return points;
    }
  
    template<int D>
    void ClusteredBoxGenerator<D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;
      while (true) {
        const std::string tag = tokenizer::findFirst(currentParsePos,next);
        if (tag == "")
          break;

        if (tag == "gaussian") {
          currentParsePos = next;
          
          std::string sMean = tokenizer::findFirst(currentParsePos,next);
          currentParsePos = next;
        
          std::string sSigma = tokenizer::findFirst(currentParsePos,next);
          currentParsePos = next;
        
          gaussianSize.mean = std::stof(sMean);
          gaussianSize.sigma = std::stof(sSigma);
        } else if (tag == "gaussian.scale") {
          currentParsePos = next;
          
          std::string scale = tokenizer::findFirst(currentParsePos,next);
          assert(scale != "");
          currentParsePos = next;
          gaussianSize.scale = std::stof(scale); 
        } else {
          break;
        }
      }
    }
    
    template<int D>
    std::vector<box_t<double,D>>
    ClusteredBoxGenerator<D>::generate(int count, int seed)
    {
      std::default_random_engine rng;
      rng.seed(seed);
      std::uniform_real_distribution<double> uniform(0.f,1.f);
  
      int numClusters = int(1+pow((double)count,(D-1.f)/D));
      // int numClusters
      //   = int(1+count/50.f);
      std::vector<vec_t<double,D>> clusterCenters;
      for (int cc=0;cc<numClusters;cc++) {
        vec_t<double,D> c;
        for (int i=0;i<D;i++)
          c[i] = uniform(rng);
        clusterCenters.push_back(c);
      }

      double sigma = 2*defaultDomainSize()/numClusters;
    
      std::normal_distribution<double> gaussian(0.f,sigma);

      double sizeMean = -1.f, sizeSigma = 0.f;
      if (gaussianSize.mean > 0) {
        std::cout << "choosing size using gaussian distribution..." << std::endl;
        sizeMean = gaussianSize.mean*gaussianSize.scale;
        sizeSigma = gaussianSize.sigma;
      } else if (uniformSize.min > 0) {
        std::cout << "choosing size using uniform min/max distribution..." << std::endl;
        sizeMean = -1.f;
      } else {
        std::cout << "choosing size using auto-chosen gaussian..." << std::endl;
        double avgClusterWidth = 4*sigma;
        sizeMean = .5f*avgClusterWidth*gaussianSize.scale;
        sizeSigma = sizeMean/3.f;
        std::cout << "choosing size using auto-config'ed gaussian"
                  << " mean=" << sizeMean
                  << " sigma=" << sizeSigma
                  << std::endl;
      }
    
      std::normal_distribution<double> sizeGaussian(sizeMean,sizeSigma);
      std::uniform_int_distribution<int> uniform_clusterID(0,numClusters-1);
      std::vector<box_t<double,D>> boxes;
      for (int sID=0;sID<count;sID++) {
        int clusterID = uniform_clusterID(rng);
        vec_t<double,D> center, halfSize;
        for (int i=0;i<D;i++)
          center[i] = gaussian(rng) + clusterCenters[clusterID][i];

        if (sizeMean > 0) {
          for (int d=0;d<D;d++)
            halfSize[d] = fabs(0.5*(double)sizeGaussian(rng));
        } else {
          for (int d=0;d<D;d++)
            halfSize[d]
              = uniformSize.min
              + (uniformSize.max-uniformSize.min) * uniform(rng);
        }
        box_t<double,D> box;
        box.lower = center - halfSize;
        box.upper = center + halfSize;
        boxes.push_back(box);
      }

      return boxes;
    }
  

#if 0
    // ==================================================================
    // "nrooks": generate N clusters of ~50 points each, then arrange
    // these N clusters in a NxNx...xN grid with a N-rooks pattern. Each
    // of these clusters has ~50 uniformly distributed points inside of
    // that clusters "field"
    // ==================================================================
    template<int D>
    std::vector<vec_t<double,D>>
    NRooksPointGenerator<D>::generate(int count, int seed)
    {
      int numClusters = (int)(1+powf((double)count,0.5f*(D-1.f)/D));
      LCG<8> rng(seed,290374);
      std::vector<vec_t<double,D>> clusterLower(numClusters);
      for (int d=0;d<D;d++) {
        for (int i=0;i<numClusters;i++) {
          clusterLower[i][d] = i/(double)numClusters;
        }
        for (int i=numClusters-1;i>0;--i) {
          int o = rng.ui32() % (i+1);
          if (i != o)
            std::swap(clusterLower[i][d],clusterLower[o][d]);
        }
      }
    
      std::vector<vec_t<double,D>> points(count);
      for (int i=0;i<count;i++) {
        int clusterID = rng.ui32() % numClusters;
        for (int d=0;d<D;d++)
          points[i][d] = clusterLower[clusterID][d] + (1.f/numClusters)*rng();
      }
    
      return points;
    }

    template<int D>
    void NRooksBoxGenerator<D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;
      while (true) {
        const std::string tag = tokenizer::findFirst(currentParsePos,next);
        if (tag == "")
          break;

        if (tag == "gaussian") {
          currentParsePos = next;
          
          std::string sMean = tokenizer::findFirst(currentParsePos,next);
          currentParsePos = next;
        
          std::string sSigma = tokenizer::findFirst(currentParsePos,next);
          currentParsePos = next;
        
          gaussianSize.mean = std::stof(sMean);
          gaussianSize.sigma = std::stof(sSigma);
        } else if (tag == "gaussian.scale") {
          currentParsePos = next;
          
          std::string scale = tokenizer::findFirst(currentParsePos,next);
          assert(scale != "");
          currentParsePos = next;
          gaussianSize.scale = std::stof(scale); 
        } else {
          break;
        }
      }
    }
  
    template<int D>
    std::vector<box_t<double,D>>
    NRooksBoxGenerator<D>::generate(int count, int seed)
    {
      int numClusters = (int)(1+powf((double)count,0.5f*(D-1.f)/D));
      LCG<8> lcg(seed,290374);
      std::vector<vec_t<double,D>> clusterLower(numClusters);
      for (int d=0;d<D;d++) {
        for (int i=0;i<numClusters;i++) {
          clusterLower[i][d] = i/(double)numClusters;
        }
        for (int i=numClusters-1;i>0;--i) {
          int o = lcg.ui32() % (i+1);
          if (i != o)
            std::swap(clusterLower[i][d],clusterLower[o][d]);
        }
      }

      double sizeMean = -1.f, sizeSigma = 0.f;
      if (gaussianSize.mean > 0) {
        sizeMean = gaussianSize.mean;
        sizeSigma = gaussianSize.sigma;
        std::cout << "choosing size using user-supplied gaussian"
                  << " mean=" << sizeMean
                  << " sigma=" << sizeSigma
                  << std::endl;
      } else if (uniformSize.min > 0) {
        std::cout << "choosing size using uniform min/max distribution..." << std::endl;
        sizeMean = -1.f;
      } else {
        std::cout << "choosing size using auto-chosen gaussian..." << std::endl;
        double avgClusterWidth = 1.f/numClusters;
        // double avgClusterWidth = 4*sigma;
        // int avgBoxesPerCluster = count / numClusters;
        sizeMean = .5f*avgClusterWidth*gaussianSize.scale;//powf(avgBoxesPerCluster,1.f/D);
        sizeSigma = sizeMean/3.f;
        std::cout << "choosing size using auto-config'ed gaussian"
                  << " mean=" << sizeMean
                  << " sigma=" << sizeSigma
                  << std::endl;
        // std::cout << "choosing size using auto-chosen gaussian..." << std::endl;
        // // int avgBoxesPerCluster = count / numClusters;
        // double avgClusterWidth = 1.f/numClusters;
        // sizeMean = .1f*avgClusterWidth;//powf(avgBoxesPerCluster,1.f/D);
        // sizeSigma = sizeMean/3.f;
        // std::cout << "choosing size using auto-config'ed gaussian"
        //           << " mean=" << sizeMean
        //           << " sigma=" << sizeSigma
        //           << std::endl;
      }
    
      std::normal_distribution<double> sizeGaussian(sizeMean,sizeSigma);
      std::default_random_engine reng;
      std::uniform_real_distribution<double> uniform(0.f,1.f);
      reng.seed(seed+29037411);

      std::vector<box_t<double,D>> boxes;

      for (int i=0;i<count;i++) {
        int clusterID = lcg.ui32() % numClusters;
        vec_t<double,D> center;
        for (int d=0;d<D;d++) {
          center[d] = clusterLower[clusterID][d] + (1.f/numClusters)*lcg();
        }
      
        vec_t<double,D> halfSize;
        if (sizeMean > 0) {
          for (int d=0;d<D;d++)
            halfSize[d] = fabsf(0.5f*(double)sizeGaussian(reng));
        } else {
          for (int d=0;d<D;d++)
            halfSize[d]
              = T(uniformSize.min
                  + (uniformSize.max-uniformSize.min) * uniform(reng));
        }
        box_t<double,D> box;
        box.lower = center - halfSize;
        box.upper = center + halfSize;
        boxes.push_back(box);
      }

      return boxes;
    }
#endif

    // ==================================================================
    template<int D>
    std::vector<box_t<double,D>>
    TrianglesBoxGenerator<D>::generate(int numRequested, int seed)
    {
      throw std::runtime_error("can generate boxes from triangles only "
                               "for T=double and D=3");
    }

    template<>
    std::vector<box_t<double,3>>
    TrianglesBoxGenerator<3>::generate(int numRequested, int seed)
    {
      std::vector<box3d> boxes;
      for (auto tri : triangles)
        boxes.push_back(box3d(tri.bounds()));
      return boxes;
    }
    
    template<int D>
    void TrianglesBoxGenerator<D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;

      std::string format = tokenizer::findFirst(currentParsePos,next);
      if (format == "") throw std::runtime_error("no triangles file format specified");
      currentParsePos = next;

      std::string fileName = tokenizer::findFirst(currentParsePos,next);
      if (fileName == "") throw std::runtime_error("no file name specified");

      std::cout << "going to start reading triangles from '"
                << fileName << "'" << std::endl;
      triangles = loadBinary<Triangle>(fileName);
      std::cout << "done loading " << prettyNumber(triangles.size())
                << " triangles..." << std::endl;
      currentParsePos = next;
    }



    // ==================================================================
    template<int D>
    std::vector<vec_t<double,D>>
    TrianglesPointGenerator<D>::generate(int numRequested, int seed)
    {
      throw std::runtime_error("can generate sample points from triangles only "
                               "for T=double and D=3");
    }
    
    template<>
    std::vector<vec_t<double,3>>
    TrianglesPointGenerator<3>::generate(int numRequested, int seed)
    {
      double sumAreas = 0.f;
      std::vector<double> areas;
      std::vector<box3f> boxes;
      for (auto tri : triangles) {
        boxes.push_back(make_box<float,3>(tri.a).grow(tri.b).grow(tri.c));
        double a = area(tri);
        areas.push_back(a);
        sumAreas += a;
      }
      std::vector<double> cdf;
      double s = 0.f;
      for (auto a : areas) {
        s += a;
        cdf.push_back(s/sumAreas);
      }

      std::vector<vec_t<double,3>> points(numRequested);
      LCG<8> rng(seed,0);
      for (int tid=0;tid<numRequested;tid++) {
        double r_which = rng();
        auto it = std::lower_bound(cdf.begin(),cdf.end(),r_which);
        size_t triID = std::min(size_t(it - cdf.begin()),cdf.size()-1);
        auto triangle = triangles[triID];
        points.push_back(vec3d(triangle.sample(rng(),rng())));
      }
      
      return points;
    }
    
    template<int D>
    void TrianglesPointGenerator<D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;

      std::string format = tokenizer::findFirst(currentParsePos,next);
      if (format == "") throw std::runtime_error("no triangles file format specified");
      currentParsePos = next;

      std::string fileName = tokenizer::findFirst(currentParsePos,next);
      if (fileName == "") throw std::runtime_error("no file name specified");

      std::cout << "going to start reading triangles from '"
                << fileName << "'" << std::endl;
      triangles = loadBinary<Triangle>(fileName);
      std::cout << "done loading " << prettyNumber(triangles.size())
                << " triangles..." << std::endl;
      currentParsePos = next;
    }

    // ==================================================================
    /*! "mixture" generator - generates a new distribution based by
      randomly picking between two input distributions */
    template<int D>
    std::vector<box_t<double,D>> MixtureBoxGenerator<D>::generate(int numRequested, int seed)
    {
      assert(gen_a);
      assert(gen_b);
      std::vector<box_t<double,D>>  boxes_a
        = gen_a->generate(numRequested,3*seed+0);
      std::vector<box_t<double,D>>  boxes_b
        = gen_b->generate(numRequested,3*seed+1);

      std::vector<box_t<double,D>> boxes(numRequested);

      for (int tid=0;tid<boxes.size();tid++) {
        LCG<8> rng(3*seed+2,tid);
        bool use_a
          = (prob_a < 1.f)
          ? (rng() < prob_a)
          : (tid < (int)prob_a);
        const auto &in = (use_a ? boxes_a : boxes_b);
        size_t inCount  = in.size();
        size_t outCount = numRequested;
        
        size_t which
          = (((inCount == outCount) ? tid : rng.ui32())) % inCount;
        boxes[tid] = in[which];
      }
      return boxes;
    }
    
    template<int D>
    void MixtureBoxGenerator<D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;

      std::string prob = tokenizer::findFirst(currentParsePos,next);
      currentParsePos = next;
      
      if (prob == "") throw std::runtime_error("no mixture probabilty specified");

      prob_a = std::stof(prob);
      gen_a = BoxGenerator<D>::createAndParse(currentParsePos);
      gen_b = BoxGenerator<D>::createAndParse(currentParsePos);
    }
    
    
    // ==================================================================
    /*! "mixture" generator - generates a new distributoin based by
      randomly picking between two input distributions */
    template<int D>
    std::vector<vec_t<double,D>>
    MixturePointGenerator<D>::generate(int numRequested, int seed)
    {
      assert(gen_a);
      assert(gen_b);
      std::vector<vec_t<double,D>>  points_a
        = gen_a->generate(numRequested,3*seed+0);
      std::vector<vec_t<double,D>>  points_b
        = gen_b->generate(numRequested,3*seed+1);

      std::vector<vec_t<double,D>> points(numRequested);
      const vec_t<double,D> *a = points_a.data();
      const vec_t<double,D> *b = points_b.data();
      for (int tid=0;tid<numRequested;tid++) {
        LCG<8> rng(3*seed+2,tid);
        bool use_a
          = (prob_a < 1.f)
          ? (rng() < prob_a)
          : (tid < (int)prob_a);
        const auto &in = (use_a ? points_a : points_b);
        size_t inCount = in.size();
        size_t outCount = numRequested;
        
        size_t which
          = (inCount == outCount)
          ? tid
          : (rng.ui32() % inCount);
        points[tid] = in[which];
      }
      return points;
    }
    
    template<int D>
    void MixturePointGenerator<D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;

      std::string prob = tokenizer::findFirst(currentParsePos,next);
      currentParsePos = next;
      
      if (prob == "") throw std::runtime_error("no mixture probabilty specified");

      prob_a = std::stof(prob);
      gen_a = PointGenerator<D>::createAndParse(currentParsePos);
      gen_b = PointGenerator<D>::createAndParse(currentParsePos);
    }


    template struct PointGenerator<2>;
    template struct PointGenerator<3>;
    template struct PointGenerator<4>;

#if CUBQL_USER_DIM
    template struct PointGenerator<CUBQL_USER_DIM>;
#endif
  } // ::cuBQL::samples
} // ::cuBQL
