// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/bvh.h"
#include "cuBQL/queries/pointData/findClosest.h"
#include "samples/common/CmdLine.h"
#include "samples/common/IO.h"
#include "testing/common/testRig.h"

namespace testing {
  
  using namespace cuBQL;
  using namespace cuBQL::samples;
      
  using vecND = cuBQL::vec_t<double,CUBQL_TEST_D>;
  using box_t = cuBQL::box_t<CUBQL_TEST_T,CUBQL_TEST_D>;
  using point_t = cuBQL::vec_t<CUBQL_TEST_T,CUBQL_TEST_D>;
  using bvh_t = cuBQL::bvh_t<CUBQL_TEST_T,CUBQL_TEST_D>;

  inline __cubql_both
  float runQuery(bvh_t bvh,
                 const point_t *data,
                 point_t query)
  {
    int closestID
      = cuBQL::points::findClosest(bvh,data,query);
    if (closestID < 0)
      return INFINITY;
    point_t closest = data[closestID];
    return sqrtf(cuBQL::fSqrDistance_rd(closest,query));
  }

  double differenceThreshold() {
    return is_real<CUBQL_TEST_T>::value
      ? 1e-3f
      /* due to rounding coords, difference can be 2 off in each dim */
      : sqrtf(CUBQL_TEST_D*2*2);
  }
   
  void computeBoxes(box_t *d_boxes,
                    const point_t *d_data,
                    int numData);
  bvh_t computeBVH(const box_t *d_boxes,
                   int numBoxes);
  void launchQueries(bvh_t bvh,
                     const point_t  *d_data,
                     float         *d_results,
                     const point_t *d_queries,
                     int            numQueries);
  void computeReferenceResults(const point_t  *d_data,
                               int            numData,
                               float      *d_results,
                               const point_t *d_queries,
                               int            numQueries);
  void free(bvh_t bvh);

  void usage(const std::string &error = "")
  {
    if (!error.empty())
      std::cout << "Error: " << error << "\n\n";
    std::cout << "Usage: ./cuBQL...closestPoint... -d dataPoints.bin -q queryPoints.bin -g goldResults.bin [--rebuild-gold]" << std::endl;
    exit(error.empty()?0:1);
  }
      
  void main(int ac, char **av,
            cuBQL::testRig::DeviceAbstraction &device)
  {
    std::string dataFileName;
    std::string queryFileName;
    std::string goldFileName;
    bool slowReference = false;
    bool rebuildGold = false;
    int numPrint = 0;

    CmdLine cmdLine(ac,av);
    while (!cmdLine.consumed()) {
      const std::string arg = cmdLine.getString();
      if (arg == "-g")
        goldFileName = cmdLine.getString();
      else if (arg == "-d")
        dataFileName = cmdLine.getString();
      else if (arg == "-q")
        queryFileName = cmdLine.getString();
      else if (arg == "-r" || arg == "--rebuild-gold")
        rebuildGold = true;
      else if (arg == "-p") 
        numPrint = cmdLine.getInt();
      else if (arg == "--slow-reference") 
        slowReference = true;
      else
        usage("un-recognized cmd-line argument '"+arg+"'");
    }
    if (dataFileName.empty()) usage("no data file name specified");
    if (queryFileName.empty()) usage("no query file name specified");
    if (goldFileName.empty()) usage("no gold file name specified");

    std::vector<point_t>  dataPoints
      = cuBQL::samples::convert<CUBQL_TEST_T>(loadBinary<vecND>(dataFileName));
    std::vector<point_t> queryPoints
      = cuBQL::samples::convert<CUBQL_TEST_T>(loadBinary<vecND>(queryFileName));
        
    const point_t *d_dataPoints
      = device.upload(dataPoints);
    int numData = int(dataPoints.size());
        
    const point_t *d_queries
      = device.upload(queryPoints);
    int numQueries = int(queryPoints.size());
        
    float *d_results
      = device.alloc<float>(queryPoints.size());

    std::cout << "computing boxes for bvh build" << std::flush << std::endl;
    int numBoxes = numData;
    box_t *d_boxes = device.alloc<box_t>(numBoxes);
    computeBoxes(d_boxes,d_dataPoints,numData);
        
    std::cout << "computing bvh" << std::flush << std::endl;
    bvh_t bvh = computeBVH(d_boxes,numBoxes);

    if (slowReference) {
      std::cout << "computing slow, non-accelerated reference" << std::flush << std::endl;
      computeReferenceResults(d_dataPoints,numData,
                              d_results,d_queries,numQueries);
    } else {
      std::cout << "launching queries" << std::flush << std::endl;
      launchQueries(bvh,d_dataPoints,
                    d_results,d_queries,numQueries);
    }
    std::cout << "queries done, downloading results..." << std::flush << std::endl;
    std::vector<float> results = device.download(d_results,queryPoints.size());
    if (numPrint != 0) 
      for (int i=0;i<(int)results.size();i++) 
        if (numPrint < 0 || i < numPrint)
          std::cout << "[" << i << "] res = " << results[i] << std::endl;
    if (rebuildGold) {
      std::cout << "#cuBQL: asked to _rebuild_ gold results, "
                << "so just saving instead of comparing" << std::endl;
      saveBinary(goldFileName,results);
      for (int i=0;i<std::min(10,int(results.size())); i++)
        std::cout << "result[" << i << "] was " << results[i] << std::endl;
    } else {
      std::cout << "loading reference 'gold' results..." << std::flush << std::endl;
      std::vector<float> gold = loadBinary<float>(goldFileName);

      bool foundResultAboveThreshold = false;
      double avg_gold = 0.f;
      int    numFinite = 0;
      for (float r : gold)
        if (std::isfinite(r)) {
          avg_gold += r;
          numFinite++;
        } 
      avg_gold /= numFinite;

      double avg_diff = 0.f;
      double max_diff = 0.f;
      numFinite = 0;
      double threshold = differenceThreshold();
      for (int i=0;i<(int)results.size();i++) {
        double difference = fabsf(results[i] - gold[i]);
        if (std::isfinite(difference)) {
          max_diff = std::max(max_diff,difference);
          avg_diff += difference;
          numFinite++;
        }
        if (difference >= threshold) {
          std::cout << "!! different result for index "
                    << i << ": ours says " << results[i] << ", gold says " << gold[i]
                    << ", that's a difference of " << difference
                    << std::endl;
          foundResultAboveThreshold = true;
        }
      }
      avg_diff /= numFinite;
      std::cout << "avg result                   " << avg_gold << std::endl;
      std::cout << "avg difference (rel to gold) " << avg_diff << std::endl;
      std::cout << "max difference (rel to gold) " << max_diff << std::endl;
      std::cout << "(note tolerance threshold (due to rounding for given type) is "
                << threshold << ")" << std::endl;
      if (foundResultAboveThreshold)
        throw std::runtime_error
          ("round (at least one) result outside of expected error tolerance threshold!");
    }
    device.free(d_results);
    device.free(d_dataPoints);
    device.free(d_queries);
  }

} // ::testing

