#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cpu.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/queries/triangleData/Triangle.h"
#include "samples/common/loadBinMesh.h"
#include <set>
#include <queue>

#define WIDTH 8

using namespace cuBQL;

using wbvh_t = WideBVH<float,3,WIDTH>;
using cbvh_t = CWideBVH<float,3,WIDTH>;
using node_t = typename cbvh_t::Node;
using wnode_t = typename wbvh_t::Node;

cbvh_t cbvh_copy(cbvh_t other);

// cbvh_t cbvh;

std::vector<box3f> boxes;

namespace cuBQL {
  inline bool operator<(const cuBQL::box3f &a,
                        const cuBQL::box3f &b)
  {
    return memcmp(&a,&b,sizeof(a)) < 0;//a.second < b.second;
  }
  
  inline std::ostream &operator<<(std::ostream &o,
                                  std::pair<int,cuBQL::box3f> p)
  { o << "{" << p.first << "," << p.second << "}"; return o; }
}

// inline bool operator<(std::pair<int,box3f> a,
//                       std::pair<int,box3f> b)
// {
//   if (a.first < b.first) return true;
//   if (b.first < a.first) return false;
//   return memcmp(&a,&b,sizeof(a)) < 0;//a.second < b.second;
// }

void addToDigest(std::vector<std::pair<int,box3f>> &digest,
                 cbvh_t bvh,
                 int nodeID,
                 int mineBegin,
                 int mineCount)
{
  auto node = bvh.nodes[nodeID];
  for (int ci=mineBegin;ci<mineBegin+mineCount;ci++) {
    auto child = node.children[ci];
    assert(child.valid);
    if (child.count) {
      box3f leafBox = child.bounds;
      digest.push_back({-1,leafBox});
      for (int li=0;li<child.count;li++) {
        int primID = bvh.primIDs[child.offset+li];
        digest.push_back({primID,leafBox});
      }
    } else {
      addToDigest(digest,bvh,child.offset,child.mineBegin,child.mineCount);
    }
  }
}



int numValid(node_t n)
{
  int sum = 0;
  for (int i=0;i<WIDTH;i++)
    if (n.children[i].valid) sum++;
  return sum;
}

int numValid(cbvh_t cbvh, uint64_t nodeID)
{
  if (nodeID >= cbvh.numNodes)
    throw std::runtime_error("INVALID NODEID IN NUMVALID "+std::to_string(nodeID));
    // std::cout << "INVALID NODEID IN NUMVALID " << nodeID << std::endl;
  // PING; PRINT(nodeID);
  return numValid(cbvh.nodes[nodeID]);
}


std::vector<std::pair<int,box3f>> computeDigest(cbvh_t bvh)
{
  std::vector<std::pair<int,box3f>> digest;
  addToDigest(digest,bvh,0,/*colorparent*/0,numValid(bvh,0));
  std::sort(digest.begin(),digest.end());
  return digest;
}
// int findMaxColorChild(node_t node)
// {
//   int mv =0;
//   for (int i=0;i<WIDTH;i++) {
//     auto c = node.children[i];
//     if (!c.valid) continue;
//     mv = std::max(mv,(int)c.colorChild);
//   }
//   return mv;
// }

// int findMaxColorChild(cbvh_t cbvh, uint64_t nodeID)
// {
//   auto node = cbvh.nodes[nodeID];
//   return findMaxColorChild(node);
// }


bool findMergablePair(cbvh_t cbvh, node_t &node, int &a, int &b)
{
  a = b = -1;
  int best = 0;
  for (int ia=0;ia<WIDTH;ia++) {
    auto ca = node.children[ia];
    if (!ca.valid) continue;
    if (ca.count > 0) continue;
    int cnt_a = numValid(cbvh,ca.offset);
    
    for (int ib=ia+1;ib<WIDTH;ib++) {
      auto cb = node.children[ib];
      if (!cb.valid) continue;
      if (cb.offset == ca.offset) continue;
      if (cb.count > 0) continue;

      int cnt_b = numValid(cbvh,cb.offset);
      if (cnt_a + cnt_b <= WIDTH && cnt_a + cnt_b > best) {
        a = ia;
        b = ib;
        best = cnt_a + cnt_b;
      }
    }
  }
  return best > 0;
  // // std::cout << "findmergerable" << std::endl;
  // std::vector<std::pair<int,int>> candidates;
  // for (int i=0;i<WIDTH;++i) {
  //   if (!node.children[i].valid) continue;
  //   if (node.children[i].count > 0) continue;

  //   // std::cout << " child " << i << " " << node.children[i].offset << ","
  //   //           << node.children[i].count
  //   //           << " val " << (int)node.children[i].valid
  //   //           << std::endl;
  //   candidates.push_back(std::pair<int,int>{numValid(cbvh,node.children[i].offset),i});
  // }
  // std::sort(candidates.begin(),candidates.end());

  // int cur = candidates.size()-1;
  // int mrg;
  // while (cur > 0) {
  //   mrg = cur-1;
  //   while(mrg >= 0) {
  //     int cur_ofs = node.children[candidates[cur].second].offset;
  //     int cur_cnt = numValid(cbvh,cur_ofs);

  //     int mrg_ofs = node.children[candidates[mrg].second].offset;
  //     int mrg_cnt = numValid(cbvh,mrg_ofs);

  //     if (cur_cnt + mrg_cnt > WIDTH || cur_ofs == mrg_ofs) {
  //       --mrg;
  //       continue;
  //     }

  //     a = candidates[cur].second;
  //     b = candidates[mrg].second;
  //     return true;
  //   }
  //   --cur;
  // }
  // return false;
}

void printNode(cbvh_t cbvh, node_t node)
{
  for (int i=0;i<WIDTH;i++) {
    std::cout << " " << i
              << " ofs=" << node.children[i].offset 
              << " cnt=" << node.children[i].count 
              << " val=" << (int)node.children[i].valid
              << " mbeg=" << (int)node.children[i].mineBegin
              << " mcnt=" << (int)node.children[i].mineCount
              // << " col=" << (int)node.children[i].colorParent
              << std::endl;
    if (0 &&
        node.children[i].valid && node.children[i].count == 0) {
      node_t nn = cbvh.nodes[node.children[i].offset];
      for (int j=0;j<WIDTH;j++) {
        if (nn.children[j].valid)
        std::cout << "  -> " << j
                  << " ofs=" << nn.children[j].offset 
                  << " cnt=" << nn.children[j].count 
                  << " val=" << (int)nn.children[j].valid
                  // << " col=" << (int)nn.children[j].colorChild
                  << std::endl;
      }
    }
  }
}

void mergeChildren(cbvh_t cbvh, node_t &node, int a, int b)
{
  // std::cout << "=== merging " << a << " " << b << std::endl;
  auto ca = node.children[a];
  auto cb = node.children[b];

  node_t &na = cbvh.nodes[ca.offset];
  // node_t &nb = cbvh.nodes[cb.offset];
  int writePos = numValid(cbvh,ca.offset);
  // int newColor = findMaxColorChild(na)+1;

  for (int j=0;j<WIDTH;j++) {
    auto &cj = node.children[j];
    if (!cj.valid) continue;
    if (cj.count > 0) continue;
    if (cj.offset != cb.offset) continue;

    node_t &nb = cbvh.nodes[cb.offset];
    for (int i=0;i<cj.mineCount;i++) 
      na.children[writePos+i] = nb.children[cj.mineBegin+i];
    cj.mineBegin = writePos;
    cj.offset = ca.offset;
    writePos += cj.mineCount;
  }
}
  
void compressSimple(cbvh_t cbvh, int nodeID, bool dbg)
{
  auto &node = cbvh.nodes[nodeID];
  
  int a, b;
#if 0
  for (int i=0;i<WIDTH;i++) {
    if (!node.children[i].valid) continue;
    if (node.children[i].count) continue;
    compressSimple(cbvh,node.children[i].offset,dbg);
  }
  while (findMergablePair(cbvh,node,a,b)) {
    // std::cout << "merging " << a << " " << b << std::endl;
    mergeChildren(cbvh,node,a,b);
  }
#else
  // if (dbg) std::cout << "compress " << nodeID << std::endl;
  while (findMergablePair(cbvh,node,a,b)) {
    // std::cout << "node " << nodeID << " merging " << a << " w/ " << b << std::endl;
    // printNode(cbvh, node);
    mergeChildren(cbvh,node,a,b);
    // printNode(cbvh, node);
  }
  for (int i=0;i<WIDTH;i++) {
    if (!node.children[i].valid) continue;
    if (node.children[i].count) continue;
    compressSimple(cbvh,node.children[i].offset,dbg);
  }
#endif
}

cbvh_t merge(cbvh_t org, bool dbg=false)
{
  cbvh_t cp = cbvh_copy(org);
  compressSimple(cp,0,dbg);
  return cp;
}

struct Merger {
  Merger(cbvh_t &bvh) : bvh(bvh)
  {
    gatherDigest(0);

    int wa, wb;
    // while (findOne(wa,wb))
    //   mergeOne(wa,wb);
  }

  bool findOne(int &wa, int &wb) {
    for (wa=WIDTH-1;wa>=0;--wa) {
      if (byWidth[wa].empty())
        continue;
      for (wb=WIDTH-1;wb>=0;--wb) {
        if (wa+wb > WIDTH) continue;
        if (byWidth[wb].empty())
          continue;
        return true;
      }
    }
    return false;
  }

#if 0
  /*! merge two nodes a and b into one (a), and update all parent's
      offsets and colors to reflect that change. will also need to
      update 'parentsOf' digest */
  void mergeNodes(int aIdx, int bIdx)
  {
    // get the two nodes ...
    node_t &na = bvh.nodes[aIdx];
    node_t &nb = bvh.nodes[bIdx];

    int writePos = numValid(na);
    int maxColor = findMaxColorChild(na);
    int translatedColor[WIDTH];
    for (int i=0;i<WIDTH;i++) translatedColor[i] = -1;
    auto translateColor = [&](int oldColor) -> int {
      if (translatedColor[oldColor] == -1)
        translatedColor[oldColor] = ++maxColor;
      return translatedColor[oldColor];
    };
    for (int ib=0;ib<WIDTH;ib++) {
      auto cb = nb.children[ib]; 
      if (!cb.valid) continue;
      
      assert(parentOf.find({bIdx,ib}) != parentOf.end());
      auto p = parentOf[{bIdx,ib}];
      assert(bvh.nodes[p.first].children[p.second].offset == bIdx);
      assert(bvh.nodes[p.first].children[p.second].valid);
      bvh.nodes[p.first].children[p.second].offset = aIdx;

      assert(bvh.nodes[p.first].children[p.second].colorChild
             ==
             cb.colorParent);
      int newColor = translateColor(bvh.nodes[p.first].children[p.second].colorChild);
      int newPosInA = writePos++;
      bvh.nodes[p.first].children[p.second].colorChild = newColor;
      cb.colorParent = newColor;

      parentOf.erase({bIdx,ib});
      parentOf[{aIdx,newPosInA}] = p;

      if (cb.count == 0) {
        const node_t &nc = bvh.nodes[cb.offset];
        for (int ic=0;ic<WIDTH;ic++) {
          auto cc = nc.children[ic];
          if (!cc.valid) continue;
          parentOf[{(int)cb.offset,ic}] = {aIdx,newPosInA};
        }
      }
    }
  }
  
  void mergeOne(int wa, int wb)
  {
    auto naIt = byWidth[wa].begin(); 
    auto nbIt = byWidth[wb].begin();

    int aIdx = *naIt;
    int bIdx = *nbIt;
    mergeNodes(aIdx,bIdx);
    
    byWidth[wa].erase(naIt);
    byWidth[wb].erase(nbIt);
    byWidth[numValid(bvh,aIdx)].insert(aIdx);
  }
#endif
  
  void gatherDigest(int nodeID) {
    node_t &node = bvh.nodes[nodeID];
    byWidth[numValid(node)].insert(nodeID);
    // parentsOf[nodeID].insert({parent,0});
    for (int i=0;i<WIDTH;i++) {
      auto &child = node.children[i];
      if (!child.valid) continue;
      if (child.count > 0) continue;
      parentOf[{(int)child.offset,nodeID}]={nodeID,i};
      gatherDigest(child.offset);
    }
  }
  std::set<int> byWidth[WIDTH];
  std::map<std::pair<int/*nodeID*/,int/*color*/>,
           std::pair<int/*nodeID*/,int/*color*/>> parentOf;
  
  cbvh_t &bvh;
};


struct Stats {
  float sahPrims = 0.f;
  float sahNodes = 0.f;
  float sahMulti = 0.f;
  
  Stats(cbvh_t &cbvh) : cbvh(cbvh)
  {
    gatherActuallyUsedNodes();
    
    numNodes = 0;
    for (int i=0;i<WIDTH;i++) nodesWithNumValid[i] = 0;

    for (auto nodeID : actuallyUsedNodes) {
      numNodes++;
      nodesWithNumValid[numValid(cbvh,nodeID)]++;
    }
    computeSAHs(0,0,numValid(cbvh,0));
  }
  
  int numNodes = 0;
  int nodesWithNumValid[WIDTH];
  void gatherActuallyUsedNodes(int nodeID=0);
  void computeSAHs(int nodeID, int mineBegin, int mineCount);
  
  std::set<int> actuallyUsedNodes;

  cbvh_t &cbvh;
  
  void print()
  {
    std::cout << "nodes with num active children:" << std::endl;
    for (int i=0;i<WIDTH;i++)
      std::cout << nodesWithNumValid[i] << "\t";
    // std::cout << std::endl;
    // for (int i=0;i<WIDTH;i++)
    //   std::cout << "nodes with num active children = " << i << " : "
    //             << nodesWithNumValid[i] << std::endl;
    std::cout << "--> (total) " << numNodes << std::endl;
    std::cout << "sah prim/node/multi : "
              << "\t" << prettyDouble(sahPrims)
              << "\t" << prettyDouble(sahNodes)
              << "\t" << prettyDouble(sahMulti)
              << std::endl
              << std::endl;
    // std::cout << "SAH (leaf) " << sahLeafNodes << "\tSAH(prims) " << sahPrims << std::endl;
  }
};

void Stats::computeSAHs(int nodeID, int mineBegin, int mineCount)
{
  auto &node = cbvh.nodes[nodeID];
  // sahMulti += getBounds(node);
  for (int i=mineBegin;i<mineBegin+mineCount;i++) {
    auto c = node.children[i];
    sahNodes += surfaceArea(c.bounds);
    if (c.count) 
      sahPrims += /*.1f * */ c.count * surfaceArea(c.bounds);
    else {
      computeSAHs(c.offset,c.mineBegin,c.mineCount);
      sahMulti += surfaceArea(c.bounds) * c.mineCount;
    }
  }
}
void Stats::gatherActuallyUsedNodes(int nodeID)
{
  actuallyUsedNodes.insert(nodeID);
  auto &node = cbvh.nodes[nodeID];
  // numNodes++;
  // nodesWithNumValid[numValid(node)]++;
  for (int i=0;i<WIDTH;i++) {
    auto child = node.children[i];
    if (!child.valid) continue;
    if (child.count) continue;
    gatherActuallyUsedNodes(child.offset);
  }
}

struct SimpleBVH {
  typedef std::shared_ptr<SimpleBVH> SP;
  struct Node {
    inline bool isLeaf() const { return inner == 0; }
    
    typedef std::shared_ptr<Node> SP;
    box3f bounds;
    SimpleBVH::SP    inner;
    std::vector<int> leaf;
  };
  std::vector<Node::SP> children;
};


int countLeaves(SimpleBVH::Node::SP node)
{
  if (node->inner) {
    assert(node->leaf.empty());
    int sum = 0;
    for (auto n : node->inner->children)
      sum += countLeaves(n);
    return sum;
  } else {
    return 1;
  }
}

int countLeaves(std::vector<SimpleBVH::Node::SP> &nodes)
{
  int sum = 0;
  for (auto node : nodes)
    sum += countLeaves(node);
  return sum;
}

SimpleBVH::Node::SP copyFrom(bvh3f bvh, int nodeID)
{
  SimpleBVH::Node::SP sn = std::make_shared<SimpleBVH::Node>();
  auto node = bvh.nodes[nodeID];
  if (node.admin.count == 0) {
    sn->inner = std::make_shared<SimpleBVH>();
    sn->inner->children.push_back(copyFrom(bvh,node.admin.offset+0));
    sn->inner->children.push_back(copyFrom(bvh,node.admin.offset+1));
  } else {
    for (int i=0;i<node.admin.count;i++)
      sn->leaf.push_back(bvh.primIDs[node.admin.offset+i]);
  }
  return sn;
}

// SimpleBVH::Node::SP copyFrom(wbvh_t wbvh, int nodeID = 0)
// {
//   const auto &wn = wbvh.nodes[nodeID];
//   SimpleBVH::Node::SP node = std::make_shared<SimpleBVH::Node>();
//   node->inner = std::make_shared<SimpleBVH>();
//   for (int i=0;i<WIDTH;i++) {
//     auto c = wn.children[i];
//     if (!c.valid) continue;
//     if (c.count) {
//       SimpleBVH::Node::SP leaf = std::make_shared<SimpleBVH::Node>();
//       for (int j=0;j<c.count;j++)
//         leaf->leaf.push_back(bvh.primIDs[c.offset+j]);
//       node->inner.push_back(leaf);
//     } else {
//       node->inner.push_back(copyFrom(wbvh,c.offset));
//     }
//   }
//   return node;
// }

void refit(SimpleBVH::Node::SP node)
{
  node->bounds = box3f();
  if (node->inner) {
    for (auto child : node->inner->children) {
      refit(child);
      node->bounds.extend(child->bounds);
    }
  } else {
    for (auto prim : node->leaf)
      node->bounds.extend(boxes[prim]);
  }
}

int countNodes(SimpleBVH::Node::SP node)
{
  int sum = 0; 
  if (node->inner) {
    ++sum;
    for (auto n : node->inner->children)
      sum += countNodes(n);
  }
  return sum;
}

float computeSAH(SimpleBVH::Node::SP node)
{
  float sum = surfaceArea(node->bounds);
  if (node->inner) {
    for (auto child : node->inner->children) {
      sum += computeSAH(child);
    }
  } else {
    sum += surfaceArea(node->bounds)*node->leaf.size();
  }
  return sum;
}

SimpleBVH::Node::SP buildBinary()
{
  box3f *d_boxes = 0;
  cudaMalloc((void**)&d_boxes,boxes.size()*sizeof(box3f));
  PRINT(boxes.size());
  cudaMemcpy(d_boxes,boxes.data(),boxes.size()*sizeof(box3f),cudaMemcpyDefault);
  cudaDeviceSynchronize();
  bvh3f d_bvh,tmpBVH;
  gpuBuilder(d_bvh,d_boxes,boxes.size(),BuildConfig().enableSAH());
  cudaDeviceSynchronize();
  tmpBVH = d_bvh;
  tmpBVH.nodes = new typename bvh3f::Node[tmpBVH.numNodes];
  tmpBVH.primIDs = new uint32_t[tmpBVH.numPrims];
  PRINT(tmpBVH.numNodes);
  PRINT(tmpBVH.numPrims);
  cudaMemcpy(tmpBVH.nodes,d_bvh.nodes,tmpBVH.numNodes*sizeof(typename bvh3f::Node),cudaMemcpyDefault);
  // PRINT(tmpBVH.nodes[0].bounds);
  // PRINT(tmpBVH.nodes[0].admin.offset);
  // PRINT(tmpBVH.nodes[0].admin.count);
  cudaMemcpy(tmpBVH.primIDs,d_bvh.primIDs,tmpBVH.numPrims*sizeof(int),cudaMemcpyDefault);
  cudaDeviceSynchronize();
  SimpleBVH::Node::SP root = copyFrom(tmpBVH,0);
  refit(root);
  std::cout << "sah right after binary build " << computeSAH(root) << std::endl;
  return root;
}



std::vector<SimpleBVH::Node::SP> _collapseBottomUp(SimpleBVH::Node::SP node, int W, int depth)
{
  // int numIn = countLeaves(node);
  if (node->inner) {
    assert(node->inner->children.size() == 2);
    std::vector<SimpleBVH::Node::SP> n0
      = _collapseBottomUp(node->inner->children[0],W,depth+1);
    if (n0.size() == W) {
      SimpleBVH::Node::SP m0 = std::make_shared<SimpleBVH::Node>();
      m0->inner = std::make_shared<SimpleBVH>();
      m0->inner->children = n0;
      n0 = { m0 };
    }
    std::vector<SimpleBVH::Node::SP> n1
      = _collapseBottomUp(node->inner->children[1],W,depth+1);
    if (n1.size() == W) {
      SimpleBVH::Node::SP m1 = std::make_shared<SimpleBVH::Node>();
      m1->inner = std::make_shared<SimpleBVH>();
      m1->inner->children = n1;
      n1 = { m1 };
    }
    if (n0.size()+n1.size() <= W) {
      for (auto n : n1) n0.push_back(n);
      // int numOut = countLeaves(n0);
      // std::cout << "case 1 : " << numIn << " -> " << numOut << std::endl;
      // assert(numIn == numOut);
      return n0;
    } else {
      // std::cout << "must collapse " << n0.size() << " + " << n1.size() << std::endl;
      if (n0.size() > n1.size())  std::swap(n0,n1);
      
      SimpleBVH::Node::SP m1 = std::make_shared<SimpleBVH::Node>();
      // SimpleBVH::Node::SP m1 = std::make_shared<SimpleBVH::Node>();
      m1->inner = std::make_shared<SimpleBVH>();
      // m1->inner = std::make_shared<SimpleBVH>();
      m1->inner->children = n1;
      // m1->inner->children = n1;
      n0.push_back(m1);
      // std::cout << " -> " << n0.size() << std::endl;
      // int numOut = countLeaves(n0);
      // std::cout << "case 2 : " << numIn << " -> " << numOut << std::endl;
      // assert(numIn == numOut);
      return n0;
    }
  } else {
    SimpleBVH::Node::SP n = std::make_shared<SimpleBVH::Node>();
    n->leaf = node->leaf;
    return { n };
  }
}

SimpleBVH::Node::SP collapseBottomUp(SimpleBVH::Node::SP node, int W)
{
  std::cout << "num leaves before bottom up collapse "
            << countLeaves(node) << std::endl;
  std::vector<SimpleBVH::Node::SP> n = _collapseBottomUp(node,W,0);
  SimpleBVH::Node::SP r;
  if (n.size() == 1) {
    r = n[0];
  } else {
    r = std::make_shared<SimpleBVH::Node>();
    r->inner = std::make_shared<SimpleBVH>();
    r->inner->children = n;
  }
  // refit(r);
  std::cout << "num leaves after bottom up collapse "
            << countLeaves(r) << std::endl;
  return r;
}


SimpleBVH::Node::SP topDown(SimpleBVH::Node::SP in, int W)
{
  if (in->isLeaf()) {
    SimpleBVH::Node::SP out = std::make_shared<SimpleBVH::Node>();
    out->leaf = in->leaf;
    return out;
  } 
  
  std::set<SimpleBVH::Node::SP> leaves;
  std::priority_queue<std::pair<float,SimpleBVH::Node::SP>>
    workQueue;
  workQueue.push({0.f,in});
  while (workQueue.size() > 0 && workQueue.size()+leaves.size() < W) {
    SimpleBVH::Node::SP next = workQueue.top().second;
    
    workQueue.pop();

    if (next->inner) {
      for (auto n : next->inner->children)
        workQueue.push({surfaceArea(n->bounds),n});
    } else {
      leaves.insert(next);
    }
  }

  SimpleBVH::Node::SP out = std::make_shared<SimpleBVH::Node>();
  out->inner = std::make_shared<SimpleBVH>();
  for (auto l : leaves) 
    out->inner->children.push_back(topDown(l,W));
  while (!workQueue.empty()) {
    out->inner->children.push_back(topDown(workQueue.top().second,W));
    workQueue.pop();
  }
  assert(out->inner->children.size() <= W);
  return out;
}

SimpleBVH::Node::SP collapseTopDown(SimpleBVH::Node::SP in, int W)
{
  refit(in);
  SimpleBVH::Node::SP root = topDown(in,W);
  refit(root);
  return root;
}




node_t emptyNode()
{
  node_t node;
  for (int i=0;i<WIDTH;i++) {
    node.children[i].valid = 0;
    node.children[i].mineBegin = 0;
    node.children[i].mineCount = 0;
  }
  return node;
}

cbvh_t cbvh_copy(cbvh_t other)
{
  cbvh_t cp;
  cp.numPrims = other.numPrims;
  cp.numNodes = other.numNodes;
  cp.nodes = new node_t[cp.numNodes];
  cp.primIDs = new uint32_t[cp.numPrims];
  std::copy(other.primIDs,other.primIDs+other.numPrims,cp.primIDs);
  std::copy(other.nodes,other.nodes+other.numNodes,cp.nodes);
  return cp;
}

struct CopyRec {
  CopyRec(std::vector<node_t> &nodes,
          std::vector<uint32_t> &primIDs,
          SimpleBVH::Node::SP node)
    : nodes(nodes), primIDs(primIDs)
  {
    assert(node->inner);
    nodes.push_back(emptyNode());
    recurse(0,node->inner);
  }
  
  std::vector<node_t> &nodes;
  std::vector<uint32_t> &primIDs;
  
  void recurse(int outID, SimpleBVH::SP inner)
  {
    for (int i=0;i<inner->children.size();i++) {
      auto child = inner->children[i];
      if (child->inner) {
        int newID = nodes.size();
        nodes.push_back(emptyNode());
        nodes[outID].children[i].valid = 1;
        nodes[outID].children[i].count = 0;
        nodes[outID].children[i].offset = newID;
        nodes[outID].children[i].mineBegin = 0;
        nodes[outID].children[i].mineCount = child->inner->children.size();
        recurse(newID,child->inner);
      } else {
        nodes[outID].children[i].valid = 1;
        nodes[outID].children[i].count = child->leaf.size();
        nodes[outID].children[i].offset = primIDs.size();
        for (auto prim : child->leaf)
          primIDs.push_back(prim);
      }
    }
  }
};


box3f refit(cbvh_t bvh, int nodeID)
{
  auto &node = bvh.nodes[nodeID];
  box3f myBounds;
  for (int ic=0;ic<WIDTH;ic++) {
    auto &child = node.children[ic];
    child.bounds = box3f();
    if (!child.valid) continue;

    if (child.count) {
      for (int i=0;i<child.count;i++) {
        child.bounds.extend(boxes[bvh.primIDs[child.offset+i]]);
      }
    } else {
      child.bounds = refit(bvh,child.offset);
    }
    myBounds.extend(child.bounds);
  }
  return myBounds;
}

cbvh_t toCBVH(SimpleBVH::Node::SP node)
{
  std::vector<node_t>  nodes;
  std::vector<uint32_t> primIDs;
  CopyRec copyRec(nodes,primIDs,node);
  cbvh_t result;
  result.numNodes = nodes.size();
  result.nodes = new node_t[result.numNodes];
  std::copy(nodes.begin(),nodes.end(),result.nodes);
  result.numPrims = primIDs.size();
  result.primIDs = new uint32_t[result.numPrims];
  std::copy(primIDs.begin(),primIDs.end(),result.primIDs);
  refit(result,0);
  return result;
}

  
cbvh_t buildNativeWide()
{
  cbvh_t cbvh;
#if 1
  box3f *d_boxes = 0;
  cudaMalloc((void**)&d_boxes,boxes.size()*sizeof(box3f));
  cudaMemcpy(d_boxes,boxes.data(),boxes.size()*sizeof(box3f),cudaMemcpyDefault);
  cudaDeviceSynchronize();
  wbvh_t d_bvh, wbvh;
  gpuBuilder(d_bvh,d_boxes,boxes.size(),BuildConfig().enableSAH());
  cudaDeviceSynchronize();
  wbvh = d_bvh;
  wbvh.nodes = new wnode_t[wbvh.numNodes];
  wbvh.primIDs = new uint32_t[wbvh.numPrims];
  cudaMemcpy(wbvh.nodes,d_bvh.nodes,wbvh.numNodes*sizeof(wnode_t),cudaMemcpyDefault);
  cudaMemcpy(wbvh.primIDs,d_bvh.primIDs,wbvh.numPrims*sizeof(int),cudaMemcpyDefault);
  cudaDeviceSynchronize();
#else
  cpuBuilder(wbvh,boxes.data(),boxes.size(),BuildConfig());
#endif
  // PRINT(wbvh.numNodes);
  cbvh.numNodes = wbvh.numNodes;
  cbvh.numPrims = wbvh.numPrims;
  cbvh.primIDs = new uint32_t[cbvh.numPrims];
  cbvh.nodes   = new typename cbvh_t::Node[cbvh.numNodes];
  memcpy(cbvh.nodes,wbvh.nodes,cbvh.numNodes*sizeof(typename cbvh_t::Node));
  memcpy(cbvh.primIDs,wbvh.primIDs,cbvh.numPrims*sizeof(int));
  
  // PRINT(bvh.numNodes);
  for (int i=0;i<cbvh.numNodes;i++) {
    node_t &node = cbvh.nodes[i];
    // if (i < 10)  PRINT(i);
    for (int j=0;j<WIDTH;j++) {
      auto &child = node.children[j];
      // if (i < 10) {
      //   PRINT(node.children[j].offset);
      //   PRINT(node.children[j].count);
      //   PRINT(node.children[j].bounds);
      // }
      if (!child.valid) continue;
      child.mineBegin = 0;
      if (child.count == 0)
        child.mineCount = numValid(cbvh,child.offset);
    }
  }
  return cbvh;
}

cbvh_t mergeCrazy(cbvh_t nativeWide)
{
  cbvh_t cp = cbvh_copy(nativeWide);
  Merger merger(cp);
  return cp;
}


void usage(const std::string &error)
{
  std::cerr << "Error : " << error << "\n\n";
  std::cout << "Usage: ./compact inFile.binmesh -o outFilePrefix [-n maxRes]" << std::endl;
  exit(0);
}


int main(int ac, char **av)
{
  // bool goodBuild = false;
  std::string inFileName = "";
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg[0] != '-') {
      inFileName = av[i];
    // } else if (arg == "--good") {
    //   goodBuild = true;
    } else
      usage("unknown cmdline arg '"+arg+"'");
  }
  const std::vector<Triangle> triangles = samples::loadBinMesh(inFileName);//loadOBJ(inFileName);
  boxes = std::vector<box3f>(triangles.size());
  for (int i=0;i<triangles.size();i++)
    boxes[i] = triangles[i].bounds();

  SimpleBVH::Node::SP bin = buildBinary();
  auto digest_binary = computeDigest(toCBVH(bin));
  auto digest_copied_binary = computeDigest(cbvh_copy(toCBVH(bin)));
  if (digest_copied_binary != digest_binary)
    throw std::runtime_error("copying seems to not work");
  
  cbvh_t binary_collapse_bu = toCBVH(collapseBottomUp(bin,WIDTH));
  auto digest_binary_collapse_bu = computeDigest(binary_collapse_bu);
  // PRINT(digest_binary_collapse_bu.size());
  // PRINT(digest_binary.size());
  // for (int i=0;i<10;i++) {
  //   PRINT(digest_binary[i]);
  //   PRINT(digest_binary_collapse_bu[i]);
  // }
  if (digest_binary_collapse_bu != digest_binary)
    throw std::runtime_error("digest_binary_collapse_bu does not match");
  std::cout << "#### binary build, bottom up collapse, no merge" << std::endl;
  {
    Stats stats(binary_collapse_bu);
    stats.print();
  }

  cbvh_t binary_collapse_bu_merged = merge(binary_collapse_bu,true);
  auto digest_binary_collapse_bu_merged = computeDigest(binary_collapse_bu_merged);
  if (digest_binary_collapse_bu_merged != digest_binary)
    throw std::runtime_error("digest_binary_collapse_bu_merged does not match");
  std::cout << "#### binary build, bottom up collapse, after merge" << std::endl;
  {
    Stats stats(binary_collapse_bu_merged);
    stats.print();
  }
  // std::cout << " sah in bottom up W : " << computeSAH(col_buW) << std::endl;
  // std::cout << " nodes in bottom up W : " << countNodes(col_buW) << std::endl;
  
  cbvh_t binary_collapse_td = toCBVH(collapseTopDown(bin,WIDTH));
  auto digest_binary_collapse_td = computeDigest(binary_collapse_td);
  if (digest_binary_collapse_td != digest_binary)
    throw std::runtime_error("digest_binary_collapse_td does not match");
  std::cout << "#### binary build, top down collapse, no merge" << std::endl;
  {
    Stats stats(binary_collapse_td);
    stats.print();
  }
  
  cbvh_t binary_collapse_td_merged = merge(binary_collapse_td);
  auto digest_binary_collapse_td_merged = computeDigest(binary_collapse_td_merged);
  if (digest_binary_collapse_td_merged != digest_binary)
    throw std::runtime_error("digest_binary_collapse_td_merged does not match");
  std::cout << "#### binary build, top down collapse, after merge" << std::endl;
  {
    Stats stats(binary_collapse_td_merged);
    stats.print();
  }
  
  // std::cout << " sah in top down W : " << computeSAH(col_tdW) << std::endl;
  // std::cout << " nodes in top down W : " << countNodes(col_tdW) << std::endl;
  
  cbvh_t nativeWide = buildNativeWide();
  std::cout << "#### native wide build, un-merged" << std::endl;
  {
    Stats stats(nativeWide);
    stats.print();
  }

  cbvh_t nativeWideMergedSimple = merge(nativeWide);
  std::cout << "#### native wide build, after SIMPLE merge" << std::endl;
  {
    Stats stats(nativeWideMergedSimple);
    stats.print();
  }
  
  cbvh_t nativeWideMergedCrazy = mergeCrazy(nativeWide);
  std::cout << "#### native wide build, after CRAZY merge" << std::endl;
  {
    Stats stats(nativeWideMergedCrazy);
    stats.print();
  }
  
}
