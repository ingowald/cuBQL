// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuBQL/math/box.h>
#include <cuBQL/queries/triangleData/Triangle.h>
#include <vector>

namespace cuBQL {
  namespace samples {
    
    std::vector<Triangle> loadBinMesh(const std::string &fileName);
    
    void loadBinMesh(std::vector<vec3i> &indices,
                     std::vector<vec3f> &vertices,
                     const std::string &fileName);
  }
}

