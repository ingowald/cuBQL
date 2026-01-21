// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace cuBQL {
  namespace omp {
    
    struct Context {
      Context(int gpuID);
      int gpuID;
      int hostID;
    };
    struct Kernel {
      inline int workIdx() const { return _workIdx; }
      int _workIdx;
    };

  }
}
