// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

namespace cuBQL {
  namespace samples {

    /*! simple helper class that allows for successively consuming
        arguments from a cmdline, with some error checking */
    struct CmdLine {
      CmdLine(int ac, char **av) : ac(ac), av(av) {}
      /*! returns true iff all items have been consumed, and there is
          no other un-consumed argument to process */
      inline bool consumed() const { return current == ac; }
      /*! get next argument off cmdline, without any type-conversion */
      inline std::string getString();
      /*! get next argument off cmdline, and convert to int */
      inline int getInt() { return std::stoi(getString()); }
      /*! get next argument off cmdline, and convert to float */
      inline float getFloat() { return std::stof(getString()); }
      /*! read a float2 from the cmdline */
      inline float get2f();
      /*! read a float3 from the cmdline */
      inline float get3f();
    private:
      int current = 1;
      const int ac;
      char **const av;
    };
    
    /*! get next argument off cmdline, without any type-conversion */
    inline std::string CmdLine::getString()
    {
      if (current >= ac)
        throw std::runtime_error
          ("CmdLine: requested to get next argument, but no more "
           "un-consumed arguments available");
      return av[current++];
    }
    
  }
}
