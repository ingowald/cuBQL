// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

namespace cuBQL {
  namespace samples {

    /*! load a vector of (binary) data from a binary-dump file. the
        data file is supposed to start with a size_t that specifies
        the *number* N of elements to be expected in that file,
        followed by the N "raw"-binary data items */
    template<typename T>
    std::vector<T> loadBinary(const std::string &fileName)
    {
      std::ifstream in(fileName.c_str(),std::ios::binary);
      if (!in.good())
        throw std::runtime_error("could not open '"+fileName+"'");
      size_t count;
      in.read((char*)&count,sizeof(count));

      std::vector<T> data(count);
      in.read((char*)data.data(),count*sizeof(T));
      return data;
    }

    /*! write a vector of (binary) data into a binary-dump file. the
        data file is supposed to start with a size_t that specifies
        the *number* N of elements to be expected in that file,
        followed by the N "raw"-binary data items */
    template<typename T>
    void saveBinary(const std::string &fileName,
                    const std::vector<T> &data)
    {
      std::ofstream out(fileName.c_str(),std::ios::binary);
      size_t count = data.size();
      out.write((char*)&count,sizeof(count));
      
      out.write((char*)data.data(),count*sizeof(T));
    }

    template<typename T, int D>
    std::vector<vec_t<T,D>> convert(const std::vector<vec_t<double,D>> &in) {
      std::vector<vec_t<T,D>> result(in.size());
      for (size_t i=0;i<in.size();i++)
        result[i] = vec_t<T,D>(in[i]);
      return result;
    };
  }
}
