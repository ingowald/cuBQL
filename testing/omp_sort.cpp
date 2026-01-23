// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuBQL/builder/omp/sort.h"
#include <iostream>

int main(int ac, char **av)
{
  cuBQL::omp::Context omp(0);
  
  int N = 13;
  // int N = 123453;
  std::vector<int> inputs(N);
  for (int i=0;i<N;i++) {
    inputs[i] = 90-i;//random() % 100;
    if (inputs[i] < 10) inputs[i] += 20;
  }

  int *d_data = 0;
  omp.alloc_and_upload(d_data,inputs);
  printf("d_data %p\n",d_data);

// #pragma omp target device(omp.gpuID)
// #pragma omp teams distribute parallel for
//   for (int i=0;i<20;i++)
//     if (1<<i < N)
//       printf("d_data[%i] = %i\n",1<<i,d_data[1<<i]);

  cuBQL::omp::sort(d_data,N,&omp);

  std::vector<int> results
    = omp.download_vector(d_data,N);
  for (int i=1;i<results.size();i++) {
    PRINT(results[i]);
    if (results[i-1] > results[i])
      throw std::runtime_error("Not sorted...");
  }
  std::cout << "sorted - perfect!" << std::endl;
}
