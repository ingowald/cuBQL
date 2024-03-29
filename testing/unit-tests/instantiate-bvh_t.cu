// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "cuBQL/bvh.h"

template<typename T, int D>
cuBQL::bvh_t<T,D> foo()
{
  using vec_t = cuBQL::vec_t<T,D>;
  using box_t = cuBQL::box_t<T,D>;
  using bvh_t = cuBQL::bvh_t<T,D>;

  bvh_t bvh;
  return bvh;
}

int main(int, char **)
{
  foo<float,2>();
  foo<float,3>();
  foo<float,4>();
  foo<float,CUBQL_TEST_N>();
  return 0;
}
