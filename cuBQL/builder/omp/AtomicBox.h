// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/omp/common.h"

namespace cuBQL {
  namespace omp {

    template<typename box_t>
    struct AtomicBox : public box_t {
      
      inline void set_empty()
      {
        *(box_t *)this = box_t();
      }
    };

    template<typename T>
    inline void atomic_min(T *ptr, T v);
    template<typename T>
    inline void atomic_max(T *ptr, T v);
    
    /*! iw - note: this implementation of atomic min/max via atomic
        compare-exchange (CAS); which is cetainly not optimal on any
        sort of modern GPU - but it works in any C++-21 compliant
        compiler, so it's what we do for now */
    inline void atomic_min(float *ptr, float value)
    {
      float current = *(volatile float *)ptr;
      while (current > value) {
        bool wasChanged
          = ((std::atomic<int>*)ptr)
          ->compare_exchange_weak((int&)current,(int&)value);
        if (wasChanged) break;
      }
    }
    
    /*! iw - note: this implementation of atomic min/max via atomic
        compare-exchange (CAS); which is cetainly not optimal on any
        sort of modern GPU - but it works in any C++-21 compliant
        compiler, so it's what we do for now */
    inline void atomic_max(float *ptr, float value)
    {
      float current = *(volatile float *)ptr;
      while (current < value) {
        bool wasChanged
          = ((std::atomic<int>*)ptr)
          ->compare_exchange_weak((int&)current,(int&)value);
        if (wasChanged) break;
      }
    }
    
    /*! iw - note: this implementation of atomic min/max via atomic
        compare-exchange (CAS); which is cetainly not optimal on any
        sort of modern GPU - but it works in any C++-21 compliant
        compiler, so it's what we do for now */
    inline void atomic_min(double *ptr, double value)
    {
      double current = *(volatile double *)ptr;
      while (current > value) {
        bool wasChanged
          = ((std::atomic<long long int>*)ptr)
          ->compare_exchange_weak((long long int&)current,
                                  (long long int&)value);
        if (wasChanged) break;
      }
    }
    
    /*! iw - note: this implementation of atomic min/max via atomic
        compare-exchange (CAS); which is cetainly not optimal on any
        sort of modern GPU - but it works in any C++-21 compliant
        compiler, so it's what we do for now */
    inline void atomic_max(double *ptr, double value)
    {
      double current = *(volatile double *)ptr;
      while (current < value) {
        bool wasChanged
          = ((std::atomic<long long int>*)ptr)
          ->compare_exchange_weak((long long int&)current,
                                  (long long int&)value);
        if (wasChanged) break;
      }
    }
    
    template<typename T, int D>
    inline void v_atomic_min(vec_t<T,D> *ptr, vec_t<T,D> v);
    template<typename T, int D>
    inline void v_atomic_max(vec_t<T,D> *ptr, vec_t<T,D> v);
    

    template<typename T>
    inline void v_atomic_min(vec_t<T,2> *ptr, vec_t<T,2> v)
    {
      atomic_min(&ptr->x,v.x); 
      atomic_min(&ptr->y,v.y);
    }
    
    template<typename T>
    inline void v_atomic_min(vec_t<T,3> *ptr, vec_t<T,3> v)
    {
      atomic_min(&ptr->x,v.x); 
      atomic_min(&ptr->y,v.y);
      atomic_min(&ptr->z,v.z);
    }
    
    template<typename T>
    inline void v_atomic_min(vec_t<T,4> *ptr, vec_t<T,4> v)
    {
      atomic_min(&ptr->x,v.x); 
      atomic_min(&ptr->y,v.y);
      atomic_min(&ptr->z,v.z);
      atomic_min(&ptr->w,v.w);
    }

    template<typename T>
    inline void v_atomic_max(vec_t<T,2> *ptr, vec_t<T,2> v)
    {
      atomic_max(&ptr->x,v.x); 
      atomic_max(&ptr->y,v.y);
    }
    
    template<typename T>
    inline void v_atomic_max(vec_t<T,3> *ptr, vec_t<T,3> v)
    {
      atomic_max(&ptr->x,v.x); 
      atomic_max(&ptr->y,v.y);
      atomic_max(&ptr->z,v.z);
    }
    
    template<typename T>
    inline void v_atomic_max(vec_t<T,4> *ptr, vec_t<T,4> v)
    {
      atomic_max(&ptr->x,v.x); 
      atomic_max(&ptr->y,v.y);
      atomic_max(&ptr->z,v.z);
      atomic_max(&ptr->w,v.w);
    }
    
    template<typename box_t>
    inline void atomic_grow(AtomicBox<box_t> &ab, typename box_t::vec_t P)
    {
      v_atomic_min(&ab.lower,P);
      v_atomic_max(&ab.upper,P);
    }
    
    template<typename box_t>
    inline void atomic_grow(AtomicBox<box_t> &ab, box_t B)
    {
      v_atomic_min(&ab.lower,B.lower);
      v_atomic_max(&ab.upper,B.upper);
    }
    
  }
}
