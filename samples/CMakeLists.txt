# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# When building the GPU code with HIP, the sample .cu sources must be compiled
# as HIP. Mark a target's CUDA sources accordingly.
function(cubql_sample_hip_lang target)
  if (CUBQL_USE_HIP)
    get_target_property(_srcs ${target} SOURCES)
    foreach(_s IN LISTS _srcs)
      if (_s MATCHES "\\.cu$")
        set_source_files_properties(${_s} PROPERTIES LANGUAGE HIP)
      endif()
    endforeach()
  endif()
endfunction()

# common stuff - data generators and IO for loading/generating test data
add_subdirectory(common)
if (NOT (TARGET stb_image))
  add_subdirectory(3rdParty/stb_image)
endif()


# ==================================================================
# sample that performs gpu-side closest-point queries between sets of
# float3 points
# ==================================================================
add_subdirectory(s01_closestPoint_points_cpu)
add_subdirectory(s01_closestPoint_points_gpu)

# ==================================================================
# sample that performs gpu-side closest-point queries between sets of
# float3 points
# ==================================================================
if (CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP)
  add_executable(sample02_distanceToTriangleMesh
    s02_distanceToTriangleMesh/distanceToTriangleMesh.cu
    )
  cubql_sample_hip_lang(sample02_distanceToTriangleMesh)
  target_link_libraries(sample02_distanceToTriangleMesh
    # the cuda-side builders for float3 data
    cuBQL_cuda_float3
    # common samples stuff
    cuBQL_samples_common
    )
endif()


# ==================================================================
# sample that performs gpu-side inside/outside test for a given input
# mesh and a generated set of query points. Input mesh is supposed to
# be a 'proper' surface mesh without holes and proper normal
# orientation (normal points to 'outside'); generated poitns lie on a
# NxNxN grid. QUeries are performed by tracing axis-aligned rays and
# computing cuqbl::crossingCount queries.
# ==================================================================
if (CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP)
  add_executable(sample03_insideOutside
    s03_insideOutsideOfClosedMesh/insideOutside.cu
    )
  cubql_sample_hip_lang(sample03_insideOutside)
  target_link_libraries(sample03_insideOutside
    # the cuda-side builders for float3 data
    cuBQL_cuda_float3
    # common samples stuff
    cuBQL_samples_common
    )
endif()


# ==================================================================
# sample that computes, for every cell of a 3D cartesian grid put
# around a surface mesh, whether the box representing that cell is
# full inside the surface (0), fully outside (1), or whether it
# intersects with one of the triangles of the surface mesh
# ==================================================================
if (CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP)
  add_executable(sample04_boxOverlapsOrInsideSurfaceMesh
    s04_boxOverlapsOrInsideSurfaceMesh/boxOverlapsOrInsideSurfaceMesh.cu
    )
  cubql_sample_hip_lang(sample04_boxOverlapsOrInsideSurfaceMesh)
  target_link_libraries(sample04_boxOverlapsOrInsideSurfaceMesh
    # the cuda-side builders for float3 data
    cuBQL_cuda_float3
    # common samples stuff
    cuBQL_samples_common
    )
endif()



# ==================================================================
# sample that traces a grid of NxM rays diagonally through the
# bounding box of a loaded triangle mesh, and computes - 'xray-style'
# whether the ray did pass through unoccluded (white) or got occluded
# (black), then saves that as an image
# ==================================================================
if (CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP)
  add_executable(sample05_lineOfSight
    s05_lineOfSight/lineOfSight.cu
    )
  cubql_sample_hip_lang(sample05_lineOfSight)
  target_link_libraries(sample05_lineOfSight
    # the cuda-side builders for float3 data
    cuBQL_cuda_float3
    # common samples stuff
    cuBQL_samples_common
    stb_image
    )
endif()


# ==================================================================
# sample that computes a volume of NxMxK floats that each store '1' if
# point is within a given distance of the loaded surface mesh, and '0'
# if not. Chosen distance for this sample is 1% of scene diameter;
# this uses the triangles::anyWithinRadius() query.
# ==================================================================
if (CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP)
  add_executable(sample06_anyTriangleWithinRadius
    s06_anyTriangleWithinRadius/anyTriangleWithinRadius.cu
    )
  cubql_sample_hip_lang(sample06_anyTriangleWithinRadius)
  target_link_libraries(sample06_anyTriangleWithinRadius
    # the cuda-side builders for float3 data
    cuBQL_cuda_float3
    # common samples stuff
    cuBQL_samples_common
    stb_image
    )
endif()

if (CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP)
  add_executable(sample07_aggregateNBody
    s07_aggregateNBody/aggregateNBody.cu
    )
  cubql_sample_hip_lang(sample07_aggregateNBody)
  target_link_libraries(sample07_aggregateNBody
    # the cuda-side builders for float3 data
    cuBQL_cuda_float3
    # common samples stuff
    cuBQL_samples_common
    stb_image
    )
  set_target_properties(sample07_aggregateNBody
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION   ON
    POSITION_INDEPENDENT_CODE    ON
    CUDA_USE_STATIC_CUDA_RUNTIME ON
    CUDA_RESOLVE_DEVICE_SYMBOLS  ON
    CXX_VISIBILITY_PRESET  hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
endif()





