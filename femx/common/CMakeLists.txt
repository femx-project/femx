add_library(femx_common)
add_library(femx::common ALIAS femx_common)

target_sources(
  femx_common
  PRIVATE Math.cpp
  PUBLIC Checks.hpp
         Cuda.hpp
         Device.hpp
         LinearInterpolation.hpp
         Math.hpp
         Types.hpp
         Vector.hpp
         View.hpp)

target_link_libraries(femx_common PUBLIC femx_config)

if(FEMX_HAS_CUDA)
  target_sources(femx_common PRIVATE Cuda.cu)
  target_link_libraries(femx_common PUBLIC CUDA::cudart)
  set_target_properties(femx_common PROPERTIES CUDA_STANDARD 17
                                               CUDA_STANDARD_REQUIRED ON)
else()
  target_sources(femx_common PRIVATE Cuda.cpp Device.cpp)
endif()
