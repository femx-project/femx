add_library(femx_linalg)
add_library(femx::linalg ALIAS femx_linalg)

target_sources(
  femx_linalg
  PRIVATE ../common/Math.cpp CsrMatrix.cpp Dense.cpp
  PUBLIC Backend.hpp
         View.hpp
         CsrGraph.hpp
         CsrMatrix.hpp
         Dense.hpp
         LinearSolver.hpp
         Vector.hpp)

target_link_libraries(femx_linalg PUBLIC femx_config femx_common)

if(FEMX_HAS_CUDA)
  target_sources(femx_linalg PRIVATE ../common/Device.cu Vector.cu CsrMatrix.cu)
  target_link_libraries(femx_linalg PUBLIC CUDA::cudart CUDA::cusparse
                                           CUDA::cublas)
  set_target_properties(femx_linalg PROPERTIES CUDA_STANDARD 17
                                               CUDA_STANDARD_REQUIRED ON)
else()
  target_sources(femx_linalg PRIVATE ../common/Device.cpp)
endif()

if(FEMX_ENABLE_RESOLVE)
  add_subdirectory(resolve)
endif()
