add_library(femx_linalg)
add_library(femx::linalg ALIAS femx_linalg)

target_sources(
  femx_linalg
  PRIVATE ../common/Math.cpp CsrTranspose.cpp DenseMatrix.cpp
  PUBLIC BlockVectorView.hpp
         CsrGraph.hpp
         CsrMatrix.hpp
         CsrTranspose.hpp
         DenseMatrix.hpp
         MatrixView.hpp
         LinearOperator.hpp
         LinearSolver.hpp
         MatrixOperator.hpp
         Vector.hpp
         VectorView.hpp)

target_link_libraries(femx_linalg PUBLIC femx_config femx_common)

if(FEMX_HAS_CUDA)
  target_sources(femx_linalg PRIVATE ../common/Device.cu)
  target_link_libraries(femx_linalg PUBLIC CUDA::cudart)
  set_target_properties(femx_linalg PROPERTIES CUDA_STANDARD 17
                                               CUDA_STANDARD_REQUIRED ON)
else()
  target_sources(femx_linalg PRIVATE ../common/Device.cpp)
endif()

add_subdirectory(native)
if(FEMX_ENABLE_RESOLVE)
  add_subdirectory(resolve)
endif()
