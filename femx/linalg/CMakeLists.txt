add_library(femx_linalg)
add_library(femx::linalg ALIAS femx_linalg)

target_sources(
  femx_linalg
  PRIVATE ../common/Math.cpp CsrPattern.cpp CsrMatrix.cpp DenseMatrix.cpp
  PUBLIC BlockVectorView.hpp
         CsrMatrix.hpp
         CsrPattern.hpp
         DenseMatrix.hpp
         MatrixView.hpp
         operator/LinearOperator.hpp
         operator/LinearSolver.hpp
         operator/MatrixBuilder.hpp
         operator/MatrixOperator.hpp
         Vector.hpp
         VectorView.hpp)

target_link_libraries(femx_linalg PUBLIC femx_config femx_common)

add_subdirectory(native)
if(FEMX_ENABLE_RESOLVE)
  add_subdirectory(resolve)
endif()
