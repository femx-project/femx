add_library(femx_linalg)
add_library(femx::linalg ALIAS femx_linalg)

target_sources(
  femx_linalg
  PRIVATE ../common/Math.cpp CsrPattern.cpp DenseLinearSolver.cpp
          DenseMatrix.cpp SparseMatrix.cpp
  PUBLIC BlockVectorView.hpp
         CsrPattern.hpp
         DenseMatrix.hpp
         DenseLinearSolver.hpp
         IndexSetList.hpp
         LinearOperator.hpp
         LinearSolver.hpp
         MatrixBuilder.hpp
         MatrixOperator.hpp
         MatrixView.hpp
         SparseMatrix.hpp
         Vector.hpp
         VectorView.hpp)

target_link_libraries(femx_linalg PUBLIC femx_config femx_common)

add_subdirectory(native)
if(FEMX_ENABLE_RESOLVE)
  add_subdirectory(resolve)
endif()
