add_library(femx_algebra)
add_library(femx::algebra ALIAS femx_algebra)

target_sources(
  femx_algebra
  PRIVATE ../core/Math.cpp
          CsrPattern.cpp
          DenseLinearSolver.cpp
          DenseMatrix.cpp
          SparseMatrix.cpp
  PUBLIC CsrPattern.hpp
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

target_link_libraries(femx_algebra PUBLIC femx_config femx_core)

add_subdirectory(backends/native)
if(FEMX_ENABLE_RESOLVE)
  add_subdirectory(backends/resolve)
endif()
