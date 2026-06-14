add_library(femx_linalg)
add_library(femx::linalg ALIAS femx_linalg)

target_sources(
  femx_linalg
  PRIVATE DenseMatrix.cpp CsrPattern.cpp HostCsrMatrix.cpp SparseMatrix.cpp
  PUBLIC DenseMatrix.hpp
         CsrPattern.hpp
         HostCsrMatrix.hpp
         MatrixBackend.hpp
         MatrixView.hpp
         SparseMatrix.hpp
         SparseMatrixImpl.hpp
         Vector.hpp
         VectorView.hpp)

target_link_libraries(
  femx_linalg
  PUBLIC femx_common femx_config)
