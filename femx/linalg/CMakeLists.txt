add_library(femx_linalg)
add_library(femx::linalg ALIAS femx_linalg)

target_sources(
  femx_linalg
  PRIVATE DenseMatrix.cpp CsrPattern.cpp SparseMatrix.cpp
  PUBLIC DenseMatrix.hpp
         CsrPattern.hpp
         IndexSetList.hpp
         MatrixView.hpp
         SparseMatrix.hpp
         Vector.hpp
         VectorView.hpp)

target_link_libraries(femx_linalg PUBLIC femx_common femx_config)
