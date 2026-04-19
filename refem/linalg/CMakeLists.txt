add_library(refem_linalg)
add_library(refem::linalg ALIAS refem_linalg)

target_sources(refem_linalg
  PRIVATE
    DenseMatrix.cpp
    FixedSparsityPattern.cpp
    HostCsrMatrix.cpp
    SparseMatrix.cpp
  PUBLIC
    DenseMatrix.hpp
    FixedSparsityPattern.hpp
    HostCsrMatrix.hpp
    LinalgBackend.hpp
    MatrixBackend.hpp
    MatrixVeiw.hpp
    MatrixView.hpp
    SparseMatrix.hpp
    SparseMatrixImpl.hpp
    Vector.hpp
    VectorView.hpp
)

target_link_libraries(refem_linalg
  PUBLIC
    refem_common
    refem_config
  PRIVATE
    refem_fe
)
