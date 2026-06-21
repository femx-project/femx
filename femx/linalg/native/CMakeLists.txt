target_sources(
  femx_linalg
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/DenseMatrixOperator.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/DenseVectorBuilder.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/SparseMatrixOperator.hpp)
