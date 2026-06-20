target_sources(
  femx_algebra
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/DenseSystemMatrix.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/DenseSystemVector.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/SparseSystemMatrix.hpp)
