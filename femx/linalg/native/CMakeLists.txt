target_sources(
  femx_linalg
  PRIVATE MapCsrMatrix.cpp DenseLinearSolver.cpp
  PUBLIC MapCsrMatrix.hpp DenseLinearSolver.hpp)
