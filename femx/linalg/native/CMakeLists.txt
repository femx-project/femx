target_sources(
  femx_linalg
  PRIVATE CsrMatrixOperator.cpp DenseLinearSolver.cpp DenseMatrixOperator.cpp
  PUBLIC CsrMatrixOperator.hpp DenseLinearSolver.hpp DenseMatrixOperator.hpp)
