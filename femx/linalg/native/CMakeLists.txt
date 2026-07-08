target_sources(
  femx_linalg
  PRIVATE CsrAssemblyMatrix.cpp DenseLinearSolver.cpp DenseAssemblyMatrix.cpp
  PUBLIC CsrAssemblyMatrix.hpp DenseLinearSolver.hpp DenseAssemblyMatrix.hpp)
