add_library(femx_linalg_petsc)
add_library(femx::linalg_petsc ALIAS femx_linalg_petsc)

target_sources(
  femx_linalg_petsc
  PRIVATE KspLinearSolver.cpp
  PUBLIC KspLinearSolver.hpp
         PETScMatrixOperator.hpp
         PETScVectorBuilder.hpp
         VectorConversion.hpp)

target_link_libraries(
  femx_linalg_petsc PUBLIC femx_linalg PkgConfig::PETSC MPI::MPI_CXX)
