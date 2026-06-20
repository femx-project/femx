add_library(femx_algebra_petsc)
add_library(femx::algebra_petsc ALIAS femx_algebra_petsc)

target_sources(
  femx_algebra_petsc
  PRIVATE KspLinearSolver.cpp
  PUBLIC KspLinearSolver.hpp
         PETScMatrixOperator.hpp
         PETScVectorBuilder.hpp
         VectorConversion.hpp)

target_link_libraries(
  femx_algebra_petsc PUBLIC femx_algebra PkgConfig::PETSC MPI::MPI_CXX)
