add_library(femx_system_petsc)
add_library(femx::system_petsc ALIAS femx_system_petsc)

target_sources(
  femx_system_petsc
  PRIVATE KspLinearSolver.cpp
  PUBLIC KspLinearSolver.hpp PETScSystemMatrix.hpp PETScSystemVector.hpp
         VectorConversion.hpp)

target_link_libraries(femx_system_petsc PUBLIC femx_system PkgConfig::PETSC
                                               MPI::MPI_CXX)
