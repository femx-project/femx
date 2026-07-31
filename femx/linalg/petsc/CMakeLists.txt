add_library(femx_linalg_petsc)
add_library(femx::linalg_petsc ALIAS femx_linalg_petsc)

target_sources(
  femx_linalg_petsc
  PRIVATE MpiContext.cpp PETScSystemMatrix.cpp PETScLinearSolver.cpp
          PETScLinearSystem.cpp PETScMatrix.cpp PETScPartition.cpp
  PUBLIC MpiContext.hpp
         PETScSystemMatrix.hpp
         PETScLinearSolver.hpp
         PETScLinearSystem.hpp
         PETScMatrix.hpp
         PETScPartition.hpp
         PETScUtilities.hpp)

target_link_libraries(femx_linalg_petsc PUBLIC femx_linalg PkgConfig::PETSC
                                               MPI::MPI_CXX)
