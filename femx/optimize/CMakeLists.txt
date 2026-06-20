add_library(femx_optimize INTERFACE)
add_library(femx::optimize ALIAS femx_optimize)

target_sources(femx_optimize INTERFACE TaoOptimizer.hpp
                                      TaoReducedFunctionalAdapter.hpp)

target_link_libraries(
  femx_optimize
  INTERFACE femx_solve femx::algebra_petsc PkgConfig::PETSC MPI::MPI_CXX)
