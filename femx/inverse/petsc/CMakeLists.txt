add_library(femx_inverse_petsc INTERFACE)
add_library(femx::inverse_petsc ALIAS femx_inverse_petsc)

target_sources(femx_inverse_petsc INTERFACE TaoOptimizer.hpp
                                            TaoReducedFunctionalAdapter.hpp)

target_link_libraries(
  femx_inverse_petsc INTERFACE femx_inverse femx_system_petsc PkgConfig::PETSC
                               MPI::MPI_CXX)
