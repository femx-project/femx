add_library(femx_opt INTERFACE)
add_library(femx::opt ALIAS femx_opt)

target_sources(femx_opt INTERFACE TaoOptimizer.hpp
                                  TaoReducedFunctionalAdapter.hpp)

target_link_libraries(femx_opt INTERFACE femx_state femx::linalg_petsc
                                         PkgConfig::PETSC MPI::MPI_CXX)
