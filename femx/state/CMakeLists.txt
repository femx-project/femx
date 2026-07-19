add_library(femx_state)
add_library(femx::state ALIAS femx_state)

target_sources(
  femx_state
  PRIVATE EnsembleBasis.cpp
  PUBLIC EnsembleBasis.hpp
         Residual.hpp
         StateSolver.hpp
         TimeResidual.hpp
         TimeIntegrator.hpp
         TimeTrajectory.hpp)

target_link_libraries(femx_state PUBLIC femx_config femx_common femx_linalg)
