add_library(femx_state)
add_library(femx::state ALIAS femx_state)

target_sources(
  femx_state
  PRIVATE EnsembleBasis.cpp
          LinearStateSolver.cpp
          NewtonStateSolver.cpp
          TimeLinearIntegrator.cpp
  PUBLIC EnsembleBasis.hpp
         Linearization.hpp
         LinearStateSolver.hpp
         NewtonStateSolver.hpp
         Residual.hpp
         StateSolver.hpp
         TimeLinearIntegrator.hpp
         TimeResidual.hpp
         TimeStateMonitor.hpp
         TimeIntegrator.hpp
         TimeTrajectory.hpp)

target_link_libraries(femx_state PUBLIC femx_config femx_common femx_linalg)
