add_library(femx_state)
add_library(femx::state ALIAS femx_state)

target_sources(
  femx_state
  PRIVATE EnsembleBasis.cpp
          LinearStateSolver.cpp
          NewtonStateSolver.cpp
          ReducedFunctional.cpp
          TimeLinearIntegrator.cpp
          TimeReducedFunctional.cpp
  PUBLIC EnsembleBasis.hpp
         LinearStateSolver.hpp
         NewtonStateSolver.hpp
         ReducedFunctional.hpp
         StateSolver.hpp
         TimeLinearIntegrator.hpp
         TimeReducedFunctional.hpp
         TimeStateMonitor.hpp
         TimeIntegrator.hpp
         TimeTrajectory.hpp)

target_link_libraries(femx_state PUBLIC femx_problem femx_linalg)
