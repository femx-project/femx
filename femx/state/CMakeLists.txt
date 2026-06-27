add_library(femx_state)
add_library(femx::state ALIAS femx_state)

target_sources(
  femx_state
  PRIVATE EnsembleBasis.cpp
          EnsembleReducedFunctional.cpp
          NewtonStateSolver.cpp
          ReducedFunctional.cpp
          DerivativeCheck.cpp
          TimeLinearIntegrator.cpp
          TimeReducedFunctional.cpp
          TimeNewtonIntegrator.cpp
  PUBLIC EnsembleBasis.hpp
         EnsembleReducedFunctional.hpp
         NewtonStateSolver.hpp
         ReducedFunctional.hpp
         StateSolver.hpp
         DerivativeCheck.hpp
         TimeLinearIntegrator.hpp
         TimeNewtonIntegrator.hpp
         TimeReducedFunctional.hpp
         TimeStateMonitor.hpp
         TimeIntegrator.hpp
         TimeTrajectory.hpp)

target_link_libraries(femx_state PUBLIC femx_problem femx_linalg)
