add_library(femx_solve)
add_library(femx::solve ALIAS femx_solve)

target_sources(
  femx_solve
  PRIVATE Newton.cpp
          ReducedFunctional.cpp
          DerivativeCheck.cpp
          TimeLinearStateSolver.cpp
          TimeReducedFunctional.cpp
          TimeStepper.cpp
  PUBLIC Newton.hpp
         ReducedFunctional.hpp
         DerivativeCheck.hpp
         TimeLinearStateSolver.hpp
         TimeReducedFunctional.hpp
         TimeStateSolver.hpp
         TimeStepper.hpp
         TimeTrajectory.hpp)

target_link_libraries(femx_solve PUBLIC femx_problem femx_algebra)
