add_library(femx_problem)
add_library(femx::problem ALIAS femx_problem)

target_sources(
  femx_problem
  PRIVATE LeastSquaresObjective.cpp SumObjective.cpp
          SumTimeObjective.cpp TimeLeastSquaresObjective.cpp
          TimeObservationData.cpp TimeRegularization.cpp
  PUBLIC Linearization.hpp
         LeastSquaresObjective.hpp
         Objective.hpp
         Residual.hpp
         SumObjective.hpp
         SumTimeObjective.hpp
         TimeLeastSquaresObjective.hpp
         TimeObjective.hpp
         TimeObservationOperator.hpp
         TimeObservationData.hpp
         TimeRegularization.hpp
         TimeResidual.hpp)

target_link_libraries(femx_problem PUBLIC femx_config femx_common femx_linalg)
