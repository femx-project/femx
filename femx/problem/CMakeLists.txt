add_library(femx_problem)
add_library(femx::problem ALIAS femx_problem)

target_sources(
  femx_problem
  PRIVATE MatrixResidualEquation.cpp
          LeastSquaresObjective.cpp
          QuadraticParameterRegularization.cpp
          SumObjectiveFunctional.cpp
          SumTimeObjectiveFunctional.cpp
          TimeDirichletControlEquation.cpp
          TimeLeastSquaresObjective.cpp
          TimeMatrixResidualEquation.cpp
          TimeObservationData.cpp
          TimeRegularization.cpp
  PUBLIC Linearization.hpp
            LeastSquaresObjective.hpp
            MatrixResidualEquation.hpp
            Objective.hpp
            ObjectiveFunctional.hpp
            Observation.hpp
            ObservationOperator.hpp
            ProblemResidualAdapter.hpp
            QuadraticParameterRegularization.hpp
            Residual.hpp
            ResidualEquation.hpp
            SumObjectiveFunctional.hpp
            SumObjective.hpp
            SumTimeObjectiveFunctional.hpp
            SumTimeObjective.hpp
            TimeDirichletControlEquation.hpp
            TimeLeastSquaresObjective.hpp
            TimeMatrixResidual.hpp
            TimeMatrixResidualEquation.hpp
            TimeObjective.hpp
            TimeObjectiveFunctional.hpp
            TimeObservation.hpp
            TimeObservationData.hpp
            TimeObservationOperator.hpp
            TimeProblemAdapter.hpp
            TimeRegularization.hpp
            TimeResidual.hpp
            TimeResidualEquation.hpp)

target_link_libraries(femx_problem PUBLIC femx_config femx_core
                                          femx_algebra femx_fem)

if(FEMX_HAS_PETSC)
  target_link_libraries(femx_problem PRIVATE femx_algebra_petsc)
endif()
