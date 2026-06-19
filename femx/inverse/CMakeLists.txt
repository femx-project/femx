add_library(femx_inverse)
add_library(femx::inverse ALIAS femx_inverse)

target_sources(
  femx_inverse
  PRIVATE AdjointReducedFunctional.cpp
          DerivativeCheck.cpp
          LeastSquaresObjective.cpp
          MatrixAdjointSolver.cpp
          ObservationGrid.cpp
          OperatorAdjointSolver.cpp
          QuadraticParameterRegularization.cpp
          ReducedFunctional.cpp
          SumObjectiveFunctional.cpp
          SumTimeObjectiveFunctional.cpp
          TimeAdjointReducedFunctional.cpp
          TimeLeastSquaresObjective.cpp
          TimeObservationData.cpp
          TimePointSampler.cpp
          TimeReducedFunctional.cpp
          TimeRegularization.cpp
  PUBLIC AdjointReducedFunctional.hpp
         AdjointSolver.hpp
         DerivativeCheck.hpp
         LeastSquaresObjective.hpp
         MatrixAdjointSolver.hpp
         ObjectiveFunctional.hpp
         ObservationGrid.hpp
         ObservationOperator.hpp
         OperatorAdjointSolver.hpp
         QuadraticParameterRegularization.hpp
         ReducedFunctional.hpp
         SumObjectiveFunctional.hpp
         SumTimeObjectiveFunctional.hpp
         TimeAdjointReducedFunctional.hpp
         TimeLeastSquaresObjective.hpp
         TimeObjectiveFunctional.hpp
         TimeObservationData.hpp
         TimeObservationOperator.hpp
         TimePointSampler.hpp
         TimeReducedFunctional.hpp
         TimeRegularization.hpp)

target_link_libraries(
  femx_inverse
  PUBLIC femx_common femx_config femx_eq femx_fem femx_linalg femx_system)

if(FEMX_HAS_PETSC)
  add_subdirectory(petsc)
endif()
