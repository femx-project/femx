add_library(femx_inverse INTERFACE)
add_library(femx::inverse ALIAS femx_inverse)

target_sources(
  femx_inverse
  INTERFACE AdjointReducedFunctional.hpp
            AdjointSolver.hpp
            DerivativeCheck.hpp
            EquationAdjointSolver.hpp
            LeastSquaresObjective.hpp
            ObjectiveFunctional.hpp
            ObservationOperator.hpp
            QuadraticParameterRegularization.hpp
            ReducedFunctional.hpp
            MatrixEquationAdjointSolver.hpp
            SumObjectiveFunctional.hpp)

target_link_libraries(
  femx_inverse INTERFACE femx_common femx_config femx_equation femx_linalg
                         femx_system)

if(FEMX_HAS_PETSC)
  add_subdirectory(petsc)
endif()
