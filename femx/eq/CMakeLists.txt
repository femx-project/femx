add_library(femx_equation INTERFACE)
add_library(femx::equation ALIAS femx_equation)

target_sources(
  femx_equation
  INTERFACE NewtonStateSolver.hpp ResidualEquation.hpp
            StateJacobianOperator.hpp StateSolver.hpp
            MatrixNewtonStateSolver.hpp AssembledResidualEquation.hpp)

target_link_libraries(femx_equation INTERFACE femx_common femx_config
                                              femx_linalg femx_system)
