add_library(femx_eq)
add_library(femx::eq ALIAS femx_eq)

target_sources(
  femx_eq
  PRIVATE MatrixLinearStateSolver.cpp
          MatrixNewtonStateSolver.cpp
          MatrixResidualEquation.cpp
          OperatorNewtonStateSolver.cpp
          StateJacobianOperator.cpp
          TimeDirichletControlEquation.cpp
          TimeMatrixLinearStateSolver.cpp
          TimeMatrixNewtonStateSolver.cpp
          TimeMatrixResidualEquation.cpp
          TimeNewtonStateSolver.cpp
          TimeStateJacobianOperator.cpp
          TimeStateTrajectory.cpp
  PUBLIC MatrixLinearStateSolver.hpp
         MatrixNewtonStateSolver.hpp
         MatrixResidualEquation.hpp
         OperatorNewtonStateSolver.hpp
         ResidualEquation.hpp
         StateJacobianOperator.hpp
         StateSolver.hpp
         TimeDirichletControlEquation.hpp
         TimeMatrixLinearStateSolver.hpp
         TimeMatrixNewtonStateSolver.hpp
         TimeMatrixResidualEquation.hpp
         TimeNewtonStateSolver.hpp
         TimeResidualEquation.hpp
         TimeStateJacobianOperator.hpp
         TimeStateSolver.hpp
         TimeStateTrajectory.hpp)

target_link_libraries(femx_eq PUBLIC femx_bc femx_common femx_config
                                     femx_linalg femx_system)

if(TARGET femx::system_petsc)
  target_link_libraries(femx_eq PRIVATE femx::system_petsc)
endif()
