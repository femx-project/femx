add_library(femx_solve)
add_library(femx::solve ALIAS femx_solve)

target_sources(femx_solve PRIVATE Newton.cpp PUBLIC Newton.hpp)

target_sources(femx_solve PRIVATE ReducedFunctional.cpp PUBLIC ReducedFunctional.hpp)

target_sources(
  femx_solve
  PRIVATE AdjointReducedFunctional.cpp
          DerivativeCheck.cpp
          MatrixAdjointSolver.cpp
          MatrixLinearStateSolver.cpp
          MatrixNewtonStateSolver.cpp
          OperatorNewtonStateSolver.cpp
          OperatorAdjointSolver.cpp
          ReducedObjective.cpp
          StateJacobianOperator.cpp
          TimeAdjointReducedFunctional.cpp
          TimeMatrixLinearStateSolver.cpp
          TimeMatrixNewtonStateSolver.cpp
          TimeNewtonStateSolver.cpp
          TimeObservationSampling.cpp
          TimeReducedFunctional.cpp
          TimeStateJacobianOperator.cpp
          TimeStateTrajectory.cpp
          TimeStepper.cpp
  PUBLIC AdjointReducedFunctional.hpp
         AdjointSolver.hpp
         DerivativeCheck.hpp
         MatrixAdjointSolver.hpp
         MatrixLinearStateSolver.hpp
         MatrixNewtonStateSolver.hpp
         OperatorAdjointSolver.hpp
         OperatorNewtonStateSolver.hpp
         ReducedObjective.hpp
         StateJacobianOperator.hpp
         StateSolver.hpp
         TimeAdjointReducedFunctional.hpp
         TimeMatrixLinearStateSolver.hpp
         TimeMatrixNewtonStateSolver.hpp
         TimeNewtonStateSolver.hpp
         TimeReducedFunctional.hpp
         TimeStateJacobianOperator.hpp
         TimeStateSolver.hpp
         TimeStateTrajectory.hpp
         TimeStepper.hpp
         TimeTrajectory.hpp)

target_link_libraries(femx_solve PUBLIC femx_problem femx_algebra)
