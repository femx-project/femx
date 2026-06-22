add_executable(runMathTests unit/common/runMathTests.cpp)

target_link_libraries(runMathTests PRIVATE femx::femx)

add_test(NAME MathTests COMMAND runMathTests)

add_executable(runNewIncludeSmokeTests unit/algebra/runNewIncludeSmokeTests.cpp)

target_link_libraries(runNewIncludeSmokeTests PRIVATE femx::linalg)

add_test(NAME NewIncludeSmokeTests COMMAND runNewIncludeSmokeTests)

add_executable(runResidualAdapterTests unit/problem/runResidualAdapterTests.cpp)

target_link_libraries(runResidualAdapterTests PRIVATE femx::femx)

add_test(NAME ResidualAdapterTests COMMAND runResidualAdapterTests)

add_executable(runTimeResidualAdapterTests
               unit/problem/runTimeResidualAdapterTests.cpp)

target_link_libraries(runTimeResidualAdapterTests PRIVATE femx::femx)

add_test(NAME TimeResidualAdapterTests COMMAND runTimeResidualAdapterTests)

add_executable(runNewtonStateSolverTests
               unit/state/runNewtonStateSolverTests.cpp)

target_link_libraries(runNewtonStateSolverTests PRIVATE femx::femx)

add_test(NAME NewtonStateSolverTests COMMAND runNewtonStateSolverTests)

add_executable(runTimeNewtonStateSolverTests
               unit/state/runTimeNewtonStateSolverTests.cpp)

target_link_libraries(runTimeNewtonStateSolverTests PRIVATE femx::femx)

add_test(NAME TimeNewtonStateSolverTests COMMAND runTimeNewtonStateSolverTests)

add_executable(runAB2TimeReducedFunctionalTests
               unit/state/runAB2TimeReducedFunctionalTests.cpp)

target_link_libraries(runAB2TimeReducedFunctionalTests PRIVATE femx::femx)

add_test(NAME AB2TimeReducedFunctionalTests
         COMMAND runAB2TimeReducedFunctionalTests)

add_executable(runStateReducedFunctionalTests
               unit/state/runReducedFunctionalTests.cpp)

target_link_libraries(runStateReducedFunctionalTests PRIVATE femx::femx)

add_test(NAME StateReducedFunctionalTests
         COMMAND runStateReducedFunctionalTests)

add_executable(runNewApiLinearControlExampleTests
               unit/state/runNewApiLinearControlExampleTests.cpp)

target_link_libraries(runNewApiLinearControlExampleTests PRIVATE femx::femx)

add_test(NAME NewApiLinearControlExampleTests
         COMMAND runNewApiLinearControlExampleTests)

add_executable(runVelocityProfileTests unit/bc/runVelocityProfileTests.cpp)

target_link_libraries(runVelocityProfileTests PRIVATE femx::femx)

add_test(NAME VelocityProfileTests COMMAND runVelocityProfileTests)

add_executable(runLagrangeQuadQ1Tests unit/fem/runLagrangeQuadQ1Tests.cpp)

target_link_libraries(runLagrangeQuadQ1Tests PRIVATE femx::femx)

add_test(NAME LagrangeQuadQ1Tests COMMAND runLagrangeQuadQ1Tests)

add_executable(runBoundaryElementValuesTests
               unit/fem/runBoundaryElementValuesTests.cpp)

target_link_libraries(runBoundaryElementValuesTests PRIVATE femx::femx)

add_test(NAME BoundaryElementValuesTests COMMAND runBoundaryElementValuesTests)

add_executable(runPublicIncludeSmokeTests
               unit/fem/runPublicIncludeSmokeTests.cpp)

target_link_libraries(runPublicIncludeSmokeTests PRIVATE femx::femx)

add_test(NAME PublicIncludeSmokeTests COMMAND runPublicIncludeSmokeTests)

add_executable(runResidualEquationTests
               unit/equation/runResidualEquationTests.cpp)

target_link_libraries(runResidualEquationTests PRIVATE femx::femx)

add_test(NAME ResidualEquationTests COMMAND runResidualEquationTests)

add_executable(runObjectiveFunctionalTests
               unit/inverse/runObjectiveFunctionalTests.cpp)

target_link_libraries(runObjectiveFunctionalTests PRIVATE femx::femx)

add_test(NAME ObjectiveFunctionalTests COMMAND runObjectiveFunctionalTests)

add_executable(runLeastSquaresObjectiveTests
               unit/inverse/runLeastSquaresObjectiveTests.cpp)

target_link_libraries(runLeastSquaresObjectiveTests PRIVATE femx::femx)

add_test(NAME LeastSquaresObjectiveTests COMMAND runLeastSquaresObjectiveTests)

add_executable(runTimeLeastSquaresObjectiveTests
               unit/inverse/runTimeLeastSquaresObjectiveTests.cpp)

target_link_libraries(runTimeLeastSquaresObjectiveTests PRIVATE femx::femx)

add_test(NAME TimeLeastSquaresObjectiveTests
         COMMAND runTimeLeastSquaresObjectiveTests)

add_executable(runTimePointInterpolatorTests
               unit/inverse/runTimePointInterpolatorTests.cpp)

target_link_libraries(runTimePointInterpolatorTests PRIVATE femx::femx)

add_test(NAME TimePointInterpolatorTests COMMAND runTimePointInterpolatorTests)

add_executable(runVtiWriterTests unit/io/runVtiWriterTests.cpp)

target_link_libraries(runVtiWriterTests PRIVATE femx::femx)

add_test(NAME VtiWriterTests COMMAND runVtiWriterTests)

add_executable(runTimeSeriesDataInTests unit/io/runTimeSeriesDataInTests.cpp)

target_link_libraries(runTimeSeriesDataInTests PRIVATE femx::femx)

add_test(NAME TimeSeriesDataInTests COMMAND runTimeSeriesDataInTests)

if(TARGET make_obs_config)
  add_executable(runMakeObsObservationVtiTests
                 unit/io/runMakeObsObservationVtiTests.cpp)

  target_link_libraries(runMakeObsObservationVtiTests PRIVATE make_obs_config
                                                              femx::femx)

  add_test(NAME MakeObsObservationVtiTests
           COMMAND runMakeObsObservationVtiTests)

  add_executable(runMakeObsTrajectoryObservationTests
                 unit/io/runMakeObsTrajectoryObservationTests.cpp)

  target_link_libraries(runMakeObsTrajectoryObservationTests
                        PRIVATE make_obs_config femx::femx)

  add_test(NAME MakeObsTrajectoryObservationTests
           COMMAND runMakeObsTrajectoryObservationTests)
endif()

add_executable(runObjectiveCompositionTests
               unit/inverse/runObjectiveCompositionTests.cpp)

target_link_libraries(runObjectiveCompositionTests PRIVATE femx::femx)

add_test(NAME ObjectiveCompositionTests COMMAND runObjectiveCompositionTests)

add_executable(runDerivativeCheckTests unit/inverse/runDerivativeCheckTests.cpp)

target_link_libraries(runDerivativeCheckTests PRIVATE femx::femx)

add_test(NAME DerivativeCheckTests COMMAND runDerivativeCheckTests)

add_executable(runOperatorSolverTests unit/equation/runOperatorSolverTests.cpp)

target_link_libraries(runOperatorSolverTests PRIVATE femx::femx)

add_test(NAME OperatorSolverTests COMMAND runOperatorSolverTests)

add_executable(runMatrixSolverTests unit/equation/runMatrixSolverTests.cpp)

target_link_libraries(runMatrixSolverTests PRIVATE femx::femx)

add_test(NAME MatrixSolverTests COMMAND runMatrixSolverTests)

add_executable(runSystemAssemblerTests
               unit/assembly/runSystemAssemblerTests.cpp)

target_link_libraries(runSystemAssemblerTests PRIVATE femx::femx)

add_test(NAME SystemAssemblerTests COMMAND runSystemAssemblerTests)

add_executable(runElementResidualEquationTests
               unit/assembly/runElementResidualEquationTests.cpp)

target_link_libraries(runElementResidualEquationTests PRIVATE femx::femx)

add_test(NAME ElementResidualEquationTests
         COMMAND runElementResidualEquationTests)

add_executable(runBoundaryResidualEquationTests
               unit/assembly/runBoundaryResidualEquationTests.cpp)

target_link_libraries(runBoundaryResidualEquationTests PRIVATE femx::femx)

add_test(NAME BoundaryResidualEquationTests
         COMMAND runBoundaryResidualEquationTests)

add_executable(runForwardADTests unit/ad/runForwardADTests.cpp)

target_link_libraries(runForwardADTests PRIVATE femx::femx)

add_test(NAME ForwardADTests COMMAND runForwardADTests)

if(ReSolve_FOUND)
  add_executable(runReSolveLinearSolverTests
                 unit/system/resolve/runReSolveLinearSolverTests.cpp)

  target_link_libraries(runReSolveLinearSolverTests PRIVATE femx::femx)

  add_test(NAME ReSolveLinearSolverTests COMMAND runReSolveLinearSolverTests)
endif()

add_executable(runReducedFunctionalTests
               unit/inverse/runReducedFunctionalTests.cpp)

target_link_libraries(runReducedFunctionalTests PRIVATE femx::femx)

add_test(NAME ReducedFunctionalTests COMMAND runReducedFunctionalTests)

add_executable(runStateSolverTests unit/equation/runStateSolverTests.cpp)

target_link_libraries(runStateSolverTests PRIVATE femx::femx)

add_test(NAME StateSolverTests COMMAND runStateSolverTests)

add_executable(runAdjointSolverTests unit/inverse/runAdjointSolverTests.cpp)

target_link_libraries(runAdjointSolverTests PRIVATE femx::femx)

add_test(NAME AdjointSolverTests COMMAND runAdjointSolverTests)

add_executable(runAdjointReducedFunctionalTests
               unit/inverse/runAdjointReducedFunctionalTests.cpp)

target_link_libraries(runAdjointReducedFunctionalTests PRIVATE femx::femx)

add_test(NAME AdjointReducedFunctionalTests
         COMMAND runAdjointReducedFunctionalTests)

add_executable(runTimeAdjointReducedFunctionalTests
               unit/inverse/runTimeAdjointReducedFunctionalTests.cpp)

target_link_libraries(runTimeAdjointReducedFunctionalTests PRIVATE femx::femx)

add_test(NAME TimeAdjointReducedFunctionalTests
         COMMAND runTimeAdjointReducedFunctionalTests)

if(TARGET femx::linalg_petsc)
  add_executable(runKspLinearSolverTests
                 unit/system/petsc/runKspLinearSolverTests.cpp)

  target_link_libraries(runKspLinearSolverTests PRIVATE femx::linalg_petsc
                                                        femx::femx)

  add_test(NAME KspLinearSolverTests COMMAND runKspLinearSolverTests)

  add_executable(runPETScSystemMatrixTests
                 unit/system/petsc/runPETScSystemMatrixTests.cpp)

  target_link_libraries(runPETScSystemMatrixTests PRIVATE femx::linalg_petsc
                                                          femx::femx)

  add_test(NAME PETScSystemMatrixTests COMMAND runPETScSystemMatrixTests)
endif()

if(TARGET femx::opt)
  add_executable(runTaoReducedFunctionalAdapterTests
                 unit/inverse/petsc/runTaoReducedFunctionalAdapterTests.cpp)

  target_link_libraries(runTaoReducedFunctionalAdapterTests PRIVATE femx::opt
                                                                    femx::femx)

  add_test(NAME TaoReducedFunctionalAdapterTests
           COMMAND runTaoReducedFunctionalAdapterTests)

  add_executable(runTaoOptimizerTests
                 unit/inverse/petsc/runTaoOptimizerTests.cpp)

  target_link_libraries(runTaoOptimizerTests PRIVATE femx::opt femx::femx)

  add_test(NAME TaoOptimizerTests COMMAND runTaoOptimizerTests)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/navier/runNavierConfigTests.cpp")
  add_executable(
    runNavierConfigTests
    unit/navier/runNavierConfigTests.cpp ../apps/forward/ns-gls/Config.cpp
    ../apps/forward/ns-gls/BCs.cpp)

  target_include_directories(
    runNavierConfigTests
    PRIVATE ${CMAKE_SOURCE_DIR}/apps/forward/ns-gls
            ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

  target_link_libraries(runNavierConfigTests PRIVATE femx::femx)

  add_test(NAME NavierConfigTests COMMAND runNavierConfigTests)
endif()

if(TARGET navier_common
   AND EXISTS
       "${CMAKE_CURRENT_SOURCE_DIR}/unit/navier/runNavierElementResidualTests.cpp"
)
  add_executable(runNavierElementResidualTests
                 unit/navier/runNavierElementResidualTests.cpp)

  target_link_libraries(runNavierElementResidualTests PRIVATE navier_common)

  add_test(NAME NavierElementResidualTests
           COMMAND runNavierElementResidualTests)
endif()

if(FEMX_HAS_ENZYME)
  add_executable(runEnzymeSmokeTests unit/ad/runEnzymeSmokeTests.cpp)

  target_link_libraries(runEnzymeSmokeTests PRIVATE femx::femx)

  add_test(NAME EnzymeSmokeTests COMMAND runEnzymeSmokeTests)

  add_executable(runEnzymeTimeVolumeKernelTests
                 unit/ad/runEnzymeTimeVolumeKernelTests.cpp)

  target_link_libraries(runEnzymeTimeVolumeKernelTests PRIVATE femx::femx)

  add_test(NAME EnzymeTimeVolumeKernelTests
           COMMAND runEnzymeTimeVolumeKernelTests)
endif()
