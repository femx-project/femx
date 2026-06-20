add_executable(runMathTests unit/common/runMathTests.cpp)

target_link_libraries(runMathTests PRIVATE femx::femx)

add_test(NAME MathTests COMMAND runMathTests)

add_executable(runNewIncludeSmokeTests unit/algebra/runNewIncludeSmokeTests.cpp)

target_link_libraries(runNewIncludeSmokeTests PRIVATE femx::algebra)

add_test(NAME NewIncludeSmokeTests COMMAND runNewIncludeSmokeTests)

add_executable(runResidualAdapterTests unit/problem/runResidualAdapterTests.cpp)

target_link_libraries(runResidualAdapterTests PRIVATE femx::femx)

add_test(NAME ResidualAdapterTests COMMAND runResidualAdapterTests)

add_executable(runTimeResidualAdapterTests
               unit/problem/runTimeResidualAdapterTests.cpp)

target_link_libraries(runTimeResidualAdapterTests PRIVATE femx::femx)

add_test(NAME TimeResidualAdapterTests COMMAND runTimeResidualAdapterTests)

add_executable(runNewtonTests unit/solve/runNewtonTests.cpp)

target_link_libraries(runNewtonTests PRIVATE femx::femx)

add_test(NAME NewtonTests COMMAND runNewtonTests)

add_executable(runTimeStepperTests unit/solve/runTimeStepperTests.cpp)

target_link_libraries(runTimeStepperTests PRIVATE femx::femx)

add_test(NAME TimeStepperTests COMMAND runTimeStepperTests)

add_executable(runSolveReducedFunctionalTests
               unit/solve/runReducedFunctionalTests.cpp)

target_link_libraries(runSolveReducedFunctionalTests PRIVATE femx::femx)

add_test(NAME SolveReducedFunctionalTests
         COMMAND runSolveReducedFunctionalTests)

add_executable(runNewApiLinearControlExampleTests
               unit/solve/runNewApiLinearControlExampleTests.cpp)

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

add_executable(runTimePointSamplerTests
               unit/inverse/runTimePointSamplerTests.cpp)

target_link_libraries(runTimePointSamplerTests PRIVATE femx::femx)

add_test(NAME TimePointSamplerTests COMMAND runTimePointSamplerTests)

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

if(TARGET navier_var_kernels)
  add_executable(runNavierVarConfigTests
                 unit/stokes/runNavierVarConfigTests.cpp)

  target_link_libraries(runNavierVarConfigTests PRIVATE navier_var_kernels
                                                        femx::femx)

  add_test(NAME NavierVarConfigTests COMMAND runNavierVarConfigTests)

  add_executable(runNavierStokesEquationTests
                 unit/stokes/runTimeNavierStokesResidualEquationTests.cpp)

  target_link_libraries(runNavierStokesEquationTests PRIVATE navier_var_kernels
                                                             femx::femx)

  add_test(NAME NavierStokesEquationTests COMMAND runNavierStokesEquationTests)

  add_executable(runNavierVarRunSupportTests
                 unit/stokes/runNavierVarRunSupportTests.cpp)

  target_link_libraries(runNavierVarRunSupportTests PRIVATE navier_var_kernels
                                                            femx::femx)

  add_test(NAME NavierVarRunSupportTests COMMAND runNavierVarRunSupportTests)

  add_executable(runTimeNavierStokesAssemblyBenchmark
                 unit/stokes/runTimeNavierStokesAssemblyBenchmark.cpp)

  target_link_libraries(runTimeNavierStokesAssemblyBenchmark
                        PRIVATE navier_var_kernels femx::femx)
endif()

if(TARGET femx::algebra_petsc)
  add_executable(runKspLinearSolverTests
                 unit/system/petsc/runKspLinearSolverTests.cpp)

  target_link_libraries(runKspLinearSolverTests PRIVATE femx::algebra_petsc
                                                        femx::femx)

  add_test(NAME KspLinearSolverTests COMMAND runKspLinearSolverTests)

  add_executable(runPETScSystemMatrixTests
                 unit/system/petsc/runPETScSystemMatrixTests.cpp)

  target_link_libraries(runPETScSystemMatrixTests PRIVATE femx::algebra_petsc
                                                          femx::femx)

  add_test(NAME PETScSystemMatrixTests COMMAND runPETScSystemMatrixTests)
endif()

if(TARGET femx::optimize)
  add_executable(runTaoReducedFunctionalAdapterTests
                 unit/inverse/petsc/runTaoReducedFunctionalAdapterTests.cpp)

  target_link_libraries(runTaoReducedFunctionalAdapterTests
                        PRIVATE femx::optimize femx::femx)

  add_test(NAME TaoReducedFunctionalAdapterTests
           COMMAND runTaoReducedFunctionalAdapterTests)

  add_executable(runTaoOptimizerTests
                 unit/inverse/petsc/runTaoOptimizerTests.cpp)

  target_link_libraries(runTaoOptimizerTests PRIVATE femx::optimize
                                                     femx::femx)

  add_test(NAME TaoOptimizerTests COMMAND runTaoOptimizerTests)
endif()

if(TARGET navier_gls_kernels)
  add_executable(
    runNavierConfigTests
    unit/navier/runNavierConfigTests.cpp ../apps/forward/ns-gls/Config.cpp
    ../apps/forward/ns-gls/BoundaryConditions.cpp)

  target_include_directories(
    runNavierConfigTests
    PRIVATE ${CMAKE_SOURCE_DIR}/apps/forward/ns-gls
            ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

  target_link_libraries(runNavierConfigTests PRIVATE femx::femx)

  add_test(NAME NavierConfigTests COMMAND runNavierConfigTests)

  add_executable(runNavierElementResidualTests
                 unit/navier/runNavierElementResidualTests.cpp)

  target_link_libraries(runNavierElementResidualTests
                        PRIVATE navier_gls_kernels)

  add_test(NAME NavierElementResidualTests
           COMMAND runNavierElementResidualTests)
endif()

if(FEMX_HAS_ENZYME)
  add_executable(runEnzymeSmokeTests unit/ad/runEnzymeSmokeTests.cpp)

  target_link_libraries(runEnzymeSmokeTests PRIVATE femx::femx)

  add_test(NAME EnzymeSmokeTests COMMAND runEnzymeSmokeTests)
endif()
