add_executable(runLagrangeQuadQ1Tests unit/fem/runLagrangeQuadQ1Tests.cpp)

target_link_libraries(runLagrangeQuadQ1Tests PRIVATE femx::femx)

add_test(NAME LagrangeQuadQ1Tests COMMAND runLagrangeQuadQ1Tests)

add_executable(runBoundaryElementValuesTests
               unit/fem/runBoundaryElementValuesTests.cpp)

target_link_libraries(runBoundaryElementValuesTests PRIVATE femx::femx)

add_test(NAME BoundaryElementValuesTests COMMAND runBoundaryElementValuesTests)

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

add_executable(runObjectiveCompositionTests
               unit/inverse/runObjectiveCompositionTests.cpp)

target_link_libraries(runObjectiveCompositionTests PRIVATE femx::femx)

add_test(NAME ObjectiveCompositionTests COMMAND runObjectiveCompositionTests)

add_executable(runDerivativeCheckTests unit/inverse/runDerivativeCheckTests.cpp)

target_link_libraries(runDerivativeCheckTests PRIVATE femx::femx)

add_test(NAME DerivativeCheckTests COMMAND runDerivativeCheckTests)

add_executable(runEquationSolverTests unit/equation/runEquationSolverTests.cpp)

target_link_libraries(runEquationSolverTests PRIVATE femx::femx)

add_test(NAME EquationSolverTests COMMAND runEquationSolverTests)

add_executable(runMatrixEquationSolverTests
               unit/equation/runMatrixEquationSolverTests.cpp)

target_link_libraries(runMatrixEquationSolverTests PRIVATE femx::femx)

add_test(NAME MatrixEquationSolverTests COMMAND runMatrixEquationSolverTests)

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

if(TARGET femx::system_petsc)
  add_executable(runKspLinearSolverTests
                 unit/system/petsc/runKspLinearSolverTests.cpp)

  target_link_libraries(runKspLinearSolverTests PRIVATE femx::system_petsc
                                                        femx::femx)

  add_test(NAME KspLinearSolverTests COMMAND runKspLinearSolverTests)

  add_executable(runPETScSystemMatrixTests
                 unit/system/petsc/runPETScSystemMatrixTests.cpp)

  target_link_libraries(runPETScSystemMatrixTests PRIVATE femx::system_petsc
                                                          femx::femx)

  add_test(NAME PETScSystemMatrixTests COMMAND runPETScSystemMatrixTests)
endif()

if(TARGET femx::inverse_petsc)
  add_executable(runTaoReducedFunctionalAdapterTests
                 unit/inverse/petsc/runTaoReducedFunctionalAdapterTests.cpp)

  target_link_libraries(runTaoReducedFunctionalAdapterTests
                        PRIVATE femx::inverse_petsc femx::femx)

  add_test(NAME TaoReducedFunctionalAdapterTests
           COMMAND runTaoReducedFunctionalAdapterTests)

  add_executable(runTaoOptimizerTests
                 unit/inverse/petsc/runTaoOptimizerTests.cpp)

  target_link_libraries(runTaoOptimizerTests PRIVATE femx::inverse_petsc
                                                     femx::femx)

  add_test(NAME TaoOptimizerTests COMMAND runTaoOptimizerTests)
endif()

if(TARGET navier_gls_kernels)
  add_executable(
    runNavierConfigTests
    unit/navier/runNavierConfigTests.cpp ../apps/forward/navier-gls/Config.cpp
    ../apps/forward/navier-gls/BoundaryConditions.cpp)

  target_include_directories(
    runNavierConfigTests
    PRIVATE ${CMAKE_SOURCE_DIR}/apps/forward/navier-gls
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
