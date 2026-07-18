add_executable(runLinalgTests unit/linalg/runLinalgTests.cpp)
target_include_directories(runLinalgTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runLinalgTests PRIVATE femx::femx)
add_test(NAME LinalgTests COMMAND runLinalgTests)

add_executable(runMemorySpaceTests unit/backend/runMemorySpaceTests.cpp)
target_include_directories(runMemorySpaceTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runMemorySpaceTests PRIVATE femx::state)
add_test(NAME MemorySpaceTests COMMAND runMemorySpaceTests)

add_executable(runBackendAssemblyTests
               unit/backend/runBackendAssemblyTests.cpp)
target_include_directories(runBackendAssemblyTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runBackendAssemblyTests PRIVATE femx::femx)
add_test(NAME BackendAssemblyTests COMMAND runBackendAssemblyTests)

add_executable(runBackendStateTests unit/backend/runBackendStateTests.cpp)
target_include_directories(runBackendStateTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runBackendStateTests PRIVATE femx::femx)
add_test(NAME BackendStateTests COMMAND runBackendStateTests)

add_executable(
  runPoissonAssemblyTests
  unit/examples/runPoissonAssemblyTests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../examples/poisson/PoissonForward.cpp)
target_include_directories(
  runPoissonAssemblyTests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}/../examples/poisson)
target_link_libraries(runPoissonAssemblyTests PRIVATE femx::femx)
add_test(NAME PoissonAssemblyTests COMMAND runPoissonAssemblyTests)

if(FEMX_HAS_CUDA)
  add_executable(runCudaAssemblyTests
                 unit/backend/runCudaAssemblyTests.cu)
  target_include_directories(runCudaAssemblyTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runCudaAssemblyTests PRIVATE femx::femx)
  set_target_properties(runCudaAssemblyTests PROPERTIES CUDA_STANDARD 17
                                                        CUDA_STANDARD_REQUIRED ON)
  add_test(NAME CudaAssemblyTests COMMAND runCudaAssemblyTests)
  set_tests_properties(CudaAssemblyTests PROPERTIES LABELS "cuda;optional")

  add_executable(runCudaLinalgTests unit/backend/runCudaLinalgTests.cu)
  target_include_directories(runCudaLinalgTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runCudaLinalgTests PRIVATE femx::linalg)
  set_target_properties(runCudaLinalgTests PROPERTIES CUDA_STANDARD 17
                                                      CUDA_STANDARD_REQUIRED ON)
  add_test(NAME CudaLinalgTests COMMAND runCudaLinalgTests)
  set_tests_properties(CudaLinalgTests PROPERTIES LABELS "cuda;optional")

  add_executable(runCudaTrajectoryTests
                 unit/state/runCudaTrajectoryTests.cu)
  target_include_directories(runCudaTrajectoryTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runCudaTrajectoryTests PRIVATE femx::state)
  set_target_properties(runCudaTrajectoryTests PROPERTIES CUDA_STANDARD 17
                                                          CUDA_STANDARD_REQUIRED ON)
  add_test(NAME CudaTrajectoryTests COMMAND runCudaTrajectoryTests)
  set_tests_properties(CudaTrajectoryTests PROPERTIES LABELS "cuda;optional")

  add_executable(runCudaObjectiveTests
                 unit/inverse/runCudaObjectiveTests.cu)
  target_include_directories(runCudaObjectiveTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runCudaObjectiveTests PRIVATE femx::inverse)
  set_target_properties(runCudaObjectiveTests PROPERTIES CUDA_STANDARD 17
                                                       CUDA_STANDARD_REQUIRED ON)
  add_test(NAME CudaObjectiveTests COMMAND runCudaObjectiveTests)
  set_tests_properties(CudaObjectiveTests
                       PROPERTIES LABELS "inverse;cuda;focused")
endif()

if(FEMX_HAS_CUDA
   AND FEMX_HAS_RESOLVE_CUDA
   AND TARGET femx_linalg_resolve
   AND NOT FEMX_RESOLVE_BACKEND_MODE STREQUAL "CPU")
  add_executable(runReSolveDeviceTests
                 unit/backend/runReSolveDeviceTests.cpp)
  target_include_directories(runReSolveDeviceTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runReSolveDeviceTests PRIVATE femx::linalg_resolve)
  add_test(NAME ReSolveDeviceTests COMMAND runReSolveDeviceTests)
  set_tests_properties(ReSolveDeviceTests
                       PROPERTIES LABELS "resolve;cuda;optional")

  add_executable(
    runResolveTimeIntegratorTests
    unit/model/ns/runResolveTimeIntegratorTests.cpp)
  target_include_directories(runResolveTimeIntegratorTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runResolveTimeIntegratorTests
                        PRIVATE femx::model::ns femx::linalg_resolve)
  add_test(NAME ResolveTimeIntegratorTests
           COMMAND runResolveTimeIntegratorTests)
  set_tests_properties(ResolveTimeIntegratorTests
                       PROPERTIES LABELS "resolve;cuda;optional")

  add_executable(
    runResolveTimeReducedTests
    unit/model/ns/runResolveTimeReducedTests.cpp)
  target_include_directories(runResolveTimeReducedTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runResolveTimeReducedTests
                        PRIVATE femx::model::ns femx::inverse
                                femx::linalg_resolve)
  add_test(NAME ResolveTimeReducedTests
           COMMAND runResolveTimeReducedTests)
  set_tests_properties(ResolveTimeReducedTests
                       PROPERTIES LABELS "resolve;cuda;inverse;optional")
endif()

add_executable(runReferenceElementTests unit/fem/runReferenceElementTests.cpp)
target_include_directories(runReferenceElementTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runReferenceElementTests PRIVATE femx::femx)
add_test(NAME ReferenceElementTests COMMAND runReferenceElementTests)

add_executable(runMeshSpaceTests unit/fem/runMeshSpaceTests.cpp)
target_include_directories(runMeshSpaceTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runMeshSpaceTests PRIVATE femx::femx)
add_test(NAME MeshSpaceTests COMMAND runMeshSpaceTests)

add_executable(runTimePointInterpolatorTests
               unit/fem/runTimePointInterpolatorTests.cpp)
target_include_directories(runTimePointInterpolatorTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runTimePointInterpolatorTests PRIVATE femx::fem)
add_test(NAME TimePointInterpolatorTests COMMAND runTimePointInterpolatorTests)
set_tests_properties(TimePointInterpolatorTests PROPERTIES LABELS "fem;focused")

add_executable(runControlMapTests unit/fem/runControlMapTests.cpp)
target_include_directories(runControlMapTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runControlMapTests PRIVATE femx::fem)
if(FEMX_HAS_CUDA)
  set_source_files_properties(unit/fem/runControlMapTests.cpp
                              PROPERTIES LANGUAGE CUDA)
  set_target_properties(runControlMapTests PROPERTIES CUDA_STANDARD 17
                                                      CUDA_STANDARD_REQUIRED ON)
endif()
add_test(NAME ControlMapTests COMMAND runControlMapTests)
set_tests_properties(ControlMapTests PROPERTIES LABELS "fem;focused")

add_executable(runBoundaryTests unit/fem/runBoundaryTests.cpp)
target_include_directories(runBoundaryTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runBoundaryTests PRIVATE femx::femx)
add_test(NAME BoundaryTests COMMAND runBoundaryTests)

add_executable(runAssemblyTests unit/assembly/runAssemblyTests.cpp)
target_include_directories(runAssemblyTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runAssemblyTests PRIVATE femx::femx)
add_test(NAME AssemblyTests COMMAND runAssemblyTests)

add_executable(runIoTests unit/io/runIoTests.cpp)
target_include_directories(runIoTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runIoTests PRIVATE femx::femx)
add_test(NAME IoTests COMMAND runIoTests)

if(TARGET femx_linalg_resolve)
  add_executable(runReSolveSolverTests unit/linalg/runReSolveSolverTests.cpp)
  target_include_directories(runReSolveSolverTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runReSolveSolverTests
                        PRIVATE femx::linalg_resolve femx::assembly)
  add_test(NAME ReSolveSolverTests COMMAND runReSolveSolverTests)
  set_tests_properties(ReSolveSolverTests PROPERTIES LABELS "resolve;optional")
endif()

if(TARGET femx_linalg_petsc)
  add_executable(runPETScSolverTests unit/linalg/runPETScSolverTests.cpp)
  target_include_directories(runPETScSolverTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runPETScSolverTests
                        PRIVATE femx::linalg_petsc femx::assembly)
  add_test(NAME PETScSolverTests COMMAND runPETScSolverTests)
  set_tests_properties(PETScSolverTests PROPERTIES LABELS "petsc;optional")
endif()

add_executable(runStateTests unit/state/runStateTests.cpp)
target_include_directories(runStateTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runStateTests PRIVATE femx::state)
add_test(NAME StateTests COMMAND runStateTests)

add_executable(runInverseTests unit/inverse/runInverseTests.cpp)
target_include_directories(runInverseTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runInverseTests PRIVATE femx::inverse)
add_test(NAME InverseTests COMMAND runInverseTests)

add_executable(runNavierStokesModelTests
               unit/model/ns/runNavierStokesModelTests.cpp)
target_include_directories(runNavierStokesModelTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runNavierStokesModelTests PRIVATE femx::femx)
add_test(NAME NavierStokesModelTests COMMAND runNavierStokesModelTests)

add_executable(runForwardConfigTests
               unit/model/ns/runForwardConfigTests.cpp)
target_include_directories(runForwardConfigTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(
  runForwardConfigTests
  PRIVATE FEMX_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(runForwardConfigTests PRIVATE femx::model::ns)
add_test(NAME ForwardConfigTests COMMAND runForwardConfigTests)
set_tests_properties(ForwardConfigTests PROPERTIES LABELS "ns;focused")

add_executable(runNavierOperatorTests
               unit/model/ns/runNavierOperatorTests.cpp)
target_include_directories(runNavierOperatorTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runNavierOperatorTests PRIVATE femx::femx)
if(FEMX_HAS_CUDA)
  set_source_files_properties(unit/model/ns/runNavierOperatorTests.cpp
                              PROPERTIES LANGUAGE CUDA)
  set_target_properties(runNavierOperatorTests
                        PROPERTIES CUDA_STANDARD 17
                                   CUDA_STANDARD_REQUIRED ON)
endif()
add_test(NAME NavierOperatorTests COMMAND runNavierOperatorTests)
set_tests_properties(NavierOperatorTests PROPERTIES LABELS "ns;focused")
