add_executable(runLinalgTests unit/linalg/runLinalgTests.cpp)
target_include_directories(runLinalgTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runLinalgTests PRIVATE femx::femx)
add_test(NAME LinalgTests COMMAND runLinalgTests)

add_executable(runHostAssemblyTests unit/assembly/runHostAssemblyTests.cpp)
target_include_directories(runHostAssemblyTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runHostAssemblyTests PRIVATE femx::femx)
add_test(NAME HostAssemblyTests COMMAND runHostAssemblyTests)

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
  add_executable(runCudaAssemblyTests unit/assembly/runCudaAssemblyTests.cu)
  target_include_directories(runCudaAssemblyTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runCudaAssemblyTests PRIVATE femx::femx)
  set_target_properties(runCudaAssemblyTests
                        PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME CudaAssemblyTests COMMAND runCudaAssemblyTests)
  set_tests_properties(CudaAssemblyTests PROPERTIES LABELS "cuda;optional")

  add_executable(runCudaLinalgTests unit/linalg/runCudaLinalgTests.cu)
  target_include_directories(runCudaLinalgTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runCudaLinalgTests PRIVATE femx::linalg)
  set_target_properties(runCudaLinalgTests PROPERTIES CUDA_STANDARD 17
                                                      CUDA_STANDARD_REQUIRED ON)
  add_test(NAME CudaLinalgTests COMMAND runCudaLinalgTests)
  set_tests_properties(CudaLinalgTests PROPERTIES LABELS "cuda;optional")

endif()

if(FEMX_HAS_CUDA
   AND FEMX_HAS_RESOLVE_CUDA
   AND TARGET femx_linalg_resolve
   AND NOT FEMX_RESOLVE_DEVICE_MODE STREQUAL "HOST")
  add_executable(runReSolveDeviceTests unit/linalg/runReSolveDeviceTests.cpp)
  target_include_directories(runReSolveDeviceTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runReSolveDeviceTests PRIVATE femx::linalg_resolve)
  add_test(NAME ReSolveDeviceTests COMMAND runReSolveDeviceTests)
  set_tests_properties(ReSolveDeviceTests PROPERTIES LABELS
                                                     "resolve;cuda;optional")

  add_executable(runResolveTimeIntegratorTests
                 unit/model/ns/runResolveTimeIntegratorTests.cpp)
  target_include_directories(runResolveTimeIntegratorTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(runResolveTimeIntegratorTests
                        PRIVATE femx::model::ns femx::linalg_resolve)
  add_test(NAME ResolveTimeIntegratorTests
           COMMAND runResolveTimeIntegratorTests)
  set_tests_properties(ResolveTimeIntegratorTests
                       PROPERTIES LABELS "resolve;cuda;optional")

  add_executable(runResolveTimeReducedTests
                 unit/model/ns/runResolveTimeReducedTests.cpp)
  target_include_directories(runResolveTimeReducedTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(
    runResolveTimeReducedTests PRIVATE femx::model::ns femx::inverse
                                       femx::linalg_resolve)
  add_test(NAME ResolveTimeReducedTests COMMAND runResolveTimeReducedTests)
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
target_include_directories(runControlMapTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
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
  target_link_libraries(runReSolveSolverTests PRIVATE femx::linalg_resolve
                                                      femx::assembly)
  add_test(NAME ReSolveSolverTests COMMAND runReSolveSolverTests)
  set_tests_properties(ReSolveSolverTests PROPERTIES LABELS "resolve;optional")
endif()

add_executable(runStateTests unit/state/runStateTests.cpp)
target_include_directories(runStateTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runStateTests PRIVATE femx::state)
add_test(NAME StateTests COMMAND runStateTests)

add_executable(runInverseTests unit/inverse/runInverseTests.cpp)
target_include_directories(runInverseTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runInverseTests PRIVATE femx::inverse)
add_test(NAME InverseTests COMMAND runInverseTests)

add_executable(runNsModelTests
               unit/model/ns/runModelTests.cpp)
target_include_directories(runNsModelTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runNsModelTests PRIVATE femx::femx)
add_test(NAME NsModelTests COMMAND runNsModelTests)

if(TARGET femx_ns_forward_support)
  add_executable(runNsForwardConfigTests
                 unit/apps/ns-forward/runConfigTests.cpp)
  target_include_directories(runNsForwardConfigTests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(
    runNsForwardConfigTests
    PRIVATE FEMX_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  target_link_libraries(runNsForwardConfigTests
                        PRIVATE femx::apps::ns_forward)
  add_test(NAME NsForwardConfigTests COMMAND runNsForwardConfigTests)
  set_tests_properties(NsForwardConfigTests PROPERTIES LABELS "ns;app;focused")
endif()
