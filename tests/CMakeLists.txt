add_executable(runLinalgTests unit/linalg/runLinalgTests.cpp)
target_include_directories(runLinalgTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runLinalgTests PRIVATE femx::femx)
add_test(NAME LinalgTests COMMAND runLinalgTests)

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
