add_executable(runLinalgTests unit/linalg/runLinalgTests.cpp)
target_include_directories(runLinalgTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runLinalgTests PRIVATE femx::femx)
add_test(NAME CoreLinalgTests COMMAND runLinalgTests)

add_executable(runReferenceElementTests unit/fem/runReferenceElementTests.cpp)
target_include_directories(runReferenceElementTests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runReferenceElementTests PRIVATE femx::femx)
add_test(NAME ReferenceElementTests COMMAND runReferenceElementTests)
