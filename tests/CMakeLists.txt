function(femx_add_test_if_exists target test_name)
  set(sources ${ARGN})
  foreach(src IN LISTS sources)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
      message(STATUS "Skipping ${test_name}: missing ${src}")
      return()
    endif()
  endforeach()

  add_executable(${target} ${sources})
  target_link_libraries(${target} PRIVATE femx::femx femx_test_helpers)
  add_test(NAME ${test_name} COMMAND ${target})
endfunction()

add_library(femx_test_helpers helpers/DerivativeCheck.cpp)
target_include_directories(femx_test_helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(femx_test_helpers PUBLIC femx::femx)

femx_add_test_if_exists(runNewtonStateSolverTests NewtonStateSolverTests
                        unit/state/runNewtonStateSolverTests.cpp)

femx_add_test_if_exists(runEnsembleBasisTests EnsembleBasisTests
                        unit/state/runEnsembleBasisTests.cpp)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/navier/runNavierConfigTests.cpp")
  add_executable(runNavierConfigTests unit/navier/runNavierConfigTests.cpp)

  target_include_directories(
    runNavierConfigTests
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

  target_link_libraries(runNavierConfigTests PRIVATE femx::model::ns)

  add_test(NAME NavierConfigTests COMMAND runNavierConfigTests)
endif()

if(TARGET femx::model::ns
   AND EXISTS
       "${CMAKE_CURRENT_SOURCE_DIR}/unit/navier/runNavierElementResidualTests.cpp"
)
  add_executable(runNavierElementResidualTests
                 unit/navier/runNavierElementResidualTests.cpp)

  target_link_libraries(runNavierElementResidualTests PRIVATE femx::model::ns)

  add_test(NAME NavierElementResidualTests
           COMMAND runNavierElementResidualTests)
endif()

if(FEMX_HAS_ENZYME)
  femx_add_test_if_exists(runEnzymeSmokeTests EnzymeSmokeTests
                          unit/ad/runEnzymeSmokeTests.cpp)

  femx_add_test_if_exists(
    runEnzymeTimeVolumeKernelTests EnzymeTimeVolumeKernelTests
    unit/ad/runEnzymeTimeVolumeKernelTests.cpp)
endif()
