cmake_minimum_required(VERSION 3.22)

project(femx_install_test LANGUAGES CXX)

find_package(femx CONFIG REQUIRED)

foreach(_target femx::femx femx::ad femx::linalg femx::model_ns
                femx::model::ns)
  if(NOT TARGET "${_target}")
    message(FATAL_ERROR "Expected installed femx target ${_target}")
  endif()
endforeach()

enable_testing()

add_executable(femx_install_test main.cpp)
target_link_libraries(femx_install_test PRIVATE femx::femx)
add_test(NAME femxInstall COMMAND femx_install_test)

add_executable(femx_linalg_install_test main.cpp)
target_link_libraries(femx_linalg_install_test PRIVATE femx::linalg)
add_test(NAME femxLinalgInstall COMMAND femx_linalg_install_test)

if(TARGET femx::linalg_resolve)
  add_executable(femx_resolve_install_test optional-resolve.cpp)
  target_link_libraries(femx_resolve_install_test PRIVATE femx::linalg_resolve)
  add_test(NAME femxReSolveInstall COMMAND femx_resolve_install_test)
endif()

if(TARGET femx::linalg_petsc)
  add_executable(femx_petsc_install_test optional-petsc.cpp)
  target_link_libraries(femx_petsc_install_test PRIVATE femx::linalg_petsc)
  add_test(NAME femxPETScInstall COMMAND femx_petsc_install_test)
endif()
