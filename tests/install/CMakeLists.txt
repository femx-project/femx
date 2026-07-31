cmake_minimum_required(VERSION 3.22)

project(femx_install_test LANGUAGES CXX)

if(FEMX_INSTALL_TEST_ENABLE_CUDA)
  enable_language(CUDA)
endif()

find_package(femx CONFIG REQUIRED)

foreach(_target femx::femx femx::common femx::ad femx::linalg
                femx::model_navier femx::model::navier)
  if(NOT TARGET "${_target}")
    message(FATAL_ERROR "Expected installed femx target ${_target}")
  endif()
endforeach()

enable_testing()

add_executable(femx_common_install_test common.cpp)
target_link_libraries(femx_common_install_test PRIVATE femx::common)
add_test(NAME femxCommonInstall COMMAND femx_common_install_test)

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

if(FEMX_INSTALL_TEST_ENABLE_CUDA)
  if(FEMX_INSTALL_TEST_EXPECT_CLANG_CUDA AND NOT CMAKE_CUDA_COMPILER_ID MATCHES
                                             "Clang")
    message(
      FATAL_ERROR "CUDA Enzyme install test requires Clang CUDA, but found "
                  "${CMAKE_CUDA_COMPILER_ID}.")
  endif()
  add_executable(femx_cuda_install_test cuda.cu)
  target_link_libraries(femx_cuda_install_test PRIVATE femx::femx)
  set_target_properties(femx_cuda_install_test
                        PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME femxCudaInstall COMMAND femx_cuda_install_test)
endif()
