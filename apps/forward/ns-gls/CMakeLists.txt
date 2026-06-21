add_library(navier_gls_support Config.cpp BoundaryConditions.cpp RunSupport.cpp)

if(FEMX_ENABLE_ENZYME)
  set_target_properties(navier_gls_support PROPERTIES UNITY_BUILD ON)
endif()

target_sources(navier_gls_support PUBLIC Config.hpp BoundaryConditions.hpp
                                         RunSupport.hpp)

target_link_libraries(navier_gls_support PUBLIC navier_common femx::femx)

target_include_directories(
  navier_gls_support
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

set(FEMX_NAVIER_GLS_GIT_COMMIT "unknown")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE FEMX_NAVIER_GLS_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()

function(configure_navier_gls_app target)
  target_link_libraries(${target} PRIVATE navier_gls_support femx::femx)

  target_include_directories(
    ${target}
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

  target_compile_definitions(
    ${target}
    PRIVATE FEMX_NAVIERSTOKES_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}"
            FEMX_GIT_COMMIT="${FEMX_NAVIER_GLS_GIT_COMMIT}"
            FEMX_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
            FEMX_CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
            FEMX_CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
            FEMX_ENABLE_HDF5_OPTION="${FEMX_ENABLE_HDF5}"
            FEMX_ENABLE_OPENMP_OPTION="${FEMX_ENABLE_OPENMP}"
            FEMX_ENABLE_RESOLVE_OPTION="${FEMX_ENABLE_RESOLVE}"
            FEMX_ENABLE_PETSC_OPTION="${FEMX_ENABLE_PETSC}"
            FEMX_ENABLE_ENZYME_OPTION="${FEMX_ENABLE_ENZYME}"
            FEMX_NAVIER_GLS_APP_NAME="${target}")
endfunction()

if(FEMX_HAS_RESOLVE AND FEMX_HAS_ENZYME)
  add_executable(ns-gls main-resolve.cpp)
  configure_navier_gls_app(ns-gls)
  target_link_libraries(ns-gls PRIVATE femx::linalg_resolve)
endif()

if(TARGET femx::linalg_petsc AND FEMX_HAS_ENZYME)
  add_executable(ns-gls-petsc main-petsc.cpp)
  configure_navier_gls_app(ns-gls-petsc)
  target_link_libraries(ns-gls-petsc PRIVATE femx::linalg_petsc)

  if(NOT TARGET ns-gls)
    add_executable(ns-gls main-petsc.cpp)
    configure_navier_gls_app(ns-gls)
    target_link_libraries(ns-gls PRIVATE femx::linalg_petsc)
  endif()
endif()
