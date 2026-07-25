get_filename_component(FEMX_APP_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.."
                       ABSOLUTE)

set(FEMX_NS_FORWARD_GIT_COMMIT "unknown")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${FEMX_APP_SOURCE_DIR}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${FEMX_APP_SOURCE_DIR}
    OUTPUT_VARIABLE FEMX_NS_FORWARD_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()

add_library(femx_ns_forward_support)
add_library(femx::apps::ns_forward ALIAS femx_ns_forward_support)

target_sources(
  femx_ns_forward_support
  PRIVATE BoundaryConditions.cpp
          CommandLine.cpp
          Config.cpp
          Monitor.cpp
          Problem.cpp
          Solve.cpp
  PUBLIC BoundaryConditions.hpp
         CommandLine.hpp
         Config.hpp
         Monitor.hpp
         Problem.hpp
         Solve.hpp)

target_include_directories(
  femx_ns_forward_support
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${FEMX_NLOHMANN_JSON_INCLUDE_DIR})

target_compile_definitions(
  femx_ns_forward_support
  PUBLIC FEMX_NS_FORWARD_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(
  femx_ns_forward_support
  PUBLIC femx::assembly
         femx::fem
         femx::io
         femx::linalg
         femx::model::ns
         femx::runtime
         femx::state)

function(configure_ns_fwd_app target)
  target_link_libraries(${target} PRIVATE femx::apps::ns_forward)

  target_compile_definitions(
    ${target}
    PRIVATE FEMX_GIT_COMMIT="${FEMX_NS_FORWARD_GIT_COMMIT}"
            FEMX_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
            FEMX_CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
            FEMX_CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
            FEMX_ENABLE_HDF5_OPTION="${FEMX_ENABLE_HDF5}"
            FEMX_ENABLE_OPENMP_OPTION="${FEMX_ENABLE_OPENMP}"
            FEMX_ENABLE_RESOLVE_OPTION="${FEMX_ENABLE_RESOLVE}"
            FEMX_ENABLE_PETSC_OPTION="${FEMX_ENABLE_PETSC}"
            FEMX_ENABLE_ENZYME_OPTION="${FEMX_ENABLE_ENZYME}"
            FEMX_NS_FORWARD_APP_NAME="${target}")
endfunction()

if(TARGET femx::linalg_resolve)
  add_executable(ns-forward-resolve main-resolve.cpp)
  configure_ns_fwd_app(ns-forward-resolve)
  target_link_libraries(ns-forward-resolve PRIVATE femx::linalg_resolve)
endif()

if(TARGET femx::linalg_petsc)
  add_executable(ns-forward-petsc main-petsc.cpp)
  configure_ns_fwd_app(ns-forward-petsc)
  target_link_libraries(ns-forward-petsc PRIVATE femx::linalg_petsc)
endif()
