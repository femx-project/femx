get_filename_component(FEMX_APP_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.."
                       ABSOLUTE)

set(FEMX_NAVIER_GIT_COMMIT "unknown")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${FEMX_APP_SOURCE_DIR}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${FEMX_APP_SOURCE_DIR}
    OUTPUT_VARIABLE FEMX_NAVIER_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()

add_library(femx_navier_support)
add_library(femx::apps::navier ALIAS femx_navier_support)

target_sources(
  femx_navier_support
  PRIVATE BoundaryConditions.cpp CommandLine.cpp Config.cpp Monitor.cpp
          NavierStokesProblem.cpp Solve.cpp
  PUBLIC BoundaryConditions.hpp CommandLine.hpp Config.hpp Monitor.hpp
         NavierStokesProblem.hpp Solve.hpp)

target_include_directories(
  femx_navier_support
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${FEMX_NLOHMANN_JSON_INCLUDE_DIR})

target_compile_definitions(
  femx_navier_support
  PUBLIC FEMX_NAVIER_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(
  femx_navier_support
  PUBLIC femx::assembly
         femx::fem
         femx::io
         femx::linalg
         femx::model::ns
         femx::runtime
         femx::state)

function(configure_navier_app target)
  target_link_libraries(${target} PRIVATE femx::apps::navier)

  target_compile_definitions(
    ${target}
    PRIVATE FEMX_GIT_COMMIT="${FEMX_NAVIER_GIT_COMMIT}"
            FEMX_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
            FEMX_CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
            FEMX_CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
            FEMX_ENABLE_HDF5_OPTION="${FEMX_ENABLE_HDF5}"
            FEMX_ENABLE_OPENMP_OPTION="${FEMX_ENABLE_OPENMP}"
            FEMX_ENABLE_RESOLVE_OPTION="${FEMX_ENABLE_RESOLVE}"
            FEMX_ENABLE_PETSC_OPTION="${FEMX_ENABLE_PETSC}"
            FEMX_ENABLE_ENZYME_OPTION="${FEMX_ENABLE_ENZYME}"
            FEMX_NAVIER_APP_NAME="${target}")
endfunction()

if(TARGET femx::linalg_resolve)
  add_executable(navier-resolve main-resolve.cpp)
  configure_navier_app(navier-resolve)
  target_link_libraries(navier-resolve PRIVATE femx::linalg_resolve)
endif()

if(TARGET femx::linalg_petsc)
  add_executable(navier-petsc main-petsc.cpp)
  configure_navier_app(navier-petsc)
  target_link_libraries(navier-petsc PRIVATE femx::linalg_petsc)
endif()
