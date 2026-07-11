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

function(configure_ns_forward_app target)
  target_link_libraries(${target} PRIVATE femx::model::ns femx::femx)

  target_compile_definitions(
    ${target}
    PRIVATE FEMX_NAVIERSTOKES_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}"
            FEMX_GIT_COMMIT="${FEMX_NS_FORWARD_GIT_COMMIT}"
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
  configure_ns_forward_app(ns-forward-resolve)
  target_link_libraries(ns-forward-resolve PRIVATE femx::linalg_resolve)
endif()

if(TARGET femx::linalg_petsc)
  add_executable(ns-forward-petsc main-petsc.cpp)
  configure_ns_forward_app(ns-forward-petsc)
  target_link_libraries(ns-forward-petsc PRIVATE femx::linalg_petsc)
endif()
