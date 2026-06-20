add_library(navier_gls_kernels Assembly.cpp Components.cpp)

target_link_libraries(navier_gls_kernels PUBLIC femx::femx)

target_include_directories(navier_gls_kernels
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(TARGET femx::system_petsc)
  add_library(navier_gls_petsc_kernels AssemblyPETSc.cpp)
  target_link_libraries(navier_gls_petsc_kernels PUBLIC navier_gls_kernels
                                                        femx::system_petsc)
  target_include_directories(navier_gls_petsc_kernels
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

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
  target_link_libraries(${target} PRIVATE navier_gls_kernels femx::femx)

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
            FEMX_NAVIER_GLS_APP_NAME="${target}")
endfunction()

set(FEMX_NAVIER_GLS_COMMON_SOURCES Config.cpp BoundaryConditions.cpp
                                   RunSupport.cpp)

if(FEMX_HAS_RESOLVE)
  add_executable(ns-gls main-resolve.cpp ${FEMX_NAVIER_GLS_COMMON_SOURCES})
  configure_navier_gls_app(ns-gls)
endif()

if(TARGET femx::system_petsc)
  add_executable(ns-gls-petsc main-petsc.cpp ${FEMX_NAVIER_GLS_COMMON_SOURCES})
  configure_navier_gls_app(ns-gls-petsc)
  target_link_libraries(ns-gls-petsc PRIVATE navier_gls_petsc_kernels)

  if(NOT TARGET ns-gls)
    add_executable(ns-gls main-petsc.cpp ${FEMX_NAVIER_GLS_COMMON_SOURCES})
    configure_navier_gls_app(ns-gls)
    target_link_libraries(ns-gls PRIVATE navier_gls_petsc_kernels)
  endif()
endif()
