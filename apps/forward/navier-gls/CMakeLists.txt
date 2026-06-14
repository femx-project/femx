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
            FEMX_ENABLE_PETSC_OPTION="${FEMX_ENABLE_PETSC}")
endfunction()

add_executable(navier-gls main-resolve.cpp Config.cpp BoundaryConditions.cpp
                          RunSupport.cpp)
configure_navier_gls_app(navier-gls)

if(TARGET femx::system_petsc)
  add_executable(navier-gls-petsc main-petsc.cpp Config.cpp
                                  BoundaryConditions.cpp RunSupport.cpp)
  configure_navier_gls_app(navier-gls-petsc)
  target_link_libraries(navier-gls-petsc PRIVATE navier_gls_petsc_kernels)
endif()
