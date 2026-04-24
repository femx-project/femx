add_executable(navier-gls main.cpp Config.cpp BoundaryConditions.cpp
                          Assembly.cpp Components.cpp)

set(REFEM_NAVIER_GLS_GIT_COMMIT "unknown")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE REFEM_NAVIER_GLS_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()

target_link_libraries(navier-gls PRIVATE refem::refem)

target_include_directories(
  navier-gls
  PRIVATE ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

target_compile_definitions(
  navier-gls
  PRIVATE REFEM_NAVIERSTOKES_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}"
          REFEM_GIT_COMMIT="${REFEM_NAVIER_GLS_GIT_COMMIT}"
          REFEM_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
          REFEM_CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
          REFEM_CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
          REFEM_ENABLE_HDF5_OPTION="${REFEM_ENABLE_HDF5}"
          REFEM_ENABLE_OPENMP_OPTION="${REFEM_ENABLE_OPENMP}"
          REFEM_ENABLE_RESOLVE_OPTION="${REFEM_ENABLE_RESOLVE}")
