cmake_minimum_required(VERSION 3.22)

option(FEMX_ENABLE_ENZYME "Enable Enzyme automatic differentiation support" OFF)

if(FEMX_ENABLE_ENZYME)
  find_package(Enzyme REQUIRED CONFIG)

  if(Enzyme_LLVM_BINARY_DIR AND NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang++")
  endif()
endif()

project(
  femx
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(FEMX_ENABLE_RESOLVE "Enable ReSolve linear solver backend" ON)
option(FEMX_ENABLE_HDF5 "Enable HDF5/XDMF output support" ON)
option(FEMX_ENABLE_OPENMP "Enable OpenMP parallel assembly" ON)
option(FEMX_ENABLE_PETSC "Enable PETSc applications" ON)
option(FEMX_BUILD_APPS "Build femx apps" ON)
option(FEMX_BUILD_EXAMPLES "Build femx examples" ON)
option(FEMX_BUILD_TESTS "Build femx tests" ON)
option(FEMX_REQUIRE_RESOLVE "Fail if ReSolve is enabled but not found" OFF)
option(RESOLVE_USE_CUDA "Request a CUDA-enabled ReSolve package" OFF)

set(FEMX_RESOLVE_BACKEND
    "AUTO"
    CACHE STRING "Requested ReSolve backend: AUTO, CPU, or CUDA")
set_property(CACHE FEMX_RESOLVE_BACKEND PROPERTY STRINGS AUTO CPU CUDA)
string(TOUPPER "${FEMX_RESOLVE_BACKEND}" FEMX_RESOLVE_BACKEND_MODE)

if(NOT FEMX_RESOLVE_BACKEND_MODE MATCHES "^(AUTO|CPU|CUDA)$")
  message(FATAL_ERROR "FEMX_RESOLVE_BACKEND must be AUTO, CPU, or CUDA.")
endif()

if(RESOLVE_USE_CUDA AND FEMX_RESOLVE_BACKEND_MODE STREQUAL "CPU")
  message(
    FATAL_ERROR "RESOLVE_USE_CUDA=ON conflicts with FEMX_RESOLVE_BACKEND=CPU.")
endif()

if(RESOLVE_USE_CUDA)
  set(FEMX_RESOLVE_BACKEND_MODE "CUDA")
endif()

set(FEMX_HAS_ENZYME OFF)
set(FEMX_HAS_PETSC OFF)
set(FEMX_HAS_RESOLVE_CUDA OFF)

if(FEMX_ENABLE_ENZYME)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "FEMX_ENABLE_ENZYME requires a Clang C++ compiler.")
  endif()

  if(NOT TARGET LLDEnzymeFlags)
    message(
      FATAL_ERROR
        "Enzyme was found, but the LLDEnzymeFlags target is unavailable.")
  endif()

  set(FEMX_HAS_ENZYME ON)
  message(STATUS "Found Enzyme: ${Enzyme_DIR}")
endif()

if(FEMX_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
endif()

if(FEMX_ENABLE_RESOLVE)
  find_package(ReSolve QUIET)

  if(ReSolve_FOUND)
    message(STATUS "Found ReSolve: ${ReSolve_DIR}")
    if(TARGET ReSolve::CUDA OR TARGET ReSolve::resolve_backend_cuda)
      set(FEMX_HAS_RESOLVE_CUDA ON)
    endif()
  elseif(FEMX_REQUIRE_RESOLVE)
    message(
      FATAL_ERROR
        "FEMX_REQUIRE_RESOLVE=ON, but ReSolve was not found. "
        "Set ReSolve_ROOT, RESOLVE_ROOT, CMAKE_PREFIX_PATH, or "
        "ReSolve_DIR to a ReSolve installation.")
  endif()
endif()

if(ReSolve_FOUND)
  if(FEMX_RESOLVE_BACKEND_MODE STREQUAL "CUDA" AND NOT FEMX_HAS_RESOLVE_CUDA)
    message(
      FATAL_ERROR "FEMX_RESOLVE_BACKEND=CUDA requires a CUDA-enabled ReSolve "
                  "package, but the found package has no CUDA target.")
  endif()

  if(FEMX_RESOLVE_BACKEND_MODE STREQUAL "CPU" AND FEMX_HAS_RESOLVE_CUDA)
    message(
      FATAL_ERROR
        "FEMX_RESOLVE_BACKEND=CPU requires a CPU-only ReSolve package, "
        "but the found package exposes CUDA. Use the resolve-cuda preset "
        "or set ReSolve_ROOT to a CPU-only ReSolve installation.")
  endif()
endif()

if(FEMX_ENABLE_HDF5)
  find_package(
    HDF5
    COMPONENTS C
    QUIET)

  if(HDF5_FOUND)
    message(STATUS "Found HDF5: ${HDF5_VERSION}")
    if(HDF5_IS_PARALLEL)
      find_package(
        MPI
        COMPONENTS CXX
        QUIET)
    endif()
    find_path(FEMX_MPI_INCLUDE_DIR mpi.h
              HINTS /usr/lib/x86_64-linux-gnu/openmpi/include
                    /usr/include/openmpi /usr/local/include)
  endif()
endif()

if(FEMX_ENABLE_PETSC)
  find_package(PkgConfig QUIET)
  find_package(
    MPI
    COMPONENTS CXX
    QUIET)

  if(PkgConfig_FOUND)
    pkg_check_modules(PETSC QUIET IMPORTED_TARGET PETSc)
    if(NOT PETSC_FOUND)
      pkg_check_modules(PETSC QUIET IMPORTED_TARGET petsc)
    endif()
  endif()

  if(PETSC_FOUND AND MPI_CXX_FOUND)
    set(FEMX_HAS_PETSC ON)
    message(STATUS "Found PETSc: ${PETSC_VERSION}")
  elseif(FEMX_ENABLE_PETSC)
    message(
      STATUS
        "PETSc application support requested but PETSc or MPI was not found")
  endif()
endif()

add_library(femx_config INTERFACE)

target_compile_features(femx_config INTERFACE cxx_std_17)

target_include_directories(
  femx_config INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                        $<INSTALL_INTERFACE:include>)

if(FEMX_ENABLE_RESOLVE)
  target_compile_definitions(femx_config INTERFACE FEMX_ENABLE_RESOLVE=1)
endif()

if(ReSolve_FOUND)
  target_compile_definitions(femx_config INTERFACE FEMX_HAS_RESOLVE=1)
  if(FEMX_HAS_RESOLVE_CUDA)
    if(NOT FEMX_RESOLVE_BACKEND_MODE STREQUAL "CPU")
      target_compile_definitions(femx_config INTERFACE RESOLVE_USE_CUDA=1)
    endif()
  endif()
endif()

if(HDF5_FOUND)
  target_compile_definitions(femx_config INTERFACE FEMX_HAS_HDF5=1)
endif()

if(FEMX_HAS_PETSC)
  target_compile_definitions(femx_config INTERFACE FEMX_HAS_PETSC=1)
endif()

if(FEMX_HAS_ENZYME)
  target_compile_definitions(femx_config INTERFACE FEMX_HAS_ENZYME=1)
  target_link_libraries(femx_config INTERFACE LLDEnzymeFlags)
endif()

if(FEMX_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(femx_config INTERFACE OpenMP::OpenMP_CXX)
endif()

add_library(femx INTERFACE)
add_library(femx::femx ALIAS femx)

target_link_libraries(femx INTERFACE femx_config)

add_subdirectory(femx)

if(FEMX_BUILD_APPS)
  add_subdirectory(apps)
endif()

if(FEMX_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(FEMX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
