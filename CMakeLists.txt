cmake_minimum_required(VERSION 3.22)

project(
  refem
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(REFEM_ENABLE_RESOLVE "Enable ReSolve linear solver backend" ON)
option(REFEM_ENABLE_HDF5 "Enable HDF5/XDMF output support" ON)
option(REFEM_ENABLE_OPENMP "Enable OpenMP parallel assembly" ON)
option(REFEM_BUILD_APPS "Build refem apps" ON)

if(REFEM_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
endif()

if(REFEM_ENABLE_RESOLVE)
  find_package(ReSolve CONFIG QUIET)

  if(ReSolve_FOUND)
    message(STATUS "Found ReSolve: ${ReSolve_DIR}")
  endif()
endif()

if(REFEM_ENABLE_HDF5)
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
    find_path(REFEM_MPI_INCLUDE_DIR mpi.h
              HINTS /usr/lib/x86_64-linux-gnu/openmpi/include
                    /usr/include/openmpi /usr/local/include)
  endif()
endif()

add_library(refem_config INTERFACE)

target_compile_features(refem_config INTERFACE cxx_std_17)

target_include_directories(
  refem_config INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                         $<INSTALL_INTERFACE:include>)

if(REFEM_ENABLE_RESOLVE)
  target_compile_definitions(refem_config INTERFACE REFEM_ENABLE_RESOLVE=1)
endif()

if(ReSolve_FOUND)
  target_compile_definitions(refem_config INTERFACE REFEM_HAS_RESOLVE=1)
endif()

if(HDF5_FOUND)
  target_compile_definitions(refem_config INTERFACE REFEM_HAS_HDF5=1)
endif()

if(REFEM_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(refem_config INTERFACE OpenMP::OpenMP_CXX)
endif()

add_library(refem INTERFACE)
add_library(refem::refem ALIAS refem)

target_link_libraries(refem INTERFACE refem_config)

add_subdirectory(refem)

if(REFEM_BUILD_APPS)
  add_subdirectory(apps)
endif()
