cmake_minimum_required(VERSION 3.22)

project(refem
  VERSION 0.1.0
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(REFEM_ENABLE_RESOLVE "Enable ReSolve linear solver backend" ON)
option(REFEM_BUILD_EXAMPLES "Build refem examples" OFF)

if(REFEM_ENABLE_RESOLVE)
  find_package(ReSolve CONFIG QUIET)

  if(ReSolve_FOUND)
    message(STATUS "Found ReSolve: ${ReSolve_DIR}")
  endif()
endif()

add_library(refem_config INTERFACE)

target_compile_features(refem_config INTERFACE cxx_std_17)

target_include_directories(refem_config
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

if(REFEM_ENABLE_RESOLVE)
  target_compile_definitions(refem_config INTERFACE REFEM_ENABLE_RESOLVE=1)
endif()

if(ReSolve_FOUND)
  target_compile_definitions(refem_config INTERFACE REFEM_HAS_RESOLVE=1)
endif()

add_library(refem INTERFACE)
add_library(refem::refem ALIAS refem)

target_link_libraries(refem INTERFACE refem_config)

add_subdirectory(refem)

if(REFEM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
