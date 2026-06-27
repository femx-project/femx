cmake_minimum_required(VERSION 3.22)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(FEMX_ENABLE_ENZYME "Enable Enzyme automatic differentiation support" OFF)

project(
  femx
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FEMX_ENABLE_RESOLVE "Enable ReSolve linear solver backend" ON)
option(FEMX_ENABLE_HDF5 "Enable HDF5/XDMF output support" ON)
option(FEMX_ENABLE_OPENMP "Enable OpenMP parallel assembly" ON)
option(FEMX_ENABLE_PETSC "Enable PETSc applications" ON)
option(FEMX_BUILD_APPS "Build femx apps" ON)
option(FEMX_BUILD_EXAMPLES "Build femx examples" ON)
option(FEMX_BUILD_TESTS "Build femx tests" ON)
option(FEMX_REQUIRE_RESOLVE "Fail if ReSolve is enabled but not found" OFF)
set(FEMX_OPENMP_ROOT
    ""
    CACHE PATH "Optional LLVM OpenMP runtime prefix for Clang builds")
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
set(FEMX_ENZYME_FLAGS_TARGET "")
set(FEMX_HAS_PETSC OFF)
set(FEMX_HAS_RESOLVE OFF)
set(FEMX_HAS_RESOLVE_CUDA OFF)

if(FEMX_ENABLE_ENZYME)
  find_package(Enzyme REQUIRED)

  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "FEMX_ENABLE_ENZYME requires a Clang C++ compiler.")
  endif()

  set(_FEMX_ENZYME_LLVM_MAJOR "${Enzyme_LLVM_VERSION_MAJOR}")
  if(NOT _FEMX_ENZYME_LLVM_MAJOR)
    string(REGEX MATCH "^[0-9]+" _FEMX_ENZYME_LLVM_MAJOR
                 "${CMAKE_CXX_COMPILER_VERSION}")
  endif()

  function(femx_import_enzyme_plugin_target target_name)
    if(TARGET "${target_name}")
      get_target_property(_is_imported "${target_name}" IMPORTED)
      if(_is_imported)
        set_target_properties("${target_name}" PROPERTIES IMPORTED_GLOBAL TRUE)
      endif()
      return()
    endif()

    set(_hints)
    if(Enzyme_DIR)
      list(APPEND _hints "${Enzyme_DIR}" "${Enzyme_DIR}/../../..")
    endif()
    if(Enzyme_LLVM_BINARY_DIR)
      list(APPEND _hints "${Enzyme_LLVM_BINARY_DIR}"
           "${Enzyme_LLVM_BINARY_DIR}/..")
    endif()

    string(MAKE_C_IDENTIFIER "${target_name}" _target_id)
    set(_cache_var "FEMX_${_target_id}_PLUGIN")
    find_library(
      ${_cache_var}
      NAMES "${target_name}"
      HINTS ${_hints}
      PATH_SUFFIXES lib lib64)

    if(${_cache_var})
      add_library("${target_name}" MODULE IMPORTED GLOBAL)
      set_target_properties("${target_name}" PROPERTIES IMPORTED_LOCATION
                                                        "${${_cache_var}}")
      if("${target_name}" MATCHES "^ClangEnzyme-")
        set_target_properties("${target_name}"
                              PROPERTIES INTERFACE_COMPILE_DEFINITIONS
                                         ENZYME_RUNPASS)
      endif()
    endif()
  endfunction()

  if(_FEMX_ENZYME_LLVM_MAJOR)
    femx_import_enzyme_plugin_target(
      "ClangEnzyme-${_FEMX_ENZYME_LLVM_MAJOR}")
    femx_import_enzyme_plugin_target("LLDEnzyme-${_FEMX_ENZYME_LLVM_MAJOR}")
  endif()

  foreach(_enzyme_target ClangEnzymeFlags LLDEnzymeFlags
                         "LLVMEnzyme-${_FEMX_ENZYME_LLVM_MAJOR}")
    if(TARGET "${_enzyme_target}")
      get_target_property(_is_imported "${_enzyme_target}" IMPORTED)
      if(_is_imported)
        set_target_properties("${_enzyme_target}" PROPERTIES IMPORTED_GLOBAL
                                                             TRUE)
      endif()
    endif()
  endforeach()

  if(TARGET ClangEnzymeFlags)
    set(FEMX_ENZYME_FLAGS_TARGET ClangEnzymeFlags)
  elseif(TARGET LLDEnzymeFlags)
    set(FEMX_ENZYME_FLAGS_TARGET LLDEnzymeFlags)
  else()
    message(
      FATAL_ERROR
        "Enzyme was found, but no supported Enzyme flags target is available.")
  endif()

  set(FEMX_HAS_ENZYME ON)
  message(STATUS "Found Enzyme: ${Enzyme_DIR} (${FEMX_ENZYME_FLAGS_TARGET})")
endif()

if(FEMX_ENABLE_OPENMP)
  find_package(OpenMP QUIET)

  if(NOT OpenMP_CXX_FOUND AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    string(REGEX MATCH "^[0-9]+" FEMX_CLANG_MAJOR
                 "${CMAKE_CXX_COMPILER_VERSION}")

    set(FEMX_CLANG_OPENMP_ROOT_HINTS)
    if(FEMX_OPENMP_ROOT)
      list(APPEND FEMX_CLANG_OPENMP_ROOT_HINTS "${FEMX_OPENMP_ROOT}")
    endif()
    if(DEFINED ENV{FEMX_OPENMP_ROOT})
      list(APPEND FEMX_CLANG_OPENMP_ROOT_HINTS "$ENV{FEMX_OPENMP_ROOT}")
    endif()
    if(DEFINED ENV{HOME} AND FEMX_CLANG_MAJOR)
      list(APPEND FEMX_CLANG_OPENMP_ROOT_HINTS
           "$ENV{HOME}/opt/libomp-${FEMX_CLANG_MAJOR}")
    endif()
    if(FEMX_CLANG_MAJOR)
      list(
        APPEND FEMX_CLANG_OPENMP_ROOT_HINTS "/usr/lib/llvm-${FEMX_CLANG_MAJOR}"
        "/usr/local/opt/llvm@${FEMX_CLANG_MAJOR}"
        "/opt/homebrew/opt/llvm@${FEMX_CLANG_MAJOR}")
    endif()

    find_path(
      FEMX_CLANG_OPENMP_INCLUDE_DIR
      NAMES omp.h
      HINTS ${FEMX_CLANG_OPENMP_ROOT_HINTS}
      PATH_SUFFIXES
        include lib/clang/${FEMX_CLANG_MAJOR}/include
        lib/llvm-${FEMX_CLANG_MAJOR}/lib/clang/${FEMX_CLANG_MAJOR}/include
        usr/lib/llvm-${FEMX_CLANG_MAJOR}/lib/clang/${FEMX_CLANG_MAJOR}/include)

    find_library(
      FEMX_CLANG_OPENMP_LIBRARY
      NAMES omp libomp
      HINTS ${FEMX_CLANG_OPENMP_ROOT_HINTS}
      PATH_SUFFIXES lib lib64 usr/lib usr/lib/llvm-${FEMX_CLANG_MAJOR}/lib
                    usr/lib/x86_64-linux-gnu)

    if(FEMX_CLANG_OPENMP_INCLUDE_DIR
       AND FEMX_CLANG_OPENMP_LIBRARY
       AND NOT TARGET OpenMP::OpenMP_CXX)
      get_filename_component(FEMX_CLANG_OPENMP_LIBRARY_DIR
                             "${FEMX_CLANG_OPENMP_LIBRARY}" DIRECTORY)
      add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
      set_target_properties(
        OpenMP::OpenMP_CXX
        PROPERTIES
          INTERFACE_COMPILE_OPTIONS "-fopenmp=libomp"
          INTERFACE_INCLUDE_DIRECTORIES "${FEMX_CLANG_OPENMP_INCLUDE_DIR}"
          INTERFACE_LINK_LIBRARIES "${FEMX_CLANG_OPENMP_LIBRARY}"
          INTERFACE_LINK_OPTIONS
          "$<$<PLATFORM_ID:Linux>:-Wl,-rpath,${FEMX_CLANG_OPENMP_LIBRARY_DIR}>")
      set(OpenMP_CXX_FOUND TRUE)
      set(OpenMP_FOUND TRUE)
      message(
        STATUS "Found OpenMP: ${FEMX_CLANG_OPENMP_LIBRARY} (Clang libomp)")
    endif()
  endif()
endif()

if(FEMX_ENABLE_RESOLVE)
  find_package(ReSolve QUIET)

  if(ReSolve_FOUND)
    message(STATUS "Found ReSolve: ${ReSolve_DIR}")
    set(FEMX_HAS_RESOLVE ON)
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
      STATUS
        "FEMX_RESOLVE_BACKEND=CPU selected; the CUDA-capable ReSolve package "
        "will be used with the CPU workspace.")
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
      target_compile_definitions(femx_config INTERFACE FEMX_RESOLVE_USE_CUDA=1)
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
  target_link_libraries(femx_config INTERFACE ${FEMX_ENZYME_FLAGS_TARGET})
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
