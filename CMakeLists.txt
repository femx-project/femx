cmake_minimum_required(VERSION 3.22)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(FEMX_ENABLE_ENZYME "Enable Enzyme automatic differentiation support" OFF)
option(FEMX_ENABLE_CUDA "Enable CUDA memory and assembly backends" OFF)

project(
  femx
  VERSION 0.4.0
  LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FEMX_ENABLE_RESOLVE "Require ReSolve linear solver backend" OFF)
option(FEMX_ENABLE_HDF5 "Require HDF5 output support" OFF)
option(FEMX_ENABLE_OPENMP "Require OpenMP parallel assembly" OFF)
option(FEMX_ENABLE_PETSC "Require PETSc applications and backends" OFF)
option(FEMX_BUILD_APPS "Build femx apps" ON)
option(FEMX_BUILD_EXAMPLES "Build femx examples" ON)
option(FEMX_BUILD_TESTS "Build femx tests" ON)
option(FEMX_BUILD_PYTHON "Build supported Python bindings" OFF)
option(FEMX_BUILD_PYTHON_EXPERIMENTAL "Build experimental Python bindings" OFF)
set(FEMX_OPENMP_ROOT
    ""
    CACHE PATH "Optional LLVM OpenMP runtime prefix for Clang builds")
set(PETSC_DIR
    ""
    CACHE PATH "Optional PETSc installation root directory")
set(PETSC_ARCH
    ""
    CACHE STRING "Optional PETSc architecture name")

set(FEMX_NLOHMANN_JSON_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third-party/nlohmann-json/single_include")
if(NOT EXISTS "${FEMX_NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
  message(
    FATAL_ERROR
      "Missing nlohmann-json submodule. Run "
      "`git submodule update --init --recursive` from the femx source tree.")
endif()

set(FEMX_RESOLVE_BACKEND
    "AUTO"
    CACHE STRING "Requested ReSolve backend: AUTO, CPU, or CUDA")
set_property(CACHE FEMX_RESOLVE_BACKEND PROPERTY STRINGS AUTO CPU CUDA)
string(TOUPPER "${FEMX_RESOLVE_BACKEND}" FEMX_RESOLVE_BACKEND_MODE)

set(FEMX_RESOLVE_REVISION
    ""
    CACHE STRING
          "Optional revision label for the installed ReSolve development tree")

if(NOT FEMX_RESOLVE_BACKEND_MODE MATCHES "^(AUTO|CPU|CUDA)$")
  message(FATAL_ERROR "FEMX_RESOLVE_BACKEND must be AUTO, CPU, or CUDA.")
endif()

if(FEMX_ENABLE_RESOLVE
   AND FEMX_RESOLVE_BACKEND_MODE STREQUAL "CUDA"
   AND NOT FEMX_ENABLE_CUDA)
  set(FEMX_ENABLE_CUDA
      ON
      CACHE BOOL "Enable CUDA memory and assembly backends" FORCE)
endif()

set(FEMX_HAS_ENZYME OFF)
set(FEMX_HAS_CUDA OFF)
set(FEMX_ENZYME_FLAGS_TARGET "")
set(FEMX_HDF5_USES_MPI OFF)
set(FEMX_HAS_PETSC OFF)
set(FEMX_HAS_RESOLVE OFF)
set(FEMX_HAS_RESOLVE_CUDA OFF)

if(FEMX_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "FEMX_ENABLE_CUDA=ON but no CUDA compiler was found.")
  endif()
  enable_language(CUDA)
  if(FEMX_ENABLE_ENZYME AND NOT CMAKE_CUDA_COMPILER_ID MATCHES "Clang")
    message(
      FATAL_ERROR
        "CUDA Enzyme requires Clang as CMAKE_CUDA_COMPILER because Enzyme "
        "differentiates LLVM IR; nvcc cannot provide that IR.")
  endif()
  find_package(CUDAToolkit REQUIRED)
  set(FEMX_HAS_CUDA ON)
  message(STATUS "CUDA execution backend: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

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
        set_target_properties(
          "${target_name}" PROPERTIES INTERFACE_COMPILE_DEFINITIONS
                                      ENZYME_RUNPASS)
      endif()
    endif()
  endfunction()

  if(_FEMX_ENZYME_LLVM_MAJOR)
    femx_import_enzyme_plugin_target("ClangEnzyme-${_FEMX_ENZYME_LLVM_MAJOR}")
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

  if(NOT OpenMP_CXX_FOUND)
    message(
      FATAL_ERROR
        "FEMX_ENABLE_OPENMP=ON but OpenMP was not found. Install OpenMP, "
        "pass -DFEMX_OPENMP_ROOT=/path/to/libomp-prefix for Clang builds, "
        "or configure with -DFEMX_ENABLE_OPENMP=OFF.")
  endif()
endif()

if(FEMX_ENABLE_RESOLVE)
  if(FEMX_HAS_ENZYME AND FEMX_HAS_CUDA)
    # The current ReSolveConfig forces nvcc in its consumer. Importing its
    # exported targets directly keeps the Clang CUDA toolchain required by
    # Enzyme while using the same installed libraries.
    find_path(
      ReSolve_DIR
      NAMES ReSolveTargets.cmake
      HINTS "${ReSolve_ROOT}" "$ENV{ReSolve_ROOT}"
      PATH_SUFFIXES share/resolve/cmake lib/cmake/ReSolve
      DOC "ReSolve development package directory")
    if(NOT EXISTS "${ReSolve_DIR}/ReSolveTargets.cmake")
      message(
        FATAL_ERROR
          "FEMX_ENABLE_RESOLVE=ON requires a ReSolve package built from the "
          "develop branch. Set ReSolve_DIR or add its installation prefix to "
          "CMAKE_PREFIX_PATH.")
    endif()
    include("${ReSolve_DIR}/ReSolveTargets.cmake")
    set(ReSolve_FOUND TRUE)
  else()
    find_package(ReSolve CONFIG QUIET)
  endif()

  if(NOT ReSolve_FOUND)
    message(
      FATAL_ERROR
        "FEMX_ENABLE_RESOLVE=ON requires a ReSolve package built from the "
        "develop branch. Set ReSolve_DIR or add its installation prefix to "
        "CMAKE_PREFIX_PATH.")
  endif()

  if(FEMX_RESOLVE_REVISION)
    set(FEMX_RESOLVE_BUILD_ID "${FEMX_RESOLVE_REVISION}")
  else()
    get_target_property(FEMX_RESOLVE_LIBRARY ReSolve::ReSolve IMPORTED_LOCATION)
    if(NOT FEMX_RESOLVE_LIBRARY)
      get_target_property(FEMX_RESOLVE_LIBRARY ReSolve::ReSolve
                          IMPORTED_LOCATION_NOCONFIG)
    endif()
    if(FEMX_RESOLVE_LIBRARY AND EXISTS "${FEMX_RESOLVE_LIBRARY}")
      file(SHA256 "${FEMX_RESOLVE_LIBRARY}" FEMX_RESOLVE_LIBRARY_SHA256)
      string(SUBSTRING "${FEMX_RESOLVE_LIBRARY_SHA256}" 0 16
                       FEMX_RESOLVE_LIBRARY_SHA256_SHORT)
      set(FEMX_RESOLVE_BUILD_ID
          "libReSolve-sha256:${FEMX_RESOLVE_LIBRARY_SHA256_SHORT}")
    else()
      set(FEMX_RESOLVE_BUILD_ID "develop")
    endif()
  endif()

  message(STATUS "Found ReSolve development package: ${ReSolve_DIR}")
  message(STATUS "ReSolve development build id: ${FEMX_RESOLVE_BUILD_ID}")
  set(FEMX_HAS_RESOLVE ON)
  if(TARGET ReSolve::CUDA OR TARGET ReSolve::resolve_backend_cuda)
    set(FEMX_HAS_RESOLVE_CUDA ON)
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
  enable_language(C)
  if(NOT HDF5_ROOT AND DEFINED ENV{CONDA_PREFIX})
    set(HDF5_ROOT "$ENV{CONDA_PREFIX}")
  endif()
  find_package(
    HDF5
    COMPONENTS C
    QUIET)

  if(NOT HDF5_FOUND AND NOT DEFINED HDF5_PREFER_PARALLEL)
    set(HDF5_PREFER_PARALLEL TRUE)
    unset(HDF5_C_COMPILER_EXECUTABLE CACHE)
    unset(HDF5_C_INCLUDE_DIR CACHE)
    unset(HDF5_hdf5_LIBRARY_DEBUG CACHE)
    unset(HDF5_hdf5_LIBRARY_RELEASE CACHE)
    find_package(
      HDF5
      COMPONENTS C
      QUIET)
  endif()

  if(NOT HDF5_FOUND)
    message(
      FATAL_ERROR
        "FEMX_ENABLE_HDF5=ON but HDF5 was not found. Install HDF5 "
        "development files, pass -DHDF5_ROOT=/path/to/hdf5, or configure "
        "with -DFEMX_ENABLE_HDF5=OFF. If using conda HDF5, make sure the "
        "conda compiler wrappers are installed or pass HDF5_C_INCLUDE_DIR "
        "and HDF5_hdf5_LIBRARY_RELEASE explicitly.")
  endif()

  message(STATUS "Found HDF5: ${HDF5_VERSION}")
  if(HDF5_IS_PARALLEL)
    find_package(
      MPI
      COMPONENTS CXX
      QUIET)
    if(MPI_CXX_FOUND)
      set(FEMX_HDF5_USES_MPI ON)
    endif()
  endif()
  find_path(FEMX_MPI_INCLUDE_DIR mpi.h
            HINTS /usr/lib/x86_64-linux-gnu/openmpi/include
                  /usr/include/openmpi /usr/local/include)
endif()

if(FEMX_ENABLE_PETSC)
  find_package(PkgConfig QUIET)
  find_package(
    MPI
    COMPONENTS CXX
    QUIET)

  if(PkgConfig_FOUND)
    set(_FEMX_PETSC_PKG_CONFIG_PATH_SAVE "$ENV{PKG_CONFIG_PATH}")
    set(_FEMX_PETSC_DIR_HINTS)
    set(_FEMX_PETSC_ARCH_HINTS)
    set(_FEMX_PETSC_PKG_CONFIG_DIRS)

    if(PETSC_DIR)
      list(APPEND _FEMX_PETSC_DIR_HINTS "${PETSC_DIR}")
    endif()
    if(DEFINED ENV{PETSC_DIR} AND NOT "$ENV{PETSC_DIR}" STREQUAL "")
      list(APPEND _FEMX_PETSC_DIR_HINTS "$ENV{PETSC_DIR}")
    endif()
    if(PETSC_ARCH)
      list(APPEND _FEMX_PETSC_ARCH_HINTS "${PETSC_ARCH}")
    endif()
    if(DEFINED ENV{PETSC_ARCH} AND NOT "$ENV{PETSC_ARCH}" STREQUAL "")
      list(APPEND _FEMX_PETSC_ARCH_HINTS "$ENV{PETSC_ARCH}")
    endif()

    foreach(_petsc_dir IN LISTS _FEMX_PETSC_DIR_HINTS)
      list(APPEND _FEMX_PETSC_PKG_CONFIG_DIRS "${_petsc_dir}/lib/pkgconfig")
      foreach(_petsc_arch IN LISTS _FEMX_PETSC_ARCH_HINTS)
        list(APPEND _FEMX_PETSC_PKG_CONFIG_DIRS
             "${_petsc_dir}/${_petsc_arch}/lib/pkgconfig")
      endforeach()
    endforeach()

    if(_FEMX_PETSC_PKG_CONFIG_DIRS)
      list(REMOVE_DUPLICATES _FEMX_PETSC_PKG_CONFIG_DIRS)
      if(WIN32)
        set(_FEMX_PKG_CONFIG_PATH_SEPARATOR ";")
      else()
        set(_FEMX_PKG_CONFIG_PATH_SEPARATOR ":")
      endif()
      list(JOIN _FEMX_PETSC_PKG_CONFIG_DIRS
           "${_FEMX_PKG_CONFIG_PATH_SEPARATOR}" _FEMX_PETSC_PKG_CONFIG_PATH)
      if(DEFINED ENV{PKG_CONFIG_PATH} AND NOT "$ENV{PKG_CONFIG_PATH}" STREQUAL
                                          "")
        set(ENV{PKG_CONFIG_PATH}
            "${_FEMX_PETSC_PKG_CONFIG_PATH}${_FEMX_PKG_CONFIG_PATH_SEPARATOR}$ENV{PKG_CONFIG_PATH}"
        )
      else()
        set(ENV{PKG_CONFIG_PATH} "${_FEMX_PETSC_PKG_CONFIG_PATH}")
      endif()
    endif()

    pkg_check_modules(PETSC QUIET IMPORTED_TARGET PETSc)
    if(NOT PETSC_FOUND)
      pkg_check_modules(PETSC QUIET IMPORTED_TARGET petsc)
    endif()
    set(ENV{PKG_CONFIG_PATH} "${_FEMX_PETSC_PKG_CONFIG_PATH_SAVE}")
  endif()

  if(NOT PkgConfig_FOUND)
    message(
      FATAL_ERROR
        "FEMX_ENABLE_PETSC=ON but pkg-config was not found. Install "
        "pkg-config and PETSc, or configure with -DFEMX_ENABLE_PETSC=OFF.")
  endif()

  if(NOT PETSC_FOUND)
    message(
      FATAL_ERROR
        "FEMX_ENABLE_PETSC=ON but PETSc was not found. Install PETSc "
        "development files; set PETSC_DIR, PETSC_ARCH, or PKG_CONFIG_PATH "
        "so pkg-config can find PETSc; or configure with "
        "-DFEMX_ENABLE_PETSC=OFF.")
  endif()

  if(NOT MPI_CXX_FOUND)
    message(
      FATAL_ERROR
        "FEMX_ENABLE_PETSC=ON but MPI C++ support was not found. Install "
        "MPI development files, or configure with -DFEMX_ENABLE_PETSC=OFF.")
  endif()

  set(FEMX_HAS_PETSC ON)
  message(STATUS "Found PETSc: ${PETSC_VERSION}")
endif()

add_library(femx_config INTERFACE)

target_compile_features(femx_config INTERFACE cxx_std_17)

target_include_directories(
  femx_config INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                        $<INSTALL_INTERFACE:include>)

if(FEMX_HAS_CUDA)
  target_compile_definitions(femx_config INTERFACE FEMX_HAS_CUDA=1)
endif()

if(FEMX_ENABLE_RESOLVE)
  target_compile_definitions(femx_config INTERFACE FEMX_ENABLE_RESOLVE=1)
endif()

if(ReSolve_FOUND)
  target_compile_definitions(femx_config INTERFACE FEMX_HAS_RESOLVE=1)
  target_compile_definitions(
    femx_config INTERFACE FEMX_RESOLVE_BUILD_ID="${FEMX_RESOLVE_BUILD_ID}")
  if(FEMX_HAS_RESOLVE_CUDA AND FEMX_HAS_CUDA)
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

if(FEMX_BUILD_PYTHON)
  add_subdirectory(python/bindings)
endif()

if(FEMX_BUILD_PYTHON_EXPERIMENTAL)
  add_subdirectory(python/experimental)
endif()

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

function(femx_export_name target export_name)
  if(TARGET "${target}")
    set_target_properties("${target}" PROPERTIES EXPORT_NAME "${export_name}")
    set_property(TARGET "${target}" PROPERTY INTERFACE_SOURCES)
  endif()
endfunction()

femx_export_name(femx femx)
femx_export_name(femx_config config)
femx_export_name(femx_common common)
femx_export_name(femx_linalg linalg)
femx_export_name(femx_ad ad)
femx_export_name(femx_state state)
femx_export_name(femx_inverse inverse)
femx_export_name(femx_fem fem)
femx_export_name(femx_assembly assembly)
femx_export_name(femx_io io)
femx_export_name(femx_runtime runtime)
femx_export_name(femx_model_ns model_ns)
femx_export_name(femx_linalg_resolve linalg_resolve)
femx_export_name(femx_linalg_petsc linalg_petsc)
femx_export_name(femx_opt opt)

set(FEMX_INSTALL_TARGETS
    femx
    femx_config
    femx_common
    femx_linalg
    femx_ad
    femx_state
    femx_inverse
    femx_fem
    femx_assembly
    femx_io
    femx_runtime
    femx_model_ns)

foreach(_femx_optional_target femx_linalg_resolve femx_linalg_petsc femx_opt)
  if(TARGET "${_femx_optional_target}")
    list(APPEND FEMX_INSTALL_TARGETS "${_femx_optional_target}")
  endif()
endforeach()

install(
  TARGETS ${FEMX_INSTALL_TARGETS}
  EXPORT femxTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  DIRECTORY femx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp")

set(FEMX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/femx")

configure_package_config_file(
  cmake/femxConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/femxConfig.cmake"
  INSTALL_DESTINATION "${FEMX_INSTALL_CMAKEDIR}")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/femxConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(
  EXPORT femxTargets
  NAMESPACE femx::
  DESTINATION "${FEMX_INSTALL_CMAKEDIR}")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/femxConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/femxConfigVersion.cmake"
        DESTINATION "${FEMX_INSTALL_CMAKEDIR}")

install(FILES cmake/FindEnzyme.cmake cmake/FindReSolve.cmake
        DESTINATION "${FEMX_INSTALL_CMAKEDIR}")

set(FEMX_INSTALL_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/install")
set(FEMX_INSTALL_TEST_PREFIX "${FEMX_INSTALL_TEST_DIR}/prefix")
set(FEMX_INSTALL_TEST_BUILD_DIR "${FEMX_INSTALL_TEST_DIR}/consumer-build")
set(FEMX_INSTALL_TEST_CONFIGURE_ARGS
    -DCMAKE_PREFIX_PATH="${FEMX_INSTALL_TEST_PREFIX}")

if(FEMX_HAS_RESOLVE AND ReSolve_DIR)
  list(APPEND FEMX_INSTALL_TEST_CONFIGURE_ARGS -DReSolve_DIR="${ReSolve_DIR}")
endif()

if(FEMX_HAS_ENZYME AND Enzyme_DIR)
  list(APPEND FEMX_INSTALL_TEST_CONFIGURE_ARGS -DEnzyme_DIR="${Enzyme_DIR}")
endif()

add_custom_target(
  test_install
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${FEMX_INSTALL_TEST_DIR}"
  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --prefix
          "${FEMX_INSTALL_TEST_PREFIX}"
  COMMAND ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/install" -B
          "${FEMX_INSTALL_TEST_BUILD_DIR}" ${FEMX_INSTALL_TEST_CONFIGURE_ARGS}
  COMMAND ${CMAKE_COMMAND} -E env MAKEFLAGS=--no-print-directory
          ${CMAKE_COMMAND} --build "${FEMX_INSTALL_TEST_BUILD_DIR}"
  COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${FEMX_INSTALL_TEST_BUILD_DIR}"
          --output-on-failure
  COMMENT "Install femx and run an external CMake consumer test"
  USES_TERMINAL)

add_dependencies(test_install ${FEMX_INSTALL_TARGETS})
