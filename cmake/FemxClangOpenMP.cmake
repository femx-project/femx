include_guard(GLOBAL)

function(femx_find_clang_openmp)
  if(TARGET OpenMP::OpenMP_CXX
     OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    return()
  endif()

  string(REGEX MATCH "^[0-9]+" _femx_clang_major
               "${CMAKE_CXX_COMPILER_VERSION}")

  set(_femx_openmp_root_hints)
  if(FEMX_OPENMP_ROOT)
    list(APPEND _femx_openmp_root_hints "${FEMX_OPENMP_ROOT}")
  endif()
  if(DEFINED ENV{FEMX_OPENMP_ROOT})
    list(APPEND _femx_openmp_root_hints "$ENV{FEMX_OPENMP_ROOT}")
  endif()
  if(DEFINED ENV{HOME} AND _femx_clang_major)
    list(APPEND _femx_openmp_root_hints
         "$ENV{HOME}/opt/libomp-${_femx_clang_major}")
  endif()
  if(_femx_clang_major)
    list(
      APPEND _femx_openmp_root_hints "/usr/lib/llvm-${_femx_clang_major}"
      "/usr/local/opt/llvm@${_femx_clang_major}"
      "/opt/homebrew/opt/llvm@${_femx_clang_major}")
  endif()

  find_path(
    FEMX_CLANG_OPENMP_INCLUDE_DIR
    NAMES omp.h
    HINTS ${_femx_openmp_root_hints}
    PATH_SUFFIXES
      include lib/clang/${_femx_clang_major}/include
      lib/llvm-${_femx_clang_major}/lib/clang/${_femx_clang_major}/include
      usr/lib/llvm-${_femx_clang_major}/lib/clang/${_femx_clang_major}/include)

  find_library(
    FEMX_CLANG_OPENMP_LIBRARY
    NAMES omp libomp
    HINTS ${_femx_openmp_root_hints}
    PATH_SUFFIXES lib lib64 usr/lib usr/lib/llvm-${_femx_clang_major}/lib
                  usr/lib/x86_64-linux-gnu)

  if(NOT FEMX_CLANG_OPENMP_INCLUDE_DIR OR NOT FEMX_CLANG_OPENMP_LIBRARY)
    return()
  endif()

  get_filename_component(_femx_openmp_library_dir
                         "${FEMX_CLANG_OPENMP_LIBRARY}" DIRECTORY)
  add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED GLOBAL)
  set_target_properties(
    OpenMP::OpenMP_CXX
    PROPERTIES
      INTERFACE_COMPILE_OPTIONS "-fopenmp=libomp"
      INTERFACE_INCLUDE_DIRECTORIES "${FEMX_CLANG_OPENMP_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${FEMX_CLANG_OPENMP_LIBRARY}"
      INTERFACE_LINK_OPTIONS
      "$<$<PLATFORM_ID:Linux>:-Wl,-rpath,${_femx_openmp_library_dir}>")

  set(OpenMP_CXX_FOUND TRUE PARENT_SCOPE)
  set(OpenMP_FOUND TRUE PARENT_SCOPE)
  message(STATUS
          "Found OpenMP: ${FEMX_CLANG_OPENMP_LIBRARY} (Clang libomp)")
endfunction()
