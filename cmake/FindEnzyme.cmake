include_guard(GLOBAL)

set(_Enzyme_HINTS)

foreach(_var Enzyme_ROOT ENZYME_ROOT LLVM_ROOT)
  if(DEFINED ${_var})
    list(APPEND _Enzyme_HINTS "${${_var}}")
  endif()

  if(DEFINED ENV{${_var}})
    list(APPEND _Enzyme_HINTS "$ENV{${_var}}")
  endif()
endforeach()

if(DEFINED Enzyme_DIR)
  list(APPEND _Enzyme_HINTS "${Enzyme_DIR}")
endif()

if(DEFINED LLVM_DIR)
  list(APPEND _Enzyme_HINTS "${LLVM_DIR}")
endif()

execute_process(
  COMMAND llvm-config --bindir
  OUTPUT_VARIABLE _Enzyme_LLVM_BINDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)

execute_process(
  COMMAND llvm-config --libdir
  OUTPUT_VARIABLE _Enzyme_LLVM_LIBDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)

find_program(
  Enzyme_CLANGXX
  NAMES clang++ clang++-21 clang++-20 clang++-19 clang++-18 clang++-17
        clang++-16 clang++-15
  HINTS ${_Enzyme_HINTS}
  PATH_SUFFIXES bin)

if(_Enzyme_LLVM_BINDIR)
  list(APPEND _Enzyme_HINTS "${_Enzyme_LLVM_BINDIR}")
endif()

if(Enzyme_CLANGXX)
  get_filename_component(_Enzyme_CLANGXX_BINDIR "${Enzyme_CLANGXX}" DIRECTORY)
  list(APPEND _Enzyme_HINTS "${_Enzyme_CLANGXX_BINDIR}")
endif()

if(_Enzyme_LLVM_LIBDIR)
  list(APPEND _Enzyme_HINTS "${_Enzyme_LLVM_LIBDIR}")
endif()

file(GLOB _Enzyme_COMMON_LLVM_DIRS
     "/usr/lib/llvm-*"
     "/usr/local/opt/llvm*"
     "/opt/homebrew/opt/llvm*")

foreach(_dir ${_Enzyme_COMMON_LLVM_DIRS})
  list(APPEND
       _Enzyme_HINTS
       "${_dir}"
       "${_dir}/bin"
       "${_dir}/lib")
endforeach()

find_package(
  Enzyme
  CONFIG
  QUIET
  HINTS
  ${_Enzyme_HINTS}
  PATH_SUFFIXES
  cmake
  lib/cmake/Enzyme
  lib64/cmake/Enzyme
  share/Enzyme/cmake
  enzyme/build
  Enzyme/enzyme/build
  build)

if(Enzyme_FOUND)
  return()
endif()

set(_Enzyme_CLANG_PLUGIN_NAMES ClangEnzyme)
set(_Enzyme_LLVM_PLUGIN_NAMES LLVMEnzyme)
foreach(_version RANGE 21 10 -1)
  list(APPEND
       _Enzyme_CLANG_PLUGIN_NAMES
       ClangEnzyme-${_version})
  list(APPEND
       _Enzyme_LLVM_PLUGIN_NAMES
       LLVMEnzyme-${_version})
endforeach()

find_library(
  Enzyme_CLANG_PLUGIN
  NAMES ${_Enzyme_CLANG_PLUGIN_NAMES}
  HINTS ${_Enzyme_HINTS}
  PATH_SUFFIXES
  lib
  lib64
  build
  build/Enzyme
  build/enzyme
  enzyme/build
  Enzyme/enzyme/build
  LLVMEnzyme
  ClangEnzyme)

find_library(
  Enzyme_LLVM_PLUGIN
  NAMES ${_Enzyme_LLVM_PLUGIN_NAMES}
  HINTS ${_Enzyme_HINTS}
  PATH_SUFFIXES
  lib
  lib64
  build
  build/Enzyme
  build/enzyme
  enzyme/build
  Enzyme/enzyme/build
  LLVMEnzyme
  ClangEnzyme)

set(_Enzyme_INVALID_PLUGIN_MESSAGE "")
foreach(_plugin_var Enzyme_CLANG_PLUGIN Enzyme_LLVM_PLUGIN)
  if(DEFINED ${_plugin_var}
     AND ${_plugin_var}
     AND NOT ${_plugin_var} MATCHES "-NOTFOUND$"
     AND NOT EXISTS "${${_plugin_var}}")
    string(APPEND
           _Enzyme_INVALID_PLUGIN_MESSAGE
           "${_plugin_var}='${${_plugin_var}}' does not exist.\n")
    set(${_plugin_var}
        "${_plugin_var}-NOTFOUND"
        CACHE FILEPATH "Enzyme compiler plugin" FORCE)
  endif()
endforeach()

if(Enzyme_CLANG_PLUGIN AND NOT TARGET ClangEnzymeFlags)
  add_library(ClangEnzymeFlags INTERFACE IMPORTED)
  set_property(
    TARGET ClangEnzymeFlags
    PROPERTY INTERFACE_COMPILE_OPTIONS
             "SHELL:-Xclang -load -Xclang ${Enzyme_CLANG_PLUGIN}")
endif()

if(Enzyme_LLVM_PLUGIN AND NOT TARGET LLDEnzymeFlags)
  add_library(LLDEnzymeFlags INTERFACE IMPORTED)
  set_property(
    TARGET LLDEnzymeFlags
    PROPERTY INTERFACE_COMPILE_OPTIONS
             "-fpass-plugin=${Enzyme_LLVM_PLUGIN}")
endif()

if(_Enzyme_LLVM_BINDIR)
  get_filename_component(_Enzyme_LLVM_ROOT "${_Enzyme_LLVM_BINDIR}" DIRECTORY)
  set(Enzyme_LLVM_BINARY_DIR "${_Enzyme_LLVM_ROOT}")
elseif(Enzyme_CLANGXX)
  get_filename_component(_Enzyme_CLANGXX_ROOT "${_Enzyme_CLANGXX_BINDIR}" DIRECTORY)
  set(Enzyme_LLVM_BINARY_DIR "${_Enzyme_CLANGXX_ROOT}")
endif()

if(Enzyme_CLANG_PLUGIN)
  set(Enzyme_PLUGIN "${Enzyme_CLANG_PLUGIN}")
elseif(Enzyme_LLVM_PLUGIN)
  set(Enzyme_PLUGIN "${Enzyme_LLVM_PLUGIN}")
endif()

include(FindPackageHandleStandardArgs)
set(_Enzyme_FAILURE_MESSAGE [=[
Set Enzyme_DIR to a directory containing EnzymeConfig.cmake, set
Enzyme_ROOT/ENZYME_ROOT to an Enzyme install or build prefix, or set
Enzyme_CLANG_PLUGIN/Enzyme_LLVM_PLUGIN to the plugin shared library.
]=])
if(_Enzyme_INVALID_PLUGIN_MESSAGE)
  string(PREPEND _Enzyme_FAILURE_MESSAGE "${_Enzyme_INVALID_PLUGIN_MESSAGE}\n")
endif()
find_package_handle_standard_args(
  Enzyme
  REQUIRED_VARS Enzyme_PLUGIN
  HANDLE_COMPONENTS
  REASON_FAILURE_MESSAGE "${_Enzyme_FAILURE_MESSAGE}")

mark_as_advanced(Enzyme_CLANGXX Enzyme_CLANG_PLUGIN Enzyme_LLVM_PLUGIN
                 Enzyme_PLUGIN Enzyme_DIR)
