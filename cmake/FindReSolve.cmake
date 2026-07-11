include_guard(GLOBAL)

set(_ReSolve_HINTS)
set(_ReSolve_VERSION_ARGS)

if(ReSolve_FIND_VERSION)
  list(APPEND _ReSolve_VERSION_ARGS "${ReSolve_FIND_VERSION}")
  if(ReSolve_FIND_VERSION_EXACT)
    list(APPEND _ReSolve_VERSION_ARGS EXACT)
  endif()
endif()

foreach(_var ReSolve_ROOT RESOLVE_ROOT)
  if(DEFINED ${_var})
    list(APPEND _ReSolve_HINTS "${${_var}}")
  endif()

  if(DEFINED ENV{${_var}})
    list(APPEND _ReSolve_HINTS "$ENV{${_var}}")
  endif()
endforeach()

find_package(
  ReSolve
  ${_ReSolve_VERSION_ARGS}
  CONFIG
  QUIET
  HINTS
  ${_ReSolve_HINTS}
  PATH_SUFFIXES
  share/resolve/cmake
  share/ReSolve/cmake
  lib/cmake/ReSolve
  lib64/cmake/ReSolve)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ReSolve CONFIG_MODE)

mark_as_advanced(ReSolve_DIR)
