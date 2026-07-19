include_guard(GLOBAL)

set(_ReSolve_DEVELOP_PREFIX "/opt/resolve/develop")
set(_ReSolve_DEVELOP_CONFIG
    "${_ReSolve_DEVELOP_PREFIX}/share/resolve/cmake")
set(ReSolve_DIR
    "${_ReSolve_DEVELOP_CONFIG}"
    CACHE PATH "Required ReSolve development package directory" FORCE)

find_package(
  ReSolve
  CONFIG
  QUIET
  PATHS "${_ReSolve_DEVELOP_PREFIX}"
  NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ReSolve CONFIG_MODE)

mark_as_advanced(ReSolve_DIR)
