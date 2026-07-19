include_guard(GLOBAL)

find_package(ReSolve CONFIG QUIET)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ReSolve CONFIG_MODE)

mark_as_advanced(ReSolve_DIR)
