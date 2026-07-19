add_library(femx_common INTERFACE)
add_library(femx::common ALIAS femx_common)

target_sources(
  femx_common
  INTERFACE Checks.hpp Context.hpp LinearInterpolation.hpp Math.hpp Types.hpp)

target_link_libraries(femx_common INTERFACE femx_config)
