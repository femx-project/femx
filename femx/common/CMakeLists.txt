add_library(femx_common INTERFACE)
add_library(femx::common ALIAS femx_common)

target_sources(femx_common INTERFACE LinearInterpolation.hpp Math.hpp Types.hpp
                                     Workspace.hpp)

target_link_libraries(femx_common INTERFACE femx_config)
