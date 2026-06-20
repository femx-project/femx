add_library(femx_core INTERFACE)
add_library(femx::core ALIAS femx_core)

target_sources(femx_core INTERFACE Math.hpp Types.hpp Workspace.hpp)

target_link_libraries(femx_core INTERFACE femx_config)
