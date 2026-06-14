add_library(femx_common INTERFACE)
add_library(femx::core ALIAS femx_common)

target_sources(femx_common INTERFACE Types.hpp Workspace.hpp)

target_link_libraries(femx_common INTERFACE femx_config)
