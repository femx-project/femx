add_library(femx_common)
add_library(femx::core ALIAS femx_common)

target_sources(femx_common
               PRIVATE Math.cpp
               PUBLIC Math.hpp Types.hpp Workspace.hpp)

target_link_libraries(femx_common PUBLIC femx_config)
