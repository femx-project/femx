add_library(femx_ad INTERFACE)
add_library(femx::ad ALIAS femx_ad)

target_sources(femx_ad INTERFACE Enzyme.hpp Forward.hpp)

target_link_libraries(femx_ad INTERFACE femx_config femx_common)
