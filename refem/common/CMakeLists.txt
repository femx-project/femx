add_library(refem_common INTERFACE)
add_library(refem::common ALIAS refem_common)

target_sources(refem_common
  INTERFACE
    Types.hpp
)

target_link_libraries(refem_common INTERFACE refem_config)
