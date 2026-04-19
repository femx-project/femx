add_library(refem_io INTERFACE)
add_library(refem::io ALIAS refem_io)

target_link_libraries(refem_io INTERFACE refem_config)
