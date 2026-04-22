add_library(refem_mesh)
add_library(refem::mesh ALIAS refem_mesh)

target_sources(
  refem_mesh
  PRIVATE Cell.cpp GmshReader.cpp Mesh.cpp
  PUBLIC Cell.hpp GmshReader.hpp Mesh.hpp)

target_link_libraries(refem_mesh PUBLIC refem_common refem_config)
