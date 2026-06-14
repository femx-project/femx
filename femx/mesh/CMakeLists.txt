add_library(femx_mesh)
add_library(femx::mesh ALIAS femx_mesh)

target_sources(
  femx_mesh
  PRIVATE Cell.cpp GmshReader.cpp Mesh.cpp
  PUBLIC Cell.hpp GmshReader.hpp Mesh.hpp)

target_link_libraries(femx_mesh PUBLIC femx_common femx_config)
