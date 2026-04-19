add_library(refem_bc)
add_library(refem::bc ALIAS refem_bc)

target_sources(refem_bc
  PRIVATE
    DirichletCondition.cpp
  PUBLIC
    DirichletCondition.hpp
)

target_link_libraries(refem_bc
  PUBLIC
    refem_config
    refem_common
    refem_fe
    refem_linalg
    refem_mesh
)
