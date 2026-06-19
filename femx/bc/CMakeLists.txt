add_library(femx_bc)
add_library(femx::bc ALIAS femx_bc)

target_sources(
  femx_bc
  PRIVATE DirichletCondition.cpp VelocityProfile.cpp
  PUBLIC DirichletCondition.hpp VelocityProfile.hpp)

target_link_libraries(femx_bc PUBLIC femx_config femx_common femx_fem
                                     femx_linalg femx_mesh)
