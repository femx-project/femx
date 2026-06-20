add_library(femx_assembly)
add_library(femx::assembly ALIAS femx_assembly)

target_sources(
  femx_assembly
  PRIVATE BoundaryDofLayout.cpp BoundaryResidualEquation.cpp DofLayout.cpp
          ElementResidualEquation.cpp SparsityPatternBuilder.cpp
          SystemAssembler.cpp
  PUBLIC BoundaryDofLayout.hpp
         BoundaryElementKernel.hpp
         BoundaryResidualEquation.hpp
         DofLayout.hpp
         ElementKernel.hpp
         ElementResidualEquation.hpp
         EnzymeBoundaryKernel.hpp
         EnzymeVolumeKernel.hpp
         SparsityPatternBuilder.hpp
         SystemAssembler.hpp)

target_link_libraries(
  femx_assembly
  PUBLIC femx_ad
         femx_common
         femx_config
         femx_eq
         femx_fem
         femx_linalg
         femx_system)
