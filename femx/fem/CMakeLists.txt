add_library(femx_fem)
add_library(femx::fem ALIAS femx_fem)

target_sources(
  femx_fem
  PRIVATE MixedFESpace.cpp DofMap.cpp ElementValues.cpp FESpace.cpp
  PUBLIC MixedFESpace.hpp
         DofMap.hpp
         ElementValues.hpp
         FESpace.hpp
         FiniteElement.hpp
         GaussQuadrature.hpp
         ReferenceElement.hpp
         elements/LagrangeQuadQ1.hpp
         elements/LagrangeTetrahedronP1.hpp
         elements/LagrangeTriangleP1.hpp)

target_link_libraries(femx_fem PUBLIC femx_common femx_config femx_mesh)
