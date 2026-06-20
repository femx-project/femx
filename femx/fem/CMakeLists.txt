add_library(femx_fem)
add_library(femx::fem ALIAS femx_fem)

target_sources(
  femx_fem
  PRIVATE BoundaryCondition.cpp
          BoundaryElementValues.cpp
          Cell.cpp
          DirichletControl.cpp
          DofMap.cpp
          ElementValues.cpp
          FESpace.cpp
          GmshReader.cpp
          Mesh.cpp
          MixedFESpace.cpp
          ObservationGrid.cpp
          TimePointSampler.cpp
          VelocityProfile.cpp
  PUBLIC BoundaryElementValues.hpp
         BoundaryCondition.hpp
         Cell.hpp
         DirichletControl.hpp
         DofLayout.hpp
         MixedFESpace.hpp
         DofMap.hpp
         ElementValues.hpp
         FESpace.hpp
         FiniteElement.hpp
         GaussQuadrature.hpp
         GmshReader.hpp
         Mesh.hpp
         ObservationGrid.hpp
         Quadrature.hpp
         ReferenceElement.hpp
         TimePointSampler.hpp
         VelocityProfile.hpp
         elements/LagrangeQuadQ1.hpp
         elements/LagrangeTetrahedronP1.hpp
         elements/LagrangeTriangleP1.hpp)

target_link_libraries(femx_fem PUBLIC femx_config femx_core femx_algebra)
