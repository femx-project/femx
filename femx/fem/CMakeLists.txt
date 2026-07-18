add_library(femx_fem)
add_library(femx::fem ALIAS femx_fem)

target_sources(
  femx_fem
  PRIVATE DirichletBC.cpp
          TimeDirichletData.cpp
          BoundarySurface.cpp
          DirichletControl.cpp
          DofLayout.cpp
          DofMap.cpp
          ElementValues.cpp
          FESpace.cpp
          Geometry.cpp
          GmshReader.cpp
          Mesh.cpp
          MixedFESpace.cpp
          ObservationGrid.cpp
          TimePointInterpolator.cpp
          VelocityProfile.cpp
  PUBLIC DirichletBC.hpp
         TimeDirichletData.hpp
         BoundarySurface.hpp
         Element.hpp
         DirichletControl.hpp
         DofLayout.hpp
         MixedFESpace.hpp
         DofMap.hpp
         ElementValues.hpp
         FESpace.hpp
         FiniteElement.hpp
         GaussQuadrature.hpp
         Geometry.hpp
         GmshReader.hpp
         Mesh.hpp
         ObservationGrid.hpp
         Quadrature.hpp
         ReferenceElement.hpp
         TimePointInterpolator.hpp
         VelocityProfile.hpp
         elements/LagrangeQuadQ1.hpp
         elements/LagrangeTetrahedronP1.hpp
         elements/LagrangeTriangleP1.hpp)

target_link_libraries(femx_fem PUBLIC femx_config femx_common femx_linalg
                                      femx_inverse)
