add_library(refem_fe)
add_library(refem::fe ALIAS refem_fe)

target_sources(refem_fe
  PRIVATE
    BlockFESpace.cpp
    DofMap.cpp
    ElementValues.cpp
    FESpace.cpp
  PUBLIC
    BlockFESpace.hpp
    DofMap.hpp
    ElementValues.hpp
    FESpace.hpp
    FiniteElement.hpp
    GaussQuadrature.hpp
    LagrangeQuadQ1.hpp
    ReferenceElement.hpp
)

target_link_libraries(refem_fe
  PUBLIC
    refem_common
    refem_config
    refem_mesh
)
