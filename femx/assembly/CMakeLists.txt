add_library(femx_assembly INTERFACE)
add_library(femx::assembly ALIAS femx_assembly)

target_sources(
  femx_assembly
  INTERFACE BoundaryDofLayout.hpp
            BoundaryElementKernel.hpp
            BoundaryResidualEquation.hpp
            DofLayout.hpp
            ElementKernel.hpp
            EnzymeBoundaryElementKernel.hpp
            EnzymeBoundaryIntegralKernel.hpp
            EnzymeElementKernel.hpp
            ElementResidualEquation.hpp
            SparsityPatternBuilder.hpp
            SystemAssembler.hpp)

target_link_libraries(
  femx_assembly
  INTERFACE femx_ad
            femx_common
            femx_config
            femx_equation
            femx_fem
            femx_linalg
            femx_system)
