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
         Assembler.hpp
         DofLayout.hpp
         ElementKernel.hpp
         ElementResidualEquation.hpp
         EnzymeBoundaryKernel.hpp
         EnzymeVolumeKernel.hpp
         FEMResidual.hpp
         Kernel.hpp
         SparsityPatternBuilder.hpp
         SystemAssembler.hpp)

target_link_libraries(
  femx_assembly
  PUBLIC femx_ad
         femx_config
         femx_algebra
         femx_problem
         femx_solve
         femx_fem)
