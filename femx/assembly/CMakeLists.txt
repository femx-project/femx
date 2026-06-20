add_library(femx_assembly)
add_library(femx::assembly ALIAS femx_assembly)

target_sources(
  femx_assembly
  PRIVATE Assembler.cpp BoundaryDofLayout.cpp FEMResidual.cpp
          SparsityPatternBuilder.cpp TimeDirichletControlResidual.cpp
  PUBLIC BoundaryDofLayout.hpp
         BoundaryElementKernel.hpp
         Assembler.hpp
         ElementKernel.hpp
         EnzymeBoundaryKernel.hpp
         EnzymeVolumeKernel.hpp
         FEMResidual.hpp
         Kernel.hpp
         SparsityPatternBuilder.hpp
         TimeDirichletControlResidual.hpp)

target_link_libraries(
  femx_assembly
  PUBLIC femx_config
         femx_ad
         femx_algebra
         femx_problem
         femx_fem)

if(TARGET femx_algebra_petsc)
  target_link_libraries(femx_assembly PRIVATE femx_algebra_petsc)
endif()
