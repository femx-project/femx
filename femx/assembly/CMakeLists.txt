add_library(femx_assembly)
add_library(femx::assembly ALIAS femx_assembly)

target_sources(
  femx_assembly
  PRIVATE Assembler.cpp BoundaryDofLayout.cpp FEMResidual.cpp
          CsrBuilder.cpp TimeDirichletControlResidual.cpp
          TimeFEMResidual.cpp
  PUBLIC BoundaryDofLayout.hpp
         BoundaryElementKernel.hpp
         Assembler.hpp
         ElementKernel.hpp
         EnzymeBoundaryKernel.hpp
         EnzymeTimeVolumeKernel.hpp
         EnzymeVolumeKernel.hpp
         FEMResidual.hpp
         CsrBuilder.hpp
         TimeDirichletControlResidual.hpp
         TimeElementKernel.hpp
         TimeFEMResidual.hpp)

target_link_libraries(femx_assembly PUBLIC femx_config femx_ad femx_linalg
                                           femx_problem femx_fem)

if(TARGET femx_linalg_petsc)
  target_link_libraries(femx_assembly PRIVATE femx_linalg_petsc)
endif()
