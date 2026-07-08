add_library(femx_assembly)
add_library(femx::assembly ALIAS femx_assembly)

target_sources(
  femx_assembly
  PRIVATE Assembler.cpp DirichletControlResidual.cpp FEMResidual.cpp
          TimeDirichletControlResidual.cpp TimeFEMResidual.cpp
  PUBLIC Assembler.hpp
         DirichletControlResidual.hpp
         ElementKernel.hpp
         EnzymeTimeVolumeKernel.hpp
         FEMResidual.hpp
         TimeDirichletControlResidual.hpp
         TimeElementKernel.hpp
         TimeFEMResidual.hpp)

target_link_libraries(femx_assembly PUBLIC femx_config femx_ad femx_linalg
                                           femx_state femx_fem)

if(TARGET femx_linalg_petsc)
  target_link_libraries(femx_assembly PRIVATE femx_linalg_petsc)
endif()
