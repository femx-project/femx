add_library(femx_assembly)
add_library(femx::assembly ALIAS femx_assembly)

target_sources(
  femx_assembly
  PRIVATE AssemblyMap.cpp
          BoundaryPlan.cpp
          DirichletControlResidual.cpp
          TimeDirichletControlResidual.cpp
          HostTimeResidual.cpp
  PUBLIC Assembly.hpp
         AssemblyMap.hpp
         BoundaryPlan.hpp
         CudaAssembly.hpp
         DirichletControlResidual.hpp
         TimeDirichletControlResidual.hpp
         HostTimeResidual.hpp)

target_link_libraries(femx_assembly PUBLIC femx_config femx_ad femx_linalg
                                           femx_state femx_fem)

if(FEMX_HAS_CUDA)
  target_sources(femx_assembly PRIVATE BoundaryPlan.cu)
  set_target_properties(femx_assembly PROPERTIES CUDA_STANDARD 17
                                                 CUDA_STANDARD_REQUIRED ON)
endif()

if(TARGET femx_linalg_petsc)
  target_link_libraries(femx_assembly PRIVATE femx_linalg_petsc)
endif()
