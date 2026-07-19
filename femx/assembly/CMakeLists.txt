add_library(femx_assembly)
add_library(femx::assembly ALIAS femx_assembly)

target_sources(
  femx_assembly
  PRIVATE Assembly.cpp
          AssemblyMap.cpp
          BoundaryMap.cpp
          ConstrainedTimeResidual.cpp
  PUBLIC Assembly.hpp
         AssemblyMap.hpp
         BoundaryMap.hpp
         ConstrainedTimeResidual.hpp
         CudaAssembly.hpp)

target_link_libraries(femx_assembly PUBLIC femx_config femx_linalg
                                           femx_state femx_fem)

if(FEMX_HAS_CUDA)
  target_sources(femx_assembly PRIVATE BoundaryMap.cu)
  set_target_properties(femx_assembly PROPERTIES CUDA_STANDARD 17
                                                 CUDA_STANDARD_REQUIRED ON)
endif()

if(TARGET femx_linalg_petsc)
  target_link_libraries(femx_assembly PUBLIC femx_linalg_petsc)
endif()
