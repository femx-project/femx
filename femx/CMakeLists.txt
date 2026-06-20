add_subdirectory(core)
add_subdirectory(algebra)
add_subdirectory(ad)
add_subdirectory(fem)
if(FEMX_HAS_PETSC)
  add_subdirectory(algebra/backends/petsc)
endif()
add_subdirectory(problem)
add_subdirectory(solve)
add_subdirectory(assembly)
if(FEMX_HAS_PETSC)
  add_subdirectory(optimize)
endif()
add_subdirectory(io)

target_link_libraries(
  femx
  INTERFACE femx_core
            femx_algebra
            femx_ad
            femx_assembly
            femx_fem
            femx_io
            femx_problem
            femx_solve)

if(FEMX_HAS_PETSC)
  target_link_libraries(femx INTERFACE femx_optimize)
endif()
