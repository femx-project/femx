add_subdirectory(common)
add_subdirectory(linalg)
add_subdirectory(ad)
add_subdirectory(problem)
add_subdirectory(fem)
if(FEMX_HAS_PETSC)
  add_subdirectory(linalg/petsc)
endif()
add_subdirectory(state)
add_subdirectory(assembly)
if(FEMX_HAS_PETSC)
  add_subdirectory(opt)
endif()
add_subdirectory(io)

target_link_libraries(
  femx
  INTERFACE femx_common
            femx_linalg
            femx_ad
            femx_assembly
            femx_fem
            femx_io
            femx_problem
            femx_state)

if(FEMX_HAS_PETSC)
  target_link_libraries(femx INTERFACE femx_opt)
endif()

if(TARGET femx_linalg_resolve)
  target_link_libraries(femx INTERFACE femx_linalg_resolve)
endif()
