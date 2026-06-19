add_subdirectory(common)
add_subdirectory(ad)
add_subdirectory(mesh)
add_subdirectory(fem)
add_subdirectory(linalg)
add_subdirectory(system)
add_subdirectory(eq)
add_subdirectory(assembly)
add_subdirectory(inverse)
add_subdirectory(bc)
add_subdirectory(io)

target_link_libraries(
  femx
  INTERFACE femx_common
            femx_ad
            femx_assembly
            femx_bc
            femx_eq
            femx_fem
            femx_inverse
            femx_io
            femx_linalg
            femx_mesh
            femx_system)
