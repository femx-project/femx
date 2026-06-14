add_subdirectory(core)
add_subdirectory(ad)
add_subdirectory(mesh)
add_subdirectory(fem)
add_subdirectory(linalg)
add_subdirectory(system)
add_subdirectory(equation)
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
            femx_equation
            femx_fem
            femx_inverse
            femx_io
            femx_linalg
            femx_mesh
            femx_system)
