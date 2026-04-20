add_executable(navierstokes
  main.cpp
  Navier.cpp
  NavierComponents.cpp
)

target_link_libraries(navierstokes PRIVATE refem::refem)

target_compile_definitions(navierstokes
  PRIVATE
    REFEM_NAVIERSTOKES_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)
