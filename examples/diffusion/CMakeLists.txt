add_executable(diffusion main.cpp)

target_link_libraries(diffusion PRIVATE refem::refem)

target_compile_definitions(diffusion
  PRIVATE
    REFEM_DIFFUSION_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)
