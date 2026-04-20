add_executable(navier-gls
  main.cpp
  NavierGLS.cpp
  Components.cpp
)

target_link_libraries(navier-gls PRIVATE refem::refem)

target_include_directories(navier-gls
  PRIVATE
    ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include
)

target_compile_definitions(navier-gls
  PRIVATE
    REFEM_NAVIERSTOKES_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    REFEM_NAVIERSTOKES_CONFIG_FILE="${CMAKE_CURRENT_SOURCE_DIR}/Config.json"
)
