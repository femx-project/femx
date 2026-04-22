add_executable(navier-gls main.cpp Config.cpp BoundaryConditions.cpp
                          Assembly.cpp Components.cpp)

target_link_libraries(navier-gls PRIVATE refem::refem)

target_include_directories(
  navier-gls
  PRIVATE ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

target_compile_definitions(
  navier-gls
  PRIVATE REFEM_NAVIERSTOKES_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}")
