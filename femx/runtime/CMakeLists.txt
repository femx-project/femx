add_library(femx_runtime BuildInfo.cpp Cli.cpp LinearSystemSelection.cpp
                         Output.cpp)
add_library(femx::runtime ALIAS femx_runtime)

target_sources(
  femx_runtime PUBLIC BuildInfo.hpp Cli.hpp LinearSystemSelection.hpp
                      Output.hpp Parallel.hpp PETScRuntime.hpp)

target_link_libraries(femx_runtime PUBLIC femx_config femx_common femx_linalg)
