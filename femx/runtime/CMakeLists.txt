add_library(femx_runtime BuildInfo.cpp Cli.cpp LinearSystemFactory.cpp
                         Output.cpp)
add_library(femx::runtime ALIAS femx_runtime)

target_sources(
  femx_runtime PUBLIC BuildInfo.hpp Cli.hpp LinearSystemFactory.hpp Output.hpp
                      Parallel.hpp PETScRuntime.hpp)

target_link_libraries(femx_runtime PUBLIC femx_config femx_common femx_linalg)

if(TARGET femx_linalg_resolve)
  target_link_libraries(femx_runtime PRIVATE femx_linalg_resolve)
endif()

if(TARGET femx_linalg_petsc)
  target_link_libraries(femx_runtime PRIVATE femx_linalg_petsc)
endif()
