set(FEMX_POISSON_OPT_SOURCES PoissonOpt.cpp)

function(configure_poisson_opt_example target)
  target_link_libraries(${target} PRIVATE femx::femx femx::opt)
  target_compile_definitions(
    ${target}
    PRIVATE
      FEMX_POISSON_OPT_APP_NAME="${target}"
      FEMX_POISSON_OPT_DEFAULT_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/output")
endfunction()

if(FEMX_HAS_ENZYME AND TARGET femx::linalg_petsc)
  add_executable(poisson-opt-petsc main-petsc.cpp ${FEMX_POISSON_OPT_SOURCES})
  configure_poisson_opt_example(poisson-opt-petsc)
  target_link_libraries(poisson-opt-petsc PRIVATE femx::linalg_petsc)
endif()

if(FEMX_HAS_ENZYME
   AND FEMX_HAS_RESOLVE
   AND TARGET femx::linalg_petsc
   AND TARGET femx::linalg_resolve)
  add_executable(poisson-opt-resolve main-resolve.cpp
                                     ${FEMX_POISSON_OPT_SOURCES})
  configure_poisson_opt_example(poisson-opt-resolve)
  target_link_libraries(poisson-opt-resolve PRIVATE femx::linalg_petsc
                                                    femx::linalg_resolve)
endif()
