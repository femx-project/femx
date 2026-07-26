set(
  FEMX_POISSON_OPT_SOURCES
  PoissonOptProblem.cpp
  PoissonOptResidual.cpp
  PoissonOptSolve.cpp)

function(configure_poisson_opt_example target)
  target_link_libraries(${target} PRIVATE femx::femx femx::opt)
  target_compile_definitions(
    ${target}
    PRIVATE
      FEMX_POISSON_OPT_APP_NAME="${target}"
      FEMX_POISSON_OPT_DEFAULT_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/output")
endfunction()

if(TARGET femx::linalg_petsc)
  add_executable(poisson-opt-petsc main-petsc.cpp ${FEMX_POISSON_OPT_SOURCES})
  configure_poisson_opt_example(poisson-opt-petsc)
  target_link_libraries(poisson-opt-petsc PRIVATE femx::linalg_petsc)
endif()

if(FEMX_HAS_RESOLVE
   AND TARGET femx::linalg_petsc
   AND TARGET femx::linalg_resolve)
  add_executable(poisson-opt-resolve main-resolve.cpp
                                     ${FEMX_POISSON_OPT_SOURCES})
  configure_poisson_opt_example(poisson-opt-resolve)
  target_link_libraries(poisson-opt-resolve PRIVATE femx::linalg_petsc
                                                    femx::linalg_resolve)

  if(FEMX_HAS_RESOLVE_CUDA
     AND FEMX_HAS_CUDA
     AND NOT FEMX_RESOLVE_DEVICE_MODE STREQUAL "HOST")
    target_sources(poisson-opt-resolve PRIVATE PoissonOptResidual.cu)
    set_target_properties(poisson-opt-resolve PROPERTIES CUDA_STANDARD 17
                                                         CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

if(FEMX_HAS_ENZYME)
  set_property(
    SOURCE PoissonOptResidual.cpp
    APPEND
    PROPERTY COMPILE_OPTIONS -O2)
  if(FEMX_HAS_CUDA)
    set_property(
      SOURCE PoissonOptResidual.cu
      APPEND
      PROPERTY COMPILE_OPTIONS -O2)
  endif()
endif()
