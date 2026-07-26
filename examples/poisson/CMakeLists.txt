set(FEMX_POISSON_SOURCES PoissonProblem.cpp PoissonResidual.cpp)

function(configure_poisson_example target)
  target_link_libraries(${target} PRIVATE femx::femx)
  target_compile_definitions(
    ${target}
    PRIVATE FEMX_POISSON_APP_NAME="${target}"
            FEMX_POISSON_DEFAULT_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/output"
  )
endfunction()

add_executable(poisson main.cpp ${FEMX_POISSON_SOURCES})
configure_poisson_example(poisson)

if(TARGET femx::linalg_petsc)
  add_executable(poisson-petsc main-petsc.cpp ${FEMX_POISSON_SOURCES})
  configure_poisson_example(poisson-petsc)
  target_link_libraries(poisson-petsc PRIVATE femx::linalg_petsc)
endif()

if(FEMX_HAS_RESOLVE)
  add_executable(poisson-resolve main-resolve.cpp ${FEMX_POISSON_SOURCES})
  configure_poisson_example(poisson-resolve)
  target_link_libraries(poisson-resolve PRIVATE femx::linalg_resolve)

  if(FEMX_HAS_RESOLVE_CUDA
     AND FEMX_HAS_CUDA
     AND NOT FEMX_RESOLVE_DEVICE_MODE STREQUAL "HOST")
    target_sources(poisson-resolve PRIVATE PoissonResidual.cu)
    set_target_properties(poisson-resolve PROPERTIES CUDA_STANDARD 17
                                                     CUDA_STANDARD_REQUIRED ON)
  endif()
endif()
