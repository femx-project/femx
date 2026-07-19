add_library(femx_linalg_resolve ReSolveLinearSolver.cpp)
add_library(femx::linalg_resolve ALIAS femx_linalg_resolve)

target_sources(femx_linalg_resolve PUBLIC ReSolveLinearSolver.hpp)

if(FEMX_HAS_RESOLVE_CUDA AND FEMX_HAS_CUDA)
  target_sources(femx_linalg_resolve PRIVATE ReSolveLinearSolver.cu)
  set_target_properties(femx_linalg_resolve PROPERTIES CUDA_STANDARD 17
                                                        CUDA_STANDARD_REQUIRED ON)
endif()

target_link_libraries(femx_linalg_resolve PUBLIC femx_linalg)

if(FEMX_ENABLE_RESOLVE AND ReSolve_FOUND)
  if(TARGET ReSolve::ReSolve)
    target_link_libraries(femx_linalg_resolve PUBLIC ReSolve::ReSolve)
  elseif(TARGET ReSolve)
    target_link_libraries(femx_linalg_resolve PUBLIC ReSolve)
  endif()
endif()
