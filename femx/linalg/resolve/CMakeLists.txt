add_library(femx_linalg_resolve ReSolveLinearSolver.cpp)
add_library(femx::linalg_resolve ALIAS femx_linalg_resolve)

target_sources(femx_linalg_resolve PUBLIC ReSolveLinearSolver.hpp)

target_link_libraries(femx_linalg_resolve PUBLIC femx_linalg)

if(FEMX_ENABLE_RESOLVE AND ReSolve_FOUND)
  if(TARGET ReSolve::ReSolve)
    target_link_libraries(femx_linalg_resolve PUBLIC ReSolve::ReSolve)
  elseif(TARGET ReSolve)
    target_link_libraries(femx_linalg_resolve PUBLIC ReSolve)
  endif()
endif()
