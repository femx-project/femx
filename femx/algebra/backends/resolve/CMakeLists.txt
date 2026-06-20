add_library(femx_algebra_resolve ReSolveLinearSolver.cpp)
add_library(femx::algebra_resolve ALIAS femx_algebra_resolve)

target_sources(femx_algebra_resolve PUBLIC ReSolveLinearSolver.hpp)

target_link_libraries(femx_algebra_resolve PUBLIC femx_algebra)

if(FEMX_ENABLE_RESOLVE AND ReSolve_FOUND)
  if(TARGET ReSolve::ReSolve)
    target_link_libraries(femx_algebra_resolve PUBLIC ReSolve::ReSolve)
  elseif(TARGET ReSolve)
    target_link_libraries(femx_algebra_resolve PUBLIC ReSolve)
  endif()
endif()
