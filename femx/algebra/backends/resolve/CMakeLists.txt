target_sources(
  femx_algebra
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/ReSolveLinearSolver.hpp)

if(FEMX_ENABLE_RESOLVE AND ReSolve_FOUND)
  if(TARGET ReSolve::ReSolve)
    target_link_libraries(femx_algebra PUBLIC ReSolve::ReSolve)
  elseif(TARGET ReSolve)
    target_link_libraries(femx_algebra PUBLIC ReSolve)
  endif()
endif()
