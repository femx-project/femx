add_library(refem_solver)
add_library(refem::solver ALIAS refem_solver)

target_sources(
  refem_solver
  PRIVATE LinearSolver.cpp ReSolveLinearSolver.cpp
  PUBLIC LinearSolver.hpp LinearSolverImpl.hpp ReSolveLinearSolver.hpp
         Workspace.hpp)

target_link_libraries(refem_solver PUBLIC refem_common refem_config
                                          refem_linalg)

if(REFEM_ENABLE_RESOLVE AND ReSolve_FOUND)
  if(TARGET ReSolve::ReSolve)
    target_link_libraries(refem_solver PUBLIC ReSolve::ReSolve)
  elseif(TARGET ReSolve)
    target_link_libraries(refem_solver PUBLIC ReSolve)
  endif()
endif()
