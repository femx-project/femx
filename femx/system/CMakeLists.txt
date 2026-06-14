add_library(femx_system)
add_library(femx::system ALIAS femx_system)

target_sources(
  femx_system
  PRIVATE resolve/ReSolveLinearSolver.cpp
  PUBLIC DenseLinearSolver.hpp
         LinearOperator.hpp
         LinearSolver.hpp
         SystemMatrix.hpp
         SystemVector.hpp
         native/DenseSystemMatrix.hpp
         native/DenseSystemVector.hpp
         native/SparseSystemMatrix.hpp
         resolve/ReSolveLinearSolver.hpp)

target_link_libraries(femx_system PUBLIC femx_common femx_config femx_linalg)

if(FEMX_ENABLE_RESOLVE AND ReSolve_FOUND)
  if(TARGET ReSolve::ReSolve)
    target_link_libraries(femx_system PUBLIC ReSolve::ReSolve)
  elseif(TARGET ReSolve)
    target_link_libraries(femx_system PUBLIC ReSolve)
  endif()
endif()

if(FEMX_HAS_PETSC)
  add_subdirectory(petsc)
endif()
