if(TARGET femx::inverse_petsc AND FEMX_HAS_ENZYME)
  add_subdirectory(poisson-bd-control)
endif()
