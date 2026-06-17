if(TARGET femx::inverse_petsc AND FEMX_HAS_ENZYME)
  add_subdirectory(opt-control-poisson)
endif()
