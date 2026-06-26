add_subdirectory(linear-control)

if(TARGET femx::opt AND FEMX_HAS_ENZYME)
  add_subdirectory(opt-control-poisson)
endif()
