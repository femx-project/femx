add_subdirectory(linear-control-new-api)
add_subdirectory(linear-control)

if(TARGET femx::optimize AND FEMX_HAS_ENZYME)
  add_subdirectory(opt-control-poisson)
endif()
