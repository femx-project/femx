if(TARGET femx::linalg_petsc)
  add_executable(make-obs-ensemble main.cpp)

  target_link_libraries(
    make-obs-ensemble PRIVATE inverse_common ns_en_var_kernels femx::femx
                              femx::linalg_petsc)

  target_compile_definitions(
    make-obs-ensemble
    PRIVATE FEMX_MAKE_OBS_ENSEMBLE_APP_NAME="make-obs-ensemble")
endif()
