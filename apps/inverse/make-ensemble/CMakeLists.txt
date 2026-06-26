add_executable(make-ensemble main.cpp)

target_link_libraries(make-ensemble PRIVATE inverse_common ns_en_var_kernels
                                            femx::femx)

target_compile_definitions(make-ensemble
                           PRIVATE FEMX_MAKE_ENSEMBLE_APP_NAME="make-ensemble")
