add_executable(poisson-boundary-flux-tao main.cpp)

target_link_libraries(poisson-boundary-flux-tao PRIVATE femx::femx
                                                        femx::inverse_petsc)
