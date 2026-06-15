add_executable(poisson-bd-control main.cpp)

target_link_libraries(poisson-bd-control PRIVATE femx::femx
                                                        femx::inverse_petsc)
