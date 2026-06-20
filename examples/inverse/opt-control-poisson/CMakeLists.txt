add_executable(opt-control-poisson main.cpp)

target_link_libraries(opt-control-poisson PRIVATE femx::femx
                                                  femx::optimize)
