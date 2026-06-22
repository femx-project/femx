if(TARGET femx::opt)
  add_executable(example-inverse-linear-control main.cpp)
  target_link_libraries(example-inverse-linear-control PRIVATE femx::opt
                                                               femx::femx)
endif()
