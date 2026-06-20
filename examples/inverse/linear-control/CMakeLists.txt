if(TARGET femx::optimize)
  add_executable(example-inverse-linear-control main.cpp)
  target_link_libraries(example-inverse-linear-control
                        PRIVATE femx::optimize femx::femx)
endif()
