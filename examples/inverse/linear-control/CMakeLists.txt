if(TARGET femx::inverse_petsc)
  add_executable(example-inverse-linear-control main.cpp)
  target_link_libraries(example-inverse-linear-control
                        PRIVATE femx::inverse_petsc femx::femx)
endif()
