add_library(make_obs_config Config.cpp ObservationVti.cpp)

target_sources(make_obs_config PUBLIC Config.hpp ObservationVti.hpp)

target_link_libraries(make_obs_config PUBLIC navier_var_kernels femx::femx)

target_include_directories(
  make_obs_config
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

function(configure_make_obs_app target)
  target_link_libraries(${target} PRIVATE make_obs_config navier_var_kernels
                                          femx::femx)
  target_compile_definitions(${target}
                             PRIVATE FEMX_MAKE_OBS_APP_NAME="${target}")
endfunction()

if(FEMX_HAS_RESOLVE)
  add_executable(make-obs main-resolve.cpp)
  configure_make_obs_app(make-obs)
endif()

if(TARGET femx::system_petsc)
  add_executable(make-obs-petsc main-petsc.cpp)
  configure_make_obs_app(make-obs-petsc)
  target_link_libraries(make-obs-petsc PRIVATE femx::system_petsc)

  if(NOT TARGET make-obs)
    add_executable(make-obs main-petsc.cpp)
    configure_make_obs_app(make-obs)
    target_link_libraries(make-obs PRIVATE femx::system_petsc)
  endif()
endif()
