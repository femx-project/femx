add_library(make_obs_config Config.cpp ObservationVti.cpp
                            TrajectoryObservation.cpp)

target_sources(make_obs_config PUBLIC Config.hpp ObservationVti.hpp
                                      TrajectoryObservation.hpp)

target_link_libraries(make_obs_config PUBLIC inverse_common ns_var_new_kernels
                                             femx::femx)

target_include_directories(
  make_obs_config
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

function(configure_make_obs_app target)
  target_link_libraries(${target} PRIVATE inverse_common make_obs_config
                                          ns_var_new_kernels femx::femx)
  target_compile_definitions(${target}
                             PRIVATE FEMX_MAKE_OBS_APP_NAME="${target}")
endfunction()

add_executable(make-obs main.cpp)
configure_make_obs_app(make-obs)
