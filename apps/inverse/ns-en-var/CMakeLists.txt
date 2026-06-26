add_library(ns_en_var_kernels Helper.cpp IO.cpp)

if(FEMX_ENABLE_ENZYME)
  set_target_properties(ns_en_var_kernels PROPERTIES UNITY_BUILD ON)
endif()

target_sources(ns_en_var_kernels PUBLIC Helper.hpp IO.hpp)

target_include_directories(
  ns_en_var_kernels
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

target_link_libraries(ns_en_var_kernels PUBLIC navier_common femx::femx)

function(configure_ns_en_var_app target)
  target_link_libraries(${target} PRIVATE inverse_common ns_en_var_kernels
                                          femx::opt femx::femx)
  target_compile_definitions(${target}
                             PRIVATE FEMX_NAVIER_EN_VAR_APP_NAME="${target}")
endfunction()

if(TARGET femx::opt AND TARGET femx::linalg_petsc)
  add_executable(ns-en-var main-petsc.cpp)
  configure_ns_en_var_app(ns-en-var)
  target_link_libraries(ns-en-var PRIVATE femx::linalg_petsc)
endif()
