add_library(ns_var_new_kernels Helper.cpp IO.cpp)

if(FEMX_ENABLE_ENZYME)
  set_target_properties(ns_var_new_kernels PROPERTIES UNITY_BUILD ON)
endif()

target_sources(ns_var_new_kernels PUBLIC Helper.hpp IO.hpp)

target_include_directories(ns_var_new_kernels
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                           PRIVATE ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

target_link_libraries(ns_var_new_kernels PUBLIC navier_common femx::femx)

function(configure_ns_var_new_app target)
  target_link_libraries(${target}
                        PRIVATE ns_var_new_kernels femx::opt femx::femx)
  target_compile_definitions(
    ${target} PRIVATE FEMX_NAVIER_VAR_NEW_APP_NAME="${target}")
endfunction()

if(TARGET femx::opt AND FEMX_HAS_RESOLVE AND FEMX_HAS_ENZYME)
  add_executable(ns-var-resolve main-resolve.cpp)
  configure_ns_var_new_app(ns-var-resolve)
  target_link_libraries(ns-var-resolve PRIVATE femx::linalg_resolve)
endif()

if(TARGET femx::opt AND TARGET femx::linalg_petsc AND FEMX_HAS_ENZYME)
  add_executable(ns-var main-petsc.cpp)
  configure_ns_var_new_app(ns-var)
  target_link_libraries(ns-var PRIVATE femx::linalg_petsc)
endif()
