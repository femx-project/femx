add_library(
  navier_var_kernels
  Assembly.cpp
  Components.cpp
  Config.cpp
  RunSupport.cpp
  DirichletControl.cpp
  DirichletControlEquation.cpp
  NavierStokesEquation.cpp)

if(FEMX_ENABLE_ENZYME)
  set_target_properties(navier_var_kernels PROPERTIES UNITY_BUILD ON)
endif()

target_sources(
  navier_var_kernels
  PUBLIC Assembly.hpp
         Config.hpp
         Components.hpp
         RunSupport.hpp
         DirichletControl.hpp
         DirichletControlEquation.hpp
         NavierStokesEquation.hpp)

target_include_directories(
  navier_var_kernels
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/third-party/nlohmann-json/single_include)

target_link_libraries(navier_var_kernels PUBLIC femx::femx)

if(TARGET femx::system_petsc)
  target_sources(navier_var_kernels PRIVATE AssemblyPETSc.cpp)
  target_link_libraries(navier_var_kernels PUBLIC femx::system_petsc)
endif()

function(configure_navier_var_app target)
  target_link_libraries(${target} PRIVATE navier_var_kernels
                                          femx::inverse_petsc femx::femx)
  target_compile_definitions(${target}
                             PRIVATE FEMX_NAVIER_VAR_APP_NAME="${target}")
endfunction()

if(TARGET femx::inverse_petsc)
  if(FEMX_HAS_RESOLVE)
    add_executable(ns-var main-resolve.cpp)
    configure_navier_var_app(ns-var)
  endif()

  if(TARGET femx::system_petsc)
    add_executable(ns-var-petsc main-petsc.cpp)
    configure_navier_var_app(ns-var-petsc)
    target_link_libraries(ns-var-petsc PRIVATE femx::system_petsc)

    if(NOT TARGET ns-var)
      add_executable(ns-var main-petsc.cpp)
      configure_navier_var_app(ns-var)
      target_link_libraries(ns-var PRIVATE femx::system_petsc)
    endif()
  endif()
endif()
