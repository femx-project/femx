add_library(femx_model_ns)
add_library(femx::model_ns ALIAS femx_model_ns)
add_library(femx::model::ns ALIAS femx_model_ns)

target_sources(
  femx_model_ns
  PRIVATE Boundary.cpp
          Components.cpp
          ForwardConfig.cpp
          ForwardProblem.cpp
          ForwardSolveMonitor.cpp
          Helper.cpp
          Kernel.cpp
          NavierStokesModel.cpp
  PUBLIC Boundary.hpp
         Components.hpp
         Config.hpp
         ForwardConfig.hpp
         ForwardProblem.hpp
         ForwardSolveMonitor.hpp
         Helper.hpp
         Kernel.hpp
         NavierStokesModel.hpp)

target_include_directories(femx_model_ns
                           PRIVATE ${FEMX_NLOHMANN_JSON_INCLUDE_DIR})

if(FEMX_ENABLE_ENZYME)
  set_target_properties(femx_model_ns PROPERTIES UNITY_BUILD ON)
endif()

target_link_libraries(
  femx_model_ns
  PUBLIC femx_config
         femx_common
         femx_linalg
         femx_fem
         femx_assembly
         femx_io
         femx_runtime
         femx_state)
