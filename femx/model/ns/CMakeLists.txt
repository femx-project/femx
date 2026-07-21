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
          NavierStokesModel.cpp
  PUBLIC Boundary.hpp
         Components.hpp
         Config.hpp
         ForwardConfig.hpp
         ForwardProblem.hpp
         ForwardSolveMonitor.hpp
         Helper.hpp
         NavierStokesModel.hpp)

target_include_directories(femx_model_ns
                           PRIVATE ${FEMX_NLOHMANN_JSON_INCLUDE_DIR})

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

if(TARGET femx_linalg_petsc)
  target_link_libraries(femx_model_ns PUBLIC femx_linalg_petsc)
endif()

if(FEMX_HAS_CUDA)
  target_sources(femx_model_ns PRIVATE Components.cu)
  set_target_properties(femx_model_ns PROPERTIES CUDA_STANDARD 17
                                                 CUDA_STANDARD_REQUIRED ON)
endif()

if(FEMX_HAS_ENZYME)
  # Enzyme's reverse lowering relies on the scalarized local work arrays. Keep
  # these two translation units optimized in Debug/default builds too.
  set_property(
    SOURCE Components.cpp
    APPEND
    PROPERTY COMPILE_OPTIONS -O2)
  if(FEMX_HAS_CUDA)
    set_property(
      SOURCE Components.cu
      APPEND
      PROPERTY COMPILE_OPTIONS -O2)
  endif()
endif()
