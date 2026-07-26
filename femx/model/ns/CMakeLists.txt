add_library(femx_model_ns)
add_library(femx::model_ns ALIAS femx_model_ns)
add_library(femx::model::ns ALIAS femx_model_ns)

target_sources(
  femx_model_ns
  PRIVATE ElementKernel.cpp Model.cpp NavierStokesResidual.cpp StateFields.cpp
  PUBLIC ElementKernel.hpp FluidProperties.hpp Model.hpp
         NavierStokesResidual.hpp StateFields.hpp)

target_link_libraries(femx_model_ns PUBLIC femx_config femx_common femx_linalg
                                           femx_fem femx_assembly femx_state)

if(TARGET femx_linalg_petsc)
  target_link_libraries(femx_model_ns PUBLIC femx_linalg_petsc)
endif()

if(FEMX_HAS_CUDA)
  target_sources(femx_model_ns PRIVATE ElementKernel.cu
                                       NavierStokesResidual.cu)
  set_target_properties(femx_model_ns PROPERTIES CUDA_STANDARD 17
                                                 CUDA_STANDARD_REQUIRED ON)
endif()

if(FEMX_HAS_ENZYME)
  # Enzyme's reverse lowering relies on the scalarized local work arrays. Keep
  # these two translation units optimized in Debug/default builds too.
  set_property(
    SOURCE ElementKernel.cpp
    APPEND
    PROPERTY COMPILE_OPTIONS -O2)
  if(FEMX_HAS_CUDA)
    set_property(
      SOURCE ElementKernel.cu
      APPEND
      PROPERTY COMPILE_OPTIONS -O2)
  endif()
endif()
