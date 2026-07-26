add_library(femx_model_navier)
add_library(femx::model_navier ALIAS femx_model_navier)
add_library(femx::model::navier ALIAS femx_model_navier)

target_sources(
  femx_model_navier
  PRIVATE NavierElementKernel.cpp NavierModel.cpp NavierResidual.cpp StateFields.cpp
  PUBLIC NavierElementKernel.hpp FluidProperties.hpp NavierModel.hpp
         NavierResidual.hpp StateFields.hpp)

target_link_libraries(femx_model_navier PUBLIC femx_config femx_common femx_linalg
                                           femx_fem femx_assembly femx_state)

if(TARGET femx_linalg_petsc)
  target_link_libraries(femx_model_navier PUBLIC femx_linalg_petsc)
endif()

if(FEMX_HAS_CUDA)
  target_sources(femx_model_navier PRIVATE NavierElementKernel.cu
                                       NavierResidual.cu)
  set_target_properties(femx_model_navier PROPERTIES CUDA_STANDARD 17
                                                 CUDA_STANDARD_REQUIRED ON)
endif()

if(FEMX_HAS_ENZYME)
  # Enzyme's reverse lowering relies on the scalarized local work arrays. Keep
  # these two translation units optimized in Debug/default builds too.
  set_property(
    SOURCE NavierElementKernel.cpp
    APPEND
    PROPERTY COMPILE_OPTIONS -O2)
  if(FEMX_HAS_CUDA)
    set_property(
      SOURCE NavierElementKernel.cu
      APPEND
      PROPERTY COMPILE_OPTIONS -O2)
  endif()
endif()
