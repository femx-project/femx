add_library(femx_inverse)
add_library(femx::inverse ALIAS femx_inverse)

target_sources(
  femx_inverse
  PRIVATE DeviceTimeObjective.cpp
          LeastSquaresObjective.cpp
          ReducedFunctional.cpp
          SumObjective.cpp
          SumTimeObjective.cpp
          TimeBlockRegularization.cpp
          TimeLeastSquaresObjective.cpp
          TimeObservationData.cpp
          TimeReducedFunctional.cpp
          TimeRegularization.cpp
  PUBLIC DeviceTimeObjective.hpp
         LeastSquaresObjective.hpp
         Objective.hpp
         ReducedFunctional.hpp
         SumObjective.hpp
         SumTimeObjective.hpp
         TimeBlockRegularization.hpp
         TimeLeastSquaresObjective.hpp
         TimeObjective.hpp
         TimeObservationData.hpp
         TimeObservationOperator.hpp
         TimeReducedFunctional.hpp
         TimeRegularization.hpp)

target_link_libraries(femx_inverse PUBLIC femx_config femx_common femx_linalg
                                          femx_state)

if(FEMX_HAS_CUDA)
  target_sources(femx_inverse PRIVATE DeviceTimeObjective.cu)
  set_target_properties(femx_inverse PROPERTIES CUDA_STANDARD 17
                                                CUDA_STANDARD_REQUIRED ON)
endif()
