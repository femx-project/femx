add_library(femx_inverse)
add_library(femx::inverse ALIAS femx_inverse)

target_sources(
  femx_inverse
  PRIVATE LeastSquaresObjective.cpp
          SumObjective.cpp
          SumTimeObjective.cpp
          TimeBlockRegularization.cpp
          TimeLeastSquaresObjective.cpp
          TimeObservationData.cpp
          TimeRegularization.cpp
  PUBLIC LeastSquaresObjective.hpp
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
