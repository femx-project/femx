add_library(navier_common Components.cpp NavierHelper.cpp NavierKernel.cpp)

if(FEMX_ENABLE_ENZYME)
  set_target_properties(navier_common PROPERTIES UNITY_BUILD ON)
endif()

target_sources(navier_common PUBLIC Components.hpp NavierConfig.hpp
                                     NavierHelper.hpp NavierKernel.hpp)

target_include_directories(navier_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(navier_common PUBLIC femx::femx)
