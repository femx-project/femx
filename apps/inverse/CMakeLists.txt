add_library(inverse_common Common.cpp)

target_sources(inverse_common PUBLIC Common.hpp PetscCommon.hpp PetscError.hpp)

target_include_directories(inverse_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(inverse_common PUBLIC femx::femx)

add_subdirectory(ns-var)
add_subdirectory(ns-en-var)
add_subdirectory(make-ensemble)
add_subdirectory(make-obs-ensemble)
add_subdirectory(make-obs)
