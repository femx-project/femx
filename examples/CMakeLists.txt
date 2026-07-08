add_subdirectory(poisson)
add_subdirectory(poisson-opt)

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/inverse/CMakeLists.txt")
  add_subdirectory(inverse)
endif()
