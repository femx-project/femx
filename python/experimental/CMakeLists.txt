find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED)

set(FEMX_PYBIND11_SOURCE_DIR
    ""
    CACHE PATH "Optional pybind11 source tree for experimental bindings")

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _FEMX_PYBIND11_CMAKE_DIR
    RESULT_VARIABLE _FEMX_PYBIND11_CMAKE_DIR_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_FEMX_PYBIND11_CMAKE_DIR_RESULT EQUAL 0 AND _FEMX_PYBIND11_CMAKE_DIR)
    set(pybind11_DIR "${_FEMX_PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG QUIET)
  endif()
endif()

if(NOT pybind11_FOUND AND FEMX_PYBIND11_SOURCE_DIR)
  add_subdirectory("${FEMX_PYBIND11_SOURCE_DIR}"
                   "${CMAKE_CURRENT_BINARY_DIR}/pybind11")
  set(pybind11_FOUND TRUE)
endif()

if(NOT pybind11_FOUND)
  message(
    FATAL_ERROR
      "FEMX_BUILD_PYTHON_EXPERIMENTAL=ON requires pybind11. "
      "Install it with `python3 -m pip install pybind11`, or pass "
      "-Dpybind11_DIR=$(python3 -m pybind11 --cmakedir), or pass "
      "-DFEMX_PYBIND11_SOURCE_DIR=/path/to/pybind11.")
endif()

pybind11_add_module(femx_experimental Module.cpp)

foreach(_target femx_linalg femx_state femx_inverse femx_fem)
  if(TARGET "${_target}")
    set_target_properties("${_target}" PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
endforeach()

target_link_libraries(femx_experimental PRIVATE femx::fem)

set_target_properties(
  femx_experimental PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                               "${PROJECT_BINARY_DIR}/python/experimental")

install(TARGETS femx_experimental LIBRARY DESTINATION ".")
