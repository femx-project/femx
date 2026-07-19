find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED)

set(FEMX_PYBIND11_SOURCE_DIR
    ""
    CACHE PATH "Optional pybind11 source tree for Python bindings")

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
      "FEMX_BUILD_PYTHON=ON requires pybind11. "
      "Install it with `python3 -m pip install pybind11`, or pass "
      "-Dpybind11_DIR=$(python3 -m pybind11 --cmakedir), or pass "
      "-DFEMX_PYBIND11_SOURCE_DIR=/path/to/pybind11.")
endif()

pybind11_add_module(_core Inverse.cpp Mesh.cpp Module.cpp NavierStokes.cpp
                    State.cpp)

foreach(_target
        femx_linalg
        femx_linalg_petsc
        femx_linalg_resolve
        femx_state
        femx_inverse
        femx_fem
        femx_assembly
        femx_io
        femx_runtime
        femx_model_ns)
  if(TARGET "${_target}")
    set_target_properties("${_target}" PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
endforeach()

target_link_libraries(_core PRIVATE femx::fem femx::inverse femx::state
                                    femx::model::ns)

if(TARGET femx_linalg_resolve)
  target_link_libraries(_core PRIVATE femx::linalg_resolve)
endif()

if(TARGET femx_linalg_petsc)
  target_link_libraries(_core PRIVATE femx::linalg_petsc femx::opt)
endif()

set_target_properties(
  _core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/femx"
                   INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS _core LIBRARY DESTINATION femx COMPONENT python)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../femx/"
        DESTINATION femx
        COMPONENT python
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE)
