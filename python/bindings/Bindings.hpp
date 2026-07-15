#pragma once

#include <pybind11/pybind11.h>

void bindMesh(pybind11::module_& module);
void bindInverse(pybind11::module_& module);
void bindNavierStokes(pybind11::module_& module);
void bindState(pybind11::module_& module);
