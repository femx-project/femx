#include "Bindings.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, module)
{
  module.doc()               = "Native Python bindings for the femx finite-element engine.";
  module.attr("__version__") = "0.2.0";
  bindMesh(module);
  bindState(module);
  bindInverse(module);
  bindNavierStokes(module);
}
