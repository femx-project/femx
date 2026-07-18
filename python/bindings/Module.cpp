#include "Bindings.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, module)
{
  module.doc()               = "Native Python bindings for the femx finite-element engine.";
  module.attr("__version__") = "0.2.0";
#if defined(FEMX_RESOLVE_USE_CUDA)
  module.attr("_resolve_uses_cuda") = true;
#else
  module.attr("_resolve_uses_cuda") = false;
#endif
  bindMesh(module);
  bindState(module);
  bindInverse(module);
  bindNavierStokes(module);
}
