#include "Bindings.hpp"
#include <femx/linalg/cuda/CudaContext.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, module)
{
  module.doc()               = "Native Python bindings for the femx finite-element engine.";
  module.attr("__version__") = "0.4.0";
#if defined(FEMX_RESOLVE_USE_CUDA)
  module.attr("_resolve_uses_cuda") = true;
#else
  module.attr("_resolve_uses_cuda") = false;
#endif
#if defined(FEMX_HAS_RESOLVE)
  module.attr("_has_resolve") = true;
#else
  module.attr("_has_resolve") = false;
#endif
#if defined(FEMX_HAS_PETSC)
  module.attr("_has_petsc") = true;
#else
  module.attr("_has_petsc") = false;
#endif
#if defined(FEMX_HAS_CUDA)
  module.attr("_cuda_available") =
      femx::linalg::CudaContext::available();
#else
  module.attr("_cuda_available") = false;
#endif
#if defined(FEMX_HAS_ENZYME)
  module.attr("_has_enzyme") = true;
#else
  module.attr("_has_enzyme") = false;
#endif
  bindMesh(module);
  bindState(module);
  bindInverse(module);
  bindNavier(module);
}
