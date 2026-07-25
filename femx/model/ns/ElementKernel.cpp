#include "ElementKernel.hpp"

#include <algorithm>

#include <femx/ad/Enzyme.hpp>
#include <femx/common/Checks.hpp>

namespace femx
{
namespace model
{
namespace ns
{

void histVjp(const HostElementKernel&             kernel,
             const assembly::HostTimeElementView& e,
             HostConstVectorView                  adj,
             HostVectorView                       out)
{
  const auto  data = kernel.data();
  const Index ncol = (data.dim() + 1) * data.numShapes();
  require(e.hist.size() == e.num_hist * ncol && e.nxt.size() == ncol
              && adj.size() == ncol && out.size() == e.hist.size(),
          "Navier history VJP element dimensions do not match");

#if defined(FEMX_HAS_ENZYME)
  std::fill(out.data(), out.data() + out.size(), 0.0);
  __enzyme_autodiff<void>(
      reinterpret_cast<void*>(detail::evalResAdj<MemorySpace::Host>),
      enzyme_const,
      data.numElems(),
      enzyme_const,
      data.numQuadraturePoints(),
      enzyme_const,
      data.numShapes(),
      enzyme_const,
      data.dim(),
      enzyme_const,
      data.NData(),
      enzyme_const,
      data.dNdxData(),
      enzyme_const,
      data.JxWData(),
      enzyme_const,
      kernel.fluid().rho,
      enzyme_const,
      kernel.fluid().mu,
      enzyme_const,
      kernel.dt(),
      enzyme_const,
      e.ie,
      enzyme_const,
      e.step,
      enzyme_const,
      e.num_hist,
      enzyme_dup,
      e.hist.data(),
      out.data(),
      enzyme_const,
      e.nxt.data(),
      enzyme_const,
      adj.data());
#else
  (void) kernel;
  (void) e;
  (void) adj;
  (void) out;
  throw std::runtime_error(
      "Navier history VJP requires Enzyme. Configure with "
      "-DFEMX_ENABLE_ENZYME=ON and provide Enzyme_DIR.");
#endif
}

} // namespace ns
} // namespace model
} // namespace femx
