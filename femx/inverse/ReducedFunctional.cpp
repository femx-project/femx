#include <femx/inverse/ReducedFunctional.hpp>

namespace femx
{
namespace inverse
{

Real ReducedFunctional::valueGrad(const Vector<Real>& prm,
                                  Vector<Real>&       grad_out)
{
  const Real obj_val = value(prm);
  grad(prm, grad_out);
  return obj_val;
}

} // namespace inverse
} // namespace femx
