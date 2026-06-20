#include <femx/solve/ReducedObjective.hpp>

namespace femx
{
namespace solve
{

Real ReducedObjective::valueGrad(const Vector<Real>& prm,
                                  Vector<Real>&       grad_out)
{
  const Real obj_val = value(prm);
  grad(prm, grad_out);
  return obj_val;
}

} // namespace solve
} // namespace femx
