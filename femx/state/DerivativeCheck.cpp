#include <cmath>
#include <stdexcept>

#include <femx/common/Math.hpp>
#include <femx/state/DerivativeCheck.hpp>

using namespace std;
using namespace femx;
using namespace femx::problem;
using namespace femx::state;

namespace femx
{
namespace state
{

bool DerivativeCheckResult::passed(Real atol,
                                   Real rtol) const
{
  return abs_err <= atol || rel_err <= rtol;
}

DerivativeCheck::DerivativeCheck(Real step)
  : step_(step)
{
  if (step_ <= 0.0)
  {
    throw runtime_error("DerivativeCheck step must be positive");
  }
}

DerivativeCheckResult DerivativeCheck::objectiveStateGrad(
    const Objective&    obj,
    const Vector<Real>& state,
    const Vector<Real>& prm,
    const Vector<Real>& dir) const
{
  const Vector<Real> state_plus  = shifted(state, dir, step_);
  const Vector<Real> state_minus = shifted(state, dir, -step_);
  const Real         reference =
      (obj.value(state_plus, prm)
       - obj.value(state_minus, prm))
      / (2.0 * step_);

  Vector<Real> grad;
  obj.stateGrad(state, prm, grad);
  return compareScalars(dot(grad, dir), reference);
}

DerivativeCheckResult DerivativeCheck::objectiveParamGrad(
    const Objective&    obj,
    const Vector<Real>& state,
    const Vector<Real>& prm,
    const Vector<Real>& dir) const
{
  const Vector<Real> param_p = shifted(prm, dir, step_);
  const Vector<Real> param_m = shifted(prm, dir, -step_);
  const Real         reference =
      (obj.value(state, param_p) - obj.value(state, param_m))
      / (2.0 * step_);

  Vector<Real> grad;
  obj.paramGrad(state, prm, grad);
  return compareScalars(dot(grad, dir), reference);
}

DerivativeCheckResult DerivativeCheck::reducedGrad(ReducedFunctional&  fn,
                                                   const Vector<Real>& prm,
                                                   const Vector<Real>& dir) const
{
  const Vector<Real> param_p = shifted(prm, dir, step_);
  const Vector<Real> param_m = shifted(prm, dir, -step_);
  const Real         reference =
      (fn.value(param_p) - fn.value(param_m))
      / (2.0 * step_);

  Vector<Real> grad;
  fn.grad(prm, grad);
  return compareScalars(dot(grad, dir), reference);
}

DerivativeCheckResult DerivativeCheck::residualStateJacobian(
    const Residual&     problem,
    Linearization&      lin,
    const Vector<Real>& state,
    const Vector<Real>& prm,
    const Vector<Real>& dir) const
{
  const Vector<Real> state_plus  = shifted(state, dir, step_);
  const Vector<Real> state_minus = shifted(state, dir, -step_);

  Vector<Real> res_p;
  Vector<Real> res_m;
  problem.res(state_plus, prm, res_p);
  problem.res(state_minus, prm, res_m);
  const Vector<Real> reference = centralDifference(res_p, res_m);

  Vector<Real> applied;
  problem.linearize(state, prm, lin);
  lin.stateJac().apply(dir, applied);
  return compareVectors(applied, reference);
}

DerivativeCheckResult DerivativeCheck::residualParamJacobian(
    const Residual&     problem,
    Linearization&      lin,
    const Vector<Real>& state,
    const Vector<Real>& prm,
    const Vector<Real>& dir) const
{
  const Vector<Real> param_p = shifted(prm, dir, step_);
  const Vector<Real> param_m = shifted(prm, dir, -step_);

  Vector<Real> res_p;
  Vector<Real> res_m;
  problem.res(state, param_p, res_p);
  problem.res(state, param_m, res_m);
  const Vector<Real> reference = centralDifference(res_p, res_m);

  Vector<Real> applied;
  problem.linearize(state, prm, lin);
  lin.paramJac().apply(dir, applied);
  return compareVectors(applied, reference);
}

DerivativeCheckResult DerivativeCheck::stateJacTranspose(
    const Residual&     problem,
    Linearization&      lin,
    const Vector<Real>& state,
    const Vector<Real>& prm,
    const Vector<Real>& dir,
    const Vector<Real>& lambda) const
{
  problem.linearize(state, prm, lin);

  Vector<Real> jac_dir;
  Vector<Real> jt_lam;
  lin.stateJac().apply(dir, jac_dir);
  lin.stateJac().applyT(lambda, jt_lam);

  return compareScalars(dot(jac_dir, lambda), dot(dir, jt_lam));
}

DerivativeCheckResult DerivativeCheck::paramJacTranspose(
    const Residual&     problem,
    Linearization&      lin,
    const Vector<Real>& state,
    const Vector<Real>& prm,
    const Vector<Real>& dir,
    const Vector<Real>& lambda) const
{
  problem.linearize(state, prm, lin);

  Vector<Real> jac_dir;
  Vector<Real> jt_lam;
  lin.paramJac().apply(dir, jac_dir);
  lin.paramJac().applyT(lambda, jt_lam);

  return compareScalars(dot(jac_dir, lambda), dot(dir, jt_lam));
}

Vector<Real> DerivativeCheck::shifted(const Vector<Real>& base,
                                      const Vector<Real>& dir,
                                      Real                scale) const
{
  if (base.size() != dir.size())
  {
    throw runtime_error("DerivativeCheck vector size mismatch");
  }

  Vector<Real> out(base.size());
  for (Index i = 0; i < base.size(); ++i)
  {
    out[i] = base[i] + scale * dir[i];
  }
  return out;
}

Vector<Real> DerivativeCheck::centralDifference(
    const Vector<Real>& plus,
    const Vector<Real>& minus) const
{
  if (plus.size() != minus.size())
  {
    throw runtime_error("DerivativeCheck residual size mismatch");
  }

  Vector<Real> out(plus.size());
  for (Index i = 0; i < plus.size(); ++i)
  {
    out[i] = (plus[i] - minus[i]) / (2.0 * step_);
  }
  return out;
}

DerivativeCheckResult DerivativeCheck::compareScalars(Real computed,
                                                      Real reference)
{
  DerivativeCheckResult result;
  result.computed  = computed;
  result.reference = reference;
  result.abs_err   = abs(computed - reference);
  result.rel_err   = result.abs_err / (1.0 + abs(reference));
  return result;
}

DerivativeCheckResult DerivativeCheck::compareVectors(
    const Vector<Real>& computed,
    const Vector<Real>& reference)
{
  if (computed.size() != reference.size())
  {
    throw runtime_error("DerivativeCheck comparison size mismatch");
  }

  Vector<Real> error(computed.size());
  for (Index i = 0; i < computed.size(); ++i)
  {
    error[i] = computed[i] - reference[i];
  }

  DerivativeCheckResult result;
  result.computed  = norm(computed);
  result.reference = norm(reference);
  result.abs_err   = norm(error);
  result.rel_err   = result.abs_err / (1.0 + result.reference);
  return result;
}

} // namespace state
} // namespace femx
