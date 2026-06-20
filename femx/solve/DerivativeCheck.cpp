#include <cmath>
#include <stdexcept>

#include <femx/core/Math.hpp>
#include <femx/solve/DerivativeCheck.hpp>

using namespace femx::solve;
using namespace femx::problem;

namespace femx
{
namespace solve
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
    throw std::runtime_error("DerivativeCheck step must be positive");
  }
}

DerivativeCheckResult DerivativeCheck::objStateGrad(
    const ObjectiveFunctional& obj,
    const Vector<Real>&        state,
    const Vector<Real>&        prm,
    const Vector<Real>&        dir) const
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

DerivativeCheckResult DerivativeCheck::objParamGrad(
    const ObjectiveFunctional& obj,
    const Vector<Real>&        state,
    const Vector<Real>&        prm,
    const Vector<Real>&        dir) const
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

DerivativeCheckResult DerivativeCheck::reducedGrad(ReducedObjective&  functional,
    const Vector<Real>& prm,
    const Vector<Real>& dir) const
{
  const Vector<Real> param_p = shifted(prm, dir, step_);
  const Vector<Real> param_m = shifted(prm, dir, -step_);
  const Real         reference =
      (functional.value(param_p) - functional.value(param_m))
      / (2.0 * step_);

  Vector<Real> grad;
  functional.grad(prm, grad);
  return compareScalars(dot(grad, dir), reference);
}

DerivativeCheckResult DerivativeCheck::resStateJac(
    const ResidualEquation& eq,
    const Vector<Real>&     state,
    const Vector<Real>&     prm,
    const Vector<Real>&     dir) const
{
  const Vector<Real> state_plus  = shifted(state, dir, step_);
  const Vector<Real> state_minus = shifted(state, dir, -step_);

  Vector<Real> res_p;
  Vector<Real> res_m;
  eq.res(state_plus, prm, res_p);
  eq.res(state_minus, prm, res_m);
  const Vector<Real> reference = centralDifference(res_p, res_m);

  Vector<Real> applied;
  eq.applyStateJac(state, prm, dir, applied);
  return compareVectors(applied, reference);
}

DerivativeCheckResult DerivativeCheck::resParamJac(
    const ResidualEquation& eq,
    const Vector<Real>&     state,
    const Vector<Real>&     prm,
    const Vector<Real>&     dir) const
{
  const Vector<Real> param_p = shifted(prm, dir, step_);
  const Vector<Real> param_m = shifted(prm, dir, -step_);

  Vector<Real> res_p;
  Vector<Real> res_m;
  eq.res(state, param_p, res_p);
  eq.res(state, param_m, res_m);
  const Vector<Real> reference = centralDifference(res_p, res_m);

  Vector<Real> applied;
  eq.applyParamJac(state, prm, dir, applied);
  return compareVectors(applied, reference);
}

DerivativeCheckResult DerivativeCheck::stateJacT(
    const ResidualEquation& eq,
    const Vector<Real>&     state,
    const Vector<Real>&     prm,
    const Vector<Real>&     dir,
    const Vector<Real>&     lambda) const
{
  Vector<Real> jac_dir;
  Vector<Real> jt_lam;
  eq.applyStateJac(state, prm, dir, jac_dir);
  eq.applyStateJacT(state, prm, lambda, jt_lam);

  return compareScalars(dot(jac_dir, lambda), dot(dir, jt_lam));
}

DerivativeCheckResult DerivativeCheck::paramJacT(
    const ResidualEquation& eq,
    const Vector<Real>&     state,
    const Vector<Real>&     prm,
    const Vector<Real>&     dir,
    const Vector<Real>&     lambda) const
{
  Vector<Real> jac_dir;
  Vector<Real> jt_lam;
  eq.applyParamJac(state, prm, dir, jac_dir);
  eq.applyParamJacT(state, prm, lambda, jt_lam);

  return compareScalars(dot(jac_dir, lambda), dot(dir, jt_lam));
}

Vector<Real> DerivativeCheck::shifted(const Vector<Real>& base,
                                      const Vector<Real>& dir,
                                      Real                scale) const
{
  if (base.size() != dir.size())
  {
    throw std::runtime_error("DerivativeCheck vector size mismatch");
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
    throw std::runtime_error("DerivativeCheck residual size mismatch");
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
  result.abs_err   = std::abs(computed - reference);
  result.rel_err   = result.abs_err / (1.0 + std::abs(reference));
  return result;
}

DerivativeCheckResult DerivativeCheck::compareVectors(
    const Vector<Real>& computed,
    const Vector<Real>& reference)
{
  if (computed.size() != reference.size())
  {
    throw std::runtime_error("DerivativeCheck comparison size mismatch");
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

} // namespace solve
} // namespace femx
