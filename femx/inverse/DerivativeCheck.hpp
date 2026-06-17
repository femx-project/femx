#pragma once

#include <cmath>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

struct DerivativeCheckResult
{
  Real computed  = 0.0;
  Real reference = 0.0;
  Real abs_err   = 0.0;
  Real rel_err   = 0.0;

  bool passed(Real atol,
              Real rtol) const
  {
    return abs_err <= atol
           || rel_err <= rtol;
  }
};

class DerivativeCheck
{
public:
  explicit DerivativeCheck(Real step = 1.0e-6)
    : step_(step)
  {
    if (step_ <= 0.0)
    {
      throw std::runtime_error("DerivativeCheck step must be positive");
    }
  }

  DerivativeCheckResult objStateGrad(const ObjectiveFunctional& objective,
                                     const Vector<Real>&        state,
                                     const Vector<Real>&        params,
                                     const Vector<Real>&        dir) const
  {
    const Vector<Real> state_plus  = shifted(state, dir, step_);
    const Vector<Real> state_minus = shifted(state, dir, -step_);
    const Real         reference =
        (objective.value(state_plus, params)
         - objective.value(state_minus, params))
        / (2.0 * step_);

    Vector<Real> grad;
    objective.stateGrad(state, params, grad);
    return compareScalars(dot(grad, dir), reference);
  }

  DerivativeCheckResult objParamGrad(const ObjectiveFunctional& objective,
                                     const Vector<Real>&        state,
                                     const Vector<Real>&        params,
                                     const Vector<Real>&        dir) const
  {
    const Vector<Real> param_p = shifted(params, dir, step_);
    const Vector<Real> param_m = shifted(params, dir, -step_);
    const Real         reference =
        (objective.value(state, param_p)
         - objective.value(state, param_m))
        / (2.0 * step_);

    Vector<Real> grad;
    objective.paramGrad(state, params, grad);
    return compareScalars(dot(grad, dir), reference);
  }

  DerivativeCheckResult reducedGrad(ReducedFunctional&  functional,
                                    const Vector<Real>& params,
                                    const Vector<Real>& dir) const
  {
    const Vector<Real> param_p = shifted(params, dir, step_);
    const Vector<Real> param_m = shifted(params, dir, -step_);
    const Real         reference =
        (functional.value(param_p) - functional.value(param_m))
        / (2.0 * step_);

    Vector<Real> grad;
    functional.grad(params, grad);
    return compareScalars(dot(grad, dir), reference);
  }

  DerivativeCheckResult resStateJac(const eq::ResidualEquation& equation,
                                    const Vector<Real>&         state,
                                    const Vector<Real>&         params,
                                    const Vector<Real>&         dir) const
  {
    const Vector<Real> state_plus  = shifted(state, dir, step_);
    const Vector<Real> state_minus = shifted(state, dir, -step_);

    Vector<Real> res_p;
    Vector<Real> res_m;
    equation.res(state_plus, params, res_p);
    equation.res(state_minus, params, res_m);
    const Vector<Real> reference = centralDifference(res_p, res_m);

    Vector<Real> applied;
    equation.applyStateJac(state, params, dir, applied);
    return compareVectors(applied, reference);
  }

  DerivativeCheckResult resParamJac(const eq::ResidualEquation& equation,
                                    const Vector<Real>&         state,
                                    const Vector<Real>&         params,
                                    const Vector<Real>&         dir) const
  {
    const Vector<Real> param_p = shifted(params, dir, step_);
    const Vector<Real> param_m = shifted(params, dir, -step_);

    Vector<Real> res_p;
    Vector<Real> res_m;
    equation.res(state, param_p, res_p);
    equation.res(state, param_m, res_m);
    const Vector<Real> reference = centralDifference(res_p, res_m);

    Vector<Real> applied;
    equation.applyParamJac(state, params, dir, applied);
    return compareVectors(applied, reference);
  }

  DerivativeCheckResult stateJacT(const eq::ResidualEquation& equation,
                                  const Vector<Real>&         state,
                                  const Vector<Real>&         params,
                                  const Vector<Real>&         dir,
                                  const Vector<Real>&         lambda) const
  {
    Vector<Real> jac_dir;
    Vector<Real> jt_lam;
    equation.applyStateJac(state, params, dir, jac_dir);
    equation.applyStateJacT(state, params, lambda, jt_lam);

    return compareScalars(dot(jac_dir, lambda),
                          dot(dir, jt_lam));
  }

  DerivativeCheckResult paramJacT(const eq::ResidualEquation& equation,
                                  const Vector<Real>&         state,
                                  const Vector<Real>&         params,
                                  const Vector<Real>&         dir,
                                  const Vector<Real>&         lambda) const
  {
    Vector<Real> jac_dir;
    Vector<Real> jt_lam;
    equation.applyParamJac(state, params, dir, jac_dir);
    equation.applyParamJacT(state, params, lambda, jt_lam);

    return compareScalars(dot(jac_dir, lambda),
                          dot(dir, jt_lam));
  }

private:
  Vector<Real> shifted(const Vector<Real>& base,
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

  Vector<Real> centralDifference(const Vector<Real>& plus,
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

  static Real dot(const Vector<Real>& x, const Vector<Real>& y)
  {
    if (x.size() != y.size())
    {
      throw std::runtime_error("DerivativeCheck dot size mismatch");
    }

    Real value = 0.0;
    for (Index i = 0; i < x.size(); ++i)
    {
      value += x[i] * y[i];
    }
    return value;
  }

  static Real norm(const Vector<Real>& x)
  {
    return std::sqrt(dot(x, x));
  }

  static DerivativeCheckResult compareScalars(Real computed,
                                              Real reference)
  {
    DerivativeCheckResult result;
    result.computed  = computed;
    result.reference = reference;
    result.abs_err   = std::abs(computed - reference);
    result.rel_err   = result.abs_err / (1.0 + std::abs(reference));
    return result;
  }

  static DerivativeCheckResult compareVectors(const Vector<Real>& computed,
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

private:
  Real step_{1.0e-6};
};

} // namespace inverse
} // namespace femx
