#pragma once

#include <cmath>
#include <stdexcept>

#include <femx/core/Types.hpp>
#include <femx/equation/ResidualEquation.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

struct DerivativeCheckResult
{
  real_type computed  = 0.0;
  real_type reference = 0.0;
  real_type abs_err   = 0.0;
  real_type rel_err   = 0.0;

  bool passed(real_type atol,
              real_type rtol) const
  {
    return abs_err <= atol
           || rel_err <= rtol;
  }
};

class DerivativeCheck
{
public:
  explicit DerivativeCheck(real_type step = 1.0e-6)
    : step_(step)
  {
    if (step_ <= 0.0)
    {
      throw std::runtime_error("DerivativeCheck step must be positive");
    }
  }

  DerivativeCheckResult objStateGrad(const ObjectiveFunctional& objective,
                                     const Vector&              state,
                                     const Vector&              params,
                                     const Vector&              dir) const
  {
    const Vector    state_plus  = shifted(state, dir, step_);
    const Vector    state_minus = shifted(state, dir, -step_);
    const real_type reference =
        (objective.value(state_plus, params)
         - objective.value(state_minus, params))
        / (2.0 * step_);

    Vector grad;
    objective.stateGrad(state, params, grad);
    return compareScalars(dot(grad, dir), reference);
  }

  DerivativeCheckResult objParamGrad(const ObjectiveFunctional& objective,
                                     const Vector&              state,
                                     const Vector&              params,
                                     const Vector&              dir) const
  {
    const Vector    param_p = shifted(params, dir, step_);
    const Vector    param_m = shifted(params, dir, -step_);
    const real_type reference =
        (objective.value(state, param_p)
         - objective.value(state, param_m))
        / (2.0 * step_);

    Vector grad;
    objective.paramGrad(state, params, grad);
    return compareScalars(dot(grad, dir), reference);
  }

  DerivativeCheckResult reducedGrad(ReducedFunctional& functional,
                                    const Vector&      params,
                                    const Vector&      dir) const
  {
    const Vector    param_p = shifted(params, dir, step_);
    const Vector    param_m = shifted(params, dir, -step_);
    const real_type reference =
        (functional.value(param_p) - functional.value(param_m))
        / (2.0 * step_);

    Vector grad;
    functional.grad(params, grad);
    return compareScalars(dot(grad, dir), reference);
  }

  DerivativeCheckResult resStateJac(const equation::ResidualEquation& equation,
                                    const Vector&                     state,
                                    const Vector&                     params,
                                    const Vector&                     dir) const
  {
    const Vector state_plus  = shifted(state, dir, step_);
    const Vector state_minus = shifted(state, dir, -step_);

    Vector res_p;
    Vector res_m;
    equation.residual(state_plus, params, res_p);
    equation.residual(state_minus, params, res_m);
    const Vector reference = centralDifference(res_p, res_m);

    Vector applied;
    equation.applyStateJac(state, params, dir, applied);
    return compareVectors(applied, reference);
  }

  DerivativeCheckResult resParamJac(const equation::ResidualEquation& equation,
                                    const Vector&                     state,
                                    const Vector&                     params,
                                    const Vector&                     dir) const
  {
    const Vector param_p = shifted(params, dir, step_);
    const Vector param_m = shifted(params, dir, -step_);

    Vector res_p;
    Vector res_m;
    equation.residual(state, param_p, res_p);
    equation.residual(state, param_m, res_m);
    const Vector reference = centralDifference(res_p, res_m);

    Vector applied;
    equation.applyParamJac(state, params, dir, applied);
    return compareVectors(applied, reference);
  }

  DerivativeCheckResult stateJacT(const equation::ResidualEquation& equation,
                                  const Vector&                     state,
                                  const Vector&                     params,
                                  const Vector&                     dir,
                                  const Vector&                     lambda) const
  {
    Vector jac_dir;
    Vector jt_lam;
    equation.applyStateJac(state, params, dir, jac_dir);
    equation.applyStateJacT(state, params, lambda, jt_lam);

    return compareScalars(dot(jac_dir, lambda),
                          dot(dir, jt_lam));
  }

  DerivativeCheckResult paramJacT(const equation::ResidualEquation& equation,
                                  const Vector&                     state,
                                  const Vector&                     params,
                                  const Vector&                     dir,
                                  const Vector&                     lambda) const
  {
    Vector jac_dir;
    Vector jt_lam;
    equation.applyParamJac(state, params, dir, jac_dir);
    equation.applyParamJacT(state, params, lambda, jt_lam);

    return compareScalars(dot(jac_dir, lambda),
                          dot(dir, jt_lam));
  }

private:
  Vector shifted(const Vector& base,
                 const Vector& dir,
                 real_type     scale) const
  {
    if (base.size() != dir.size())
    {
      throw std::runtime_error("DerivativeCheck vector size mismatch");
    }

    Vector out(base.size());
    for (index_type i = 0; i < base.size(); ++i)
    {
      out[i] = base[i] + scale * dir[i];
    }
    return out;
  }

  Vector centralDifference(const Vector& plus,
                           const Vector& minus) const
  {
    if (plus.size() != minus.size())
    {
      throw std::runtime_error("DerivativeCheck residual size mismatch");
    }

    Vector out(plus.size());
    for (index_type i = 0; i < plus.size(); ++i)
    {
      out[i] = (plus[i] - minus[i]) / (2.0 * step_);
    }
    return out;
  }

  static real_type dot(const Vector& x, const Vector& y)
  {
    if (x.size() != y.size())
    {
      throw std::runtime_error("DerivativeCheck dot size mismatch");
    }

    real_type value = 0.0;
    for (index_type i = 0; i < x.size(); ++i)
    {
      value += x[i] * y[i];
    }
    return value;
  }

  static real_type norm(const Vector& x)
  {
    return std::sqrt(dot(x, x));
  }

  static DerivativeCheckResult compareScalars(real_type computed,
                                              real_type reference)
  {
    DerivativeCheckResult result;
    result.computed  = computed;
    result.reference = reference;
    result.abs_err   = std::abs(computed - reference);
    result.rel_err =
        result.abs_err / (1.0 + std::abs(reference));
    return result;
  }

  static DerivativeCheckResult compareVectors(const Vector& computed,
                                              const Vector& reference)
  {
    if (computed.size() != reference.size())
    {
      throw std::runtime_error("DerivativeCheck comparison size mismatch");
    }

    Vector error(computed.size());
    for (index_type i = 0; i < computed.size(); ++i)
    {
      error[i] = computed[i] - reference[i];
    }

    DerivativeCheckResult result;
    result.computed  = norm(computed);
    result.reference = norm(reference);
    result.abs_err   = norm(error);
    result.rel_err =
        result.abs_err / (1.0 + result.reference);
    return result;
  }

private:
  real_type step_{1.0e-6};
};

} // namespace inverse
} // namespace femx
