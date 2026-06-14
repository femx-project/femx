#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/eq/StateSolver.hpp>
#include <femx/inverse/AdjointSolver.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Reduced objective using one state solve and one adjoint solve. */
class AdjointReducedFunctional : public ReducedFunctional
{
public:
  AdjointReducedFunctional(eq::StateSolver&            state_solver,
                           AdjointSolver&              adj_solver,
                           const eq::ResidualEquation& equation,
                           const ObjectiveFunctional&  objective)
    : state_solver_(state_solver),
      adjoint_solver_(adj_solver),
      eq_(equation),
      objective_(objective)
  {
    checkDimensions();
  }

  Index numParams() const override
  {
    return state_solver_.numParams();
  }

  Real value(const Vector& params) override
  {
    Vector state;
    state_solver_.solve(params, state);
    return objective_.value(state, params);
  }

  void grad(const Vector& params, Vector& out) override
  {
    Vector state;
    state_solver_.solve(params, state);
    gradientAtState(state, params, out);
  }

  Real valueGrad(const Vector& params, Vector& grad_out) override
  {
    Vector state;
    state_solver_.solve(params, state);
    const Real obj_val = objective_.value(state, params);
    gradientAtState(state, params, grad_out);
    return obj_val;
  }

private:
  void checkDimensions() const
  {
    if (state_solver_.numStates() != objective_.numStates()
        || state_solver_.numParams() != objective_.numParams()
        || state_solver_.numStates() != eq_.numStates()
        || state_solver_.numParams() != eq_.numParams()
        || state_solver_.numStates() != adjoint_solver_.numStates()
        || state_solver_.numParams() != adjoint_solver_.numParams()
        || eq_.numRes() != adjoint_solver_.numRes())
    {
      throw std::runtime_error(
          "AdjointReducedFunctional received inconsistent dimensions");
    }
  }

  void gradientAtState(const Vector& state,
                       const Vector& params,
                       Vector&       out) const
  {
    Vector state_grad;
    objective_.stateGrad(state, params, state_grad);

    Vector adjoint;
    adjoint_solver_.solve(state, params, state_grad, adjoint);

    Vector param_grad;
    objective_.paramGrad(state, params, param_grad);

    Vector res_param_adj;
    eq_.applyParamJacT(state, params, adjoint, res_param_adj);

    if (param_grad.size() != numParams()
        || res_param_adj.size() != numParams())
    {
      throw std::runtime_error(
          "AdjointReducedFunctional gradient size mismatch");
    }

    if (out.size() != numParams())
    {
      out.resize(numParams());
    }
    else
    {
      out.setZero();
    }

    for (Index i = 0; i < numParams(); ++i)
    {
      out[i] = param_grad[i] - res_param_adj[i];
    }
  }

private:
  eq::StateSolver&            state_solver_;
  AdjointSolver&              adjoint_solver_;
  const eq::ResidualEquation& eq_;
  const ObjectiveFunctional&  objective_;
};

} // namespace inverse
} // namespace femx
