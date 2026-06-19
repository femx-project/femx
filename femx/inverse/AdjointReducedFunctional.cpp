#include <stdexcept>

#include <femx/inverse/AdjointReducedFunctional.hpp>

using namespace femx::eq;

namespace femx
{
namespace inverse
{

AdjointReducedFunctional::AdjointReducedFunctional(
    StateSolver&            state_solver,
    AdjointSolver&              adj_solver,
    const ResidualEquation& eq,
    const ObjectiveFunctional&  obj)
  : state_solver_(state_solver),
    adj_solver_(adj_solver),
    eq_(eq),
    obj_(obj)
{
  checkDims();
}

Index AdjointReducedFunctional::numParams() const
{
  return state_solver_.numParams();
}

Real AdjointReducedFunctional::value(const Vector<Real>& prm)
{
  Vector<Real> state;
  state_solver_.solve(prm, state);
  return obj_.value(state, prm);
}

void AdjointReducedFunctional::grad(const Vector<Real>& prm,
                                    Vector<Real>&       out)
{
  Vector<Real> state;
  state_solver_.solve(prm, state);
  gradAt(state, prm, out);
}

Real AdjointReducedFunctional::valueGrad(const Vector<Real>& prm,
                                         Vector<Real>&       grad_out)
{
  Vector<Real> state;
  state_solver_.solve(prm, state);
  const Real obj_val = obj_.value(state, prm);
  gradAt(state, prm, grad_out);
  return obj_val;
}

void AdjointReducedFunctional::checkDims() const
{
  if (state_solver_.numStates() != obj_.numStates()
      || state_solver_.numParams() != obj_.numParams()
      || state_solver_.numStates() != eq_.numStates()
      || state_solver_.numParams() != eq_.numParams()
      || state_solver_.numStates() != adj_solver_.numStates()
      || state_solver_.numParams() != adj_solver_.numParams()
      || eq_.numRes() != adj_solver_.numRes())
  {
    throw std::runtime_error(
        "AdjointReducedFunctional received inconsistent dimensions");
  }
}

void AdjointReducedFunctional::gradAt(
    const Vector<Real>& state,
    const Vector<Real>& prm,
    Vector<Real>&       out) const
{
  Vector<Real> state_grad;
  obj_.stateGrad(state, prm, state_grad);

  Vector<Real> adjoint;
  adj_solver_.solve(state, prm, state_grad, adjoint);

  Vector<Real> param_grad;
  obj_.paramGrad(state, prm, param_grad);

  Vector<Real> res_param_adj;
  eq_.applyParamJacT(state, prm, adjoint, res_param_adj);

  if (param_grad.size() != numParams() || res_param_adj.size() != numParams())
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

} // namespace inverse
} // namespace femx
