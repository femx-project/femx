#include <stdexcept>

#include <femx/solve/ReducedFunctional.hpp>

namespace femx
{
namespace solve
{

ReducedFunctional::ReducedFunctional(
    const problem::Residual&  problem,
    const problem::Objective& objective,
    Newton&                   state_solver,
    algebra::LinearSolver&    adjoint_linear_solver)
  : problem_(problem),
    objective_(objective),
    state_solver_(state_solver),
    adjoint_linear_solver_(adjoint_linear_solver),
    dims_(problem.dimensions())
{
  checkDims();
}

Index ReducedFunctional::numParams() const
{
  return dims_.num_params;
}

Real ReducedFunctional::value(const Vector<Real>& prm)
{
  Vector<Real> state;
  state_solver_.solve(prm, state);
  return objective_.value(state, prm);
}

void ReducedFunctional::grad(const Vector<Real>& prm,
                             Vector<Real>&       out)
{
  Vector<Real> state;
  state_solver_.solve(prm, state);
  gradAt(state, prm, out);
}

Real ReducedFunctional::valueGrad(const Vector<Real>& prm,
                                  Vector<Real>&       grad_out)
{
  Vector<Real> state;
  state_solver_.solve(prm, state);
  const Real obj_val = objective_.value(state, prm);
  gradAt(state, prm, grad_out);
  return obj_val;
}

void ReducedFunctional::checkDims() const
{
  if (dims_.num_states != state_solver_.numStates()
      || dims_.num_params != state_solver_.numParams()
      || dims_.num_residuals != state_solver_.numResiduals()
      || dims_.num_states != objective_.numStates()
      || dims_.num_params != objective_.numParams()
      || dims_.num_residuals != dims_.num_states)
  {
    throw std::runtime_error(
        "ReducedFunctional received inconsistent dimensions");
  }
}

void ReducedFunctional::gradAt(const Vector<Real>& state,
                               const Vector<Real>& prm,
                               Vector<Real>&       out)
{
  Vector<Real> state_grad;
  objective_.stateGrad(state, prm, state_grad);
  if (state_grad.size() != dims_.num_states)
  {
    throw std::runtime_error("ReducedFunctional state gradient size mismatch");
  }

  problem_.linearize(state, prm, state_solver_.linearization());
  const problem::Linearization& linearization = state_solver_.linearization();

  Vector<Real> adjoint;
  adjoint_linear_solver_.solveT(
      linearization.stateJacobian(), state_grad, adjoint);
  if (adjoint.size() != dims_.num_residuals)
  {
    throw std::runtime_error("ReducedFunctional adjoint size mismatch");
  }

  Vector<Real> param_grad;
  objective_.paramGrad(state, prm, param_grad);

  Vector<Real> res_param_adj;
  linearization.paramJacobian().applyT(adjoint, res_param_adj);

  if (param_grad.size() != numParams() || res_param_adj.size() != numParams())
  {
    throw std::runtime_error("ReducedFunctional gradient size mismatch");
  }

  resize(out, numParams());
  for (Index i = 0; i < numParams(); ++i)
  {
    out[i] = param_grad[i] - res_param_adj[i];
  }
}

void ReducedFunctional::resize(Vector<Real>& out,
                               Index         size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

} // namespace solve
} // namespace femx
