#include <stdexcept>

#include <femx/inverse/Objective.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/Linearization.hpp>
#include <femx/state/StateSolver.hpp>
using namespace femx::state;
using namespace femx::inverse;
using namespace femx::linalg;

namespace femx
{
namespace inverse
{

ReducedFunctional::ReducedFunctional(
    const Residual&  problem,
    const Objective& obj,
    StateSolver&     state_solver,
    Linearization&   lin,
    LinearSolver&    adj_lin_solver)
  : problem_(problem),
    obj_(obj),
    state_solver_(state_solver),
    lin_(lin),
    adj_lin_solver_(adj_lin_solver),
    dims_(problem.dims())
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
  return obj_.value(state, prm);
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
  const Real obj_val = obj_.value(state, prm);
  gradAt(state, prm, grad_out);
  return obj_val;
}

void ReducedFunctional::checkDims() const
{
  if (dims_.num_states != state_solver_.numStates()
      || dims_.num_params != state_solver_.numParams()
      || dims_.num_residuals != state_solver_.numResiduals()
      || dims_.num_states != obj_.numStates()
      || dims_.num_params != obj_.numParams()
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
  obj_.stateGrad(state, prm, state_grad);
  checkSize(state_grad, dims_.num_states);

  problem_.linearize(state, prm, lin_);

  Vector<Real> adj;
  adj_lin_solver_.solveT(lin_.stateJac(), state_grad, adj);
  checkSize(adj, dims_.num_residuals);

  Vector<Real> param_grad;
  obj_.paramGrad(state, prm, param_grad);

  Vector<Real> res_param_adj;
  lin_.paramJac().applyT(adj, res_param_adj);

  checkSize(param_grad, numParams());
  checkSize(res_param_adj, numParams());

  resizeOrZero(out, numParams());
  for (Index i = 0; i < numParams(); ++i)
  {
    out[i] = param_grad[i] - res_param_adj[i];
  }
}

void ReducedFunctional::checkSize(const Vector<Real>& value, Index exp)
{
  if (value.size() != exp)
  {
    throw std::runtime_error("ReducedFunctional vector size mismatch");
  }
}

} // namespace inverse
} // namespace femx
