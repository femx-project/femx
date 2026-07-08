#include <stdexcept>

#include <femx/state/LinearStateSolver.hpp>
using namespace femx::state;
using namespace femx::linalg;

namespace femx
{
namespace state
{

LinearStateSolver::LinearStateSolver(const Residual& problem,
                                     Linearization&  lin,
                                     LinearSolver&   lin_solver)
  : problem_(problem),
    linearization_(lin),
    lin_solver_(lin_solver),
    dims_(problem.dims())
{
  if (dims_.num_residuals != dims_.num_states)
  {
    throw std::runtime_error(
        "LinearStateSolver requires square state residual dimensions");
  }
}

Index LinearStateSolver::numStates() const
{
  return dims_.num_states;
}

Index LinearStateSolver::numParams() const
{
  return dims_.num_params;
}

Index LinearStateSolver::numResiduals() const
{
  return dims_.num_residuals;
}

void LinearStateSolver::solve(const Vector<Real>& prm,
                              Vector<Real>&       state)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error("LinearStateSolver parameter size mismatch");
  }

  Vector<Real> zero_state(numStates());
  Vector<Real> res;
  problem_.res(zero_state, prm, res);
  if (res.size() != numResiduals())
  {
    throw std::runtime_error("LinearStateSolver residual size mismatch");
  }

  Vector<Real> rhs(res.size());
  for (Index i = 0; i < res.size(); ++i)
  {
    rhs[i] = -res[i];
  }

  problem_.linearize(zero_state, prm, linearization_);
  lin_solver_.solve(linearization_.stateJac(), rhs, state);
  if (state.size() != numStates())
  {
    throw std::runtime_error("LinearStateSolver solution size mismatch");
  }
}

} // namespace state
} // namespace femx
