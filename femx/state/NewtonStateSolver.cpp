#include <stdexcept>

#include <femx/common/Math.hpp>
#include <femx/state/NewtonStateSolver.hpp>

using namespace std;
using namespace femx::problem;
using namespace femx::linalg;

namespace femx
{
namespace state
{

NewtonStateSolver::NewtonStateSolver(const Residual& problem,
                                     Linearization&  lin,
                                     LinearSolver&   lin_solver)
  : problem_(problem),
    linearization_(lin),
    lin_solver_(lin_solver),
    dims_(problem.dims())
{
  if (dims_.num_residuals != dims_.num_states)
  {
    throw runtime_error(
        "NewtonStateSolver requires square state residual dimensions");
  }
}

NewtonStateOptions& NewtonStateSolver::opts()
{
  return opts_;
}

const NewtonStateOptions& NewtonStateSolver::opts() const
{
  return opts_;
}

void NewtonStateSolver::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw runtime_error("NewtonStateSolver initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void NewtonStateSolver::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

Index NewtonStateSolver::numStates() const
{
  return dims_.num_states;
}

Index NewtonStateSolver::numParams() const
{
  return dims_.num_params;
}

Index NewtonStateSolver::numResiduals() const
{
  return dims_.num_residuals;
}

void NewtonStateSolver::solve(const Vector<Real>& prm,
                              Vector<Real>&       state)
{
  if (prm.size() != numParams())
  {
    throw runtime_error("NewtonStateSolver parameter size mismatch");
  }

  initializeState(state);

  Vector<Real> res;
  Vector<Real> rhs;
  Vector<Real> step;
  for (Index i = 0; i <= opts_.max_its; ++i)
  {
    problem_.res(state, prm, res);
    if (res.size() != numResiduals())
    {
      throw runtime_error("NewtonStateSolver residual size mismatch");
    }

    if (squaredNorm(res)
        <= opts_.residual_tolerance * opts_.residual_tolerance)
    {
      return;
    }
    if (i == opts_.max_its)
    {
      break;
    }

    resizeOrZero(rhs, res.size());
    for (Index k = 0; k < res.size(); ++k)
    {
      rhs[k] = -res[k];
    }

    problem_.linearize(state, prm, linearization_);
    lin_solver_.solve(linearization_.stateJac(), rhs, step);
    if (step.size() != numStates())
    {
      throw runtime_error("NewtonStateSolver step size mismatch");
    }

    for (Index k = 0; k < numStates(); ++k)
    {
      state[k] += step[k];
    }

    if (squaredNorm(step) <= opts_.step_tolerance * opts_.step_tolerance)
    {
      return;
    }
  }

  throw runtime_error("NewtonStateSolver failed to converge");
}

void NewtonStateSolver::initializeState(Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }
  resizeOrZero(state, numStates());
}

} // namespace state
} // namespace femx
