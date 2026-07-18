#include <stdexcept>

#include <femx/common/Math.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/state/Linearization.hpp>
#include <femx/state/NewtonStateSolver.hpp>
using namespace femx::state;
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
  if (dims_.num_res != dims_.num_states)
  {
    throw std::runtime_error(
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

void NewtonStateSolver::setInitialState(const HostVector& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error("NewtonStateSolver initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void NewtonStateSolver::clearInitialState()
{
  init_state_     = HostVector{};
  has_init_state_ = false;
}

Index NewtonStateSolver::numStates() const
{
  return dims_.num_states;
}

Index NewtonStateSolver::numParams() const
{
  return dims_.num_param;
}

Index NewtonStateSolver::numRes() const
{
  return dims_.num_res;
}

void NewtonStateSolver::solve(const HostVector& prm,
                              HostVector&       state)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error("NewtonStateSolver parameter size mismatch");
  }

  initializeState(state);

  HostVector res;
  HostVector rhs;
  HostVector step;
  for (Index i = 0; i <= opts_.max_its; ++i)
  {
    problem_.res(state, prm, res);
    if (res.size() != numRes())
    {
      throw std::runtime_error("NewtonStateSolver residual size mismatch");
    }

    if (squaredNorm(res)
        <= opts_.res_tol * opts_.res_tol)
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
      throw std::runtime_error("NewtonStateSolver step size mismatch");
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

  throw std::runtime_error("NewtonStateSolver failed to converge");
}

void NewtonStateSolver::initializeState(HostVector& state) const
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
