#include <stdexcept>

#include <femx/core/Math.hpp>
#include <femx/solve/TimeMatrixNewtonStateSolver.hpp>

using namespace femx::algebra;

namespace femx
{
namespace solve
{

TimeMatrixNewtonStateSolver::TimeMatrixNewtonStateSolver(
    const TimeMatrixResidualEquation& eq,
    SystemMatrix&                     next_state_jac,
    LinearSolver&                     lin_solver)
  : eq_(eq),
    next_state_jac_(next_state_jac),
    lin_solver_(lin_solver)
{
  if (eq_.numRes() != eq_.numStates())
  {
    throw std::runtime_error(
        "TimeMatrixNewtonStateSolver requires square state residual dimensions");
  }
}

TimeMatrixNewtonStateSolverOptions& TimeMatrixNewtonStateSolver::options()
{
  return options_;
}

const TimeMatrixNewtonStateSolverOptions& TimeMatrixNewtonStateSolver::options()
    const
{
  return options_;
}

void TimeMatrixNewtonStateSolver::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error(
        "TimeMatrixNewtonStateSolver initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void TimeMatrixNewtonStateSolver::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

Index TimeMatrixNewtonStateSolver::numSteps() const
{
  return eq_.numSteps();
}

Index TimeMatrixNewtonStateSolver::numStates() const
{
  return eq_.numStates();
}

Index TimeMatrixNewtonStateSolver::numParams() const
{
  return eq_.numParams();
}

void TimeMatrixNewtonStateSolver::solve(const Vector<Real>&  prm,
                                        TimeStateTrajectory& tr)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "TimeMatrixNewtonStateSolver parameter size mismatch");
  }

  tr.resize(numSteps(), numStates());
  Vector<Real> init = tr[0];
  initializeInitialState(init);

  for (Index step = 0; step < numSteps(); ++step)
  {
    tr[step + 1]              = tr[step];
    const Vector<Real> x      = tr[step];
    Vector<Real>       x_next = tr[step + 1];
    solveStep(step, prm, x, x_next);
  }
}

void TimeMatrixNewtonStateSolver::solveStep(Index               step,
                                            const Vector<Real>& prm,
                                            const Vector<Real>& x,
                                            Vector<Real>&       x_next)
{
  Vector<Real> res;
  Vector<Real> rhs;
  Vector<Real> dx;

  for (Index iter = 0; iter <= options_.max_its; ++iter)
  {
    eq_.res(step, x_next, x, prm, res);
    if (res.size() != eq_.numRes())
    {
      throw std::runtime_error(
          "TimeMatrixNewtonStateSolver residual size mismatch");
    }

    if (squaredNorm(res) <= options_.res_tol * options_.res_tol)
    {
      return;
    }
    if (iter == options_.max_its)
    {
      break;
    }

    resize(rhs, res.size());
    for (Index i = 0; i < res.size(); ++i)
    {
      rhs[i] = -res[i];
    }

    eq_.assembleNextStateJac(step, x_next, x, prm, next_state_jac_);
    next_state_jac_.finalize();
    lin_solver_.solve(next_state_jac_, rhs, dx);
    if (dx.size() != numStates())
    {
      throw std::runtime_error(
          "TimeMatrixNewtonStateSolver update size mismatch");
    }

    for (Index i = 0; i < numStates(); ++i)
    {
      x_next[i] += dx[i];
    }

    if (squaredNorm(dx) <= options_.step_tolerance * options_.step_tolerance)
    {
      return;
    }
  }

  throw std::runtime_error("TimeMatrixNewtonStateSolver failed to converge");
}

void TimeMatrixNewtonStateSolver::initializeInitialState(
    Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }
  resize(state, numStates());
}

void TimeMatrixNewtonStateSolver::resize(Vector<Real>& out,
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
