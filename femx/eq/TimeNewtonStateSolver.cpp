#include <stdexcept>

#include <femx/common/Math.hpp>
#include <femx/eq/TimeNewtonStateSolver.hpp>
#include <femx/eq/TimeStateJacobianOperator.hpp>

using namespace femx::system;

namespace femx
{
namespace eq
{

TimeNewtonStateSolver::TimeNewtonStateSolver(
    const TimeResidualEquation& eq,
    LinearSolver&       lin_solver)
  : eq_(eq),
    lin_solver_(lin_solver)
{
  if (eq_.numRes() != eq_.numStates())
  {
    throw std::runtime_error(
        "TimeNewtonStateSolver requires square state residual dimensions");
  }
}

TimeNewtonStateSolverOptions& TimeNewtonStateSolver::options()
{
  return options_;
}

const TimeNewtonStateSolverOptions& TimeNewtonStateSolver::options() const
{
  return options_;
}

void TimeNewtonStateSolver::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error(
        "TimeNewtonStateSolver initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void TimeNewtonStateSolver::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

Index TimeNewtonStateSolver::numSteps() const
{
  return eq_.numSteps();
}

Index TimeNewtonStateSolver::numStates() const
{
  return eq_.numStates();
}

Index TimeNewtonStateSolver::numParams() const
{
  return eq_.numParams();
}

void TimeNewtonStateSolver::solve(const Vector<Real>&  prm,
                                  TimeStateTrajectory& tr)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error("TimeNewtonStateSolver parameter size mismatch");
  }

  tr.resize(numSteps(), numStates());
  Vector<Real> init = tr[0];
  initializeInitialState(init);

  for (Index step = 0; step < numSteps(); ++step)
  {
    tr[step + 1] = tr[step];
    const Vector<Real> x      = tr[step];
    Vector<Real>       x_next = tr[step + 1];
    solveStep(step, prm, x, x_next);
  }
}

void TimeNewtonStateSolver::solveStep(Index               step,
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
      throw std::runtime_error("TimeNewtonStateSolver residual size mismatch");
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

    const TimeStateJacobianOperator jac(
        eq_, step, x_next, x, prm, TimeStateSlot::Next);
    lin_solver_.solve(jac, rhs, dx);
    if (dx.size() != numStates())
    {
      throw std::runtime_error("TimeNewtonStateSolver update size mismatch");
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

  throw std::runtime_error("TimeNewtonStateSolver failed to converge");
}

void TimeNewtonStateSolver::initializeInitialState(Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }
  resize(state, numStates());
}

void TimeNewtonStateSolver::resize(Vector<Real>& out,
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

} // namespace eq
} // namespace femx
