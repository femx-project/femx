#include <chrono>
#include <stdexcept>
#include <utility>

#include <femx/eq/TimeMatrixLinearStateSolver.hpp>

using namespace femx::system;

namespace femx
{
namespace eq
{

namespace
{

using Clock = std::chrono::steady_clock;

Real elapsedSeconds(const Clock::time_point& begin)
{
  return std::chrono::duration<Real>(Clock::now() - begin).count();
}

} // namespace

TimeMatrixLinearStateSolver::TimeMatrixLinearStateSolver(
    const TimeMatrixResidualEquation& eq,
    SystemMatrix&             next_state_jac,
    LinearSolver&             lin_solver)
  : eq_(eq),
    next_state_jac_(next_state_jac),
    lin_solver_(lin_solver)
{
  if (eq_.numRes() != eq_.numStates())
  {
    throw std::runtime_error(
        "TimeMatrixLinearStateSolver requires square state residual dimensions");
  }
}

void TimeMatrixLinearStateSolver::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error(
        "TimeMatrixLinearStateSolver initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void TimeMatrixLinearStateSolver::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

void TimeMatrixLinearStateSolver::setStepMonitor(StepMonitor monitor)
{
  step_monitor_ = std::move(monitor);
}

void TimeMatrixLinearStateSolver::clearStepMonitor()
{
  step_monitor_ = nullptr;
}

void TimeMatrixLinearStateSolver::resetTiming()
{
  assembly_seconds_ = 0.0;
  solve_seconds_    = 0.0;
  assembly_calls_   = 0;
  solve_calls_      = 0;
}

Real TimeMatrixLinearStateSolver::assemblySeconds() const
{
  return assembly_seconds_;
}

Real TimeMatrixLinearStateSolver::solveSeconds() const
{
  return solve_seconds_;
}

Index TimeMatrixLinearStateSolver::assemblyCalls() const
{
  return assembly_calls_;
}

Index TimeMatrixLinearStateSolver::solveCalls() const
{
  return solve_calls_;
}

Index TimeMatrixLinearStateSolver::numSteps() const
{
  return eq_.numSteps();
}

Index TimeMatrixLinearStateSolver::numStates() const
{
  return eq_.numStates();
}

Index TimeMatrixLinearStateSolver::numParams() const
{
  return eq_.numParams();
}

void TimeMatrixLinearStateSolver::solve(const Vector<Real>&  prm,
                                        TimeStateTrajectory& tr)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "TimeMatrixLinearStateSolver parameter size mismatch");
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
    if (step_monitor_)
    {
      step_monitor_(step + 1, numSteps());
    }
  }
}

void TimeMatrixLinearStateSolver::solveStep(Index               step,
                                            const Vector<Real>& prm,
                                            const Vector<Real>& x,
                                            Vector<Real>&       x_next)
{
  Vector<Real> res;
  eq_.res(step, x_next, x, prm, res);
  if (res.size() != eq_.numRes())
  {
    throw std::runtime_error(
        "TimeMatrixLinearStateSolver residual size mismatch");
  }

  Vector<Real> rhs;
  resize(rhs, res.size());
  for (Index i = 0; i < res.size(); ++i)
  {
    rhs[i] = -res[i];
  }

  const auto assembly_begin = Clock::now();
  eq_.assembleNextStateJac(step, x_next, x, prm, next_state_jac_);
  next_state_jac_.finalize();
  assembly_seconds_ += elapsedSeconds(assembly_begin);
  ++assembly_calls_;

  Vector<Real> dx;
  const auto solve_begin = Clock::now();
  lin_solver_.solve(next_state_jac_, rhs, dx);
  solve_seconds_ += elapsedSeconds(solve_begin);
  ++solve_calls_;
  if (dx.size() != numStates())
  {
    throw std::runtime_error(
        "TimeMatrixLinearStateSolver update size mismatch");
  }

  for (Index i = 0; i < numStates(); ++i)
  {
    x_next[i] += dx[i];
  }
}

void TimeMatrixLinearStateSolver::initializeInitialState(
    Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }

  resize(state, numStates());
}

void TimeMatrixLinearStateSolver::resize(Vector<Real>& out,
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
