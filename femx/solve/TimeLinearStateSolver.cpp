#include <chrono>
#include <stdexcept>
#include <utility>

#include <femx/solve/TimeLinearStateSolver.hpp>

namespace femx
{
namespace solve
{

namespace
{

using Clock = std::chrono::steady_clock;

Real elapsedSeconds(const Clock::time_point& begin)
{
  return std::chrono::duration<Real>(Clock::now() - begin).count();
}

} // namespace

TimeLinearStateSolver::TimeLinearStateSolver(
    const problem::TimeResidual& eq,
    linalg::MatrixOperator&      next_state_jac,
    linalg::LinearSolver&        lin_solver)
  : eq_(eq),
    next_state_jac_(next_state_jac),
    lin_solver_(lin_solver),
    dims_(eq.dimensions())
{
  if (dims_.num_residuals != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeLinearStateSolver requires square state residual dimensions");
  }
}

void TimeLinearStateSolver::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error(
        "TimeLinearStateSolver initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void TimeLinearStateSolver::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

void TimeLinearStateSolver::setStepMonitor(StepMonitor monitor)
{
  step_monitor_ = std::move(monitor);
}

void TimeLinearStateSolver::clearStepMonitor()
{
  step_monitor_ = nullptr;
}

void TimeLinearStateSolver::resetTiming()
{
  assembly_seconds_ = 0.0;
  solve_seconds_    = 0.0;
  assembly_calls_   = 0;
  solve_calls_      = 0;
}

Real TimeLinearStateSolver::assemblySeconds() const
{
  return assembly_seconds_;
}

Real TimeLinearStateSolver::solveSeconds() const
{
  return solve_seconds_;
}

Index TimeLinearStateSolver::assemblyCalls() const
{
  return assembly_calls_;
}

Index TimeLinearStateSolver::solveCalls() const
{
  return solve_calls_;
}

Index TimeLinearStateSolver::numSteps() const
{
  return dims_.num_steps;
}

Index TimeLinearStateSolver::numStates() const
{
  return dims_.num_states;
}

Index TimeLinearStateSolver::numParams() const
{
  return dims_.num_params;
}

void TimeLinearStateSolver::solve(const Vector<Real>& prm,
                                  TimeTrajectory&     tr)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "TimeLinearStateSolver parameter size mismatch");
  }

  tr.resize(numSteps(), numStates());
  Vector<Real> init;
  initializeInitialState(init);
  tr[0] = init;

  for (Index step = 0; step < numSteps(); ++step)
  {
    tr[step + 1]              = tr[step];
    const Vector<Real> x      = tr[step];
    Vector<Real>       x_next = tr[step + 1];
    solveStep(step, prm, x, x_next);
    tr[step + 1] = x_next;
    if (step_monitor_)
    {
      step_monitor_(step + 1, numSteps());
    }
  }
}

void TimeLinearStateSolver::solveStep(Index               step,
                                      const Vector<Real>& prm,
                                      const Vector<Real>& x,
                                      Vector<Real>&       x_next)
{
  problem::TimeContext ctx;
  ctx.step           = step;
  ctx.prev_state = &x;
  ctx.next_state     = &x_next;
  ctx.prm            = &prm;

  Vector<Real> res;
  eq_.residual(ctx, res);
  if (res.size() != dims_.num_residuals)
  {
    throw std::runtime_error("TimeLinearStateSolver residual size mismatch");
  }

  Vector<Real> rhs;
  resize(rhs, res.size());
  for (Index i = 0; i < res.size(); ++i)
  {
    rhs[i] = -res[i];
  }

  const auto assembly_begin = Clock::now();
  if (!eq_.assembleJacobian(
          ctx, problem::VariableBlock::NextState, next_state_jac_))
  {
    throw std::runtime_error(
        "TimeLinearStateSolver requires an assembled next-state Jacobian");
  }
  next_state_jac_.finalize();
  assembly_seconds_ += elapsedSeconds(assembly_begin);
  ++assembly_calls_;

  Vector<Real> dx;
  const auto   solve_begin = Clock::now();
  lin_solver_.solve(next_state_jac_, rhs, dx);
  solve_seconds_ += elapsedSeconds(solve_begin);
  ++solve_calls_;
  if (dx.size() != numStates())
  {
    throw std::runtime_error(
        "TimeLinearStateSolver update size mismatch");
  }

  for (Index i = 0; i < numStates(); ++i)
  {
    x_next[i] += dx[i];
  }
}

void TimeLinearStateSolver::initializeInitialState(Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }
  resize(state, numStates());
}

void TimeLinearStateSolver::resize(Vector<Real>& out, Index size)
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
