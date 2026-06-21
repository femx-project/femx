#include <stdexcept>

#include <femx/solve/TimeStepper.hpp>

namespace femx
{
namespace solve
{

TimeStepper::TimeStepper(const problem::TimeResidual& problem,
                         linalg::LinearSolver&        linear_solver)
  : problem_(problem),
    linear_solver_(linear_solver),
    dims_(problem.dimensions())
{
  if (dims_.num_residuals != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeStepper requires square next-state residual dimensions");
  }
}

TimeStepperOptions& TimeStepper::options()
{
  return options_;
}

const TimeStepperOptions& TimeStepper::options() const
{
  return options_;
}

void TimeStepper::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error("TimeStepper initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void TimeStepper::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

Index TimeStepper::numSteps() const
{
  return dims_.num_steps;
}

Index TimeStepper::numStates() const
{
  return dims_.num_states;
}

Index TimeStepper::numParams() const
{
  return dims_.num_params;
}

Index TimeStepper::numResiduals() const
{
  return dims_.num_residuals;
}

void TimeStepper::solve(const Vector<Real>& prm,
                        TimeTrajectory&     trajectory)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error("TimeStepper parameter size mismatch");
  }

  trajectory.resize(numSteps(), numStates());
  Vector<Real> initial_state = trajectory[0];
  initializeInitialState(initial_state);

  for (Index step = 0; step < numSteps(); ++step)
  {
    trajectory[step + 1]              = trajectory[step];
    const Vector<Real> prev_state = trajectory[step];
    Vector<Real>       next_state     = trajectory[step + 1];
    solveStep(step, prm, prev_state, next_state);
  }
}

TimeStepper::NextStateJacobian::NextStateJacobian(
    const TimeStepper& owner)
  : owner_(owner)
{
}

void TimeStepper::NextStateJacobian::reset(problem::TimeContext ctx)
{
  ctx_ = ctx;
}

Index TimeStepper::NextStateJacobian::numRows() const
{
  return owner_.numResiduals();
}

Index TimeStepper::NextStateJacobian::numCols() const
{
  return owner_.numStates();
}

void TimeStepper::NextStateJacobian::apply(const Vector<Real>& dir,
                                           Vector<Real>&       out) const
{
  owner_.problem_.applyJac(
      ctx_, problem::VariableBlock::NextState, dir, out);
}

void TimeStepper::NextStateJacobian::applyT(const Vector<Real>& dir,
                                            Vector<Real>&       out) const
{
  owner_.problem_.applyJacT(
      ctx_, problem::VariableBlock::NextState, dir, out);
}

void TimeStepper::solveStep(Index               step,
                            const Vector<Real>& prm,
                            const Vector<Real>& prev_state,
                            Vector<Real>&       next_state)
{
  Vector<Real>      res;
  Vector<Real>      rhs;
  Vector<Real>      update;
  NextStateJacobian next_jac(*this);

  problem::TimeContext ctx;
  ctx.step           = step;
  ctx.prev_state = &prev_state;
  ctx.next_state     = &next_state;
  ctx.prm            = &prm;

  for (Index iter = 0; iter <= options_.max_its; ++iter)
  {
    problem_.residual(ctx, res);
    if (res.size() != numResiduals())
    {
      throw std::runtime_error("TimeStepper residual size mismatch");
    }

    if (squaredNorm(res)
        <= options_.residual_tolerance * options_.residual_tolerance)
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

    next_jac.reset(ctx);
    linear_solver_.solve(next_jac, rhs, update);
    if (update.size() != numStates())
    {
      throw std::runtime_error("TimeStepper update size mismatch");
    }

    for (Index i = 0; i < numStates(); ++i)
    {
      next_state[i] += update[i];
    }

    if (squaredNorm(update)
        <= options_.step_tolerance * options_.step_tolerance)
    {
      return;
    }
  }

  throw std::runtime_error("TimeStepper failed to converge");
}

void TimeStepper::initializeInitialState(Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }
  resize(state, numStates());
}

void TimeStepper::resize(Vector<Real>& out,
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

Real TimeStepper::squaredNorm(const Vector<Real>& x)
{
  Real value = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    value += x[i] * x[i];
  }
  return value;
}

} // namespace solve
} // namespace femx
