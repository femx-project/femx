#include <chrono>
#include <stdexcept>
#include <utility>

#include <femx/solve/TimeReducedFunctional.hpp>

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

TimeReducedFunctional::TimeReducedFunctional(
    TimeStateSolver&              state_solver,
    const problem::TimeResidual&  eq,
    linalg::MatrixOperator&       next_state_jac,
    linalg::MatrixOperator&       prev_state_jac,
    linalg::LinearSolver&         adjoint_solver,
    const problem::TimeObjective& obj)
  : state_solver_(state_solver),
    eq_(eq),
    next_state_jac_(next_state_jac),
    prev_state_jac_(prev_state_jac),
    adj_solver_(adjoint_solver),
    obj_(obj),
    dims_(eq.dimensions())
{
  checkDims();
}

void TimeReducedFunctional::setProgress(ProgressCallback callback)
{
  callback_ = std::move(callback);
}

void TimeReducedFunctional::clearProgress()
{
  callback_ = nullptr;
}

void TimeReducedFunctional::setInitialStateParamJacT(
    InitialStateParamJacT callback)
{
  init_param_jac_t_ = std::move(callback);
}

void TimeReducedFunctional::clearInitialStateParamJacT()
{
  init_param_jac_t_ = nullptr;
}

void TimeReducedFunctional::resetTiming()
{
  assembly_seconds_ = 0.0;
  solve_seconds_    = 0.0;
  assembly_calls_   = 0;
  solve_calls_      = 0;
}

Real TimeReducedFunctional::assemblySeconds() const
{
  return assembly_seconds_;
}

Real TimeReducedFunctional::solveSeconds() const
{
  return solve_seconds_;
}

Index TimeReducedFunctional::assemblyCalls() const
{
  return assembly_calls_;
}

Index TimeReducedFunctional::solveCalls() const
{
  return solve_calls_;
}

Index TimeReducedFunctional::numParams() const
{
  return state_solver_.numParams();
}

Real TimeReducedFunctional::value(const Vector<Real>& prm)
{
  TimeTrajectory tr;
  solveFwd(prm, tr);
  return obj_.value(tr, prm);
}

void TimeReducedFunctional::grad(const Vector<Real>& prm,
                                 Vector<Real>&       out)
{
  TimeTrajectory tr;
  solveFwd(prm, tr);
  gradAt(tr, prm, out);
}

Real TimeReducedFunctional::valueGrad(const Vector<Real>& prm,
                                      Vector<Real>&       grad_out)
{
  TimeTrajectory tr;
  solveFwd(prm, tr);
  const Real obj_val = obj_.value(tr, prm);
  gradAt(tr, prm, grad_out);
  return obj_val;
}

void TimeReducedFunctional::checkDims() const
{
  if (state_solver_.numSteps() != dims_.num_steps
      || state_solver_.numSteps() != obj_.numSteps()
      || state_solver_.numStates() != dims_.num_states
      || state_solver_.numStates() != obj_.numStates()
      || state_solver_.numParams() != dims_.num_params
      || state_solver_.numParams() != obj_.numParams()
      || dims_.num_residuals != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeReducedFunctional received inconsistent dimensions");
  }
}

void TimeReducedFunctional::solveFwd(const Vector<Real>& prm,
                                     TimeTrajectory&     tr)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "TimeReducedFunctional parameter size mismatch");
  }

  notify("forward-begin", 0, state_solver_.numSteps());
  state_solver_.solve(prm, tr);
  notify("forward-end", state_solver_.numSteps(), state_solver_.numSteps());
  if (tr.numSteps() != state_solver_.numSteps()
      || tr.numStates() != state_solver_.numStates())
  {
    throw std::runtime_error(
        "TimeReducedFunctional forward trajectory size mismatch");
  }
}

void TimeReducedFunctional::gradAt(const TimeTrajectory& tr,
                                   const Vector<Real>&   prm,
                                   Vector<Real>&         out)
{
  obj_.paramGrad(tr, prm, out);
  checkSize(out,
            numParams(),
            "TimeReducedFunctional parameter gradient size mismatch");

  const Index  steps = state_solver_.numSteps();
  Vector<Real> next_adjoint;
  Vector<Real> rhs;
  Vector<Real> carry;
  Vector<Real> adjoint;
  Vector<Real> param_adj;
  Vector<Real> initial_state_grad;

  notify("adjoint-begin", 0, steps);
  for (Index step = steps; step-- > 0;)
  {
    notify("adjoint-step", steps - step, steps);
    obj_.stateGrad(step + 1, tr, prm, rhs);
    checkSize(rhs,
              dims_.num_states,
              "TimeReducedFunctional state gradient size mismatch");

    if (step + 1 < steps)
    {
      const Vector<Real>   carry_prev = tr[step + 1];
      const Vector<Real>   carry_next = tr[step + 2];
      problem::TimeContext carry_ctx;
      carry_ctx.step           = step + 1;
      carry_ctx.prev_state = &carry_prev;
      carry_ctx.next_state     = &carry_next;
      carry_ctx.prm            = &prm;
      assemble(carry_ctx, problem::VariableBlock::PrevState, prev_state_jac_);
      prev_state_jac_.applyT(next_adjoint, carry);
      checkSize(carry,
                dims_.num_states,
                "TimeReducedFunctional carry size mismatch");
      for (Index i = 0; i < rhs.size(); ++i)
      {
        rhs[i] -= carry[i];
      }
    }

    const Vector<Real>   prev_state = tr[step];
    const Vector<Real>   next_state = tr[step + 1];
    problem::TimeContext ctx;
    ctx.step           = step;
    ctx.prev_state = &prev_state;
    ctx.next_state     = &next_state;
    ctx.prm            = &prm;
    assemble(ctx, problem::VariableBlock::NextState, next_state_jac_);

    const auto solve_begin = Clock::now();
    adj_solver_.solveT(next_state_jac_, rhs, adjoint);
    solve_seconds_ += elapsedSeconds(solve_begin);
    ++solve_calls_;
    checkSize(adjoint,
              dims_.num_residuals,
              "TimeReducedFunctional adjoint size mismatch");

    eq_.applyJacT(
        ctx, problem::VariableBlock::Param, adjoint, param_adj);
    checkSize(param_adj,
              numParams(),
              "TimeReducedFunctional residual parameter gradient size mismatch");
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] -= param_adj[i];
    }
    next_adjoint = adjoint;
  }

  if (init_param_jac_t_)
  {
    obj_.stateGrad(0, tr, prm, initial_state_grad);
    checkSize(initial_state_grad,
              dims_.num_states,
              "TimeReducedFunctional initial state gradient size mismatch");

    if (steps > 0)
    {
      const Vector<Real>   prev_state = tr[0];
      const Vector<Real>   next_state = tr[1];
      problem::TimeContext ctx;
      ctx.step           = 0;
      ctx.prev_state = &prev_state;
      ctx.next_state     = &next_state;
      ctx.prm            = &prm;
      assemble(ctx, problem::VariableBlock::PrevState, prev_state_jac_);
      prev_state_jac_.applyT(next_adjoint, carry);
      checkSize(carry,
                dims_.num_states,
                "TimeReducedFunctional initial carry size mismatch");
      for (Index i = 0; i < initial_state_grad.size(); ++i)
      {
        initial_state_grad[i] -= carry[i];
      }
    }

    init_param_jac_t_(prm, initial_state_grad, param_adj);
    checkSize(param_adj,
              numParams(),
              "TimeReducedFunctional initial parameter gradient size mismatch");
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] += param_adj[i];
    }
  }
  notify("adjoint-end", steps, steps);
}

void TimeReducedFunctional::assemble(problem::TimeContext    ctx,
                                     problem::VariableBlock  wrt,
                                     linalg::MatrixOperator& out)
{
  const auto assembly_begin = Clock::now();
  if (!eq_.assembleJacobian(ctx, wrt, out))
  {
    throw std::runtime_error(
        "TimeReducedFunctional requires assembled state Jacobians");
  }
  out.finalize();
  assembly_seconds_ += elapsedSeconds(assembly_begin);
  ++assembly_calls_;
}

void TimeReducedFunctional::notify(const char* phase,
                                   Index       step,
                                   Index       total_steps)
{
  if (callback_)
  {
    callback_(phase, step, total_steps);
  }
}

void TimeReducedFunctional::checkSize(const Vector<Real>& value,
                                      Index               expected,
                                      const char*         message)
{
  if (value.size() != expected)
  {
    throw std::runtime_error(message);
  }
}

} // namespace solve
} // namespace femx
