#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <femx/inverse/TimeObjective.hpp>
#include <femx/inverse/TimeReducedFunctional.hpp>
#include <femx/linalg/BlockVectorView.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <femx/state/TimeTrajectory.hpp>
using namespace femx::state;
using namespace femx::inverse;
using namespace femx::linalg;

namespace femx
{
namespace inverse
{

namespace
{

using Clock = std::chrono::steady_clock;

Real elapsedSeconds(const Clock::time_point& begin)
{
  return std::chrono::duration<Real>(Clock::now() - begin).count();
}

class ForwardProgressMonitor final : public TimeStateMonitor
{
public:
  explicit ForwardProgressMonitor(TimeReducedProgressMonitor& monitor)
    : monitor_(monitor)
  {
  }

  void observe(Index, const HostVector&) override
  {
  }

  bool observeStep(const TimeStepStateContext& ctx) override
  {
    monitor_.progress("forward-step", ctx.level, ctx.total_steps);
    return false;
  }

private:
  TimeReducedProgressMonitor& monitor_;
};

class ScopedForwardProgress
{
public:
  ScopedForwardProgress(TimeIntegrator&             integrator,
                        TimeReducedProgressMonitor* monitor)
    : integrator_(integrator)
  {
    if (monitor != nullptr)
    {
      monitor_ = std::make_unique<ForwardProgressMonitor>(*monitor);
      integrator_.setMonitor(monitor_.get());
    }
  }

  ScopedForwardProgress(const ScopedForwardProgress&)            = delete;
  ScopedForwardProgress& operator=(const ScopedForwardProgress&) = delete;

  ~ScopedForwardProgress()
  {
    if (monitor_ != nullptr)
    {
      integrator_.clearMonitor();
    }
  }

private:
  TimeIntegrator&                         integrator_;
  std::unique_ptr<ForwardProgressMonitor> monitor_;
};

Index historyLevel(Index step, Index lag)
{
  return step > lag ? step - lag : 0;
}

void checkFinite(const HostVector& x, const std::string& name, Index step)
{
  Real max_abs = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    const Real value = std::abs(x[i]);
    if (!std::isfinite(value))
    {
      throw std::runtime_error(
          name + " contains a non-finite value at step "
          + std::to_string(static_cast<long long>(step)) + ", index "
          + std::to_string(static_cast<long long>(i)));
    }
    max_abs = std::max(max_abs, value);
  }
  if (max_abs > std::sqrt(std::numeric_limits<Real>::max()))
  {
    std::ostringstream msg;
    msg << name << " is too large at step " << step
        << " (max_abs=" << max_abs << ")";
    throw std::runtime_error(msg.str());
  }
}

void fillHistory(const TimeTrajectory& tr,
                 Index                 step,
                 Index                 num_hist,
                 HostVector&           hist)
{
  const Index num_states = tr.numStates();
  if (num_hist < 0 || num_states < 0)
  {
    throw std::runtime_error(
        "TimeReducedFunctional received invalid history dimensions");
  }
  if (hist.size() != num_hist * num_states)
  {
    hist.resize(num_hist * num_states);
  }
  BlockVectorView<Real> hist_view(hist.data(), num_hist, num_states);
  for (Index i = 0; i < num_hist; ++i)
  {
    const HostVector& st = tr[historyLevel(step, i)];
    HostVectorView    h  = hist_view.block(i);
    for (Index j = 0; j < num_states; ++j)
    {
      h[j] = st[j];
    }
  }
}

TimeContext makeContext(const TimeTrajectory& tr,
                        Index                 step,
                        Index                 num_hist,
                        const HostVector&     prm,
                        HostVector&           hist,
                        HostVector&           nxt)
{
  fillHistory(tr, step, num_hist, hist);
  nxt = tr[step + 1];

  TimeContext ctx;
  ctx.step = step;
  ctx.nxt  = &nxt;
  ctx.prm  = &prm;
  ctx.hist = TimeHistoryView(hist.data(), num_hist, tr.numStates());
  return ctx;
}

} // namespace

TimeReducedFunctional::TimeReducedFunctional(
    TimeIntegrator&      integrator,
    const TimeResidual&  problem,
    TimeLinearization&   lin,
    MatrixOperator&      J_next,
    MatrixOperator&      J_hist,
    LinearSolver&        adj_solver,
    const TimeObjective& obj)
  : integrator_(integrator),
    problem_(problem),
    lin_(lin),
    J_next_(J_next),
    J_hist_(J_hist),
    adj_solver_(adj_solver),
    obj_(obj),
    dims_(problem.dims())
{
  checkDims();
}

void TimeReducedFunctional::setMonitor(
    TimeReducedProgressMonitor* monitor)
{
  progress_monitor_ = monitor;
}

void TimeReducedFunctional::clearMonitor()
{
  progress_monitor_ = nullptr;
}

void TimeReducedFunctional::setInitialStateParamJacT(
    InitialStateGradientMap* map)
{
  init_grad_map_ = map;
}

void TimeReducedFunctional::clearInitialStateParamJacT()
{
  init_grad_map_ = nullptr;
}

void TimeReducedFunctional::resetTiming()
{
  assm_sec_    = 0.0;
  solve_sec_   = 0.0;
  assm_calls_  = 0;
  solve_calls_ = 0;
}

Real TimeReducedFunctional::assemblySeconds() const
{
  return assm_sec_;
}

Real TimeReducedFunctional::solveSeconds() const
{
  return solve_sec_;
}

Index TimeReducedFunctional::assemblyCalls() const
{
  return assm_calls_;
}

Index TimeReducedFunctional::solveCalls() const
{
  return solve_calls_;
}

Index TimeReducedFunctional::numParams() const
{
  return integrator_.numParams();
}

Real TimeReducedFunctional::value(const HostVector& prm)
{
  TimeTrajectory tr;
  solveFwd(prm, tr);
  return obj_.value(tr, prm);
}

void TimeReducedFunctional::grad(const HostVector& prm,
                                 HostVector&       out)
{
  TimeTrajectory tr;
  solveFwd(prm, tr);
  gradAt(tr, prm, out);
}

Real TimeReducedFunctional::valueGrad(const HostVector& prm,
                                      HostVector&       grad_out)
{
  TimeTrajectory tr;
  solveFwd(prm, tr);
  const Real obj_val = obj_.value(tr, prm);
  gradAt(tr, prm, grad_out);
  return obj_val;
}

void TimeReducedFunctional::checkDims() const
{
  if (integrator_.numSteps() != dims_.num_steps
      || integrator_.numSteps() != obj_.numSteps()
      || integrator_.numStates() != dims_.num_states
      || integrator_.numStates() != obj_.numStates()
      || integrator_.numParams() != dims_.num_param
      || integrator_.numParams() != obj_.numParams()
      || dims_.num_res != dims_.num_states
      || dims_.num_hist <= 0)
  {
    throw std::runtime_error(
        "TimeReducedFunctional received inconsistent dimensions");
  }
}

void TimeReducedFunctional::solveFwd(const HostVector& prm,
                                     TimeTrajectory&   tr)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "TimeReducedFunctional parameter size mismatch");
  }

  notify("forward-begin", 0, integrator_.numSteps());
  {
    ScopedForwardProgress progress(integrator_, progress_monitor_);
    integrator_.solve(prm, tr);
  }
  notify("forward-end", integrator_.numSteps(), integrator_.numSteps());
  if (tr.numSteps() != integrator_.numSteps()
      || tr.numStates() != integrator_.numStates())
  {
    throw std::runtime_error(
        "TimeReducedFunctional forward trajectory size mismatch");
  }
}

void TimeReducedFunctional::gradAt(const TimeTrajectory& tr,
                                   const HostVector&     prm,
                                   HostVector&           out)
{
  obj_.paramGrad(tr, prm, out);
  checkSize(out, numParams());

  const Index steps = integrator_.numSteps();
  HostVector  rhs;
  HostVector  carry;
  HostVector  adj;
  HostVector  param_adj;
  HostVector  init_state_grad;
  HostVector  ctx_next_state;
  HostVector  carry_next_state;
  HostVector  hist;
  HostVector  carry_history;

  Array<HostVector> adjoints(steps);

  notify("adjoint-begin", 0, steps);
  for (Index t = steps; t-- > 0;)
  {
    notify("adjoint-step", steps - t, steps);
    obj_.stateGrad(t + 1, tr, prm, rhs);
    checkSize(rhs, dims_.num_states);
    checkFinite(rhs, "adjoint objective gradient", t);

    const Index level = t + 1;
    for (Index i = 0; i < dims_.num_hist; ++i)
    {
      const Index ft = level + i;
      if (ft >= steps)
      {
        break;
      }

      TimeContext carry_ctx = makeContext(tr,
                                          ft,
                                          dims_.num_hist,
                                          prm,
                                          carry_history,
                                          carry_next_state);

      assemble(carry_ctx, VariableBlock::hist(i), J_hist_);
      J_hist_.applyT(adjoints[ft], carry);

      checkSize(carry, dims_.num_states);
      checkFinite(carry, "adjoint history carry", t);

      for (Index j = 0; j < rhs.size(); ++j)
      {
        rhs[j] -= carry[j];
      }
    }
    checkFinite(rhs, "adjoint right-hand side", t);

    TimeContext ctx = makeContext(tr,
                                  t,
                                  dims_.num_hist,
                                  prm,
                                  hist,
                                  ctx_next_state);

    assemble(ctx, VariableBlock::NextState, J_next_);

    const auto solve_begin = Clock::now();
    adj_solver_.solveT(J_next_, rhs, adj);
    solve_sec_ += elapsedSeconds(solve_begin);

    ++solve_calls_;
    checkSize(adj, dims_.num_res);
    checkFinite(adj, "adjoint solution", t);

    problem_.linearize(ctx, lin_);
    lin_.applyJacT(VariableBlock::Param, adj, param_adj);

    checkSize(param_adj, numParams());
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] -= param_adj[i];
    }
    adjoints[t] = adj;
  }

  if (init_grad_map_ != nullptr)
  {
    obj_.stateGrad(0, tr, prm, init_state_grad);
    checkSize(init_state_grad, dims_.num_states);

    for (Index t = 0; t < steps; ++t)
    {
      for (Index i = 0; i < dims_.num_hist; ++i)
      {
        if (historyLevel(t, i) != 0)
        {
          continue;
        }

        TimeContext carry_ctx =
            makeContext(tr,
                        t,
                        dims_.num_hist,
                        prm,
                        carry_history,
                        carry_next_state);

        assemble(carry_ctx, VariableBlock::hist(i), J_hist_);
        J_hist_.applyT(adjoints[t], carry);

        checkSize(carry, dims_.num_states);

        for (Index j = 0; j < init_state_grad.size(); ++j)
        {
          init_state_grad[j] -= carry[j];
        }
      }
    }

    if (init_grad_map_ != nullptr)
    {
      init_grad_map_->apply(prm, init_state_grad, param_adj);
    }
    checkSize(param_adj, numParams());
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] += param_adj[i];
    }
  }
  notify("adjoint-end", steps, steps);
}

void TimeReducedFunctional::assemble(TimeContext     ctx,
                                     VariableBlock   wrt,
                                     MatrixOperator& out)
{
  const auto assembly_begin = Clock::now();
  problem_.linearize(ctx, lin_);
  if (!lin_.assembleJac(wrt, out))
  {
    throw std::runtime_error(
        "TimeReducedFunctional requires assembled state Jacobians");
  }
  out.finalize();
  assm_sec_ += elapsedSeconds(assembly_begin);
  ++assm_calls_;
}

void TimeReducedFunctional::notify(const char* phase,
                                   Index       step,
                                   Index       total_steps)
{
  if (progress_monitor_ != nullptr)
  {
    progress_monitor_->progress(phase, step, total_steps);
  }
}

void TimeReducedFunctional::checkSize(const HostVector& value,
                                      Index             exp)
{
  if (value.size() != exp)
  {
    throw std::runtime_error("TimeReducedFunctional vector size mismatch");
  }
}

} // namespace inverse
} // namespace femx
