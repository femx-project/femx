#include <chrono>
#include <stdexcept>
#include <utility>

#include <femx/linalg/BlockVectorView.hpp>
#include <femx/state/TimeReducedFunctional.hpp>

using namespace std;
using namespace femx::problem;
using namespace femx::linalg;

namespace femx
{
namespace state
{

namespace
{

using Clock = chrono::steady_clock;

Real elapsedSeconds(const Clock::time_point& begin)
{
  return chrono::duration<Real>(Clock::now() - begin).count();
}

Index historyLevel(Index step, Index lag)
{
  return step > lag ? step - lag : 0;
}

void fillHistory(const TimeTrajectory& tr,
                 Index                 step,
                 Index                 nhst,
                 Vector<Real>&         hist)
{
  const Index nst = tr.numStates();
  if (nhst < 0 || nst < 0)
  {
    throw runtime_error(
        "TimeReducedFunctional received invalid history dimensions");
  }
  if (hist.size() != nhst * nst)
  {
    hist.resize(nhst * nst);
  }
  BlockVectorView<Real> hist_view(hist.data(), nhst, nst);
  for (Index i = 0; i < nhst; ++i)
  {
    const Vector<Real>& st = tr[historyLevel(step, i)];
    VectorView<Real>    h  = hist_view.block(i);
    for (Index j = 0; j < nst; ++j)
    {
      h[j] = st[j];
    }
  }
}

TimeContext makeContext(
    const TimeTrajectory& tr,
    Index                 step,
    Index                 nhst,
    const Vector<Real>&   prm,
    Vector<Real>&         hist,
    Vector<Real>&         prev,
    Vector<Real>&         nxt)
{
  fillHistory(tr, step, nhst, hist);
  prev = tr[step];
  nxt  = tr[step + 1];

  TimeContext ctx;
  ctx.step = step;
  ctx.prev = &prev;
  ctx.nxt  = &nxt;
  ctx.prm  = &prm;
  ctx.hist = TimeHistoryView(hist.data(), nhst, tr.numStates());
  return ctx;
}

} // namespace

TimeReducedFunctional::TimeReducedFunctional(
    TimeStateSolver&     state_solver,
    const TimeResidual&  problem,
    TimeLinearization&   lin,
    MatrixOperator&      nxt_jac,
    MatrixOperator&      hist_jac,
    LinearSolver&        adj_solver,
    const TimeObjective& obj)
  : state_solver_(state_solver),
    problem_(problem),
    lin_(lin),
    nxt_jac_(nxt_jac),
    hist_jac_(hist_jac),
    adj_solver_(adj_solver),
    obj_(obj),
    dims_(problem.dims())
{
  checkDims();
}

void TimeReducedFunctional::setProgress(ProgressCallback cb)
{
  callback_ = std::move(cb);
}

void TimeReducedFunctional::clearProgress()
{
  callback_ = nullptr;
}

void TimeReducedFunctional::setInitialStateParamJacT(
    InitialStateParamJacT cb)
{
  init_param_jac_t_ = std::move(cb);
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
  if (state_solver_.numSteps() != dims_.nt
      || state_solver_.numSteps() != obj_.numSteps()
      || state_solver_.numStates() != dims_.nst
      || state_solver_.numStates() != obj_.numStates()
      || state_solver_.numParams() != dims_.nprm
      || state_solver_.numParams() != obj_.numParams()
      || dims_.nres != dims_.nst
      || dims_.nhst <= 0)
  {
    throw runtime_error(
        "TimeReducedFunctional received inconsistent dimensions");
  }
}

void TimeReducedFunctional::solveFwd(const Vector<Real>& prm,
                                     TimeTrajectory&     tr)
{
  if (prm.size() != numParams())
  {
    throw runtime_error(
        "TimeReducedFunctional parameter size mismatch");
  }

  notify("forward-begin", 0, state_solver_.numSteps());
  state_solver_.solve(prm, tr);
  notify("forward-end", state_solver_.numSteps(), state_solver_.numSteps());
  if (tr.numSteps() != state_solver_.numSteps()
      || tr.numStates() != state_solver_.numStates())
  {
    throw runtime_error(
        "TimeReducedFunctional forward trajectory size mismatch");
  }
}

void TimeReducedFunctional::gradAt(const TimeTrajectory& tr,
                                   const Vector<Real>&   prm,
                                   Vector<Real>&         out)
{
  obj_.paramGrad(tr, prm, out);
  checkSize(out, numParams());

  const Index  steps = state_solver_.numSteps();
  Vector<Real> rhs;
  Vector<Real> carry;
  Vector<Real> adj;
  Vector<Real> param_adj;
  Vector<Real> init_state_grad;
  Vector<Real> ctx_next_state;
  Vector<Real> carry_next_state;
  Vector<Real> hist;
  Vector<Real> prev;
  Vector<Real> carry_history;
  Vector<Real> carry_prev_state;

  Vector<Vector<Real>> adjoints(steps);

  notify("adjoint-begin", 0, steps);
  for (Index t = steps; t-- > 0;)
  {
    notify("adjoint-step", steps - t, steps);
    obj_.stateGrad(t + 1, tr, prm, rhs);
    checkSize(rhs, dims_.nst);

    const Index level = t + 1;
    for (Index i = 0; i < dims_.nhst; ++i)
    {
      const Index ft = level + i;
      if (ft >= steps)
      {
        break;
      }

      TimeContext carry_ctx = makeContext(tr,
                                          ft,
                                          dims_.nhst,
                                          prm,
                                          carry_history,
                                          carry_prev_state,
                                          carry_next_state);

      assemble(carry_ctx, VariableBlock::hist(i), hist_jac_);
      hist_jac_.applyT(adjoints[ft], carry);

      checkSize(carry, dims_.nst);

      for (Index j = 0; j < rhs.size(); ++j)
      {
        rhs[j] -= carry[j];
      }
    }

    TimeContext ctx = makeContext(tr,
                                  t,
                                  dims_.nhst,
                                  prm,
                                  hist,
                                  prev,
                                  ctx_next_state);

    assemble(ctx, VariableBlock::NextState, nxt_jac_);

    const auto solve_begin = Clock::now();
    adj_solver_.solveT(nxt_jac_, rhs, adj);
    solve_seconds_ += elapsedSeconds(solve_begin);

    ++solve_calls_;
    checkSize(adj, dims_.nres);

    problem_.linearize(ctx, lin_);
    lin_.applyJacT(VariableBlock::Param, adj, param_adj);

    checkSize(param_adj, numParams());
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] -= param_adj[i];
    }
    adjoints[t] = adj;
  }

  if (init_param_jac_t_)
  {
    obj_.stateGrad(0, tr, prm, init_state_grad);
    checkSize(init_state_grad, dims_.nst);

    for (Index t = 0; t < steps; ++t)
    {
      for (Index i = 0; i < dims_.nhst; ++i)
      {
        if (historyLevel(t, i) != 0)
        {
          continue;
        }

        TimeContext carry_ctx =
            makeContext(tr,
                        t,
                        dims_.nhst,
                        prm,
                        carry_history,
                        carry_prev_state,
                        carry_next_state);

        assemble(carry_ctx, VariableBlock::hist(i), hist_jac_);
        hist_jac_.applyT(adjoints[t], carry);

        checkSize(carry, dims_.nst);

        for (Index j = 0; j < init_state_grad.size(); ++j)
        {
          init_state_grad[j] -= carry[j];
        }
      }
    }

    init_param_jac_t_(prm, init_state_grad, param_adj);
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
    throw runtime_error(
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
                                      Index               exp)
{
  if (value.size() != exp)
  {
    throw runtime_error("TimeReducedFunctional vector size mismatch");
  }
}

} // namespace state
} // namespace femx
