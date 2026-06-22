#include <chrono>
#include <stdexcept>
#include <utility>

#include <femx/linalg/BlockVectorView.hpp>
#include <femx/state/TimeLinearStateSolver.hpp>

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

void copyHistoryState(const Vector<Real>& hist,
                      Index               nst,
                      Vector<Real>&       out)
{
  if (out.size() != nst)
  {
    out.resize(nst);
  }
  for (Index i = 0; i < nst; ++i)
  {
    out[i] = hist[i];
  }
}

void copyStateToView(const Vector<Real>& src,
                     VectorView<Real>    dst)
{
  if (src.size() != dst.size())
  {
    throw runtime_error(
        "TimeLinearStateSolver state view size mismatch");
  }
  for (Index i = 0; i < src.size(); ++i)
  {
    dst[i] = src[i];
  }
}

void copyView(VectorView<Real> src,
              VectorView<Real> dst)
{
  if (src.size() != dst.size())
  {
    throw runtime_error(
        "TimeLinearStateSolver history view size mismatch");
  }
  for (Index i = 0; i < src.size(); ++i)
  {
    dst[i] = src[i];
  }
}

void initializeHistoryWindow(const Vector<Real>& initial,
                             Index               depth,
                             Index               nst,
                             Vector<Real>&       hist)
{
  if (depth < 0 || nst < 0 || initial.size() != nst)
  {
    throw runtime_error(
        "TimeLinearStateSolver received invalid history window dimensions");
  }
  hist.resize(depth * nst);
  BlockVectorView<Real> hist_view(hist.data(), depth, nst);
  for (Index lag = 0; lag < depth; ++lag)
  {
    copyStateToView(initial, hist_view.block(lag));
  }
}

void advanceHistoryWindow(Vector<Real>&       hist,
                          Index               depth,
                          Index               nst,
                          const Vector<Real>& next)
{
  if (depth < 0 || nst < 0 || hist.size() != depth * nst
      || next.size() != nst)
  {
    throw runtime_error(
        "TimeLinearStateSolver history window size mismatch");
  }
  BlockVectorView<Real> hist_view(hist.data(), depth, nst);
  for (Index lag = depth - 1; lag > 0; --lag)
  {
    copyView(hist_view.block(lag - 1), hist_view.block(lag));
  }
  if (depth > 0)
  {
    copyStateToView(next, hist_view.block(0));
  }
}

} // namespace

TimeLinearStateSolver::TimeLinearStateSolver(
    const TimeResidual& problem,
    MatrixOperator&     next_state_jac,
    LinearSolver&       lin_solver)
  : problem_(problem),
    next_state_jac_(next_state_jac),
    lin_solver_(lin_solver),
    dims_(problem.dims())
{
  if (dims_.nres != dims_.nst)
  {
    throw runtime_error(
        "TimeLinearStateSolver requires square state residual dimensions");
  }
  if (dims_.nhst <= 0)
  {
    throw runtime_error(
        "TimeLinearStateSolver requires at least one history state");
  }
}

void TimeLinearStateSolver::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw runtime_error(
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
  return dims_.nt;
}

Index TimeLinearStateSolver::numStates() const
{
  return dims_.nst;
}

Index TimeLinearStateSolver::numParams() const
{
  return dims_.nprm;
}

void TimeLinearStateSolver::solve(const Vector<Real>& prm,
                                  TimeTrajectory&     tr)
{
  tr.resize(numSteps(), numStates());
  solve(
      prm,
      [&](Index level, const Vector<Real>& state)
      {
        tr[level] = state;
      });
}

void TimeLinearStateSolver::solve(const Vector<Real>&  prm,
                                  const StateObserver& observer)
{
  if (prm.size() != numParams())
  {
    throw runtime_error(
        "TimeLinearStateSolver parameter size mismatch");
  }

  Vector<Real> init;
  initializeInitialState(init);
  if (observer)
  {
    observer(0, init);
  }

  Vector<Real> hist;
  initializeHistoryWindow(init, dims_.nhst, numStates(), hist);
  for (Index step = 0; step < numSteps(); ++step)
  {
    Vector<Real> cur_state = Vector<Real>::view(hist.data(), numStates());
    Vector<Real> x_next    = cur_state;
    solveStep(step, prm, hist, x_next);
    advanceHistoryWindow(hist, dims_.nhst, numStates(), x_next);
    if (observer)
    {
      observer(step + 1, x_next);
    }
    if (step_monitor_)
    {
      step_monitor_(step + 1, numSteps());
    }
  }
}

void TimeLinearStateSolver::solveStep(
    Index               step,
    const Vector<Real>& prm,
    const Vector<Real>& hist,
    Vector<Real>&       x_next)
{
  Vector<Real> prev;
  copyHistoryState(hist, numStates(), prev);

  TimeContext ctx;
  ctx.step = step;
  ctx.prev = &prev;
  ctx.nxt  = &x_next;
  ctx.prm  = &prm;
  ctx.hist = TimeHistoryView(hist.data(), dims_.nhst, numStates());

  Vector<Real> res;
  problem_.res(ctx, res);
  if (res.size() != dims_.nres)
  {
    throw runtime_error("TimeLinearStateSolver residual size mismatch");
  }

  const auto assembly_begin = Clock::now();
  if (!problem_.assembleJac(
          ctx, VariableBlock::NextState, next_state_jac_))
  {
    throw runtime_error(
        "TimeLinearStateSolver requires an assembled next-state Jacobian");
  }
  next_state_jac_.finalize();

  Vector<Real> rhs;
  next_state_jac_.apply(x_next, rhs);
  if (rhs.size() != res.size())
  {
    throw runtime_error("TimeLinearStateSolver RHS size mismatch");
  }
  for (Index i = 0; i < res.size(); ++i)
  {
    rhs[i] -= res[i];
  }
  problem_.prepareLinearSolve(
      ctx, VariableBlock::NextState, next_state_jac_, rhs);
  assembly_seconds_ += elapsedSeconds(assembly_begin);
  ++assembly_calls_;

  Vector<Real> next;
  const auto   solve_begin = Clock::now();
  lin_solver_.solve(next_state_jac_, rhs, next);
  solve_seconds_ += elapsedSeconds(solve_begin);
  ++solve_calls_;
  if (next.size() != numStates())
  {
    throw runtime_error(
        "TimeLinearStateSolver update size mismatch");
  }

  for (Index i = 0; i < numStates(); ++i)
  {
    x_next[i] = next[i];
  }
}

void TimeLinearStateSolver::initializeInitialState(Vector<Real>& state) const
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
