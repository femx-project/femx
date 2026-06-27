#include <chrono>
#include <stdexcept>

#include <femx/linalg/BlockVectorView.hpp>
#include <femx/state/TimeLinearIntegrator.hpp>

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
        "TimeLinearIntegrator state view size mismatch");
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
        "TimeLinearIntegrator history view size mismatch");
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
        "TimeLinearIntegrator received invalid history window dimensions");
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
        "TimeLinearIntegrator history window size mismatch");
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

TimeLinearIntegrator::TimeLinearIntegrator(
    const TimeResidual& problem,
    MatrixOperator&     J_next,
    LinearSolver&       lin_solver)
  : problem_(problem),
    J_next_(J_next),
    lin_solver_(lin_solver),
    dims_(problem.dims())
{
  if (dims_.nres != dims_.nst)
  {
    throw runtime_error(
        "TimeLinearIntegrator requires square state residual dimensions");
  }
  if (dims_.nhst <= 0)
  {
    throw runtime_error(
        "TimeLinearIntegrator requires at least one history state");
  }
}

void TimeLinearIntegrator::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw runtime_error(
        "TimeLinearIntegrator initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void TimeLinearIntegrator::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

void TimeLinearIntegrator::resetTiming()
{
  assm_sec_       = 0.0;
  solve_sec_      = 0.0;
  last_assm_sec_  = 0.0;
  last_solve_sec_ = 0.0;
  assm_calls_     = 0;
  solve_calls_    = 0;
}

Real TimeLinearIntegrator::assemblySeconds() const
{
  return assm_sec_;
}

Real TimeLinearIntegrator::solveSeconds() const
{
  return solve_sec_;
}

Real TimeLinearIntegrator::lastAssemblySeconds() const
{
  return last_assm_sec_;
}

Real TimeLinearIntegrator::lastSolveSeconds() const
{
  return last_solve_sec_;
}

Index TimeLinearIntegrator::assemblyCalls() const
{
  return assm_calls_;
}

Index TimeLinearIntegrator::solveCalls() const
{
  return solve_calls_;
}

Index TimeLinearIntegrator::numSteps() const
{
  return dims_.nt;
}

Index TimeLinearIntegrator::numStates() const
{
  return dims_.nst;
}

Index TimeLinearIntegrator::numParams() const
{
  return dims_.nprm;
}

void TimeLinearIntegrator::solve(const Vector<Real>& prm,
                                 TimeTrajectory&     tr)
{
  solveImpl(prm, &tr);
}

void TimeLinearIntegrator::solve(const Vector<Real>& prm)
{
  solveImpl(prm, nullptr);
}

void TimeLinearIntegrator::solveImpl(const Vector<Real>& prm,
                                     TimeTrajectory*     tr)
{
  if (prm.size() != numParams())
  {
    throw runtime_error(
        "TimeLinearIntegrator parameter size mismatch");
  }

  if (tr != nullptr)
  {
    tr->resize(numSteps(), numStates());
  }

  MonitorScope monitor_scope(*this);

  Vector<Real> init;
  initializeInitialState(init);
  if (tr != nullptr)
  {
    (*tr)[0] = init;
  }
  observeState(0, init);

  Vector<Real> hist;
  initializeHistoryWindow(init, dims_.nhst, numStates(), hist);
  for (Index step = 0; step < numSteps(); ++step)
  {
    Vector<Real> cur_state = VectorView<Real>(hist.data(), numStates());
    Vector<Real> x_next    = cur_state;

    solveStep(step, prm, hist, x_next);
    if (tr != nullptr)
    {
      (*tr)[step + 1] = x_next;
    }

    const TimeStepStateContext ctx{
        step + 1,
        numSteps(),
        cur_state,
        x_next,
        last_assm_sec_,
        last_solve_sec_};

    const bool stop = observeStep(ctx);
    advanceHistoryWindow(hist, dims_.nhst, numStates(), x_next);
    if (stop)
    {
      break;
    }
  }
}

void TimeLinearIntegrator::solveStep(
    Index               step,
    const Vector<Real>& prm,
    const Vector<Real>& hist,
    Vector<Real>&       x_next)
{
  last_assm_sec_  = 0.0;
  last_solve_sec_ = 0.0;

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
    throw runtime_error("TimeLinearIntegrator residual size mismatch");
  }

  const auto assembly_begin = Clock::now();
  if (!problem_.assembleJac(
          ctx, VariableBlock::NextState, J_next_))
  {
    throw runtime_error(
        "TimeLinearIntegrator requires an assembled next-state Jacobian");
  }
  J_next_.finalize();

  Vector<Real> rhs;
  J_next_.apply(x_next, rhs);
  if (rhs.size() != res.size())
  {
    throw runtime_error("TimeLinearIntegrator RHS size mismatch");
  }
  for (Index i = 0; i < res.size(); ++i)
  {
    rhs[i] -= res[i];
  }
  problem_.prepareLinearSolve(ctx, VariableBlock::NextState, J_next_, rhs);
  last_assm_sec_  = elapsedSeconds(assembly_begin);
  assm_sec_      += last_assm_sec_;
  ++assm_calls_;

  Vector<Real> next;
  const auto   solve_begin = Clock::now();

  lin_solver_.solve(J_next_, rhs, next);

  last_solve_sec_  = elapsedSeconds(solve_begin);
  solve_sec_      += last_solve_sec_;
  
  ++solve_calls_;
  if (next.size() != numStates())
  {
    throw runtime_error(
        "TimeLinearIntegrator update size mismatch");
  }

  for (Index i = 0; i < numStates(); ++i)
  {
    x_next[i] = next[i];
  }
}

void TimeLinearIntegrator::initializeInitialState(Vector<Real>& state) const
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
