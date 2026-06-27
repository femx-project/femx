#include <stdexcept>

#include <femx/linalg/BlockVectorView.hpp>
#include <femx/state/TimeNewtonIntegrator.hpp>

using namespace std;
using namespace femx::problem;
using namespace femx::linalg;

namespace femx
{
namespace state
{

namespace
{

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
        "TimeNewtonIntegrator state view size mismatch");
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
        "TimeNewtonIntegrator history view size mismatch");
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
        "TimeNewtonIntegrator received invalid history window dimensions");
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
        "TimeNewtonIntegrator history window size mismatch");
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

TimeNewtonIntegrator::TimeNewtonIntegrator(
    const TimeResidual& problem,
    LinearSolver&       lin_solver)
  : problem_(problem),
    lin_solver_(lin_solver),
    dims_(problem.dims())
{
  if (dims_.nres != dims_.nst)
  {
    throw runtime_error(
        "TimeNewtonIntegrator requires square next-state residual dimensions");
  }
  if (dims_.nhst <= 0)
  {
    throw runtime_error(
        "TimeNewtonIntegrator requires at least one history state");
  }
}

TimeNewtonOptions& TimeNewtonIntegrator::opts()
{
  return opts_;
}

const TimeNewtonOptions& TimeNewtonIntegrator::opts() const
{
  return opts_;
}

void TimeNewtonIntegrator::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw runtime_error("TimeNewtonIntegrator initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void TimeNewtonIntegrator::clearInitialState()
{
  init_state_     = Vector<Real>{};
  has_init_state_ = false;
}

Index TimeNewtonIntegrator::numSteps() const
{
  return dims_.nt;
}

Index TimeNewtonIntegrator::numStates() const
{
  return dims_.nst;
}

Index TimeNewtonIntegrator::numParams() const
{
  return dims_.nprm;
}

Index TimeNewtonIntegrator::numResiduals() const
{
  return dims_.nres;
}

void TimeNewtonIntegrator::solve(const Vector<Real>& prm,
                                  TimeTrajectory&     tr)
{
  solveImpl(prm, &tr);
}

void TimeNewtonIntegrator::solve(const Vector<Real>& prm)
{
  solveImpl(prm, nullptr);
}

void TimeNewtonIntegrator::solveImpl(const Vector<Real>& prm,
                                      TimeTrajectory*     tr)
{
  if (prm.size() != numParams())
  {
    throw runtime_error("TimeNewtonIntegrator parameter size mismatch");
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
    Vector<Real> cur_state = Vector<Real>::view(hist.data(), numStates());
    Vector<Real> nxt       = cur_state;
    solveStep(step, prm, hist, nxt);
    if (tr != nullptr)
    {
      (*tr)[step + 1] = nxt;
    }
    const TimeStepStateContext ctx{
        step + 1,
        numSteps(),
        cur_state,
        nxt,
        0.0,
        0.0};
    const bool stop = observeStep(ctx);
    advanceHistoryWindow(hist, dims_.nhst, numStates(), nxt);
    if (stop)
    {
      break;
    }
  }
}

TimeNewtonIntegrator::NextStateJacobian::NextStateJacobian(
    const TimeNewtonIntegrator& owner)
  : owner_(owner)
{
}

void TimeNewtonIntegrator::NextStateJacobian::reset(TimeContext ctx)
{
  ctx_ = ctx;
}

Index TimeNewtonIntegrator::NextStateJacobian::numRows() const
{
  return owner_.numResiduals();
}

Index TimeNewtonIntegrator::NextStateJacobian::numCols() const
{
  return owner_.numStates();
}

void TimeNewtonIntegrator::NextStateJacobian::apply(
    const Vector<Real>& dir,
    Vector<Real>&       out) const
{
  owner_.problem_.applyJac(
      ctx_, VariableBlock::NextState, dir, out);
}

void TimeNewtonIntegrator::NextStateJacobian::applyT(
    const Vector<Real>& dir,
    Vector<Real>&       out) const
{
  owner_.problem_.applyJacT(
      ctx_, VariableBlock::NextState, dir, out);
}

void TimeNewtonIntegrator::solveStep(Index               step,
                                      const Vector<Real>& prm,
                                      const Vector<Real>& hist,
                                      Vector<Real>&       nxt)
{
  Vector<Real>      res;
  Vector<Real>      rhs;
  Vector<Real>      update;
  Vector<Real>      prev;
  NextStateJacobian J_next(*this);
  copyHistoryState(hist, numStates(), prev);

  TimeContext ctx;
  ctx.step = step;
  ctx.prev = &prev;
  ctx.nxt  = &nxt;
  ctx.prm  = &prm;
  ctx.hist = TimeHistoryView(hist.data(), dims_.nhst, numStates());

  for (Index iter = 0; iter <= opts_.max_its; ++iter)
  {
    problem_.res(ctx, res);
    if (res.size() != numResiduals())
    {
      throw runtime_error("TimeNewtonIntegrator residual size mismatch");
    }

    if (squaredNorm(res)
        <= opts_.residual_tolerance * opts_.residual_tolerance)
    {
      return;
    }
    if (iter == opts_.max_its)
    {
      break;
    }

    resizeOrZero(rhs, res.size());
    for (Index i = 0; i < res.size(); ++i)
    {
      rhs[i] = -res[i];
    }

    J_next.reset(ctx);
    lin_solver_.solve(J_next, rhs, update);
    if (update.size() != numStates())
    {
      throw runtime_error("TimeNewtonIntegrator update size mismatch");
    }

    for (Index i = 0; i < numStates(); ++i)
    {
      nxt[i] += update[i];
    }

    if (squaredNorm(update)
        <= opts_.step_tolerance * opts_.step_tolerance)
    {
      return;
    }
  }

  throw runtime_error("TimeNewtonIntegrator failed to converge");
}

void TimeNewtonIntegrator::initializeInitialState(Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }
  resizeOrZero(state, numStates());
}

Real TimeNewtonIntegrator::squaredNorm(const Vector<Real>& x)
{
  Real value = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    value += x[i] * x[i];
  }
  return value;
}

} // namespace state
} // namespace femx
