#include <stdexcept>

#include <femx/linalg/BlockVectorView.hpp>
#include <femx/solve/TimeStepper.hpp>

using namespace std;
using namespace femx::problem;
using namespace femx::linalg;

namespace femx
{
namespace solve
{

namespace
{

void fillHistory(const TimeTrajectory& tr,
                 Index                 step,
                 Index                 depth,
                 Index                 nst,
                 Vector<Real>&         storage)
{
  if (depth < 0 || nst < 0)
  {
    throw runtime_error("TimeStepper received invalid history dimensions");
  }
  if (storage.size() != depth * nst)
  {
    storage.resize(depth * nst);
  }
  BlockVectorView<Real> hist(storage.data(), depth, nst);
  for (Index lag = 0; lag < depth; ++lag)
  {
    const Index        level = step > lag ? step - lag : 0;
    const Vector<Real> state = tr[level];
    VectorView<Real>   h     = hist.block(lag);
    for (Index i = 0; i < nst; ++i)
    {
      h[i] = state[i];
    }
  }
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

} // namespace

TimeStepper::TimeStepper(const TimeResidual& problem,
                         LinearSolver&       lin_solver)
  : problem_(problem),
    lin_solver_(lin_solver),
    dims_(problem.dims())
{
  if (dims_.nres != dims_.nst)
  {
    throw runtime_error(
        "TimeStepper requires square next-state residual dimensions");
  }
  if (dims_.nhst <= 0)
  {
    throw runtime_error(
        "TimeStepper requires at least one history state");
  }
}

TimeStepperOptions& TimeStepper::opts()
{
  return opts_;
}

const TimeStepperOptions& TimeStepper::opts() const
{
  return opts_;
}

void TimeStepper::setInitialState(const Vector<Real>& state)
{
  if (state.size() != numStates())
  {
    throw runtime_error("TimeStepper initial state size mismatch");
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
  return dims_.nt;
}

Index TimeStepper::numStates() const
{
  return dims_.nst;
}

Index TimeStepper::numParams() const
{
  return dims_.nprm;
}

Index TimeStepper::numResiduals() const
{
  return dims_.nres;
}

void TimeStepper::solve(const Vector<Real>& prm,
                        TimeTrajectory&     tr)
{
  if (prm.size() != numParams())
  {
    throw runtime_error("TimeStepper parameter size mismatch");
  }

  tr.resize(numSteps(), numStates());
  Vector<Real> init_state = tr[0];
  initializeInitialState(initial_state);

  Vector<Real> hist;
  for (Index step = 0; step < numSteps(); ++step)
  {
    fillHistory(tr,
                step,
                dims_.nhst,
                numStates(),
                hist);
    Vector<Real> cur_state =
        Vector<Real>::view(hist.data(), numStates());
    tr[step + 1]     = cur_state;
    Vector<Real> nxt = tr[step + 1];
    solveStep(step, prm, hist, nxt);
    tr[step + 1] = nxt;
  }
}

TimeStepper::NextStateJacobian::NextStateJacobian(
    const TimeStepper& owner)
  : owner_(owner)
{
}

void TimeStepper::NextStateJacobian::reset(TimeContext ctx)
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
      ctx_, VariableBlock::NextState, dir, out);
}

void TimeStepper::NextStateJacobian::applyT(const Vector<Real>& dir,
                                            Vector<Real>&       out) const
{
  owner_.problem_.applyJacT(
      ctx_, VariableBlock::NextState, dir, out);
}

void TimeStepper::solveStep(Index               step,
                            const Vector<Real>& prm,
                            const Vector<Real>& hist,
                            Vector<Real>&       nxt)
{
  Vector<Real>      res;
  Vector<Real>      rhs;
  Vector<Real>      update;
  Vector<Real>      prev;
  NextStateJacobian next_jac(*this);
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
      throw runtime_error("TimeStepper residual size mismatch");
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

    next_jac.reset(ctx);
    lin_solver_.solve(next_jac, rhs, update);
    if (update.size() != numStates())
    {
      throw runtime_error("TimeStepper update size mismatch");
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

  throw runtime_error("TimeStepper failed to converge");
}

void TimeStepper::initializeInitialState(Vector<Real>& state) const
{
  if (has_init_state_)
  {
    state = init_state_;
    return;
  }
  resizeOrZero(state, numStates());
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
