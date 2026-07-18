#include <chrono>
#include <stdexcept>

#include <femx/linalg/BlockVectorView.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/state/TimeLinearIntegrator.hpp>
#include <femx/state/TimeTrajectory.hpp>
using namespace femx::state;
using namespace femx::linalg;

namespace femx
{
namespace state
{

namespace
{

using Clock = std::chrono::steady_clock;

Real elapsedSeconds(const Clock::time_point& begin)
{
  return std::chrono::duration<Real>(Clock::now() - begin).count();
}

void copyStateToView(const HostVector& src,
                     HostVectorView    dst)
{
  if (src.size() != dst.size())
  {
    throw std::runtime_error(
        "TimeLinearIntegrator state view size mismatch");
  }
  for (Index i = 0; i < src.size(); ++i)
  {
    dst[i] = src[i];
  }
}

void copyView(HostVectorView src,
              HostVectorView dst)
{
  if (src.size() != dst.size())
  {
    throw std::runtime_error(
        "TimeLinearIntegrator history view size mismatch");
  }
  for (Index i = 0; i < src.size(); ++i)
  {
    dst[i] = src[i];
  }
}

void initializeHistoryWindow(const HostVector& initial,
                             Index             depth,
                             Index             num_states,
                             HostVector&       hist)
{
  if (depth < 0 || num_states < 0 || initial.size() != num_states)
  {
    throw std::runtime_error(
        "TimeLinearIntegrator received invalid history window dimensions");
  }
  hist.resize(depth * num_states);
  BlockVectorView<Real> hist_view(hist.data(), depth, num_states);
  for (Index lag = 0; lag < depth; ++lag)
  {
    copyStateToView(initial, hist_view.block(lag));
  }
}

void advanceHistoryWindow(HostVector&       hist,
                          Index             depth,
                          Index             num_states,
                          const HostVector& next)
{
  if (depth < 0 || num_states < 0 || hist.size() != depth * num_states
      || next.size() != num_states)
  {
    throw std::runtime_error(
        "TimeLinearIntegrator history window size mismatch");
  }
  BlockVectorView<Real> hist_view(hist.data(), depth, num_states);
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
  if (dims_.num_res != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeLinearIntegrator requires square state residual dimensions");
  }
  if (dims_.num_history_states <= 0)
  {
    throw std::runtime_error(
        "TimeLinearIntegrator requires at least one history state");
  }
}

void TimeLinearIntegrator::setInitialState(const HostVector& state)
{
  if (state.size() != numStates())
  {
    throw std::runtime_error(
        "TimeLinearIntegrator initial state size mismatch");
  }
  init_state_     = state;
  has_init_state_ = true;
}

void TimeLinearIntegrator::clearInitialState()
{
  init_state_     = HostVector{};
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
  return dims_.num_steps;
}

Index TimeLinearIntegrator::numStates() const
{
  return dims_.num_states;
}

Index TimeLinearIntegrator::numParams() const
{
  return dims_.num_param;
}

void TimeLinearIntegrator::solve(const HostVector& prm,
                                 TimeTrajectory&   tr)
{
  solveImpl(prm, &tr);
}

void TimeLinearIntegrator::solve(const HostVector& prm)
{
  solveImpl(prm, nullptr);
}

void TimeLinearIntegrator::solveImpl(const HostVector& prm,
                                     TimeTrajectory*   tr)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "TimeLinearIntegrator parameter size mismatch");
  }

  if (tr != nullptr)
  {
    tr->resize(numSteps(), numStates());
  }

  MonitorScope monitor_scope(*this);

  HostVector init;
  initializeInitialState(init);
  if (tr != nullptr)
  {
    (*tr)[0] = init;
  }
  observeState(0, init);

  HostVector hist;
  initializeHistoryWindow(init, dims_.num_history_states, numStates(), hist);
  for (Index step = 0; step < numSteps(); ++step)
  {
    HostVector cur_state = HostVectorView(hist.data(), numStates());
    HostVector x_next    = cur_state;

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
    advanceHistoryWindow(hist, dims_.num_history_states, numStates(), x_next);
    if (stop)
    {
      break;
    }
  }
}

void TimeLinearIntegrator::solveStep(
    Index             step,
    const HostVector& prm,
    const HostVector& hist,
    HostVector&       x_next)
{
  last_assm_sec_  = 0.0;
  last_solve_sec_ = 0.0;

  TimeContext ctx;
  ctx.step = step;
  ctx.nxt  = &x_next;
  ctx.prm  = &prm;
  ctx.hist = TimeHistoryView(hist.data(), dims_.num_history_states, numStates());

  HostVector res;
  problem_.res(ctx, res);
  if (res.size() != dims_.num_res)
  {
    throw std::runtime_error("TimeLinearIntegrator residual size mismatch");
  }

  const auto assembly_begin = Clock::now();
  if (!problem_.assembleJac(
          ctx, VariableBlock::NextState, J_next_))
  {
    throw std::runtime_error(
        "TimeLinearIntegrator requires an assembled next-state Jacobian");
  }
  J_next_.finalize();

  HostVector rhs;
  J_next_.apply(x_next, rhs);
  if (rhs.size() != res.size())
  {
    throw std::runtime_error("TimeLinearIntegrator RHS size mismatch");
  }
  for (Index i = 0; i < res.size(); ++i)
  {
    rhs[i] -= res[i];
  }
  problem_.prepareLinearSolve(ctx, VariableBlock::NextState, J_next_, rhs);
  last_assm_sec_  = elapsedSeconds(assembly_begin);
  assm_sec_      += last_assm_sec_;
  ++assm_calls_;

  HostVector next;
  const auto solve_begin = Clock::now();

  lin_solver_.solve(J_next_, rhs, next);

  last_solve_sec_  = elapsedSeconds(solve_begin);
  solve_sec_      += last_solve_sec_;

  ++solve_calls_;
  if (next.size() != numStates())
  {
    throw std::runtime_error(
        "TimeLinearIntegrator update size mismatch");
  }

  for (Index i = 0; i < numStates(); ++i)
  {
    x_next[i] = next[i];
  }
}

void TimeLinearIntegrator::initializeInitialState(HostVector& state) const
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
