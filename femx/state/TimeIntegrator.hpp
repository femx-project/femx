#pragma once

#include <chrono>
#include <functional>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/LinearSystem.hpp>
#include <femx/state/TimeResidual.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx::state
{

/**
 * @brief Host state and timing data observed after one time level.
 */
struct TimeStepStateContext
{
  Index                      level{0};
  Index                      total_steps{0};
  HostVectorView<const Real> prev;
  HostVectorView<const Real> curr;
  Real                       assm_sec{0.0};
  Real                       lin_solve_sec{0.0};
};

/**
 * @brief Timings and operation counts for one time solve.
 */
struct SolveStats
{
  Real  assm_sec{0.0};
  Real  lin_solve_sec{0.0};
  Index assm_calls{0};
  Index lin_solve_calls{0};
};

/**
 * @brief Integrate an implicit time residual in one memory space.
 */
template <MemorySpace Space>
class TimeIntegrator final
{
public:
  static constexpr MemorySpace space = Space;

  using Vec       = Vector<Space, Real>;
  using VecView   = VectorView<Space, Real>;
  using ConstView = VectorView<Space, const Real>;
  using Ctx       = linalg::Context<space>;
  using Res       = TimeResidual<Space>;
  using System    = linalg::LinearSystem<space>;
  using Tr        = TimeTrajectory;
  using StepCtx   = TimeStepStateContext;
  using Observer  = std::function<bool(const StepCtx&)>;

  TimeIntegrator(const Res& res, System& system);

  TimeIntegrator(const TimeIntegrator&)            = delete;
  TimeIntegrator& operator=(const TimeIntegrator&) = delete;
  TimeIntegrator(TimeIntegrator&&)                 = delete;
  TimeIntegrator& operator=(TimeIntegrator&&)      = delete;

  Index numSteps() const noexcept;
  Index numStates() const noexcept;
  Index numParams() const noexcept;

  const Res& residual() const noexcept;
  Ctx&       context() const noexcept;

  void setInitialState(ConstView state);
  void setInitialState(const Vec& state);
  void clearInitialState() noexcept;

  SolveStats solve(ConstView prm, Observer observer = {});
  SolveStats solve(ConstView prm, Tr& traj, Observer observer = {});

  const SolveStats& lastStats() const noexcept;
  void              resetStats() noexcept;

private:
  VecView            histState(Index lag);
  TimeContext<space> timeCtx(Index step, ConstView prm) const;
  void               initialize(ConstView prm);
  void               advanceHist();
  SolveStats         solveStep(Index step, ConstView prm);
  SolveStats         solveImpl(ConstView prm, Tr* traj, Observer observer);

  const Res& res_;
  System&    system_;
  Ctx&       ctx_;
  TimeDims   dims_;
  Vec        init_;
  Vec        hist_;
  Vec        nxt_;
  Vec        res_vec_;
  Vec        rhs_;
  Vec        x_;
  bool       has_init_{false};
  SolveStats stats_;
};

using HostTimeIntegrator   = TimeIntegrator<MemorySpace::Host>;
using DeviceTimeIntegrator = TimeIntegrator<MemorySpace::Device>;

namespace detail
{

using TimeClock = std::chrono::steady_clock;

inline Real elapsedSec(const TimeClock::time_point& begin)
{
  return std::chrono::duration<Real>(TimeClock::now() - begin).count();
}

} // namespace detail

template <MemorySpace Space>
TimeIntegrator<Space>::TimeIntegrator(const Res& res,
                                      System&    system)
  : res_(res),
    system_(system),
    ctx_(system.context()),
    dims_(res.dims())
{
  require(dims_.num_res == dims_.num_states,
          "TimeIntegrator requires square residual dimensions");
  require(dims_.num_hist > 0,
          "TimeIntegrator requires at least one history state");
  require(res_.hostPattern().rows() == dims_.num_res
              && res_.hostPattern().cols() == dims_.num_states,
          "TimeIntegrator residual pattern dimensions do not match");

  init_.resize(numStates());
  hist_.resize(dims_.num_hist * numStates());
  nxt_.resize(numStates());
  res_vec_.resize(dims_.num_res);
  rhs_.resize(dims_.num_res);
  x_.resize(numStates());
  ctx_.sync();
}

template <MemorySpace Space>
Index TimeIntegrator<Space>::numSteps() const noexcept
{
  return dims_.num_steps;
}

template <MemorySpace Space>
Index TimeIntegrator<Space>::numStates() const noexcept
{
  return dims_.num_states;
}

template <MemorySpace Space>
Index TimeIntegrator<Space>::numParams() const noexcept
{
  return dims_.num_param;
}

template <MemorySpace Space>
const typename TimeIntegrator<Space>::Res&
TimeIntegrator<Space>::residual() const noexcept
{
  return res_;
}

template <MemorySpace Space>
typename TimeIntegrator<Space>::Ctx&
TimeIntegrator<Space>::context() const noexcept
{
  return ctx_;
}

template <MemorySpace Space>
void TimeIntegrator<Space>::setInitialState(ConstView state)
{
  require(state.size() == numStates(),
          "TimeIntegrator initial state size mismatch");
  auto& vec_handler = ctx_.vectorHandler();
  vec_handler.copy(state, init_);
  ctx_.sync();
  has_init_ = true;
}

template <MemorySpace Space>
void TimeIntegrator<Space>::setInitialState(const Vec& state)
{
  setInitialState(state.view());
}

template <MemorySpace Space>
void TimeIntegrator<Space>::clearInitialState() noexcept
{
  has_init_ = false;
}

template <MemorySpace Space>
SolveStats TimeIntegrator<Space>::solve(ConstView prm, Observer observer)
{
  return solveImpl(prm, nullptr, std::move(observer));
}

template <MemorySpace Space>
SolveStats TimeIntegrator<Space>::solve(ConstView prm,
                                        Tr&       traj,
                                        Observer  observer)
{
  return solveImpl(prm, &traj, std::move(observer));
}

template <MemorySpace Space>
const SolveStats& TimeIntegrator<Space>::lastStats() const noexcept
{
  return stats_;
}

template <MemorySpace Space>
void TimeIntegrator<Space>::resetStats() noexcept
{
  stats_ = {};
}

template <MemorySpace Space>
typename TimeIntegrator<Space>::VecView
TimeIntegrator<Space>::histState(Index lag)
{
  return hist_.view().subview(lag * numStates(), numStates());
}

template <MemorySpace Space>
TimeContext<TimeIntegrator<Space>::space>
TimeIntegrator<Space>::timeCtx(Index step, ConstView prm) const
{
  return {step,
          nxt_.view(),
          prm,
          {hist_.data(), dims_.num_hist, numStates()}};
}

template <MemorySpace Space>
void TimeIntegrator<Space>::initialize(ConstView prm)
{
  auto& vec_handler = ctx_.vectorHandler();
  if (!has_init_)
  {
    res_.initialState(prm, init_, ctx_);
  }
  vec_handler.copy(init_.view(), nxt_);
  for (Index lag = 0; lag < dims_.num_hist; ++lag)
  {
    vec_handler.copy(init_.view(), histState(lag));
  }
}

template <MemorySpace Space>
void TimeIntegrator<Space>::advanceHist()
{
  auto& vec_handler = ctx_.vectorHandler();
  for (Index lag = dims_.num_hist - 1; lag > 0; --lag)
  {
    vec_handler.copy(histState(lag - 1), histState(lag));
  }
  vec_handler.copy(x_.view(), histState(0));
  vec_handler.copy(x_.view(), nxt_);
}

template <MemorySpace Space>
SolveStats TimeIntegrator<Space>::solveStep(Index step, ConstView prm)
{
  auto&                    vec_handler = ctx_.vectorHandler();
  auto&                    jac         = system_.matrix();
  const TimeContext<space> time        = timeCtx(step, prm);
  const auto               assm_begin  = detail::TimeClock::now();

  jac.setup(res_.hostPattern());
  res_.assembleNext(time, res_vec_, jac, ctx_);
  require(res_vec_.size() == dims_.num_res, "TimeIntegrator residual size mismatch");

  jac.finalize();
  jac.matvec(nxt_.view(), rhs_);
  vec_handler.axpby(-1.0, res_vec_.view(), 1.0, rhs_.view());

  res_.setup(time, jac, rhs_, ctx_);
  ctx_.sync();

  const Real assm_sec = detail::elapsedSec(assm_begin);

  const auto solve_begin = detail::TimeClock::now();
  system_.solve(rhs_.view(), x_);
  const Real solve_sec = detail::elapsedSec(solve_begin);

  require(x_.size() == numStates(), "TimeIntegrator solution size mismatch");
  advanceHist();
  return {assm_sec, solve_sec, 1, 1};
}

template <MemorySpace Space>
SolveStats TimeIntegrator<Space>::solveImpl(ConstView prm,
                                            Tr*       traj,
                                            Observer  observer)
{
  auto& vec_handler = ctx_.vectorHandler();
  require(prm.size() == numParams(),
          "TimeIntegrator parameter size mismatch");

  stats_ = {};
  if (traj != nullptr)
  {
    traj->resize(numSteps(), numStates());
  }

  initialize(prm);
  if (traj != nullptr)
  {
    vec_handler.copy(nxt_.view(), traj->level(0));
  }

  HostVector<Real> obs_prev;
  HostVector<Real> obs_curr;
  if (observer)
  {
    HostVectorView<const Real> init;
    if (traj != nullptr)
    {
      ctx_.sync();
      init = static_cast<const Tr&>(*traj).level(0);
    }
    else
    {
      obs_prev.resize(numStates());
      obs_curr.resize(numStates());
      vec_handler.copy(nxt_.view(), obs_prev.view());
      ctx_.sync();
      init = static_cast<const HostVector<Real>&>(obs_prev).view();
    }
    if (observer({0, numSteps(), init, init, 0.0, 0.0}))
    {
      return stats_;
    }
  }

  for (Index step = 0; step < numSteps(); ++step)
  {
    const SolveStats step_stats  = solveStep(step, prm);
    stats_.assm_sec             += step_stats.assm_sec;
    stats_.lin_solve_sec        += step_stats.lin_solve_sec;
    stats_.assm_calls           += step_stats.assm_calls;
    stats_.lin_solve_calls      += step_stats.lin_solve_calls;

    if (traj != nullptr)
    {
      vec_handler.copy(nxt_.view(), traj->level(step + 1));
    }

    if (observer)
    {
      HostVectorView<const Real> prev;
      HostVectorView<const Real> curr;
      if (traj != nullptr)
      {
        ctx_.sync();
        const Tr& const_traj = *traj;
        prev                 = const_traj.level(step);
        curr                 = const_traj.level(step + 1);
      }
      else
      {
        vec_handler.copy(nxt_.view(), obs_curr.view());
        ctx_.sync();
        prev = static_cast<const HostVector<Real>&>(obs_prev).view();
        curr = static_cast<const HostVector<Real>&>(obs_curr).view();
      }

      const bool stop = observer({step + 1,
                                  numSteps(),
                                  prev,
                                  curr,
                                  step_stats.assm_sec,
                                  step_stats.lin_solve_sec});
      if (traj == nullptr)
      {
        std::swap(obs_prev, obs_curr);
      }
      if (stop)
      {
        break;
      }
    }
  }

  ctx_.sync();
  return stats_;
}

} // namespace femx::state
