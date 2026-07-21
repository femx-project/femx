#pragma once

#include <chrono>
#include <functional>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Backend.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>
#include <femx/state/TimeResidual.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx::state
{

/** @brief Host state and timing data observed after one time level. */
struct TimeStepStateContext
{
  Index               level{0};
  Index               total_steps{0};
  HostConstVectorView prev;
  HostConstVectorView curr;
  Real                assm_sec{0.0};
  Real                lin_solve_sec{0.0};
};

/** @brief Timings and operation counts for one time solve. */
struct SolveStats
{
  Real  assm_sec{0.0};
  Real  lin_solve_sec{0.0};
  Index assm_calls{0};
  Index lin_solve_calls{0};
};

/** @brief Backend-independent implicit time integrator. */
template <class Backend>
class TimeIntegrator final
{
  static_assert(linalg::is_backend_v<Backend>,
                "TimeIntegrator requires a valid backend type");

public:
  static constexpr MemorySpace space = Backend::space;

  using Vec       = typename Backend::Vec;
  using VecView   = typename Backend::VecView;
  using ConstView = typename Backend::ConstView;
  using Mat       = typename Backend::Mat;
  using Ctx       = typename Backend::Ctx;
  using Res       = TimeResidual<Backend>;
  using Solver    = linalg::LinearSolver<Backend>;
  using Tr        = TimeTrajectory;
  using StepCtx   = TimeStepStateContext;
  using Observer  = std::function<bool(const StepCtx&)>;

  TimeIntegrator(const Res& res, Mat& jac, Solver& solver, Ctx& ctx);

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
  SolveStats solve(ConstView prm, Tr& tr, Observer observer = {});

  const SolveStats& lastStats() const noexcept;
  void              resetStats() noexcept;

private:
  VecView            histState(Index lag);
  TimeContext<space> timeCtx(Index step, ConstView prm) const;
  void               initialize(ConstView prm);
  void               advanceHist();
  SolveStats         solveStep(Index step, ConstView prm);
  SolveStats         solveImpl(ConstView prm, Tr* tr, Observer observer);

  const Res& res_;
  Mat&       jac_;
  Solver&    solver_;
  Ctx&       ctx_;
  TimeDims   dims_;
  Vec        init_;
  Vec        hist_;
  Vec        nxt_;
  Vec        res_vec_;
  Vec        rhs_;
  Vec        sol_;
  bool       has_init_{false};
  SolveStats stats_;
};

using HostTimeIntegrator   = TimeIntegrator<linalg::HostCsrBackend>;
using DeviceTimeIntegrator = TimeIntegrator<linalg::CudaCsrBackend>;

namespace detail
{

using TimeClock = std::chrono::steady_clock;

inline Real elapsedSec(const TimeClock::time_point& begin)
{
  return std::chrono::duration<Real>(TimeClock::now() - begin).count();
}

} // namespace detail

template <class Backend>
TimeIntegrator<Backend>::TimeIntegrator(const Res& res,
                                        Mat&       jac,
                                        Solver&    solver,
                                        Ctx&       ctx)
  : res_(res), jac_(jac), solver_(solver), ctx_(ctx), dims_(res.dims())
{
  require(dims_.num_res == dims_.num_states,
          "TimeIntegrator requires square residual dimensions");
  require(dims_.num_hist > 0,
          "TimeIntegrator requires at least one history state");
  require(res_.pattern().rows() == dims_.num_res
              && res_.pattern().cols() == dims_.num_states,
          "TimeIntegrator residual pattern dimensions do not match");

  init_.resize(numStates());
  hist_.resize(dims_.num_hist * numStates());
  nxt_.resize(numStates());
  res_vec_.resize(dims_.num_res);
  rhs_.resize(dims_.num_res);
  sol_.resize(numStates());
  ctx_.sync();
}

template <class Backend>
Index TimeIntegrator<Backend>::numSteps() const noexcept
{
  return dims_.num_steps;
}

template <class Backend>
Index TimeIntegrator<Backend>::numStates() const noexcept
{
  return dims_.num_states;
}

template <class Backend>
Index TimeIntegrator<Backend>::numParams() const noexcept
{
  return dims_.num_param;
}

template <class Backend>
const typename TimeIntegrator<Backend>::Res&
TimeIntegrator<Backend>::residual() const noexcept
{
  return res_;
}

template <class Backend>
typename TimeIntegrator<Backend>::Ctx&
TimeIntegrator<Backend>::context() const noexcept
{
  return ctx_;
}

template <class Backend>
void TimeIntegrator<Backend>::setInitialState(ConstView state)
{
  require(state.size() == numStates(),
          "TimeIntegrator initial state size mismatch");
  linalg::VectorHandler<Backend> vec_handler(ctx_);
  vec_handler.copy(state, init_);
  ctx_.sync();
  has_init_ = true;
}

template <class Backend>
void TimeIntegrator<Backend>::setInitialState(const Vec& state)
{
  setInitialState(state.view());
}

template <class Backend>
void TimeIntegrator<Backend>::clearInitialState() noexcept
{
  has_init_ = false;
}

template <class Backend>
SolveStats TimeIntegrator<Backend>::solve(ConstView prm, Observer observer)
{
  return solveImpl(prm, nullptr, std::move(observer));
}

template <class Backend>
SolveStats TimeIntegrator<Backend>::solve(ConstView prm,
                                          Tr&       tr,
                                          Observer  observer)
{
  return solveImpl(prm, &tr, std::move(observer));
}

template <class Backend>
const SolveStats& TimeIntegrator<Backend>::lastStats() const noexcept
{
  return stats_;
}

template <class Backend>
void TimeIntegrator<Backend>::resetStats() noexcept
{
  stats_ = {};
}

template <class Backend>
typename TimeIntegrator<Backend>::VecView
TimeIntegrator<Backend>::histState(Index lag)
{
  return hist_.view().subview(lag * numStates(), numStates());
}

template <class Backend>
TimeContext<TimeIntegrator<Backend>::space>
TimeIntegrator<Backend>::timeCtx(Index step, ConstView prm) const
{
  return {step,
          nxt_.view(),
          prm,
          {hist_.data(), dims_.num_hist, numStates()}};
}

template <class Backend>
void TimeIntegrator<Backend>::initialize(ConstView prm)
{
  linalg::VectorHandler<Backend> vec_handler(ctx_);
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

template <class Backend>
void TimeIntegrator<Backend>::advanceHist()
{
  linalg::VectorHandler<Backend> vec_handler(ctx_);
  for (Index lag = dims_.num_hist - 1; lag > 0; --lag)
  {
    vec_handler.copy(histState(lag - 1), histState(lag));
  }
  vec_handler.copy(sol_.view(), histState(0));
  vec_handler.copy(sol_.view(), nxt_);
}

template <class Backend>
SolveStats TimeIntegrator<Backend>::solveStep(Index step, ConstView prm)
{
  linalg::VectorHandler<Backend> vec_handler(ctx_);
  linalg::MatrixHandler<Backend> mat_handler(ctx_);
  const TimeContext<space>       time       = timeCtx(step, prm);
  const auto                     assm_begin = detail::TimeClock::now();

  res_.assembleNext(time, res_vec_, jac_, ctx_);
  require(res_vec_.size() == dims_.num_res, "TimeIntegrator residual size mismatch");

  mat_handler.finalize(jac_);
  mat_handler.matvec(jac_, nxt_.view(), rhs_);
  vec_handler.axpby(-1.0, res_vec_.view(), 1.0, rhs_.view());

  res_.prepareLinearSolve(time, jac_, rhs_, ctx_);
  ctx_.sync();

  const Real assm_sec = detail::elapsedSec(assm_begin);

  const auto solve_begin = detail::TimeClock::now();
  solver_.solve(jac_, rhs_, sol_, ctx_);
  const Real lin_solve_sec = detail::elapsedSec(solve_begin);

  require(sol_.size() == numStates(), "TimeIntegrator solution size mismatch");
  advanceHist();
  return {assm_sec, lin_solve_sec, 1, 1};
}

template <class Backend>
SolveStats TimeIntegrator<Backend>::solveImpl(ConstView prm,
                                              Tr*       tr,
                                              Observer  observer)
{
  linalg::VectorHandler<Backend> vec_handler(ctx_);
  require(prm.size() == numParams(),
          "TimeIntegrator parameter size mismatch");

  stats_ = {};
  if (tr != nullptr)
  {
    tr->resize(numSteps(), numStates());
  }

  initialize(prm);
  if (tr != nullptr)
  {
    vec_handler.copy(nxt_.view(), tr->level(0));
  }

  HostVector obs_prev;
  HostVector obs_curr;
  if (observer)
  {
    HostConstVectorView init;
    if (tr != nullptr)
    {
      ctx_.sync();
      init = static_cast<const Tr&>(*tr).level(0);
    }
    else
    {
      obs_prev.resize(numStates());
      obs_curr.resize(numStates());
      vec_handler.copy(nxt_.view(), obs_prev.view());
      ctx_.sync();
      init = static_cast<const HostVector&>(obs_prev).view();
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

    if (tr != nullptr)
    {
      vec_handler.copy(nxt_.view(), tr->level(step + 1));
    }

    if (observer)
    {
      HostConstVectorView prev;
      HostConstVectorView curr;
      if (tr != nullptr)
      {
        ctx_.sync();
        const Tr& const_tr = *tr;
        prev               = const_tr.level(step);
        curr               = const_tr.level(step + 1);
      }
      else
      {
        vec_handler.copy(nxt_.view(), obs_curr.view());
        ctx_.sync();
        prev = static_cast<const HostVector&>(obs_prev).view();
        curr = static_cast<const HostVector&>(obs_curr).view();
      }

      const bool stop = observer({step + 1,
                                  numSteps(),
                                  prev,
                                  curr,
                                  step_stats.assm_sec,
                                  step_stats.lin_solve_sec});
      if (tr == nullptr)
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
