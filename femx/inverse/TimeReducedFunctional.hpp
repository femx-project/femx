#pragma once

#include <chrono>
#include <functional>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/inverse/TimeObjective.hpp>
#include <femx/linalg/Backend.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <femx/state/TimeResidual.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx::inverse
{

using TimeReducedProgress =
    std::function<void(const char* phase, Index step, Index total_steps)>;

namespace detail
{

using Clock = std::chrono::steady_clock;

inline Real elapsedSec(const Clock::time_point& begin)
{
  return std::chrono::duration<Real>(Clock::now() - begin).count();
}

inline Index histLevel(Index step, Index lag)
{
  return step > lag ? step - lag : 0;
}

} // namespace detail

/** @brief Backend-independent transient reduced functional and adjoint solve. */
template <class Backend>
class TimeReducedFunctional final
{
  static_assert(linalg::is_backend_v<Backend>,
                "TimeReducedFunctional requires a valid backend type");

public:
  static constexpr MemorySpace space = Backend::space;

  using Vec        = typename Backend::Vec;
  using VecView    = typename Backend::VecView;
  using ConstView  = typename Backend::ConstView;
  using Mat        = typename Backend::Mat;
  using Ctx        = typename Backend::Ctx;
  using Integrator = state::TimeIntegrator<Backend>;
  using Res        = state::TimeResidual<Backend>;
  using Tr         = state::TimeTrajectory;
  using Solver     = linalg::LinearSolver<Backend>;
  using StepCtx    = state::TimeStepStateContext;
  using Observer   = typename Integrator::Observer;

  TimeReducedFunctional(Integrator&          integrator,
                        Mat&                 jac,
                        Solver&              adj_solver,
                        const TimeObjective& obj);

  TimeReducedFunctional(const TimeReducedFunctional&)            = delete;
  TimeReducedFunctional& operator=(const TimeReducedFunctional&) = delete;
  TimeReducedFunctional(TimeReducedFunctional&&)                 = delete;
  TimeReducedFunctional& operator=(TimeReducedFunctional&&)      = delete;

  Index numParams() const noexcept;

  Real value(HostConstVectorView prm, TimeReducedProgress progress = {});
  void grad(HostConstVectorView prm,
            HostVectorView      out,
            TimeReducedProgress progress = {});
  Real valueGrad(HostConstVectorView prm,
                 HostVectorView      out,
                 TimeReducedProgress progress = {});

  void  resetTiming() noexcept;
  Real  assemblySeconds() const noexcept;
  Real  solveSeconds() const noexcept;
  Index assemblyCalls() const noexcept;
  Index solveCalls() const noexcept;

private:
  Index numSteps() const noexcept;
  Index numStates() const noexcept;

  VecView histState(Index lag);
  VecView carry(Index lag);
  void    resetCarry();
  void    advanceCarry();

  state::TimeContext<space> timeCtx(Index step) const;
  void                      loadStep(Index step);
  void                      solveFwd(HostConstVectorView        prm,
                                     const TimeReducedProgress& progress);
  Mat&                      assembleNext(Index step);
  void                      solveAdj(HostVectorView             out,
                                     const TimeReducedProgress& progress);
  void                      notify(const TimeReducedProgress& progress,
                                   const char*                phase,
                                   Index                      step) const;
  static void               checkSize(const Vec& vec, Index expected);

  Integrator&          integrator_;
  const Res&           res_;
  Mat&                 jac_;
  Solver&              adj_solver_;
  Ctx&                 ctx_;
  const TimeObjective& obj_;
  state::TimeDims      dims_;
  Tr                   tr_;
  HostVector           host_prm_;
  HostVector           host_rhs_;
  HostVector           host_grad_;
  Vec                  prm_;
  Vec                  hist_;
  Vec                  nxt_;
  Vec                  rhs_;
  Vec                  sol_;
  Vec                  carry_;
  Vec                  prm_adj_;
  Vec                  init_grad_;
  Vec                  grad_;
  Real                 assm_sec_{0.0};
  Real                 solve_sec_{0.0};
  Index                assm_calls_{0};
  Index                solve_calls_{0};
  Index                carry_head_{0};
};

using HostTimeReducedFunctional   = TimeReducedFunctional<linalg::HostCsrBackend>;
using DeviceTimeReducedFunctional = TimeReducedFunctional<linalg::CudaCsrBackend>;

template <class Backend>
TimeReducedFunctional<Backend>::TimeReducedFunctional(
    Integrator&          integrator,
    Mat&                 jac,
    Solver&              adj_solver,
    const TimeObjective& obj)
  : integrator_(integrator),
    res_(integrator.residual()),
    jac_(jac),
    adj_solver_(adj_solver),
    ctx_(integrator.context()),
    obj_(obj),
    dims_(res_.dims()),
    prm_(dims_.num_param),
    hist_(dims_.num_hist * dims_.num_states),
    nxt_(dims_.num_states),
    rhs_(dims_.num_states),
    sol_(dims_.num_states),
    carry_(dims_.num_hist * dims_.num_states),
    prm_adj_(dims_.num_param),
    init_grad_(dims_.num_states),
    grad_(dims_.num_param)
{
  require(dims_.num_hist > 0 && dims_.num_res == dims_.num_states,
          "TimeReducedFunctional requires square residuals and history states");
  require(obj_.numSteps() == numSteps()
              && obj_.numStates() == numStates()
              && obj_.numParams() == numParams(),
          "TimeReducedFunctional objective dimensions do not match");
  ctx_.synchronize();
}

template <class Backend>
Index TimeReducedFunctional<Backend>::numParams() const noexcept
{
  return integrator_.numParams();
}

template <class Backend>
Real TimeReducedFunctional<Backend>::value(
    HostConstVectorView prm,
    TimeReducedProgress progress)
{
  resetTiming();
  solveFwd(prm, progress);
  return obj_.value(tr_, host_prm_);
}

template <class Backend>
void TimeReducedFunctional<Backend>::grad(HostConstVectorView prm,
                                          HostVectorView      out,
                                          TimeReducedProgress progress)
{
  resetTiming();
  solveFwd(prm, progress);
  solveAdj(out, progress);
}

template <class Backend>
Real TimeReducedFunctional<Backend>::valueGrad(
    HostConstVectorView prm,
    HostVectorView      out,
    TimeReducedProgress progress)
{
  resetTiming();
  solveFwd(prm, progress);
  const Real val = obj_.value(tr_, host_prm_);
  solveAdj(out, progress);
  return val;
}

template <class Backend>
void TimeReducedFunctional<Backend>::resetTiming() noexcept
{
  assm_sec_    = 0.0;
  solve_sec_   = 0.0;
  assm_calls_  = 0;
  solve_calls_ = 0;
}

template <class Backend>
Real TimeReducedFunctional<Backend>::assemblySeconds() const noexcept
{
  return assm_sec_;
}

template <class Backend>
Real TimeReducedFunctional<Backend>::solveSeconds() const noexcept
{
  return solve_sec_;
}

template <class Backend>
Index TimeReducedFunctional<Backend>::assemblyCalls() const noexcept
{
  return assm_calls_;
}

template <class Backend>
Index TimeReducedFunctional<Backend>::solveCalls() const noexcept
{
  return solve_calls_;
}

template <class Backend>
Index TimeReducedFunctional<Backend>::numSteps() const noexcept
{
  return integrator_.numSteps();
}

template <class Backend>
Index TimeReducedFunctional<Backend>::numStates() const noexcept
{
  return integrator_.numStates();
}

template <class Backend>
typename TimeReducedFunctional<Backend>::VecView
TimeReducedFunctional<Backend>::histState(Index lag)
{
  return hist_.view().subview(lag * numStates(), numStates());
}

template <class Backend>
typename TimeReducedFunctional<Backend>::VecView
TimeReducedFunctional<Backend>::carry(Index lag)
{
  require(lag >= 0 && lag < dims_.num_hist,
          "TimeReducedFunctional carry lag is out of range");
  const Index block = (carry_head_ + lag) % dims_.num_hist;
  return carry_.view().subview(block * numStates(), numStates());
}

template <class Backend>
void TimeReducedFunctional<Backend>::resetCarry()
{
  zero(carry_.view(), ctx_);
  carry_head_ = 0;
}

template <class Backend>
void TimeReducedFunctional<Backend>::advanceCarry()
{
  zero(carry(0), ctx_);
  carry_head_ = (carry_head_ + 1) % dims_.num_hist;
}

template <class Backend>
state::TimeContext<TimeReducedFunctional<Backend>::space>
TimeReducedFunctional<Backend>::timeCtx(Index step) const
{
  return {step,
          nxt_.view(),
          prm_.view(),
          {hist_.data(), dims_.num_hist, numStates()}};
}

template <class Backend>
void TimeReducedFunctional<Backend>::loadStep(Index step)
{
  const Tr& tr = tr_;
  for (Index lag = 0; lag < dims_.num_hist; ++lag)
  {
    copy(tr.level(detail::histLevel(step, lag)), histState(lag), ctx_);
  }
  copy(tr.level(step + 1), nxt_.view(), ctx_);
}

template <class Backend>
void TimeReducedFunctional<Backend>::solveFwd(
    HostConstVectorView        prm,
    const TimeReducedProgress& progress)
{
  require(prm.size() == numParams(),
          "TimeReducedFunctional parameter size mismatch");
  host_prm_ = prm;
  copy(host_prm_.view(), prm_.view(), ctx_);

  notify(progress, "forward-begin", 0);
  Observer observer;
  if (progress)
  {
    observer = [&progress](const StepCtx& ctx)
    {
      if (ctx.level > 0)
      {
        progress("forward-step", ctx.level, ctx.total_steps);
      }
      return false;
    };
  }

  const state::SolveStats stats = integrator_.solve(prm_.view(), tr_, std::move(observer));

  assm_sec_    += stats.assm_sec;
  solve_sec_   += stats.lin_solve_sec;
  assm_calls_  += stats.assm_calls;
  solve_calls_ += stats.lin_solve_calls;

  notify(progress, "forward-end", numSteps());
  require(tr_.numSteps() == numSteps() && tr_.numStates() == numStates(),
          "TimeReducedFunctional forward trajectory size mismatch");
}

template <class Backend>
typename TimeReducedFunctional<Backend>::Mat&
TimeReducedFunctional<Backend>::assembleNext(Index step)
{
  const auto begin = detail::Clock::now();
  loadStep(step);
  res_.assembleNext(timeCtx(step), sol_, jac_, ctx_);
  finalize(jac_, ctx_);
  ctx_.synchronize();
  assm_sec_ += detail::elapsedSec(begin);
  ++assm_calls_;
  return jac_;
}

template <class Backend>
void TimeReducedFunctional<Backend>::solveAdj(
    HostVectorView             out,
    const TimeReducedProgress& progress)
{
  require(out.size() == numParams(),
          "TimeReducedFunctional gradient size mismatch");
  obj_.paramGrad(tr_, host_prm_, host_grad_);
  copy(host_grad_.view(), grad_.view(), ctx_);
  checkSize(grad_, numParams());
  obj_.stateGrad(0, tr_, host_prm_, host_rhs_);
  copy(host_rhs_.view(), init_grad_.view(), ctx_);
  checkSize(init_grad_, numStates());
  resetCarry();

  notify(progress, "adjoint-begin", 0);
  for (Index step = numSteps(); step-- > 0;)
  {
    notify(progress, "adjoint-step", numSteps() - step);
    obj_.stateGrad(step + 1, tr_, host_prm_, host_rhs_);
    copy(host_rhs_.view(), rhs_.view(), ctx_);
    checkSize(rhs_, numStates());

    axpby(1.0, carry(0), 1.0, rhs_.view(), ctx_);
    advanceCarry();

    Mat&       nxt_jac     = assembleNext(step);
    const auto solve_begin = detail::Clock::now();
    adj_solver_.solveT(nxt_jac, rhs_, sol_, ctx_);
    solve_sec_ += detail::elapsedSec(solve_begin);
    ++solve_calls_;
    checkSize(sol_, dims_.num_res);

    for (Index lag = 0; lag < dims_.num_hist; ++lag)
    {
      res_.applyJacT(timeCtx(step),
                     state::VariableBlock::hist(lag),
                     sol_.view(),
                     rhs_,
                     ctx_);
      checkSize(rhs_, numStates());
      if (detail::histLevel(step, lag) == 0)
      {
        axpby(-1.0, rhs_.view(), 1.0, init_grad_.view(), ctx_);
      }
      else
      {
        axpby(-1.0, rhs_.view(), 1.0, carry(lag), ctx_);
      }
    }

    res_.applyJacT(timeCtx(step),
                   state::VariableBlock::Param,
                   sol_.view(),
                   prm_adj_,
                   ctx_);
    checkSize(prm_adj_, numParams());
    axpby(-1.0, prm_adj_.view(), 1.0, grad_.view(), ctx_);
  }

  res_.addInitialStateJacobianTranspose(init_grad_.view(), grad_.view(), ctx_);
  copy(grad_.view(), out, ctx_);
  ctx_.synchronize();
  notify(progress, "adjoint-end", numSteps());
}

template <class Backend>
void TimeReducedFunctional<Backend>::notify(
    const TimeReducedProgress& progress,
    const char*                phase,
    Index                      step) const
{
  if (progress)
  {
    progress(phase, step, numSteps());
  }
}

template <class Backend>
void TimeReducedFunctional<Backend>::checkSize(const Vec& vec,
                                               Index      expected)
{
  require(vec.size() == expected,
          "TimeReducedFunctional vector size mismatch");
}

} // namespace femx::inverse
