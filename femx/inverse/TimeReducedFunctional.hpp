#pragma once

#include <chrono>
#include <functional>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/inverse/TimeObjective.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/LinearSystem.hpp>
#include <femx/linalg/Vector.hpp>
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

/** @brief Evaluate a transient reduced objective and its adjoint gradient. */
template <MemorySpace Space>
class TimeReducedFunctional final
{
public:
  static constexpr MemorySpace space = Space;

  using Vec        = Vector<Space, Real>;
  using VecView    = VectorView<Space, Real>;
  using ConstView  = VectorView<Space, const Real>;
  using Ctx        = linalg::Context<space>;
  using Integrator = state::TimeIntegrator<Space>;
  using Res        = state::TimeResidual<Space>;
  using Tr         = state::TimeTrajectory;
  using System     = linalg::LinearSystem<space>;
  using StepCtx    = state::TimeStepStateContext;
  using Observer   = typename Integrator::Observer;

  TimeReducedFunctional(Integrator&          integrator,
                        System&              adj_system,
                        const TimeObjective& obj);

  TimeReducedFunctional(const TimeReducedFunctional&)            = delete;
  TimeReducedFunctional& operator=(const TimeReducedFunctional&) = delete;
  TimeReducedFunctional(TimeReducedFunctional&&)                 = delete;
  TimeReducedFunctional& operator=(TimeReducedFunctional&&)      = delete;

  Index numParams() const noexcept;

  Real value(HostVectorView<const Real> prm, TimeReducedProgress progress = {});
  void grad(HostVectorView<const Real> prm,
            HostVectorView<Real>       out,
            TimeReducedProgress        progress = {});
  Real valueGrad(HostVectorView<const Real> prm,
                 HostVectorView<Real>       out,
                 TimeReducedProgress        progress = {});

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
  void                      solveFwd(HostVectorView<const Real> prm,
                                     const TimeReducedProgress& progress);
  void                      assembleNext(Index step);
  void                      solveAdj(HostVectorView<Real>       out,
                                     const TimeReducedProgress& progress);
  void                      notify(const TimeReducedProgress& progress,
                                   const char*                phase,
                                   Index                      step) const;
  static void               checkSize(const Vec& vec, Index expected);

  Integrator&          integrator_;
  const Res&           res_;
  System&              adj_system_;
  Ctx&                 ctx_;
  const TimeObjective& obj_;
  state::TimeDims      dims_;
  Tr                   tr_;
  HostVector<Real>     h_prm_;
  HostVector<Real>     h_rhs_;
  HostVector<Real>     h_grad_;
  Vec                  prm_;
  Vec                  hist_;
  Vec                  nxt_;
  Vec                  rhs_;
  Vec                  result_;
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

using HostTimeReducedFunctional =
    TimeReducedFunctional<MemorySpace::Host>;
using DeviceTimeReducedFunctional =
    TimeReducedFunctional<MemorySpace::Device>;

template <MemorySpace Space>
TimeReducedFunctional<Space>::TimeReducedFunctional(
    Integrator&          integrator,
    System&              adj_system,
    const TimeObjective& obj)
  : integrator_(integrator),
    res_(integrator.residual()),
    adj_system_(adj_system),
    ctx_(adj_system.context()),
    obj_(obj),
    dims_(res_.dims()),
    prm_(dims_.num_param),
    hist_(dims_.num_hist * dims_.num_states),
    nxt_(dims_.num_states),
    rhs_(dims_.num_states),
    result_(dims_.num_states),
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
  ctx_.sync();
}

template <MemorySpace Space>
Index TimeReducedFunctional<Space>::numParams() const noexcept
{
  return integrator_.numParams();
}

template <MemorySpace Space>
Real TimeReducedFunctional<Space>::value(
    HostVectorView<const Real> prm,
    TimeReducedProgress        progress)
{
  resetTiming();
  solveFwd(prm, progress);
  return obj_.value(tr_, h_prm_);
}

template <MemorySpace Space>
void TimeReducedFunctional<Space>::grad(HostVectorView<const Real> prm,
                                        HostVectorView<Real>       out,
                                        TimeReducedProgress        progress)
{
  resetTiming();
  solveFwd(prm, progress);
  solveAdj(out, progress);
}

template <MemorySpace Space>
Real TimeReducedFunctional<Space>::valueGrad(
    HostVectorView<const Real> prm,
    HostVectorView<Real>       out,
    TimeReducedProgress        progress)
{
  resetTiming();
  solveFwd(prm, progress);
  const Real val = obj_.value(tr_, h_prm_);
  solveAdj(out, progress);
  return val;
}

template <MemorySpace Space>
void TimeReducedFunctional<Space>::resetTiming() noexcept
{
  assm_sec_    = 0.0;
  solve_sec_   = 0.0;
  assm_calls_  = 0;
  solve_calls_ = 0;
}

template <MemorySpace Space>
Real TimeReducedFunctional<Space>::assemblySeconds() const noexcept
{
  return assm_sec_;
}

template <MemorySpace Space>
Real TimeReducedFunctional<Space>::solveSeconds() const noexcept
{
  return solve_sec_;
}

template <MemorySpace Space>
Index TimeReducedFunctional<Space>::assemblyCalls() const noexcept
{
  return assm_calls_;
}

template <MemorySpace Space>
Index TimeReducedFunctional<Space>::solveCalls() const noexcept
{
  return solve_calls_;
}

template <MemorySpace Space>
Index TimeReducedFunctional<Space>::numSteps() const noexcept
{
  return integrator_.numSteps();
}

template <MemorySpace Space>
Index TimeReducedFunctional<Space>::numStates() const noexcept
{
  return integrator_.numStates();
}

template <MemorySpace Space>
typename TimeReducedFunctional<Space>::VecView
TimeReducedFunctional<Space>::histState(Index lag)
{
  return hist_.view().subview(lag * numStates(), numStates());
}

template <MemorySpace Space>
typename TimeReducedFunctional<Space>::VecView
TimeReducedFunctional<Space>::carry(Index lag)
{
  require(lag >= 0 && lag < dims_.num_hist,
          "TimeReducedFunctional carry lag is out of range");
  const Index block = (carry_head_ + lag) % dims_.num_hist;
  return carry_.view().subview(block * numStates(), numStates());
}

template <MemorySpace Space>
void TimeReducedFunctional<Space>::resetCarry()
{
  auto& vec_handler = ctx_.vectors();
  vec_handler.zero(carry_.view());
  carry_head_ = 0;
}

template <MemorySpace Space>
void TimeReducedFunctional<Space>::advanceCarry()
{
  auto& vec_handler = ctx_.vectors();
  vec_handler.zero(carry(0));
  carry_head_ = (carry_head_ + 1) % dims_.num_hist;
}

template <MemorySpace Space>
state::TimeContext<TimeReducedFunctional<Space>::space>
TimeReducedFunctional<Space>::timeCtx(Index step) const
{
  return {step,
          nxt_.view(),
          prm_.view(),
          {hist_.data(), dims_.num_hist, numStates()}};
}

template <MemorySpace Space>
void TimeReducedFunctional<Space>::loadStep(Index step)
{
  auto&     vec_handler = ctx_.vectors();
  const Tr& tr          = tr_;
  for (Index lag = 0; lag < dims_.num_hist; ++lag)
  {
    vec_handler.copy(tr.level(detail::histLevel(step, lag)), histState(lag));
  }
  vec_handler.copy(tr.level(step + 1), nxt_.view());
}

template <MemorySpace Space>
void TimeReducedFunctional<Space>::solveFwd(
    HostVectorView<const Real> prm,
    const TimeReducedProgress& progress)
{
  auto& vec_handler = ctx_.vectors();
  require(prm.size() == numParams(),
          "TimeReducedFunctional parameter size mismatch");
  h_prm_ = prm;
  vec_handler.copy(h_prm_.view(), prm_.view());

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

template <MemorySpace Space>
void TimeReducedFunctional<Space>::assembleNext(Index step)
{
  const auto begin = detail::Clock::now();
  loadStep(step);
  auto& jac = adj_system_.matrix();
  jac.setup(res_.hostPattern());
  res_.assembleNext(timeCtx(step), result_, jac, ctx_);
  jac.finalize();
  ctx_.sync();
  assm_sec_ += detail::elapsedSec(begin);
  ++assm_calls_;
}

template <MemorySpace Space>
void TimeReducedFunctional<Space>::solveAdj(
    HostVectorView<Real>       out,
    const TimeReducedProgress& progress)
{
  auto& vec_handler = ctx_.vectors();
  require(out.size() == numParams(),
          "TimeReducedFunctional gradient size mismatch");
  obj_.paramGrad(tr_, h_prm_, h_grad_);
  vec_handler.copy(h_grad_.view(), grad_.view());
  checkSize(grad_, numParams());
  obj_.stateGrad(0, tr_, h_prm_, h_rhs_);
  vec_handler.copy(h_rhs_.view(), init_grad_.view());
  checkSize(init_grad_, numStates());
  resetCarry();

  notify(progress, "adjoint-begin", 0);
  for (Index step = numSteps(); step-- > 0;)
  {
    notify(progress, "adjoint-step", numSteps() - step);
    obj_.stateGrad(step + 1, tr_, h_prm_, h_rhs_);
    vec_handler.copy(h_rhs_.view(), rhs_.view());
    checkSize(rhs_, numStates());

    vec_handler.axpby(1.0, carry(0), 1.0, rhs_.view());
    advanceCarry();

    assembleNext(step);
    const auto solve_begin = detail::Clock::now();
    adj_system_.solveT(rhs_.view(), result_);
    solve_sec_ += detail::elapsedSec(solve_begin);
    ++solve_calls_;
    checkSize(result_, dims_.num_res);

    for (Index lag = 0; lag < dims_.num_hist; ++lag)
    {
      res_.applyJacT(timeCtx(step),
                     state::VariableBlock::hist(lag),
                     result_.view(),
                     rhs_,
                     ctx_);
      checkSize(rhs_, numStates());
      if (detail::histLevel(step, lag) == 0)
      {
        vec_handler.axpby(-1.0, rhs_.view(), 1.0, init_grad_.view());
      }
      else
      {
        vec_handler.axpby(-1.0, rhs_.view(), 1.0, carry(lag));
      }
    }

    res_.applyJacT(timeCtx(step),
                   state::VariableBlock::Param,
                   result_.view(),
                   prm_adj_,
                   ctx_);
    checkSize(prm_adj_, numParams());
    vec_handler.axpby(-1.0, prm_adj_.view(), 1.0, grad_.view());
  }

  res_.addInitialStateJacT(init_grad_.view(), grad_.view(), ctx_);
  vec_handler.copy(grad_.view(), out);
  ctx_.sync();
  notify(progress, "adjoint-end", numSteps());
}

template <MemorySpace Space>
void TimeReducedFunctional<Space>::notify(
    const TimeReducedProgress& progress,
    const char*                phase,
    Index                      step) const
{
  if (progress)
  {
    progress(phase, step, numSteps());
  }
}

template <MemorySpace Space>
void TimeReducedFunctional<Space>::checkSize(const Vec& vec,
                                             Index      expected)
{
  require(vec.size() == expected,
          "TimeReducedFunctional vector size mismatch");
}

} // namespace femx::inverse
