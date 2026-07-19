#pragma once

#include <chrono>
#include <functional>
#include <memory>
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

template <MemorySpace Space>
class ObjectiveEval;

/** @brief Direct evaluation of a Host objective for any Host-storage backend. */
template <>
class ObjectiveEval<MemorySpace::Host> final
{
public:
  template <class Ctx>
  ObjectiveEval(const TimeObjective& obj, Ctx&)
    : obj_(obj)
  {
  }

  Index numSteps() const noexcept
  {
    return obj_.numSteps();
  }

  Index numStates() const noexcept
  {
    return obj_.numStates();
  }

  Index numParams() const noexcept
  {
    return obj_.numParams();
  }

  template <class Ctx>
  Real value(const state::TimeTrajectory& tr,
             const HostVector&            prm,
             Ctx&) const
  {
    return obj_.value(tr, prm);
  }

  template <class Ctx>
  void stateGrad(Index                        level,
                 const state::TimeTrajectory& tr,
                 const HostVector&            prm,
                 HostVector&                  out,
                 Ctx&) const
  {
    obj_.stateGrad(level, tr, prm, out);
  }

  template <class Ctx>
  void paramGrad(const state::TimeTrajectory& tr,
                 const HostVector&            prm,
                 HostVector&                  out,
                 Ctx&) const
  {
    obj_.paramGrad(tr, prm, out);
  }

private:
  const TimeObjective& obj_;
};

/** @brief Device execution adapter for the flattened CUDA objective plan. */
template <>
class ObjectiveEval<MemorySpace::Device> final
{
public:
  ObjectiveEval(const TimeObjective& obj, CudaContext& ctx);
  ~ObjectiveEval();

  ObjectiveEval(const ObjectiveEval&)            = delete;
  ObjectiveEval& operator=(const ObjectiveEval&) = delete;
  ObjectiveEval(ObjectiveEval&&)                 = delete;
  ObjectiveEval& operator=(ObjectiveEval&&)      = delete;

  Index numSteps() const noexcept;
  Index numStates() const noexcept;
  Index numParams() const noexcept;

  Real value(const state::DeviceTimeTrajectory& tr,
             const DeviceVector&                prm,
             CudaContext&                       ctx) const;

  void stateGrad(Index                              level,
                 const state::DeviceTimeTrajectory& tr,
                 const DeviceVector&                prm,
                 DeviceVector&                      out,
                 CudaContext&                       ctx) const;

  void paramGrad(const state::DeviceTimeTrajectory& tr,
                 const DeviceVector&                prm,
                 DeviceVector&                      out,
                 CudaContext&                       ctx) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

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
  using Tr         = state::Trajectory<space>;
  using Solver     = linalg::LinearSolver<Backend>;
  using StepCtx    = state::TimeStepStateContext<space>;
  using Observer   = typename Integrator::Observer;
  using Obj        = detail::ObjectiveEval<space>;

  TimeReducedFunctional(Integrator&          integrator,
                        Mat&                 jac,
                        Solver&              adj_solver,
                        const TimeObjective& obj);

  TimeReducedFunctional(const TimeReducedFunctional&)            = delete;
  TimeReducedFunctional& operator=(const TimeReducedFunctional&) = delete;
  TimeReducedFunctional(TimeReducedFunctional&&)                 = delete;
  TimeReducedFunctional& operator=(TimeReducedFunctional&&)      = delete;

  Index numParams() const noexcept;

  Real value(ConstView prm, TimeReducedProgress progress = {});
  void grad(ConstView           prm,
            VecView             out,
            TimeReducedProgress progress = {});
  Real valueGrad(ConstView           prm,
                 VecView             out,
                 TimeReducedProgress progress = {});

  void  resetTiming() noexcept;
  Real  assemblySeconds() const noexcept;
  Real  solveSeconds() const noexcept;
  Index assemblyCalls() const noexcept;
  Index solveCalls() const noexcept;

private:
  Index numSteps() const noexcept;
  Index numStates() const noexcept;

  VecView   histState(Index lag);
  VecView   adj(Index step);
  ConstView adj(Index step) const;

  state::TimeContext<space> timeCtx(Index step) const;
  void                      loadStep(Index step);
  void                      solveFwd(ConstView prm, const TimeReducedProgress& progress);
  Mat&                      assemble(Index step, state::VariableBlock wrt);
  void                      solveAdj(VecView out, const TimeReducedProgress& progress);
  void                      notify(const TimeReducedProgress& progress,
                                   const char*                phase,
                                   Index                      step) const;
  static void               checkSize(const Vec& vec, Index expected);

  Integrator&     integrator_;
  const Res&      res_;
  Mat&            jac_;
  Solver&         adj_solver_;
  Ctx&            ctx_;
  Obj             obj_;
  state::TimeDims dims_;
  Tr              tr_;
  Vec             prm_;
  Vec             hist_;
  Vec             nxt_;
  Vec             rhs_;
  Vec             sol_;
  Vec             adjs_;
  Vec             carry_;
  Vec             prm_adj_;
  Vec             init_grad_;
  Vec             grad_;
  Real            assm_sec_{0.0};
  Real            solve_sec_{0.0};
  Index           assm_calls_{0};
  Index           solve_calls_{0};
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
    obj_(obj, ctx_),
    dims_(res_.dims()),
    prm_(dims_.num_param),
    hist_(dims_.num_hist * dims_.num_states),
    nxt_(dims_.num_states),
    rhs_(dims_.num_states),
    sol_(dims_.num_states),
    adjs_(dims_.num_steps * dims_.num_states),
    carry_(dims_.num_states),
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
Real TimeReducedFunctional<Backend>::value(ConstView           prm,
                                           TimeReducedProgress progress)
{
  resetTiming();
  solveFwd(prm, progress);
  return obj_.value(tr_, prm_, ctx_);
}

template <class Backend>
void TimeReducedFunctional<Backend>::grad(ConstView           prm,
                                          VecView             out,
                                          TimeReducedProgress progress)
{
  resetTiming();
  solveFwd(prm, progress);
  solveAdj(out, progress);
}

template <class Backend>
Real TimeReducedFunctional<Backend>::valueGrad(
    ConstView           prm,
    VecView             out,
    TimeReducedProgress progress)
{
  resetTiming();
  solveFwd(prm, progress);
  const Real val = obj_.value(tr_, prm_, ctx_);
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
TimeReducedFunctional<Backend>::adj(Index step)
{
  return adjs_.view().subview(step * numStates(), numStates());
}

template <class Backend>
typename TimeReducedFunctional<Backend>::ConstView
TimeReducedFunctional<Backend>::adj(Index step) const
{
  return adjs_.view().subview(step * numStates(), numStates());
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
  for (Index lag = 0; lag < dims_.num_hist; ++lag)
  {
    copy(tr_.level(detail::histLevel(step, lag)), histState(lag), ctx_);
  }
  copy(tr_.level(step + 1), nxt_, ctx_);
}

template <class Backend>
void TimeReducedFunctional<Backend>::solveFwd(
    ConstView                  prm,
    const TimeReducedProgress& progress)
{
  require(prm.size() == numParams(),
          "TimeReducedFunctional parameter size mismatch");
  copy(prm, prm_, ctx_);

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
TimeReducedFunctional<Backend>::assemble(Index                step,
                                         state::VariableBlock wrt)
{
  const auto begin = detail::Clock::now();
  loadStep(step);
  res_.assembleJac(timeCtx(step), wrt, jac_, ctx_);
  finalize(jac_, ctx_);
  ctx_.synchronize();
  assm_sec_ += detail::elapsedSec(begin);
  ++assm_calls_;
  return jac_;
}

template <class Backend>
void TimeReducedFunctional<Backend>::solveAdj(
    VecView                    out,
    const TimeReducedProgress& progress)
{
  require(out.size() == numParams(),
          "TimeReducedFunctional gradient size mismatch");
  obj_.paramGrad(tr_, prm_, grad_, ctx_);
  checkSize(grad_, numParams());

  notify(progress, "adjoint-begin", 0);
  for (Index step = numSteps(); step-- > 0;)
  {
    notify(progress, "adjoint-step", numSteps() - step);
    obj_.stateGrad(step + 1, tr_, prm_, rhs_, ctx_);
    checkSize(rhs_, numStates());

    const Index level = step + 1;
    for (Index lag = 0; lag < dims_.num_hist; ++lag)
    {
      const Index future = level + lag;
      if (future >= numSteps())
      {
        break;
      }
      Mat& hist_jac = assemble(future, state::VariableBlock::hist(lag));
      applyT(hist_jac, adj(future), carry_, ctx_);
      checkSize(carry_, numStates());
      axpby(-1.0, carry_.view(), 1.0, rhs_.view(), ctx_);
    }

    Mat&       nxt_jac     = assemble(step, state::VariableBlock::NextState);
    const auto solve_begin = detail::Clock::now();
    adj_solver_.solveT(nxt_jac, rhs_, sol_, ctx_);
    solve_sec_ += detail::elapsedSec(solve_begin);
    ++solve_calls_;
    checkSize(sol_, dims_.num_res);
    copy(sol_.view(), adj(step), ctx_);

    res_.applyJacT(timeCtx(step),
                   state::VariableBlock::Param,
                   sol_.view(),
                   prm_adj_,
                   ctx_);
    checkSize(prm_adj_, numParams());
    axpby(-1.0, prm_adj_.view(), 1.0, grad_.view(), ctx_);
  }

  obj_.stateGrad(0, tr_, prm_, init_grad_, ctx_);
  checkSize(init_grad_, numStates());
  for (Index step = 0; step < numSteps(); ++step)
  {
    for (Index lag = 0; lag < dims_.num_hist; ++lag)
    {
      if (detail::histLevel(step, lag) != 0)
      {
        continue;
      }
      Mat& hist_jac = assemble(step, state::VariableBlock::hist(lag));
      applyT(hist_jac, adj(step), carry_, ctx_);
      checkSize(carry_, numStates());
      axpby(-1.0,
            carry_.view(),
            1.0,
            init_grad_.view(),
            ctx_);
    }
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
