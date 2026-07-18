#include <chrono>
#include <functional>
#include <stdexcept>
#include <utility>

#include "ResolveTimeIntegrator.hpp"
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/BoundaryMap.hpp>
#include <femx/assembly/CudaAssembly.hpp>
#include <femx/inverse/DeviceTimeObjective.hpp>
#include <femx/inverse/TimeObjective.hpp>
#include <femx/inverse/TimeReducedFunctional.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/CsrTranspose.hpp>
#include <femx/model/ns/Components.hpp>
#include <femx/model/ns/NavierStokesModel.hpp>

namespace femx::model::ns
{
namespace
{

using Clock = std::chrono::steady_clock;

Real secondsSince(const Clock::time_point& start)
{
  return std::chrono::duration<Real>(Clock::now() - start).count();
}

Index historyLevel(Index step, Index lag)
{
  return step > lag ? step - lag : 0;
}

Array<Index> boundaryDofs(const fem::HostControlMap& ctr)
{
  Array<Index> out(ctr.numBcs());
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = ctr.dofs()[i];
  }
  return out;
}

fem::HostControlMap fixedControl(const NavierStokesModel& model,
                                 Array<Index>             dofs,
                                 HostVector               vals)
{
  return fem::makeControlMap(model.numSteps(),
                             model.numStates(),
                             {},
                             std::move(dofs),
                             std::move(vals),
                             {},
                             0,
                             0);
}

} // namespace

class ResolveTimeIntegrator::Impl
{
public:
  using InitObserver = std::function<void(const HostVector&)>;
  using Observer =
      std::function<bool(const state::TimeStepStateContext&)>;
  using StepNotify = std::function<void(Index)>;

  Impl(const NavierStokesModel&        model,
       fem::HostControlMap             ctr,
       HostVector                      init,
       const fem::HostInitialStateMap* init_map,
       linalg::ReSolveOptions          opts)
    : num_steps_(model.numSteps()),
      num_states_(model.numStates()),
      num_prm_(ctr.numParams()),
      host_graph_(model.map().graph()),
      init_(num_states_),
      hist_(2 * num_states_),
      nxt_(num_states_),
      res_(num_states_),
      rhs_(num_states_),
      sol_(num_states_),
      prm_(num_prm_),
      bc_vals_(ctr.numBcs()),
      solver_(opts),
      adj_solver_(std::move(opts))
  {
    if (ctr.numSteps() != num_steps_
        || ctr.numStates() != num_states_)
    {
      throw std::runtime_error(
          "ResolveTimeIntegrator control dimensions do not match the model");
    }
    if ((!init.empty() && init.size() != num_states_)
        || (init_map != nullptr
            && (init_map->numStates() != num_states_
                || init_map->numParams() != num_prm_)))
    {
      throw std::runtime_error(
          "ResolveTimeIntegrator initial-state dimensions do not match control");
    }

    copy(model.map(), map_, ctx_);
    const assembly::HostBoundaryMap host_bc =
        assembly::makeBoundaryMap(boundaryDofs(ctr), host_graph_);
    assembly::copy(host_bc, bc_, ctx_);
    fem::copy(ctr, ctr_, ctx_);
    if (init_map != nullptr)
    {
      fem::copy(*init_map, init_map_, ctx_);
      has_init_map_ = true;
    }
    else if (init.empty())
    {
      init_.setZero(ctx_);
    }
    else
    {
      femx::copy(init, init_, ctx_);
    }
    ns::copy(model.data(), data_, ctx_);

    jac_       = std::make_unique<DeviceCsrMatrix>(map_.graph());
    solve_mat_ = std::make_unique<DeviceCsrMatrix>(map_.graph());
    op_        = NavierOperator<MemorySpace::Device>(
        data_.view(),
        {model.fluid().rho, model.fluid().mu},
        model.dt());
    ctx_.synchronize();
  }

  Index numSteps() const noexcept
  {
    return num_steps_;
  }

  Index numStates() const noexcept
  {
    return num_states_;
  }

  Index numParams() const noexcept
  {
    return num_prm_;
  }

  CudaContext& ctx() noexcept
  {
    return ctx_;
  }

  DeviceConstVectorView prm() const noexcept
  {
    return prm_.view();
  }

  void setInitialState(const HostVector& state)
  {
    if (has_init_map_)
    {
      throw std::runtime_error(
          "ResolveTimeIntegrator controlled initial state is set by its map");
    }
    if (state.size() != num_states_)
    {
      throw std::runtime_error(
          "ResolveTimeIntegrator initial state size mismatch");
    }
    femx::copy(state, init_, ctx_);
    ctx_.synchronize();
  }

  void setParam(const HostVector& prm)
  {
    if (prm.size() != num_prm_)
    {
      throw std::runtime_error(
          "ResolveTimeIntegrator parameter size mismatch");
    }
    femx::copy(prm, prm_, ctx_);
  }

  void solve(state::DeviceTimeTrajectory& tr,
             const InitObserver&          init_obs,
             const Observer&              observe,
             const StepNotify&            notify)
  {
    tr.resize(num_steps_, num_states_);
    initialize(tr);

    HostVector prev;
    HostVector curr;
    if (observe || init_obs)
    {
      femx::copy(init_, prev, ctx_);
      ctx_.synchronize();
      if (init_obs)
      {
        init_obs(prev);
      }
      if (observe)
      {
        curr.resize(num_states_);
      }
    }

    for (Index step = 0; step < num_steps_; ++step)
    {
      const auto assm_start = Clock::now();
      assembly::assemble(op_,
                         step,
                         2,
                         state::VariableBlock::NextState,
                         map_,
                         hist_,
                         nxt_,
                         res_,
                         *jac_,
                         ctx_);
      fem::controlVals(
          ctr_.view(), step, prm_.view(), bc_vals_.view(), ctx_);
      assembly::replaceRes(bc_, nxt_, bc_vals_, res_, ctx_);
      assembly::replaceRows(bc_, *jac_, 1.0, ctx_);
      femx::apply(*jac_, nxt_.view(), rhs_.view(), ctx_);
      femx::axpby(-1.0, res_.view(), 1.0, rhs_.view(), ctx_);
      femx::copy(jac_->vals().view(), solve_mat_->vals().view(), ctx_);
      assembly::prepareForwardSolve(
          bc_, *solve_mat_, rhs_, bc_vals_, ctx_);

      last_assm_sec_  = secondsSince(assm_start);
      assm_sec_      += last_assm_sec_;
      ++assm_calls_;

      const auto solve_start = Clock::now();
      solver_.setOperator(*solve_mat_);
      solver_.solve(rhs_, sol_, ctx_);
      last_solve_sec_  = secondsSince(solve_start);
      solve_sec_      += last_solve_sec_;
      ++solve_calls_;

      femx::copy(sol_.view(), tr.view().level(step + 1), ctx_);
      advance();
      if (notify)
      {
        notify(step + 1);
      }

      if (observe)
      {
        femx::copy(sol_, curr, ctx_);
        ctx_.synchronize();
        const state::TimeStepStateContext step_ctx{
            step + 1,
            num_steps_,
            prev,
            curr,
            last_assm_sec_,
            last_solve_sec_};
        if (observe(step_ctx))
        {
          break;
        }
        prev = curr;
      }
    }
  }

  void copyToHost(const state::DeviceTimeTrajectory& src,
                  state::TimeTrajectory&             dst)
  {
    state::copy(src, dst, ctx_);
    ctx_.synchronize();
  }

  state::DeviceTimeTrajectory& trajectory() noexcept
  {
    return tr_;
  }

  void prepareAdjoint()
  {
    if (tr_mat_ != nullptr)
    {
      return;
    }

    const HostCsrTransposeMap tr(host_graph_);
    femx::copy(tr, map_.graph(), tr_map_, ctx_);
    tr_mat_ = std::make_unique<DeviceCsrMatrix>(tr_map_.trGraph());
    adj_solver_.setOperator(*tr_mat_);
    adjs_.resize(num_steps_ * num_states_);
    carry_.resize(num_states_);
    prm_adj_.resize(num_prm_);
    init_grad_.resize(num_states_);
    grad_.resize(num_prm_);
    ctx_.synchronize();
  }

  void gradAt(const inverse::DeviceTimeObjective& obj,
              const StepNotify&                   notify)
  {
    prepareAdjoint();
    obj.paramGrad(tr_, prm_.view(), grad_.view(), ctx_);

    for (Index step = num_steps_; step-- > 0;)
    {
      if (notify)
      {
        notify(num_steps_ - step);
      }
      obj.stateGrad(
          step + 1, tr_, prm_.view(), rhs_.view(), ctx_);

      const Index level = step + 1;
      for (Index lag = 0; lag < 2; ++lag)
      {
        const Index future = level + lag;
        if (future >= num_steps_)
        {
          break;
        }
        assembleAt(tr_,
                   future,
                   state::VariableBlock::hist(lag),
                   0.0);
        applyJacT(adj(future), carry_.view());
        femx::axpby(-1.0, carry_.view(), 1.0, rhs_.view(), ctx_);
      }

      assembleAt(
          tr_, step, state::VariableBlock::NextState, 1.0);
      femx::trVals(*jac_, tr_map_, *tr_mat_, ctx_);
      const auto solve_start = Clock::now();
      adj_solver_.solve(rhs_, sol_, ctx_);
      solve_sec_ += secondsSince(solve_start);
      ++solve_calls_;
      femx::copy(sol_.view(), adj(step), ctx_);

      prm_adj_.setZero(ctx_);
      fem::addControlJacT(
          ctr_.view(), step, sol_.view(), prm_adj_.view(), ctx_);
      femx::axpby(
          -1.0, prm_adj_.view(), 1.0, grad_.view(), ctx_);
    }

    obj.stateGrad(0, tr_, prm_.view(), init_grad_.view(), ctx_);
    for (Index step = 0; step < num_steps_; ++step)
    {
      for (Index lag = 0; lag < 2; ++lag)
      {
        if (historyLevel(step, lag) != 0)
        {
          continue;
        }
        assembleAt(tr_,
                   step,
                   state::VariableBlock::hist(lag),
                   0.0);
        applyJacT(adj(step), carry_.view());
        femx::axpby(
            -1.0, carry_.view(), 1.0, init_grad_.view(), ctx_);
      }
    }
    if (has_init_map_)
    {
      fem::addInitialJacT(
          init_map_.view(), init_grad_.view(), grad_.view(), ctx_);
    }
  }

  void copyGrad(HostVector& out)
  {
    femx::copy(grad_, out, ctx_);
    ctx_.synchronize();
  }

  Real assemblySeconds() const noexcept
  {
    return assm_sec_;
  }

  Real solveSeconds() const noexcept
  {
    return solve_sec_;
  }

  Index assemblyCalls() const noexcept
  {
    return assm_calls_;
  }

  Index solveCalls() const noexcept
  {
    return solve_calls_;
  }

  void resetTiming() noexcept
  {
    assm_sec_       = 0.0;
    solve_sec_      = 0.0;
    last_assm_sec_  = 0.0;
    last_solve_sec_ = 0.0;
    assm_calls_     = 0;
    solve_calls_    = 0;
  }

private:
  void initialize(state::DeviceTimeTrajectory& tr)
  {
    if (has_init_map_)
    {
      fem::initialState(
          init_map_.view(), prm_.view(), init_.view(), ctx_);
    }
    femx::copy(init_.view(), nxt_.view(), ctx_);
    femx::copy(init_.view(), hist_.view().subview(0, num_states_), ctx_);
    femx::copy(init_.view(),
               hist_.view().subview(num_states_, num_states_),
               ctx_);
    femx::copy(init_.view(), tr.view().level(0), ctx_);
  }

  void advance()
  {
    femx::copy(hist_.view().subview(0, num_states_),
               hist_.view().subview(num_states_, num_states_),
               ctx_);
    femx::copy(sol_.view(), hist_.view().subview(0, num_states_), ctx_);
    femx::copy(sol_.view(), nxt_.view(), ctx_);
  }

  void loadStep(const state::DeviceTimeTrajectory& tr, Index step)
  {
    for (Index lag = 0; lag < 2; ++lag)
    {
      femx::copy(
          tr.level(historyLevel(step, lag)),
          hist_.view().subview(lag * num_states_, num_states_),
          ctx_);
    }
    femx::copy(tr.level(step + 1), nxt_.view(), ctx_);
  }

  void assembleAt(const state::DeviceTimeTrajectory& tr,
                  Index                              step,
                  state::VariableBlock               wrt,
                  Real                               diag)
  {
    const auto start = Clock::now();
    loadStep(tr, step);
    assembly::assemble(
        op_, step, 2, wrt, map_, hist_, nxt_, res_, *jac_, ctx_);
    assembly::replaceRows(bc_, *jac_, diag, ctx_);
    assm_sec_ += secondsSince(start);
    ++assm_calls_;
  }

  DeviceVectorView adj(Index step)
  {
    return adjs_.view().subview(step * num_states_, num_states_);
  }

  DeviceConstVectorView adj(Index step) const
  {
    return adjs_.view().subview(step * num_states_, num_states_);
  }

  void applyJacT(DeviceConstVectorView dir, DeviceVectorView out)
  {
    femx::trVals(*jac_, tr_map_, *tr_mat_, ctx_);
    femx::apply(*tr_mat_, dir, out, ctx_);
  }

private:
  Index num_steps_{0};
  Index num_states_{0};
  Index num_prm_{0};
  bool  has_init_map_{false};

  CudaContext                         ctx_;
  HostCsrGraph                        host_graph_;
  assembly::DeviceAssemblyMap         map_;
  assembly::DeviceBoundaryMap         bc_;
  fem::DeviceControlMap               ctr_;
  fem::DeviceInitialStateMap          init_map_;
  DeviceNavierData                    data_;
  NavierOperator<MemorySpace::Device> op_;

  std::unique_ptr<DeviceCsrMatrix> jac_;
  std::unique_ptr<DeviceCsrMatrix> solve_mat_;
  DeviceCsrTransposeMap            tr_map_;
  std::unique_ptr<DeviceCsrMatrix> tr_mat_;

  DeviceVector init_;
  DeviceVector hist_;
  DeviceVector nxt_;
  DeviceVector res_;
  DeviceVector rhs_;
  DeviceVector sol_;
  DeviceVector prm_;
  DeviceVector bc_vals_;
  DeviceVector adjs_;
  DeviceVector carry_;
  DeviceVector prm_adj_;
  DeviceVector init_grad_;
  DeviceVector grad_;

  linalg::ReSolveLinearSolver solver_;
  linalg::ReSolveLinearSolver adj_solver_;
  state::DeviceTimeTrajectory tr_;

  Real  assm_sec_{0.0};
  Real  solve_sec_{0.0};
  Real  last_assm_sec_{0.0};
  Real  last_solve_sec_{0.0};
  Index assm_calls_{0};
  Index solve_calls_{0};
};

ResolveTimeIntegrator::ResolveTimeIntegrator(
    const NavierStokesModel& model,
    Array<Index>             bc_dofs,
    HostVector               bc_vals,
    linalg::ReSolveOptions   opts)
  : impl_(std::make_unique<Impl>(model,
                                 fixedControl(model,
                                              std::move(bc_dofs),
                                              std::move(bc_vals)),
                                 HostVector{},
                                 nullptr,
                                 std::move(opts)))
{
}

ResolveTimeIntegrator::ResolveTimeIntegrator(
    const NavierStokesModel& model,
    fem::HostControlMap      ctr,
    HostVector               init,
    linalg::ReSolveOptions   opts)
  : impl_(std::make_unique<Impl>(model,
                                 std::move(ctr),
                                 std::move(init),
                                 nullptr,
                                 std::move(opts)))
{
}

ResolveTimeIntegrator::ResolveTimeIntegrator(
    const NavierStokesModel& model,
    fem::HostControlMap      ctr,
    fem::HostInitialStateMap init,
    linalg::ReSolveOptions   opts)
  : impl_(std::make_unique<Impl>(
        model, std::move(ctr), HostVector{}, &init, std::move(opts)))
{
}

ResolveTimeIntegrator::~ResolveTimeIntegrator() = default;

Index ResolveTimeIntegrator::numSteps() const
{
  return impl_->numSteps();
}

Index ResolveTimeIntegrator::numStates() const
{
  return impl_->numStates();
}

Index ResolveTimeIntegrator::numParams() const
{
  return impl_->numParams();
}

void ResolveTimeIntegrator::setInitialState(const HostVector& state)
{
  impl_->setInitialState(state);
}

void ResolveTimeIntegrator::solve(const HostVector&      prm,
                                  state::TimeTrajectory& tr)
{
  auto& dtr = impl_->trajectory();
  solve(prm, dtr);
  impl_->copyToHost(dtr, tr);
}

void ResolveTimeIntegrator::solve(const HostVector&            prm,
                                  state::DeviceTimeTrajectory& tr)
{
  impl_->setParam(prm);
  MonitorScope scope(*this);

  Impl::InitObserver init_obs;
  Impl::Observer     observer;
  if (hasMonitor())
  {
    init_obs = [this](const HostVector& state)
    {
      observeState(0, state);
    };
    observer = [this](const state::TimeStepStateContext& ctx)
    {
      return observeStep(ctx);
    };
  }
  impl_->solve(tr, init_obs, observer, {});
}

Real ResolveTimeIntegrator::assemblySeconds() const
{
  return impl_->assemblySeconds();
}

Real ResolveTimeIntegrator::solveSeconds() const
{
  return impl_->solveSeconds();
}

Index ResolveTimeIntegrator::assemblyCalls() const
{
  return impl_->assemblyCalls();
}

Index ResolveTimeIntegrator::solveCalls() const
{
  return impl_->solveCalls();
}

void ResolveTimeIntegrator::resetTiming()
{
  impl_->resetTiming();
}

ResolveTimeReducedFunctional::ResolveTimeReducedFunctional(
    ResolveTimeIntegrator&        integ,
    const inverse::TimeObjective& obj)
  : integ_(integ),
    obj_(std::make_unique<inverse::DeviceTimeObjective>())
{
  obj_->add(obj, integ_.impl_->ctx());
  if (obj_->numSteps() != integ_.numSteps()
      || obj_->numStates() != integ_.numStates()
      || obj_->numParams() != integ_.numParams())
  {
    throw std::runtime_error(
        "ResolveTimeReducedFunctional objective dimensions do not match");
  }
}

ResolveTimeReducedFunctional::~ResolveTimeReducedFunctional() = default;

Index ResolveTimeReducedFunctional::numParams() const
{
  return integ_.numParams();
}

Real ResolveTimeReducedFunctional::value(const HostVector& prm)
{
  solveForward(prm);
  return obj_->value(integ_.impl_->trajectory(),
                     integ_.impl_->prm(),
                     integ_.impl_->ctx());
}

void ResolveTimeReducedFunctional::grad(const HostVector& prm,
                                        HostVector&       out)
{
  solveForward(prm);
  solveAdjoint(out);
}

Real ResolveTimeReducedFunctional::valueGrad(const HostVector& prm,
                                             HostVector&       out)
{
  solveForward(prm);
  const Real val = obj_->value(integ_.impl_->trajectory(),
                               integ_.impl_->prm(),
                               integ_.impl_->ctx());
  solveAdjoint(out);
  return val;
}

void ResolveTimeReducedFunctional::setMonitor(
    inverse::TimeReducedProgressMonitor* monitor)
{
  monitor_ = monitor;
}

void ResolveTimeReducedFunctional::clearMonitor()
{
  monitor_ = nullptr;
}

Real ResolveTimeReducedFunctional::assemblySeconds() const
{
  return integ_.assemblySeconds();
}

Real ResolveTimeReducedFunctional::solveSeconds() const
{
  return integ_.solveSeconds();
}

Index ResolveTimeReducedFunctional::assemblyCalls() const
{
  return integ_.assemblyCalls();
}

Index ResolveTimeReducedFunctional::solveCalls() const
{
  return integ_.solveCalls();
}

void ResolveTimeReducedFunctional::resetTiming()
{
  integ_.resetTiming();
}

void ResolveTimeReducedFunctional::solveForward(const HostVector& prm)
{
  integ_.impl_->setParam(prm);
  notify("forward-begin", 0);
  integ_.impl_->solve(
      integ_.impl_->trajectory(), {}, {}, [this](Index step)
      { notify("forward-step", step); });
  notify("forward-end", integ_.numSteps());
}

void ResolveTimeReducedFunctional::solveAdjoint(HostVector& out)
{
  notify("adjoint-begin", 0);
  integ_.impl_->gradAt(*obj_, [this](Index step)
                       { notify("adjoint-step", step); });
  integ_.impl_->copyGrad(out);
  notify("adjoint-end", integ_.numSteps());
}

void ResolveTimeReducedFunctional::notify(const char* phase,
                                          Index       step)
{
  if (monitor_ != nullptr)
  {
    monitor_->progress(phase, step, integ_.numSteps());
  }
}

} // namespace femx::model::ns
