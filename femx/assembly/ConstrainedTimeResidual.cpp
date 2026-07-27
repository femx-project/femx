#include <utility>

#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/common/Checks.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>

namespace femx::assembly
{
namespace
{

HostVector<Index> boundaryDofs(const fem::HostControlMap& control)
{
  HostVector<Index> dofs(control.numBcs());
  for (Index i = 0; i < dofs.size(); ++i)
  {
    dofs[i] = control.dofs()[i];
  }
  return dofs;
}

linalg::CudaContext& cudaContext(
    linalg::Context<MemorySpace::Device>& ctx)
{
  return dynamic_cast<linalg::CudaContext&>(ctx);
}

template <class Ctx>
void controlVals(const fem::HostControlMap& map,
                 Index                      step,
                 HostVectorView<const Real> prm,
                 HostVectorView<Real>       out,
                 Ctx&)
{
  fem::controlVals(map, step, prm, out);
}

void controlVals(const fem::DeviceControlMap&          map,
                 Index                                 step,
                 DeviceVectorView<const Real>          prm,
                 DeviceVectorView<Real>                out,
                 linalg::Context<MemorySpace::Device>& ctx)
{
  fem::controlVals(map, step, prm, out, cudaContext(ctx));
}

template <class Ctx>
void evalInitState(const fem::HostInitialStateMap& map,
                   HostVectorView<const Real>      prm,
                   HostVectorView<Real>            out,
                   Ctx&)
{
  fem::initialState(map, prm, out);
}

void evalInitState(const fem::DeviceInitialStateMap&     map,
                   DeviceVectorView<const Real>          prm,
                   DeviceVectorView<Real>                out,
                   linalg::Context<MemorySpace::Device>& ctx)
{
  fem::initialState(map, prm, out, cudaContext(ctx));
}

template <class Ctx>
void addInitJacT(const fem::HostInitialStateMap& map,
                 HostVectorView<const Real>      adj,
                 HostVectorView<Real>            out,
                 Ctx&)
{
  fem::addInitialJacT(map, adj, out);
}

void addInitJacT(const fem::DeviceInitialStateMap&     map,
                 DeviceVectorView<const Real>          adj,
                 DeviceVectorView<Real>                out,
                 linalg::Context<MemorySpace::Device>& ctx)
{
  fem::addInitialJacT(map, adj, out, cudaContext(ctx));
}

template <class Ctx>
void addControlJacT(const fem::HostControlMap& map,
                    Index                      step,
                    HostVectorView<const Real> adj,
                    HostVectorView<Real>       out,
                    Ctx&)
{
  fem::addControlJacT(map, step, adj, out);
}

void addControlJacT(const fem::DeviceControlMap&          map,
                    Index                                 step,
                    DeviceVectorView<const Real>          adj,
                    DeviceVectorView<Real>                out,
                    linalg::Context<MemorySpace::Device>& ctx)
{
  fem::addControlJacT(map, step, adj, out, cudaContext(ctx));
}

template <class Ctx>
void applyDirichletConditionsForContext(
    const HostBoundaryMap&     map,
    HostVectorView<const Real> state,
    HostVectorView<const Real> vals,
    HostVectorView<Real>       res,
    Ctx&)
{
  assembly::applyDirichletConditions(map, state, vals, res);
}

void applyDirichletConditionsForContext(
    const DeviceBoundaryMap&              map,
    DeviceVectorView<const Real>          state,
    DeviceVectorView<const Real>          vals,
    DeviceVectorView<Real>                res,
    linalg::Context<MemorySpace::Device>& ctx)
{
  assembly::applyDirichletConditions(map, state, vals, res, cudaContext(ctx));
}

template <class Ctx>
void zeroBoundaryVals(const HostBoundaryMap& map, HostVectorView<Real> vals, Ctx&)
{
  assembly::zeroBoundary(map, vals);
}

void zeroBoundaryVals(const DeviceBoundaryMap&              map,
                      DeviceVectorView<Real>                vals,
                      linalg::Context<MemorySpace::Device>& ctx)
{
  assembly::zeroBoundary(map, vals, cudaContext(ctx));
}

template <MemorySpace Space>
void eliminateJacColumns(
    const BoundaryMap<Space>&    boundary,
    linalg::SystemMatrix<Space>& jac,
    Vector<Space, Real>&         rhs,
    const Vector<Space, Real>&   values)
{
  jac.eliminateColumns(boundary.view().constrained_rows,
                       values.view(),
                       rhs.view());
}

template <MemorySpace Space>
void resizeAndZero(Vector<Space, Real>&    out,
                   Index                   size,
                   linalg::Context<Space>& ctx)
{
  auto& vec_handler = ctx.vectors();
  vec_handler.resizeOrZero(out, size);
}

} // namespace

template <MemorySpace Space>
ConstrainedTimeResidual<Space>::ConstrainedTimeResidual(
    const Base&              base,
    fem::HostControlMap      control,
    fem::HostInitialStateMap init)
  : base_(base)
{
  if constexpr (Space == MemorySpace::Host)
  {
    initDims(control, init);
    control_  = std::move(control);
    boundary_ = makeBoundaryMap(boundaryDofs(control_));

    setInitialStateMap(std::move(init));

    base_prm_.resize(base_dims_.num_param);
    base_adj_.resize(dims_.num_res);
    boundary_vals_.resize(control_.numBcs());
  }
  else
  {
    require(false, "The non-owning constructor requires Host storage");
  }
}

template <MemorySpace Space>
ConstrainedTimeResidual<Space>::ConstrainedTimeResidual(
    const Base&              base,
    fem::HostControlMap      control,
    fem::HostInitialStateMap init,
    Ctx&                     ctx)
  : base_(base)
{
  if constexpr (Space == MemorySpace::Device)
  {
    initDims(control, init);

    const HostBoundaryMap h_boundary = makeBoundaryMap(boundaryDofs(control));
    auto&                 cuda_ctx   = cudaContext(ctx);
    copy(h_boundary, boundary_, cuda_ctx);
    fem::copy(control, control_, cuda_ctx);
    if (init.numStates() != 0)
    {
      fem::copy(init, init_, cuda_ctx);
    }
    base_prm_.resize(base_dims_.num_param);
    base_adj_.resize(dims_.num_res);
    boundary_vals_.resize(control.numBcs());
  }
  else
  {
    require(false, "The context constructor requires Device storage");
  }
}

template <MemorySpace Space>
state::TimeDims ConstrainedTimeResidual<Space>::dims() const
{
  return dims_;
}

template <MemorySpace Space>
const HostCsrPattern& ConstrainedTimeResidual<Space>::hostPattern() const
{
  return base_.hostPattern();
}

template <MemorySpace Space>
const typename ConstrainedTimeResidual<Space>::Control&
ConstrainedTimeResidual<Space>::controlMap() const noexcept
{
  return control_;
}

template <MemorySpace Space>
void ConstrainedTimeResidual<Space>::setInitialStateMap(
    fem::HostInitialStateMap init)
{
  checkInitMap(init);
  if constexpr (Space == MemorySpace::Host)
  {
    init_ = std::move(init);
  }
  else
  {
    require(false,
            "Device initial-state updates require an explicit CUDA context");
  }
}

template <MemorySpace Space>
void ConstrainedTimeResidual<Space>::clearInitialStateMap() noexcept
{
  init_ = {};
}

template <MemorySpace Space>
void ConstrainedTimeResidual<Space>::initialState(ConstView prm,
                                                  Vec&      out,
                                                  Ctx&      ctx) const
{
  require(prm.size() == dims_.num_param,
          "ConstrainedTimeResidual initial-state parameter size mismatch");

  if (init_.numStates() == 0)
  {
    base_.initialState(base_prm_.view(), out, ctx);
    return;
  }
  if (out.size() != dims_.num_states)
  {
    out.resize(dims_.num_states);
  }
  evalInitState(init_, prm, out.view(), ctx);
}

template <MemorySpace Space>
void ConstrainedTimeResidual<Space>::addInitialStateJacT(
    ConstView state_grad,
    VecView   out,
    Ctx&      ctx) const
{
  require(state_grad.size() == dims_.num_states
              && out.size() == dims_.num_param,
          "ConstrainedTimeResidual initial-state transpose size mismatch");
  if (init_.numStates() != 0)
  {
    assembly::addInitJacT(init_, state_grad, out, ctx);
  }
}

template <MemorySpace Space>
void ConstrainedTimeResidual<Space>::assembleNext(const StepCtx& time,
                                                  Vec&           res,
                                                  Jac&           jac,
                                                  Ctx&           ctx) const
{
  checkCtx(time);
  base_.assembleNext(baseCtx(time), res, jac, ctx);
  require(res.size() == dims_.num_res,
          "ConstrainedTimeResidual base residual size mismatch");

  assembly::controlVals(
      control_, time.step, time.prm, boundary_vals_.view(), ctx);
  applyDirichletConditionsForContext(
      boundary_, time.nxt, boundary_vals_.view(), res.view(), ctx);
  assembly::applyDirichletConditions(boundary_, jac);
}

template <MemorySpace Space>
void ConstrainedTimeResidual<Space>::applyJacT(
    const StepCtx&       time,
    state::VariableBlock wrt,
    ConstView            adj,
    Vec&                 out,
    Ctx&                 ctx) const
{
  auto& vec_handler = ctx.vectors();
  checkCtx(time);

  require(!wrt.isNextState(),
          "Constrained transpose apply supports only history and parameter blocks");
  require(adj.size() == dims_.num_res,
          "ConstrainedTimeResidual adjoint size mismatch");

  if (wrt.isParam())
  {
    resizeAndZero<Space>(out, dims_.num_param, ctx);
    assembly::addControlJacT(
        control_, time.step, adj, out.view(), ctx);
    return;
  }

  vec_handler.copy(adj, base_adj_.view());

  zeroBoundaryVals(boundary_, base_adj_.view(), ctx);
  base_.applyJacT(baseCtx(time), wrt, base_adj_.view(), out, ctx);

  require(out.size() == dims_.num_states,
          "ConstrainedTimeResidual transpose apply size mismatch");
}

template <MemorySpace Space>
void ConstrainedTimeResidual<Space>::setup(
    const StepCtx& time,
    Jac&           jac,
    Vec&           rhs,
    Ctx&           ctx) const
{
  checkCtx(time);
  base_.setup(baseCtx(time), jac, rhs, ctx);

  assembly::controlVals(control_, time.step, time.prm, boundary_vals_.view(), ctx);
  eliminateJacColumns(boundary_, jac, rhs, boundary_vals_);
}

template <MemorySpace Space>
typename ConstrainedTimeResidual<Space>::StepCtx
ConstrainedTimeResidual<Space>::baseCtx(const StepCtx& time) const
{
  StepCtx base_time = time;
  base_time.prm     = base_prm_.view();
  return base_time;
}

template <MemorySpace Space>
void ConstrainedTimeResidual<Space>::initDims(
    const fem::HostControlMap&      control,
    const fem::HostInitialStateMap& init)
{
  base_dims_ = base_.dims();
  dims_      = base_dims_;
  require(base_dims_.num_res == base_dims_.num_states,
          "ConstrainedTimeResidual requires square state residuals");
  require(base_dims_.num_param == 0,
          "ConstrainedTimeResidual requires a parameter-free base residual");
  require(control.numSteps() == dims_.num_steps
              && control.numStates() == dims_.num_states,
          "ConstrainedTimeResidual control dimensions do not match");
  dims_.num_param = control.numParams();
  checkInitMap(init);
}

template <MemorySpace Space>
void ConstrainedTimeResidual<Space>::checkCtx(
    const StepCtx& time) const
{
  require(time.step >= 0 && time.step < dims_.num_steps
              && time.hist.count() >= dims_.num_hist
              && time.hist.stateSize() == dims_.num_states
              && time.nxt.size() == dims_.num_states
              && time.prm.size() == dims_.num_param,
          "ConstrainedTimeResidual context dimensions do not match");
}

template <MemorySpace Space>
void ConstrainedTimeResidual<Space>::checkInitMap(
    const fem::HostInitialStateMap& map) const
{
  require(map.numStates() == 0
              || (map.numStates() == dims_.num_states
                  && map.numParams() == dims_.num_param),
          "ConstrainedTimeResidual initial-state dimensions do not match");
}

template class ConstrainedTimeResidual<MemorySpace::Host>;

#if defined(FEMX_HAS_CUDA)
template class ConstrainedTimeResidual<MemorySpace::Device>;
#endif

} // namespace femx::assembly
