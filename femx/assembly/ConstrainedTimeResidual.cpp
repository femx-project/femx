#include <utility>

#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/common/Checks.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

#if defined(FEMX_HAS_PETSC)
#include <femx/linalg/petsc/PETScBackend.hpp>
#endif

namespace femx::assembly
{
namespace
{

Array<Index> boundaryDofs(const fem::HostControlMap& control)
{
  Array<Index> dofs(control.numBcs());
  for (Index i = 0; i < dofs.size(); ++i)
  {
    dofs[i] = control.dofs()[i];
  }
  return dofs;
}

template <class Ctx>
void controlVals(const fem::HostControlMap& map,
                 Index                      step,
                 HostConstVectorView        prm,
                 HostVectorView             out,
                 Ctx&)
{
  fem::controlVals(map, step, prm, out);
}

void controlVals(const fem::DeviceControlMap& map,
                 Index                        step,
                 DeviceConstVectorView        prm,
                 DeviceVectorView             out,
                 CudaContext&                 ctx)
{
  fem::controlVals(map, step, prm, out, ctx);
}

template <class Ctx>
void evalInitState(const fem::HostInitialStateMap& map,
                   HostConstVectorView             prm,
                   HostVectorView                  out,
                   Ctx&)
{
  fem::initialState(map, prm, out);
}

void evalInitState(const fem::DeviceInitialStateMap& map,
                   DeviceConstVectorView             prm,
                   DeviceVectorView                  out,
                   CudaContext&                      ctx)
{
  fem::initialState(map, prm, out, ctx);
}

template <class Ctx>
void addInitJacT(const fem::HostInitialStateMap& map,
                 HostConstVectorView             adj,
                 HostVectorView                  out,
                 Ctx&)
{
  fem::addInitialJacT(map, adj, out);
}

void addInitJacT(const fem::DeviceInitialStateMap& map,
                 DeviceConstVectorView             adj,
                 DeviceVectorView                  out,
                 CudaContext&                      ctx)
{
  fem::addInitialJacT(map, adj, out, ctx);
}

template <class Ctx>
void addControlJacT(const fem::HostControlMap& map,
                    Index                      step,
                    HostConstVectorView        adj,
                    HostVectorView             out,
                    Ctx&)
{
  fem::addControlJacT(map, step, adj, out);
}

void addControlJacT(const fem::DeviceControlMap& map,
                    Index                        step,
                    DeviceConstVectorView        adj,
                    DeviceVectorView             out,
                    CudaContext&                 ctx)
{
  fem::addControlJacT(map, step, adj, out, ctx);
}

template <class Ctx>
void replaceResCtx(const HostBoundaryMap& map,
                   HostConstVectorView    state,
                   HostConstVectorView    vals,
                   HostVectorView         res,
                   Ctx&)
{
  assembly::replaceRes(map, state, vals, res);
}

void replaceResCtx(const DeviceBoundaryMap& map,
                   DeviceConstVectorView    state,
                   DeviceConstVectorView    vals,
                   DeviceVectorView         res,
                   CudaContext&             ctx)
{
  assembly::replaceRes(map, state, vals, res, ctx);
}

template <class Ctx>
void zeroBoundaryVals(const HostBoundaryMap& map, HostVectorView vals, Ctx&)
{
  assembly::zeroBoundary(map, vals);
}

void zeroBoundaryVals(const DeviceBoundaryMap& map,
                      DeviceVectorView         vals,
                      CudaContext&             ctx)
{
  assembly::zeroBoundary(map, vals, ctx);
}

template <class Ctx>
void replaceJacRows(const HostBoundaryMap& boundary,
                    HostCsrMatrix&         jac,
                    Real                   diag,
                    Ctx&)
{
  replaceRows(boundary, jac, diag);
}

void replaceJacRows(const DeviceBoundaryMap& boundary,
                    DeviceCsrMatrix&         jac,
                    Real                     diag,
                    CudaContext&             ctx)
{
  replaceRows(boundary, jac, diag, ctx);
}

template <class Ctx>
void prepareForward(const HostBoundaryMap& boundary,
                    HostCsrMatrix&         jac,
                    HostVector&            rhs,
                    const HostVector&      vals,
                    Ctx&)
{
  prepareForwardSolve(boundary, jac, rhs, vals);
}

void prepareForward(const DeviceBoundaryMap& boundary,
                    DeviceCsrMatrix&         jac,
                    DeviceVector&            rhs,
                    const DeviceVector&      vals,
                    CudaContext&             ctx)
{
  prepareForwardSolve(boundary, jac, rhs, vals, ctx);
}

#if defined(FEMX_HAS_PETSC)
Array<Index> boundaryRows(const HostBoundaryMap& boundary)
{
  const auto   view = boundary.view();
  Array<Index> rows(view.num_bcs);
  for (Index i = 0; i < rows.size(); ++i)
  {
    rows[i] = view.bcRow(i);
  }
  return rows;
}

void replaceJacRows(const HostBoundaryMap& boundary,
                    linalg::PETScOperator& jac,
                    Real                   diag,
                    linalg::PetscContext&)
{
  jac.replaceRows(boundaryRows(boundary), diag);
}

void prepareForward(const HostBoundaryMap&,
                    linalg::PETScOperator&,
                    HostVector&,
                    const HostVector&,
                    linalg::PetscContext&)
{
  // PETSc row replacement already gives the exact nonsymmetric system.
}
#endif

template <class Backend>
void resizeAndZero(typename Backend::Vec& out,
                   Index                  size,
                   typename Backend::Ctx& ctx)
{
  linalg::VectorHandler<Backend> vec_handler(ctx);
  vec_handler.resizeOrZero(out, size);
}

} // namespace

template <class Backend>
ConstrainedTimeResidual<Backend>::ConstrainedTimeResidual(
    const Base&              base,
    fem::HostControlMap      control,
    fem::HostInitialStateMap init)
  : base_(&base)
{
  if constexpr (Backend::space == MemorySpace::Host)
  {
    initDims(control, init);
    control_  = std::move(control);
    boundary_ = makeBoundaryMap(boundaryDofs(control_), base_->hostPattern());
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

template <class Backend>
ConstrainedTimeResidual<Backend>::ConstrainedTimeResidual(
    std::unique_ptr<Base>    base,
    fem::HostControlMap      control,
    fem::HostInitialStateMap init,
    Ctx&                     ctx)
  : owned_base_(std::move(base)), base_(owned_base_.get())
{
  if constexpr (Backend::space == MemorySpace::Device)
  {
    require(base_ != nullptr,
            "ConstrainedTimeResidual requires a base residual");
    initDims(control, init);

    const HostBoundaryMap host_boundary =
        makeBoundaryMap(boundaryDofs(control), base_->hostPattern());
    copy(host_boundary, boundary_, ctx);
    fem::copy(control, control_, ctx);
    if (init.numStates() != 0)
    {
      fem::copy(init, init_, ctx);
    }
    base_prm_.resize(base_dims_.num_param);
    base_adj_.resize(dims_.num_res);
    boundary_vals_.resize(control.numBcs());
  }
  else
  {
    require(false, "The owning copy constructor requires Device storage");
  }
}

template <class Backend>
state::TimeDims ConstrainedTimeResidual<Backend>::dims() const
{
  return dims_;
}

template <class Backend>
const HostCsrPattern& ConstrainedTimeResidual<Backend>::hostPattern() const
{
  return base_->hostPattern();
}

template <class Backend>
const typename ConstrainedTimeResidual<Backend>::Pattern&
ConstrainedTimeResidual<Backend>::pattern() const
{
  return base_->pattern();
}

template <class Backend>
const typename ConstrainedTimeResidual<Backend>::Control&
ConstrainedTimeResidual<Backend>::controlMap() const noexcept
{
  return control_;
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::setInitialStateMap(
    fem::HostInitialStateMap init)
{
  checkInitMap(init);
  if constexpr (Backend::space == MemorySpace::Host)
  {
    init_ = std::move(init);
  }
  else
  {
    require(false,
            "Device initial-state updates require an explicit CUDA context");
  }
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::clearInitialStateMap() noexcept
{
  init_ = {};
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::initialState(ConstView prm,
                                                    Vec&      out,
                                                    Ctx&      ctx) const
{
  require(prm.size() == dims_.num_param,
          "ConstrainedTimeResidual initial-state parameter size mismatch");
  if (init_.numStates() == 0)
  {
    base_->initialState(base_prm_.view(), out, ctx);
    return;
  }
  if (out.size() != dims_.num_states)
  {
    out.resize(dims_.num_states);
  }
  evalInitState(init_, prm, out.view(), ctx);
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::addInitialStateJacobianTranspose(
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

template <class Backend>
void ConstrainedTimeResidual<Backend>::res(const StepCtx& time,
                                           Vec&           out,
                                           Ctx&           ctx) const
{
  checkCtx(time);
  base_->res(baseCtx(time), out, ctx);
  require(out.size() == dims_.num_res,
          "ConstrainedTimeResidual base residual size mismatch");

  assembly::controlVals(
      control_, time.step, time.prm, boundary_vals_.view(), ctx);
  replaceResCtx(
      boundary_, time.nxt, boundary_vals_.view(), out.view(), ctx);
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::assembleNext(const StepCtx& time,
                                                    Vec&           res,
                                                    Mat&           jac,
                                                    Ctx&           ctx) const
{
  checkCtx(time);
  base_->assembleNext(baseCtx(time), res, jac, ctx);
  require(res.size() == dims_.num_res,
          "ConstrainedTimeResidual base residual size mismatch");

  assembly::controlVals(
      control_, time.step, time.prm, boundary_vals_.view(), ctx);
  replaceResCtx(
      boundary_, time.nxt, boundary_vals_.view(), res.view(), ctx);
  replaceJacRows(boundary_, jac, 1.0, ctx);
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::applyJacT(
    const StepCtx&       time,
    state::VariableBlock wrt,
    ConstView            adj,
    Vec&                 out,
    Ctx&                 ctx) const
{
  linalg::VectorHandler<Backend> vec_handler(ctx);
  checkCtx(time);
  require(!wrt.isNextState(),
          "Constrained transpose apply supports only history and parameter blocks");
  require(adj.size() == dims_.num_res,
          "ConstrainedTimeResidual adjoint size mismatch");
  if (wrt.isParam())
  {
    resizeAndZero<Backend>(out, dims_.num_param, ctx);
    assembly::addControlJacT(
        control_, time.step, adj, out.view(), ctx);
    return;
  }

  vec_handler.copy(adj, base_adj_.view());
  zeroBoundaryVals(boundary_, base_adj_.view(), ctx);
  base_->applyJacT(baseCtx(time), wrt, base_adj_.view(), out, ctx);
  require(out.size() == dims_.num_states,
          "ConstrainedTimeResidual transpose apply size mismatch");
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::prepareLinearSolve(
    const StepCtx& time,
    Mat&           jac,
    Vec&           rhs,
    Ctx&           ctx) const
{
  checkCtx(time);
  base_->prepareLinearSolve(baseCtx(time), jac, rhs, ctx);
  assembly::controlVals(
      control_, time.step, time.prm, boundary_vals_.view(), ctx);
  prepareForward(boundary_, jac, rhs, boundary_vals_, ctx);
}

template <class Backend>
typename ConstrainedTimeResidual<Backend>::StepCtx
ConstrainedTimeResidual<Backend>::baseCtx(const StepCtx& time) const
{
  StepCtx base_time = time;
  base_time.prm     = base_prm_.view();
  return base_time;
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::initDims(
    const fem::HostControlMap&      control,
    const fem::HostInitialStateMap& init)
{
  require(base_ != nullptr,
          "ConstrainedTimeResidual requires a base residual");
  base_dims_ = base_->dims();
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

template <class Backend>
void ConstrainedTimeResidual<Backend>::checkCtx(
    const StepCtx& time) const
{
  require(time.step >= 0 && time.step < dims_.num_steps
              && time.hist.count() >= dims_.num_hist
              && time.hist.stateSize() == dims_.num_states
              && time.nxt.size() == dims_.num_states
              && time.prm.size() == dims_.num_param,
          "ConstrainedTimeResidual context dimensions do not match");
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::checkInitMap(
    const fem::HostInitialStateMap& map) const
{
  require(map.numStates() == 0
              || (map.numStates() == dims_.num_states
                  && map.numParams() == dims_.num_param),
          "ConstrainedTimeResidual initial-state dimensions do not match");
}

template class ConstrainedTimeResidual<linalg::HostCsrBackend>;

#if defined(FEMX_HAS_CUDA)
template class ConstrainedTimeResidual<linalg::CudaCsrBackend>;
#endif

#if defined(FEMX_HAS_PETSC)
template class ConstrainedTimeResidual<linalg::PetscBackend>;
#endif

} // namespace femx::assembly
