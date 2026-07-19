#include <utility>

#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/common/Checks.hpp>

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

void replaceJacRows(const HostBoundaryMap& boundary,
                    HostCsrMatrix&         jac,
                    Real                   diag)
{
  replaceRows(boundary, jac, diag);
}

void prepareForward(const HostBoundaryMap& boundary,
                    HostCsrMatrix&         jac,
                    HostVector&            rhs,
                    const HostVector&      vals)
{
  prepareForwardSolve(boundary, jac, rhs, vals);
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
                    Real                   diag)
{
  jac.replaceRows(boundaryRows(boundary), diag);
}

void prepareForward(const HostBoundaryMap&,
                    linalg::PETScOperator&,
                    HostVector&,
                    const HostVector&)
{
  // Row replacement already gives the exact nonsymmetric Dirichlet system.
}
#endif

} // namespace

template <class Backend>
ConstrainedTimeResidual<Backend>::ConstrainedTimeResidual(
    const Base&              base,
    fem::HostControlMap      control,
    fem::HostInitialStateMap init_state)
  : base_(base),
    control_(std::move(control)),
    boundary_(makeBoundaryMap(boundaryDofs(control_), base.hostGraph())),
    base_dims_(base.dims()),
    dims_(base_dims_)
{
  require(base_dims_.num_res == base_dims_.num_states,
          "ConstrainedTimeResidual requires square state residuals");
  require(base_dims_.num_param == 0,
          "ConstrainedTimeResidual requires a parameter-free base residual");
  require(control_.numSteps() == base_dims_.num_steps
              && control_.numStates() == base_dims_.num_states,
          "ConstrainedTimeResidual control dimensions do not match");

  dims_.num_param = control_.numParams();
  base_prm_.resize(base_dims_.num_param);
  boundary_vals_.resize(control_.numBcs());
  setInitialStateMap(std::move(init_state));
}

template <class Backend>
state::TimeDims ConstrainedTimeResidual<Backend>::dims() const
{
  return dims_;
}

template <class Backend>
const HostCsrGraph& ConstrainedTimeResidual<Backend>::hostGraph() const
{
  return base_.hostGraph();
}

template <class Backend>
const typename ConstrainedTimeResidual<Backend>::Base::Graph&
ConstrainedTimeResidual<Backend>::graph() const
{
  return base_.graph();
}

template <class Backend>
const fem::HostControlMap&
ConstrainedTimeResidual<Backend>::controlMap() const noexcept
{
  return control_;
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::setInitialStateMap(
    fem::HostInitialStateMap init_state)
{
  checkInitialStateMap(init_state);
  init_state_ = std::move(init_state);
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::clearInitialStateMap() noexcept
{
  init_state_ = {};
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::initialState(ConstView   prm,
                                                    HostVector& out,
                                                    Ctx&        ctx) const
{
  require(prm.size() == dims_.num_param,
          "ConstrainedTimeResidual initial-state parameter size mismatch");
  if (init_state_.numStates() == 0)
  {
    base_.initialState(base_prm_.view(), out, ctx);
    return;
  }
  out.resize(dims_.num_states);
  fem::initialState(init_state_, prm, out.view());
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::addInitialStateJacobianTranspose(
    ConstView      state_grad,
    HostVectorView out,
    Ctx&) const
{
  require(state_grad.size() == dims_.num_states
              && out.size() == dims_.num_param,
          "ConstrainedTimeResidual initial-state transpose size mismatch");
  if (init_state_.numStates() != 0)
  {
    fem::addInitialJacT(init_state_, state_grad, out);
  }
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::res(
    const state::HostTimeContext& time,
    HostVector&                   out,
    Ctx&                          ctx) const
{
  checkContext(time);
  base_.res(baseContext(time), out, ctx);
  require(out.size() == dims_.num_res,
          "ConstrainedTimeResidual base residual size mismatch");

  fem::controlVals(control_, time.step, time.prm, boundary_vals_.view());
  replaceRes(boundary_, time.nxt, boundary_vals_.view(), out.view());
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::assemble(
    const state::HostTimeContext& time,
    state::VariableBlock          wrt,
    HostVector&                   res,
    Mat&                          jac,
    Ctx&                          ctx) const
{
  checkContext(time);
  require(!wrt.isParam(),
          "ConstrainedTimeResidual parameter Jacobian is matrix-free");
  base_.assemble(baseContext(time), wrt, res, jac, ctx);
  require(res.size() == dims_.num_res,
          "ConstrainedTimeResidual base residual size mismatch");

  fem::controlVals(control_, time.step, time.prm, boundary_vals_.view());
  replaceRes(boundary_, time.nxt, boundary_vals_.view(), res.view());
  replaceJacRows(boundary_, jac, wrt.isNextState() ? 1.0 : 0.0);
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::applyJac(
    const state::HostTimeContext& time,
    state::VariableBlock          wrt,
    ConstView                     dir,
    HostVector&                   out,
    Ctx&                          ctx) const
{
  checkContext(time);
  if (wrt.isParam())
  {
    resizeOrZero(out, dims_.num_res);
    fem::controlJac(control_, time.step, dir, out.view());
    return;
  }

  base_.applyJac(baseContext(time), wrt, dir, out, ctx);
  require(out.size() == dims_.num_res,
          "ConstrainedTimeResidual Jacobian apply size mismatch");
  const auto view = boundary_.view();
  const Real diag = wrt.isNextState() ? 1.0 : 0.0;
  for (Index i = 0; i < view.num_bcs; ++i)
  {
    const Index row = view.bcRow(i);
    out[row]        = diag * dir[row];
  }
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::applyJacT(
    const state::HostTimeContext& time,
    state::VariableBlock          wrt,
    ConstView                     adj,
    HostVector&                   out,
    Ctx&                          ctx) const
{
  checkContext(time);
  if (wrt.isParam())
  {
    resizeOrZero(out, dims_.num_param);
    fem::addControlJacT(control_, time.step, adj, out.view());
    return;
  }

  HostVector base_adj(adj);
  const auto view = boundary_.view();
  for (Index i = 0; i < view.num_bcs; ++i)
  {
    base_adj[view.bcRow(i)] = 0.0;
  }
  base_.applyJacT(baseContext(time), wrt, base_adj.view(), out, ctx);
  require(out.size() == dims_.num_states,
          "ConstrainedTimeResidual transpose apply size mismatch");
  if (wrt.isNextState())
  {
    for (Index i = 0; i < view.num_bcs; ++i)
    {
      const Index row  = view.bcRow(i);
      out[row]        += adj[row];
    }
  }
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::assembleJac(
    const state::HostTimeContext& time,
    state::VariableBlock          wrt,
    Mat&                          out,
    Ctx&                          ctx) const
{
  checkContext(time);
  require(!wrt.isParam(),
          "ConstrainedTimeResidual parameter Jacobian is matrix-free");
  base_.assembleJac(baseContext(time), wrt, out, ctx);
  replaceJacRows(boundary_, out, wrt.isNextState() ? 1.0 : 0.0);
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::prepareLinearSolve(
    const state::HostTimeContext& time,
    state::VariableBlock          wrt,
    Mat&                          jac,
    HostVector&                   rhs,
    Ctx&                          ctx) const
{
  checkContext(time);
  base_.prepareLinearSolve(baseContext(time), wrt, jac, rhs, ctx);
  if (wrt.isNextState())
  {
    fem::controlVals(control_, time.step, time.prm, boundary_vals_.view());
    prepareForward(boundary_, jac, rhs, boundary_vals_);
  }
}

template <class Backend>
state::HostTimeContext ConstrainedTimeResidual<Backend>::baseContext(
    const state::HostTimeContext& time) const
{
  state::HostTimeContext base_time = time;
  base_time.prm                    = base_prm_.view();
  return base_time;
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::checkContext(
    const state::HostTimeContext& time) const
{
  require(time.step >= 0 && time.step < dims_.num_steps
              && time.hist.count() >= dims_.num_hist
              && time.hist.stateSize() == dims_.num_states
              && time.nxt.size() == dims_.num_states
              && time.prm.size() == dims_.num_param,
          "ConstrainedTimeResidual context dimensions do not match");
}

template <class Backend>
void ConstrainedTimeResidual<Backend>::checkInitialStateMap(
    const fem::HostInitialStateMap& map) const
{
  require(map.numStates() == 0
              || (map.numStates() == dims_.num_states
                  && map.numParams() == dims_.num_param),
          "ConstrainedTimeResidual initial-state dimensions do not match");
}

ConstrainedTimeResidual<linalg::CudaCsrBackend>::ConstrainedTimeResidual(
    std::unique_ptr<Base>    base,
    fem::HostControlMap      control,
    fem::HostInitialStateMap init_state,
    CudaContext&             ctx)
  : base_(std::move(base))
{
  require(base_ != nullptr,
          "Device ConstrainedTimeResidual requires a base residual");
  base_dims_ = base_->dims();
  dims_      = base_dims_;
  require(base_dims_.num_res == base_dims_.num_states
              && base_dims_.num_param == 0,
          "Device ConstrainedTimeResidual requires a square, parameter-free base residual");
  require(control.numSteps() == dims_.num_steps
              && control.numStates() == dims_.num_states,
          "Device ConstrainedTimeResidual control dimensions do not match");

  dims_.num_param = control.numParams();
  has_init_state_ = init_state.numStates() != 0;
  require(!has_init_state_
              || (init_state.numStates() == dims_.num_states
                  && init_state.numParams() == dims_.num_param),
          "Device ConstrainedTimeResidual initial-state dimensions do not match");

  const HostBoundaryMap host_boundary =
      makeBoundaryMap(boundaryDofs(control), base_->hostGraph());
  copy(host_boundary, boundary_, ctx);
  fem::copy(control, control_, ctx);
  if (has_init_state_)
  {
    fem::copy(init_state, init_state_, ctx);
  }
  boundary_vals_.resize(control.numBcs());
}

ConstrainedTimeResidual<linalg::CudaCsrBackend>::~ConstrainedTimeResidual() =
    default;

state::TimeDims
ConstrainedTimeResidual<linalg::CudaCsrBackend>::dims() const
{
  return dims_;
}

const HostCsrGraph&
ConstrainedTimeResidual<linalg::CudaCsrBackend>::hostGraph() const
{
  return base_->hostGraph();
}

const DeviceCsrGraph&
ConstrainedTimeResidual<linalg::CudaCsrBackend>::graph() const
{
  return base_->graph();
}

void ConstrainedTimeResidual<linalg::CudaCsrBackend>::initialState(
    DeviceConstVectorView prm,
    DeviceVector&         out,
    CudaContext&          ctx) const
{
  checkParameters(prm);
  if (has_init_state_)
  {
    if (out.size() != dims_.num_states)
    {
      out.resize(dims_.num_states);
    }
    fem::initialState(init_state_, prm, out.view(), ctx);
    return;
  }
  base_->initialState({}, out, ctx);
}

void ConstrainedTimeResidual<linalg::CudaCsrBackend>::
    addInitialStateJacobianTranspose(DeviceConstVectorView state_grad,
                                     DeviceVectorView      out,
                                     CudaContext&          ctx) const
{
  require(state_grad.size() == dims_.num_states
              && out.size() == dims_.num_param,
          "Device ConstrainedTimeResidual initial-state transpose size mismatch");
  if (has_init_state_)
  {
    fem::addInitialJacT(init_state_, state_grad, out, ctx);
  }
}

void ConstrainedTimeResidual<linalg::CudaCsrBackend>::res(
    const state::DeviceTimeContext& time,
    DeviceVector&                   out,
    CudaContext&                    ctx) const
{
  checkContext(time);
  base_->res(baseContext(time), out, ctx);
  fem::controlVals(control_, time.step, time.prm, boundary_vals_.view(), ctx);
  replaceRes(boundary_, time.nxt, boundary_vals_.view(), out.view(), ctx);
}

void ConstrainedTimeResidual<linalg::CudaCsrBackend>::assemble(
    const state::DeviceTimeContext& time,
    state::VariableBlock            wrt,
    DeviceVector&                   res,
    DeviceCsrMatrix&                jac,
    CudaContext&                    ctx) const
{
  checkContext(time);
  require(!wrt.isParam(),
          "Device ConstrainedTimeResidual parameter Jacobian is matrix-free");
  base_->assemble(baseContext(time), wrt, res, jac, ctx);
  fem::controlVals(control_, time.step, time.prm, boundary_vals_.view(), ctx);
  replaceRes(boundary_, time.nxt, boundary_vals_.view(), res.view(), ctx);
  replaceRows(boundary_, jac, wrt.isNextState() ? 1.0 : 0.0, ctx);
}

void ConstrainedTimeResidual<linalg::CudaCsrBackend>::applyJac(
    const state::DeviceTimeContext& time,
    state::VariableBlock            wrt,
    DeviceConstVectorView           dir,
    DeviceVector&                   out,
    CudaContext&                    ctx) const
{
  checkContext(time);
  if (wrt.isParam())
  {
    if (out.size() != dims_.num_res)
    {
      out.resize(dims_.num_res);
    }
    out.setZero(ctx);
    fem::controlJac(control_, time.step, dir, out.view(), ctx);
    return;
  }
  DeviceCsrMatrix jac(graph());
  assembleJac(time, wrt, jac, ctx);
  if (out.size() != dims_.num_res)
  {
    out.resize(dims_.num_res);
  }
  femx::apply(jac, dir, out.view(), ctx);
}

void ConstrainedTimeResidual<linalg::CudaCsrBackend>::applyJacT(
    const state::DeviceTimeContext& time,
    state::VariableBlock            wrt,
    DeviceConstVectorView           adj,
    DeviceVector&                   out,
    CudaContext&                    ctx) const
{
  checkContext(time);
  if (wrt.isParam())
  {
    if (out.size() != dims_.num_param)
    {
      out.resize(dims_.num_param);
    }
    out.setZero(ctx);
    fem::addControlJacT(control_, time.step, adj, out.view(), ctx);
    return;
  }
  DeviceCsrMatrix jac(graph());
  assembleJac(time, wrt, jac, ctx);
  if (out.size() != dims_.num_states)
  {
    out.resize(dims_.num_states);
  }
  femx::applyT(jac, adj, out.view(), ctx);
}

void ConstrainedTimeResidual<linalg::CudaCsrBackend>::assembleJac(
    const state::DeviceTimeContext& time,
    state::VariableBlock            wrt,
    DeviceCsrMatrix&                out,
    CudaContext&                    ctx) const
{
  checkContext(time);
  require(!wrt.isParam(),
          "Device ConstrainedTimeResidual parameter Jacobian is matrix-free");
  base_->assembleJac(baseContext(time), wrt, out, ctx);
  replaceRows(boundary_, out, wrt.isNextState() ? 1.0 : 0.0, ctx);
}

void ConstrainedTimeResidual<linalg::CudaCsrBackend>::prepareLinearSolve(
    const state::DeviceTimeContext& time,
    state::VariableBlock            wrt,
    DeviceCsrMatrix&                jac,
    DeviceVector&                   rhs,
    CudaContext&                    ctx) const
{
  checkContext(time);
  base_->prepareLinearSolve(baseContext(time), wrt, jac, rhs, ctx);
  if (wrt.isNextState())
  {
    fem::controlVals(control_, time.step, time.prm, boundary_vals_.view(), ctx);
    prepareForwardSolve(boundary_, jac, rhs, boundary_vals_, ctx);
  }
}

state::DeviceTimeContext
ConstrainedTimeResidual<linalg::CudaCsrBackend>::baseContext(
    const state::DeviceTimeContext& time) const
{
  state::DeviceTimeContext base_time = time;
  base_time.prm                      = {};
  return base_time;
}

void ConstrainedTimeResidual<linalg::CudaCsrBackend>::checkContext(
    const state::DeviceTimeContext& time) const
{
  require(time.step >= 0 && time.step < dims_.num_steps
              && time.nxt.size() == dims_.num_states
              && time.hist.count() == dims_.num_hist
              && time.hist.stateSize() == dims_.num_states,
          "Device ConstrainedTimeResidual context dimensions do not match");
  checkParameters(time.prm);
}

void ConstrainedTimeResidual<linalg::CudaCsrBackend>::checkParameters(
    DeviceConstVectorView prm) const
{
  require(prm.size() == dims_.num_param,
          "Device ConstrainedTimeResidual parameter size mismatch");
}

template class ConstrainedTimeResidual<linalg::HostCsrBackend>;

#if defined(FEMX_HAS_PETSC)
template class ConstrainedTimeResidual<linalg::PetscBackend>;
#endif

} // namespace femx::assembly
