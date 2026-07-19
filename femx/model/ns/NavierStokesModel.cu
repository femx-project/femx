#include <memory>
#include <utility>

#include "NavierStokesModel.hpp"
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/assembly/CudaAssembly.hpp>
#include <femx/common/Checks.hpp>
#include <femx/linalg/CsrMatrix.hpp>

namespace femx::model::ns
{
namespace
{

class DeviceNavierResidual final : public state::DeviceTimeResidual
{
public:
  DeviceNavierResidual(const NavierStokesModel& model, CudaContext& ctx)
    : dims_{model.numSteps(),
            model.numStates(),
            0,
            model.map().numRes(),
            model.residual().dims().num_hist},
      host_graph_(model.map().graph())
  {
    require(dims_.num_res == dims_.num_states,
            "Device Navier-Stokes residual must be square");
    require(dims_.num_hist > 0,
            "Device Navier-Stokes residual requires history states");

    copy(model.map(), map_, ctx);
    ns::copy(model.data(), data_, ctx);

    op_ = NavierOperator<MemorySpace::Device>(
        data_.view(),
        {model.fluid().rho, model.fluid().mu},
        model.dt());
  }

  state::TimeDims dims() const override
  {
    return dims_;
  }

  const HostCsrGraph& hostGraph() const override
  {
    return host_graph_;
  }

  const DeviceCsrGraph& graph() const override
  {
    return map_.graph();
  }

  void initialState(DeviceConstVectorView prm,
                    DeviceVector&         out,
                    CudaContext&          ctx) const override
  {
    require(prm.empty(),
            "Device Navier physics residual is parameter-free");
    if (out.size() != dims_.num_states)
    {
      out.resize(dims_.num_states);
    }
    out.setZero(ctx);
  }

  void res(const state::DeviceTimeContext& time,
           DeviceVector&                   out,
           CudaContext&                    ctx) const override
  {
    DeviceCsrMatrix unused(map_.graph());
    assemble(time,
             state::VariableBlock::NextState,
             out,
             unused,
             ctx);
  }

  void assemble(const state::DeviceTimeContext& time,
                state::VariableBlock            wrt,
                DeviceVector&                   res,
                DeviceCsrMatrix&                jac,
                CudaContext&                    ctx) const override
  {
    checkCtx(time);
    require(!wrt.isParam(), "Device Navier parameter Jacobian is matrix-free");

    assembly::assemble(op_,
                       time.step,
                       dims_.num_hist,
                       wrt,
                       map_,
                       time.hist.values(),
                       time.nxt,
                       res,
                       jac,
                       ctx);
  }

  void applyJac(const state::DeviceTimeContext& time,
                state::VariableBlock            wrt,
                DeviceConstVectorView           dir,
                DeviceVector&                   out,
                CudaContext&                    ctx) const override
  {
    require(!wrt.isParam(),
            "Device Navier parameter Jacobian is empty");
    DeviceVector    unused;
    DeviceCsrMatrix jac(map_.graph());
    assemble(time, wrt, unused, jac, ctx);
    if (out.size() != dims_.num_res)
    {
      out.resize(dims_.num_res);
    }
    femx::apply(jac, dir, out.view(), ctx);
  }

  void applyJacT(const state::DeviceTimeContext& time,
                 state::VariableBlock            wrt,
                 DeviceConstVectorView           adj,
                 DeviceVector&                   out,
                 CudaContext&                    ctx) const override
  {
    if (wrt.isParam())
    {
      out.resize(0);
      return;
    }
    DeviceVector    unused;
    DeviceCsrMatrix jac(map_.graph());

    assemble(time, wrt, unused, jac, ctx);
    if (out.size() != dims_.num_states)
    {
      out.resize(dims_.num_states);
    }
    
    femx::applyT(jac, adj, out.view(), ctx);
  }

private:
  void checkCtx(const state::DeviceTimeContext& ctx) const
  {
    require(ctx.step >= 0 && ctx.step < dims_.num_steps,
            "Device Navier-Stokes step is out of range");
    require(ctx.nxt.size() == dims_.num_states
                && ctx.hist.count() == dims_.num_hist
                && ctx.hist.stateSize() == dims_.num_states,
            "Device Navier-Stokes state dimensions do not match");
    require(ctx.prm.empty(),
            "Device Navier-Stokes physics residual is parameter-free");
  }

  state::TimeDims                     dims_;
  HostCsrGraph                        host_graph_;
  assembly::DeviceAssemblyMap         map_;
  DeviceNavierData                    data_;
  NavierOperator<MemorySpace::Device> op_;
};

} // namespace

std::unique_ptr<state::DeviceTimeResidual> makeDeviceTimeResidual(
    const NavierStokesModel& model,
    fem::HostControlMap      control,
    fem::HostInitialStateMap init_state)
{
  CudaContext ctx;
  auto        base = std::make_unique<DeviceNavierResidual>(model, ctx);
  auto        out  = std::make_unique<assembly::DeviceConstrainedTimeResidual>(
      std::move(base), std::move(control), std::move(init_state), ctx);
  ctx.synchronize();
  return out;
}

} // namespace femx::model::ns
