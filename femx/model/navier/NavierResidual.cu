#include <stdexcept>

#include "NavierResidual.hpp"
#include <femx/ad/Enzyme.hpp>
#include <femx/common/Checks.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>
#include <femx/model/navier/NavierModel.hpp>

namespace femx::model::navier
{
namespace
{

void validateTimeContext(const state::DeviceTimeContext& time,
                         Index                           num_steps,
                         Index                           num_states)
{
  require(time.step >= 0 && time.step < num_steps,
          "Navier-Stokes residual step is out of range");
  require(time.hist.count() >= kNumHist
              && time.hist.stateSize() == num_states
              && time.nxt.size() == num_states && time.prm.empty(),
          "Navier-Stokes residual vector size mismatch");
}

} // namespace

CudaNavierResidual::CudaNavierResidual(
    const NavierModel&                    model,
    linalg::Context<MemorySpace::Device>& ctx)
  : num_steps_(model.numSteps()),
    h_pattern_(model.assemblyMap().pattern())
{
  fem::copy(model.elementData(), elem_data_, ctx);
  assembly::copy(model.assemblyMap(), assm_map_, ctx);
  kernel_ = DeviceNavierElementKernel(elem_data_.view(), model.fluid(), model.dt());
}

state::TimeDims CudaNavierResidual::dims() const
{
  return {num_steps_,
          assm_map_.numStates(),
          0,
          assm_map_.numRes(),
          kNumHist};
}

const HostCsrPattern& CudaNavierResidual::hostPattern() const
{
  return h_pattern_;
}

void CudaNavierResidual::initialState(
    ConstView prm,
    Vec&      out,
    Ctx&      base_ctx) const
{
  require(prm.empty(), "Navier-Stokes physics residual is parameter-free");
  auto& ctx = static_cast<linalg::CudaContext&>(base_ctx);
  ctx.vectorHandler().assign(out, assm_map_.numStates(), 0);
}

void CudaNavierResidual::assembleNext(
    const StepCtx& time,
    Vec&           res,
    Jac&           base_jac,
    Ctx&           base_ctx) const
{
  validateTimeContext(time, num_steps_, assm_map_.numStates());

  auto& ctx = static_cast<linalg::CudaContext&>(base_ctx);
  auto& jac = static_cast<linalg::CudaSystemMatrix&>(base_jac);

  const auto range = ctx.elementRange(assm_map_.numElems());
  const auto hist  = ConstView(time.hist.data(), kNumHist * assm_map_.numStates());

  detail::assembleNext(kernel_,
                       time.step,
                       kNumHist,
                       range.begin,
                       range.end,
                       assm_map_,
                       hist,
                       time.nxt,
                       res,
                       jac,
                       ctx);
}

void CudaNavierResidual::applyJacT(
    const StepCtx&       time,
    state::VariableBlock with_respect_to,
    ConstView            adj,
    Vec&                 out,
    Ctx&                 base_ctx) const
{
  validateTimeContext(time, num_steps_, assm_map_.numStates());

  require(!with_respect_to.isNextState(),
          "Navier-Stokes transpose apply expects a history or parameter block");
  require(adj.size() == assm_map_.numRes(),
          "Navier-Stokes residual adjoint size mismatch");

  if (with_respect_to.isParam())
  {
    out.resize(0);
    return;
  }

  require(with_respect_to.historyLag() >= 0
              && with_respect_to.historyLag() < kNumHist,
          "Navier-Stokes residual history lag is out of range");

  if (!ad::has_enzyme)
  {
    throw std::runtime_error(
        "Navier-Stokes history VJP requires Enzyme");
  }

  auto&      ctx   = static_cast<linalg::CudaContext&>(base_ctx);
  const auto range = ctx.elementRange(assm_map_.numElems());
  const auto hist  = ConstView(time.hist.data(), kNumHist * assm_map_.numStates());

  detail::applyHistJacT(kernel_,
                        time.step,
                        kNumHist,
                        with_respect_to.historyLag(),
                        range.begin,
                        range.end,
                        assm_map_,
                        hist,
                        time.nxt,
                        adj,
                        out,
                        ctx);
}

} // namespace femx::model::navier
