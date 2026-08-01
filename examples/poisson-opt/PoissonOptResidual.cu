#include "../poisson/PoissonElementKernel.hpp"
#include "PoissonOptResidual.hpp"
#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/CudaAssembly.hpp>
#include <femx/common/Checks.hpp>
#include <femx/common/Cuda.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>

namespace femx::examples::poisson_opt
{
namespace
{

#if defined(FEMX_HAS_ENZYME)

__global__ void parameterVjpKernel(
    DeviceVectorView<const Index> ctr_dofs,
    DeviceVectorView<const Real>  state,
    DeviceVectorView<const Real>  prm,
    DeviceVectorView<const Real>  adj,
    DeviceVectorView<Real>        out)
{
  const Index ip = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ip >= prm.size())
  {
    return;
  }

  const Index state_dof = ctr_dofs[ip];
  const Real  derivative =
      __enzyme_fwddiff<Real>(
          reinterpret_cast<void*>(detail::controlResidual),
          enzyme_const,
          state[state_dof],
          enzyme_dup,
          prm[ip],
          1.0);

  out[ip] = derivative * adj[state_dof];
}

#endif

} // namespace

CudaPoissonOptResidual::CudaPoissonOptResidual(
    const PoissonOptProblem& problem,
    linalg::CudaContext&     ctx)
  : num_states_(problem.numStates()),
    num_prm_(problem.numParameters()),
    h_pattern_(problem.assemblyMap().pattern())
{
  fem::copy(problem.mesh(), mesh_, ctx);
  fem::copy(problem.elementData(), elem_data_, ctx);
  assembly::copy(problem.assemblyMap(), assm_map_, ctx);
  assembly::copy(problem.boundaryMap(), boundary_map_, ctx);
  ctx.vectorHandler().copy(problem.controlDofs(), ctr_dofs_);
  boundary_vals_.resize(problem.boundaryMap().numBcs());
  ctx.sync();
}

state::Dimensions CudaPoissonOptResidual::dims() const
{
  return {num_states_, num_prm_, num_states_};
}

const HostCsrPattern& CudaPoissonOptResidual::hostPattern() const
{
  return h_pattern_;
}

void CudaPoissonOptResidual::assembleResidual(
    const DeviceVector<Real>&             state,
    const DeviceVector<Real>&             prm,
    DeviceVector<Real>&                   out,
    linalg::Context<MemorySpace::Device>& base_ctx) const
{
  checkVectors(state, prm);
  auto& ctx = static_cast<linalg::CudaContext&>(base_ctx);

  assembly::assembleResidual(
      poisson::DevicePoissonElementKernel(elem_data_.view()),
      mesh_,
      assm_map_,
      state,
      out,
      ctx);

  assembly::applyDirichletConditions(boundary_map_,
                                     state.view(),
                                     boundaryValues(prm, ctx),
                                     out.view(),
                                     ctx);
}

void CudaPoissonOptResidual::assembleJacobian(
    const DeviceVector<Real>&                  state,
    const DeviceVector<Real>&                  prm,
    linalg::SystemMatrix<MemorySpace::Device>& out,
    linalg::Context<MemorySpace::Device>&      base_ctx) const
{
  checkVectors(state, prm);
  auto& ctx = static_cast<linalg::CudaContext&>(base_ctx);
  auto& jac = static_cast<linalg::CudaSystemMatrix&>(out);

  assembly::assembleJacobian(
      poisson::DevicePoissonElementKernel(elem_data_.view()),
      mesh_,
      assm_map_,
      state,
      jac,
      ctx);

  assembly::applyDirichletConditions(boundary_map_, jac);
}

void CudaPoissonOptResidual::applyParamJacT(
    const DeviceVector<Real>&             state,
    const DeviceVector<Real>&             prm,
    const DeviceVector<Real>&             adj,
    DeviceVector<Real>&                   out,
    linalg::Context<MemorySpace::Device>& base_ctx) const
{
  checkVectors(state, prm);
  require(adj.size() == dims().num_res,
          "Poisson optimization adjoint size mismatch");

  auto& ctx = static_cast<linalg::CudaContext&>(base_ctx);
  ctx.vectorHandler().assign(out, num_prm_, 0);

#if defined(FEMX_HAS_ENZYME)

  constexpr unsigned int threads = 128;
  const auto             stream  = static_cast<cudaStream_t>(ctx.stream());

  parameterVjpKernel<<<cuda::numBlocks(num_prm_, threads),
                       threads,
                       0,
                       stream>>>(ctr_dofs_.view(),
                                 state.view(),
                                 prm.view(),
                                 adj.view(),
                                 out.view());
  cuda::checkLastError();

#else

  ctx.vectorHandler().gather(adj.view(), ctr_dofs_.view(), out.view());
  ctx.vectorHandler().axpby(-1.0, out.view(), 0.0, out.view());

#endif
}

void CudaPoissonOptResidual::checkVectors(
    const DeviceVector<Real>& state,
    const DeviceVector<Real>& prm) const
{
  require(state.size() == num_states_,
          "Poisson optimization state size mismatch");
  require(prm.size() == num_prm_,
          "Poisson optimization parameter size mismatch");
}

DeviceVectorView<const Real>
CudaPoissonOptResidual::boundaryValues(
    const DeviceVector<Real>& prm,
    linalg::CudaContext&      ctx) const
{
  ctx.vectorHandler().zero(boundary_vals_.view());
  ctx.vectorHandler().copy(
      prm.view(),
      DeviceVectorView<Real>(
          boundary_vals_.data(), num_prm_));
  return boundary_vals_.view();
}

} // namespace femx::examples::poisson_opt
