#include "PoissonElementKernel.hpp"
#include "PoissonResidual.hpp"
#include <femx/assembly/CudaAssembly.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>

namespace femx::examples::poisson
{

CudaPoissonResidual::CudaPoissonResidual(
    const PoissonProblem& problem,
    linalg::CudaContext&  ctx)
  : num_dofs_(problem.numDofs()),
    h_pattern_(problem.assemblyMap().pattern())
{
  fem::copy(problem.mesh(), mesh_, ctx);
  fem::copy(problem.elementData(), elem_data_, ctx);
  assembly::copy(problem.assemblyMap(), assm_map_, ctx);
  assembly::copy(problem.boundaryMap(), boundary_map_, ctx);
  ctx.vectorHandler().copy(problem.boundaryValues(), boundary_vals_);
}

state::Dimensions CudaPoissonResidual::dims() const
{
  return {num_dofs_, 0, num_dofs_};
}

const HostCsrPattern& CudaPoissonResidual::hostPattern() const
{
  return h_pattern_;
}

void CudaPoissonResidual::assembleResidual(
    const DeviceVector<Real>& state,
    const DeviceVector<Real>& /* prm */,
    DeviceVector<Real>&                   out,
    linalg::Context<MemorySpace::Device>& base_ctx) const
{
  auto& ctx = static_cast<linalg::CudaContext&>(base_ctx);

  assembly::assembleResidual(
      DevicePoissonElementKernel(elem_data_.view()),
      mesh_,
      assm_map_,
      state,
      out,
      ctx);

  assembly::applyDirichletConditions(
      boundary_map_,
      state.view(),
      boundary_vals_.view(),
      out.view(),
      ctx);
}

void CudaPoissonResidual::assembleJacobian(
    const DeviceVector<Real>& state,
    const DeviceVector<Real>& /* prm */,
    linalg::SystemMatrix<MemorySpace::Device>& out,
    linalg::Context<MemorySpace::Device>&      base_ctx) const
{
  auto& ctx = static_cast<linalg::CudaContext&>(base_ctx);
  auto& jac = static_cast<linalg::CudaSystemMatrix&>(out);

  assembly::assembleJacobian(
      DevicePoissonElementKernel(elem_data_.view()),
      mesh_,
      assm_map_,
      state,
      jac,
      ctx);

  assembly::applyDirichletConditions(boundary_map_, jac);
}

void CudaPoissonResidual::applyParamJacT(
    const DeviceVector<Real>& /* state */,
    const DeviceVector<Real>& /* prm */,
    const DeviceVector<Real>& /* adj */,
    DeviceVector<Real>& out,
    linalg::Context<MemorySpace::Device>&) const
{
  // This example has no parameters.
  out.resize(0);
}

} // namespace femx::examples::poisson
