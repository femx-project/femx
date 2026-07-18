#include <cuda_runtime_api.h>

#include <stdexcept>

#include <femx/assembly/BoundaryPlan.hpp>
#include <femx/common/Context.hpp>

namespace femx
{
namespace assembly
{
namespace
{

constexpr int kThreads = 256;

__global__ void replaceRowsKernel(BoundaryPlanView<MemorySpace::Device> plan,
                                  const Index*                          row_ptr,
                                  Real*                                 mat_vals,
                                  Real*                                 rhs,
                                  const Real*                           bc_vals)
{
  const Index ib =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ib >= plan.num_bcs)
  {
    return;
  }

  const Index row = plan.bcRow(ib);
  for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
  {
    mat_vals[k] = 0.0;
  }
  mat_vals[plan.diag(ib)] = 1.0;
  if (rhs != nullptr)
  {
    rhs[row] = bc_vals[ib];
  }
}

__global__ void eliminateColsKernel(
    BoundaryPlanView<MemorySpace::Device> plan,
    Real*                                 mat_vals,
    Real*                                 rhs,
    const Real*                           bc_vals)
{
  const Index ib = static_cast<Index>(blockIdx.x);
  if (ib >= plan.num_bcs)
  {
    return;
  }

  const Real bc = bc_vals[ib];
  for (Index i  = plan.colBegin(ib) + threadIdx.x; i < plan.colEnd(ib);
       i       += blockDim.x)
  {
    const Index row = plan.col_rows[i];
    const Index k   = plan.col_entries[i];
    const Real  val = mat_vals[k];
    if (!plan.isBc(row))
    {
      atomicAdd(rhs + row, -val * bc);
    }
    mat_vals[k] = 0.0;
  }
}

__global__ void replaceResKernel(
    BoundaryPlanView<MemorySpace::Device> plan,
    const Real*                           state,
    const Real*                           bc_vals,
    Real*                                 res)
{
  const Index ib =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ib < plan.num_bcs)
  {
    const Index row = plan.bcRow(ib);
    res[row]        = state[row] - bc_vals[ib];
  }
}

void checkMat(const DeviceBoundaryPlan& plan, const DeviceCsrMatrix& mat)
{
  if (mat.graph().layoutId() != plan.layoutId())
  {
    throw std::runtime_error(
        "BoundaryPlan matrix does not match the planned CSR layout");
  }
}

void checkForward(const DeviceBoundaryPlan& plan,
                  const DeviceCsrMatrix&    mat,
                  const DeviceVector&       rhs,
                  const DeviceVector&       bc_vals)
{
  checkMat(plan, mat);
  if (rhs.size() != plan.rows() || bc_vals.size() != plan.numBcs())
  {
    throw std::runtime_error(
        "BoundaryPlan forward vectors have incompatible sizes");
  }
  if (&rhs == &bc_vals)
  {
    throw std::runtime_error(
        "BoundaryPlan RHS and prescribed values must not alias");
  }
  const DeviceVector& mat_vals = mat.vals();
  if (&rhs == &mat_vals || &bc_vals == &mat_vals)
  {
    throw std::runtime_error(
        "BoundaryPlan vectors must not alias matrix values");
  }
}

cudaStream_t cudaStream(CudaContext& ctx)
{
  return static_cast<cudaStream_t>(ctx.stream());
}

void launchRows(const DeviceBoundaryPlan& plan,
                DeviceCsrMatrix&          mat,
                DeviceVector*             rhs,
                const DeviceVector*       bc_vals,
                CudaContext&              ctx)
{
  if (plan.numBcs() == 0)
  {
    return;
  }
  const unsigned int blocks = static_cast<unsigned int>(
      (plan.numBcs() + kThreads - 1) / kThreads);
  replaceRowsKernel<<<blocks, kThreads, 0, cudaStream(ctx)>>>(
      plan.view(),
      mat.rowPtrData(),
      mat.valsData(),
      rhs == nullptr ? nullptr : rhs->data(),
      bc_vals == nullptr ? nullptr : bc_vals->data());
  device::checkLastError();
}

} // namespace

void replaceRows(const DeviceBoundaryPlan& plan,
                 DeviceCsrMatrix&          jac,
                 CudaContext&              ctx)
{
  checkMat(plan, jac);
  launchRows(plan, jac, nullptr, nullptr, ctx);
}

void replaceRes(const DeviceBoundaryPlan& plan,
                const DeviceVector&       state,
                const DeviceVector&       bc_vals,
                DeviceVector&             res,
                CudaContext&              ctx)
{
  if (state.size() != plan.rows() || res.size() != plan.rows()
      || bc_vals.size() != plan.numBcs())
  {
    throw std::runtime_error(
        "BoundaryPlan residual vectors have incompatible sizes");
  }
  if (&state == &res || &bc_vals == &res)
  {
    throw std::runtime_error(
        "BoundaryPlan residual output must not alias its inputs");
  }
  if (plan.numBcs() == 0)
  {
    return;
  }
  const unsigned int blocks = static_cast<unsigned int>(
      (plan.numBcs() + kThreads - 1) / kThreads);
  replaceResKernel<<<blocks, kThreads, 0, cudaStream(ctx)>>>(
      plan.view(), state.data(), bc_vals.data(), res.data());
  device::checkLastError();
}

void prepareForwardSolve(const DeviceBoundaryPlan& plan,
                         DeviceCsrMatrix&          solve_mat,
                         DeviceVector&             rhs,
                         const DeviceVector&       bc_vals,
                         CudaContext&              ctx)
{
  checkForward(plan, solve_mat, rhs, bc_vals);
  if (plan.numBcs() == 0)
  {
    return;
  }

  eliminateColsKernel<<<static_cast<unsigned int>(plan.numBcs()),
                        kThreads,
                        0,
                        cudaStream(ctx)>>>(
      plan.view(), solve_mat.valsData(), rhs.data(), bc_vals.data());
  device::checkLastError();
  launchRows(plan, solve_mat, &rhs, &bc_vals, ctx);
}

} // namespace assembly
} // namespace femx
