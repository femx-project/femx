#include <cuda_runtime_api.h>

#include <stdexcept>

#include <femx/assembly/BoundaryMap.hpp>
#include <femx/common/Context.hpp>

namespace femx
{
namespace assembly
{
namespace
{

constexpr int kThreads = 256;

__global__ void replaceRowsKernel(BoundaryMapView<MemorySpace::Device> map,
                                  const Index*                         row_ptr,
                                  Real*                                mat_vals,
                                  Real*                                rhs,
                                  const Real*                          bc_vals)
{
  const Index ib =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ib >= map.num_bcs)
  {
    return;
  }

  const Index row = map.bcRow(ib);
  for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
  {
    mat_vals[k] = 0.0;
  }
  mat_vals[map.diag(ib)] = 1.0;
  if (rhs != nullptr)
  {
    rhs[row] = bc_vals[ib];
  }
}

__global__ void eliminateColsKernel(
    BoundaryMapView<MemorySpace::Device> map,
    Real*                                mat_vals,
    Real*                                rhs,
    const Real*                          bc_vals)
{
  const Index ib = static_cast<Index>(blockIdx.x);
  if (ib >= map.num_bcs)
  {
    return;
  }

  const Real bc = bc_vals[ib];
  for (Index i  = map.colBegin(ib) + threadIdx.x; i < map.colEnd(ib);
       i       += blockDim.x)
  {
    const Index row = map.col_rows[i];
    const Index k   = map.col_entries[i];
    const Real  val = mat_vals[k];
    if (!map.isBc(row))
    {
      atomicAdd(rhs + row, -val * bc);
    }
    mat_vals[k] = 0.0;
  }
}

__global__ void replaceResKernel(
    BoundaryMapView<MemorySpace::Device> map,
    const Real*                          state,
    const Real*                          bc_vals,
    Real*                                res)
{
  const Index ib =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ib < map.num_bcs)
  {
    const Index row = map.bcRow(ib);
    res[row]        = state[row] - bc_vals[ib];
  }
}

void checkMat(const DeviceBoundaryMap& map, const DeviceCsrMatrix& mat)
{
  if (mat.graph().layoutId() != map.layoutId())
  {
    throw std::runtime_error(
        "BoundaryMap matrix does not match the mapped CSR layout");
  }
}

void checkForward(const DeviceBoundaryMap& map,
                  const DeviceCsrMatrix&   mat,
                  const DeviceVector&      rhs,
                  const DeviceVector&      bc_vals)
{
  checkMat(map, mat);
  if (rhs.size() != map.rows() || bc_vals.size() != map.numBcs())
  {
    throw std::runtime_error(
        "BoundaryMap forward vectors have incompatible sizes");
  }
  if (&rhs == &bc_vals)
  {
    throw std::runtime_error(
        "BoundaryMap RHS and prescribed values must not alias");
  }
  const DeviceVector& mat_vals = mat.vals();
  if (&rhs == &mat_vals || &bc_vals == &mat_vals)
  {
    throw std::runtime_error(
        "BoundaryMap vectors must not alias matrix values");
  }
}

cudaStream_t cudaStream(CudaContext& ctx)
{
  return static_cast<cudaStream_t>(ctx.stream());
}

void launchRows(const DeviceBoundaryMap& map,
                DeviceCsrMatrix&         mat,
                DeviceVector*            rhs,
                const DeviceVector*      bc_vals,
                CudaContext&             ctx)
{
  if (map.numBcs() == 0)
  {
    return;
  }
  const unsigned int blocks = static_cast<unsigned int>(
      (map.numBcs() + kThreads - 1) / kThreads);
  replaceRowsKernel<<<blocks, kThreads, 0, cudaStream(ctx)>>>(
      map.view(),
      mat.rowPtrData(),
      mat.valsData(),
      rhs == nullptr ? nullptr : rhs->data(),
      bc_vals == nullptr ? nullptr : bc_vals->data());
  device::checkLastError();
}

} // namespace

void replaceRows(const DeviceBoundaryMap& map,
                 DeviceCsrMatrix&         jac,
                 CudaContext&             ctx)
{
  checkMat(map, jac);
  launchRows(map, jac, nullptr, nullptr, ctx);
}

void replaceRes(const DeviceBoundaryMap& map,
                const DeviceVector&      state,
                const DeviceVector&      bc_vals,
                DeviceVector&            res,
                CudaContext&             ctx)
{
  if (state.size() != map.rows() || res.size() != map.rows()
      || bc_vals.size() != map.numBcs())
  {
    throw std::runtime_error(
        "BoundaryMap residual vectors have incompatible sizes");
  }
  if (&state == &res || &bc_vals == &res)
  {
    throw std::runtime_error(
        "BoundaryMap residual output must not alias its inputs");
  }
  if (map.numBcs() == 0)
  {
    return;
  }
  const unsigned int blocks = static_cast<unsigned int>(
      (map.numBcs() + kThreads - 1) / kThreads);
  replaceResKernel<<<blocks, kThreads, 0, cudaStream(ctx)>>>(
      map.view(), state.data(), bc_vals.data(), res.data());
  device::checkLastError();
}

void prepareForwardSolve(const DeviceBoundaryMap& map,
                         DeviceCsrMatrix&         solve_mat,
                         DeviceVector&            rhs,
                         const DeviceVector&      bc_vals,
                         CudaContext&             ctx)
{
  checkForward(map, solve_mat, rhs, bc_vals);
  if (map.numBcs() == 0)
  {
    return;
  }

  eliminateColsKernel<<<static_cast<unsigned int>(map.numBcs()),
                        kThreads,
                        0,
                        cudaStream(ctx)>>>(
      map.view(), solve_mat.valsData(), rhs.data(), bc_vals.data());
  device::checkLastError();
  launchRows(map, solve_mat, &rhs, &bc_vals, ctx);
}

} // namespace assembly
} // namespace femx
