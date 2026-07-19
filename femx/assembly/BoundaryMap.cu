#include <cuda_runtime_api.h>

#include <femx/assembly/BoundaryMap.hpp>
#include <femx/common/Checks.hpp>
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
                                  Real                                 diag,
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
  mat_vals[map.diag(ib)] = diag;
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

__global__ void zeroBoundaryKernel(
    BoundaryMapView<MemorySpace::Device> map,
    Real*                                vals)
{
  const Index ib =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ib < map.num_bcs)
  {
    vals[map.bcRow(ib)] = 0.0;
  }
}

void checkMat(const DeviceBoundaryMap& map, const DeviceCsrMatrix& mat)
{
  require(mat.graph().layoutId() == map.layoutId(),
          "BoundaryMap matrix does not match the mapped CSR layout");
}

void checkForward(const DeviceBoundaryMap& map,
                  const DeviceCsrMatrix&   mat,
                  const DeviceVector&      rhs,
                  const DeviceVector&      bc_vals)
{
  checkMat(map, mat);
  require(rhs.size() == map.rows() && bc_vals.size() == map.numBcs(),
          "BoundaryMap forward vectors have incompatible sizes");
  require(&rhs != &bc_vals,
          "BoundaryMap RHS and prescribed values must not alias");
  const DeviceVector& mat_vals = mat.vals();
  require(&rhs != &mat_vals && &bc_vals != &mat_vals,
          "BoundaryMap vectors must not alias matrix values");
}

cudaStream_t cudaStream(CudaContext& ctx)
{
  return static_cast<cudaStream_t>(ctx.stream());
}

void launchRows(const DeviceBoundaryMap& map,
                DeviceCsrMatrix&         mat,
                Real                     diag,
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
      diag,
      rhs == nullptr ? nullptr : rhs->data(),
      bc_vals == nullptr ? nullptr : bc_vals->data());
  device::checkLastError();
}

} // namespace

void replaceRows(const DeviceBoundaryMap& map,
                 DeviceCsrMatrix&         jac,
                 Real                     diag,
                 CudaContext&             ctx)
{
  checkMat(map, jac);
  launchRows(map, jac, diag, nullptr, nullptr, ctx);
}

void replaceRes(const DeviceBoundaryMap& map,
                DeviceConstVectorView    state,
                DeviceConstVectorView    bc_vals,
                DeviceVectorView         res,
                CudaContext&             ctx)
{
  require(state.size() == map.rows() && res.size() == map.rows()
              && bc_vals.size() == map.numBcs(),
          "BoundaryMap residual vectors have incompatible sizes");
  require(state.data() != res.data() && bc_vals.data() != res.data(),
          "BoundaryMap residual output must not alias its inputs");
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

void zeroBoundary(const DeviceBoundaryMap& map,
                  DeviceVectorView         vals,
                  CudaContext&             ctx)
{
  require(vals.size() == map.rows(),
          "BoundaryMap vector has incompatible size");
  if (map.numBcs() == 0)
  {
    return;
  }
  const unsigned int blocks = static_cast<unsigned int>(
      (map.numBcs() + kThreads - 1) / kThreads);
  zeroBoundaryKernel<<<blocks, kThreads, 0, cudaStream(ctx)>>>(map.view(),
                                                               vals.data());
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
  launchRows(map, solve_mat, 1.0, &rhs, &bc_vals, ctx);
}

} // namespace assembly
} // namespace femx
