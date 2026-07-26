#include <cuda_runtime_api.h>

#include <femx/assembly/BoundaryMap.hpp>
#include <femx/common/Checks.hpp>

namespace femx::assembly
{
namespace
{

constexpr int kThreads = 256;

__global__ void replaceResKernel(const Index* rows,
                                 Index        count,
                                 const Real*  state,
                                 const Real*  prescribed_values,
                                 Real*        residual)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count)
  {
    const Index row = rows[i];
    residual[row]   = state[row] - prescribed_values[i];
  }
}

__global__ void zeroBoundaryKernel(const Index* rows,
                                   Index        count,
                                   Real*        values)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count)
  {
    values[rows[i]] = 0.0;
  }
}

cudaStream_t stream(linalg::CudaContext& ctx)
{
  return static_cast<cudaStream_t>(ctx.stream());
}

} // namespace

void replaceRes(const DeviceBoundaryMap&     map,
                DeviceVectorView<const Real> state,
                DeviceVectorView<const Real> prescribed_values,
                DeviceVectorView<Real>       residual,
                linalg::CudaContext&         ctx)
{
  const auto rows = map.view().constrained_rows;
  require(prescribed_values.size() == rows.size(),
          "BoundaryMap prescribed-value size mismatch");
  require(state.data() != residual.data()
              && prescribed_values.data() != residual.data(),
          "BoundaryMap residual output must not alias its inputs");
  if (rows.empty())
  {
    return;
  }
  replaceResKernel<<<cuda::numBlocks(rows.size(), kThreads),
                     kThreads,
                     0,
                     stream(ctx)>>>(rows.data(),
                                    rows.size(),
                                    state.data(),
                                    prescribed_values.data(),
                                    residual.data());
  cuda::checkLastError();
}

void zeroBoundary(const DeviceBoundaryMap& map,
                  DeviceVectorView<Real>   values,
                  linalg::CudaContext&     ctx)
{
  const auto rows = map.view().constrained_rows;
  if (rows.empty())
  {
    return;
  }
  zeroBoundaryKernel<<<cuda::numBlocks(rows.size(), kThreads),
                       kThreads,
                       0,
                       stream(ctx)>>>(rows.data(), rows.size(), values.data());
  cuda::checkLastError();
}

} // namespace femx::assembly
