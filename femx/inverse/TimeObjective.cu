#include <cuda_runtime_api.h>

#include "TimeObjectivePlan.hpp"

namespace femx
{
namespace inverse
{
namespace detail
{
namespace
{

constexpr int kThreads = 256;

__global__ void lsValueKernel(Index       size,
                              const Real* lo,
                              const Real* hi,
                              const Real* data,
                              const Real* wts,
                              Real        lo_wt,
                              Real        hi_wt,
                              Real        row_wt,
                              Real*       scalar)
{
  const Index i = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < size)
  {
    const Real res = lo_wt * lo[i] + hi_wt * hi[i] - data[i];
    atomicAdd(scalar, 0.5 * row_wt * wts[i] * res * res);
  }
}

__global__ void lsDirKernel(Index       size,
                            const Real* lo,
                            const Real* hi,
                            const Real* data,
                            const Real* wts,
                            Real        lo_wt,
                            Real        hi_wt,
                            Real        scale,
                            Real*       dir)
{
  const Index i = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < size)
  {
    const Real res = lo_wt * lo[i] + hi_wt * hi[i] - data[i];
    dir[i]         = scale * wts[i] * res;
  }
}

unsigned int blocks(Index size)
{
  return static_cast<unsigned int>((size + kThreads - 1) / kThreads);
}

cudaStream_t stream(CudaContext& ctx)
{
  return static_cast<cudaStream_t>(ctx.stream());
}

} // namespace

void launchLsValue(DeviceConstVectorView lo,
                   DeviceConstVectorView hi,
                   DeviceConstVectorView data,
                   DeviceConstVectorView wts,
                   Real                  lo_wt,
                   Real                  hi_wt,
                   Real                  row_wt,
                   DeviceVectorView      scalar,
                   CudaContext&          ctx)
{
  if (lo.empty())
  {
    return;
  }
  lsValueKernel<<<blocks(lo.size()), kThreads, 0, stream(ctx)>>>(
      lo.size(),
      lo.data(),
      hi.data(),
      data.data(),
      wts.data(),
      lo_wt,
      hi_wt,
      row_wt,
      scalar.data());
  device::checkLastError();
}

void launchLsDir(DeviceConstVectorView lo,
                 DeviceConstVectorView hi,
                 DeviceConstVectorView data,
                 DeviceConstVectorView wts,
                 Real                  lo_wt,
                 Real                  hi_wt,
                 Real                  scale,
                 DeviceVectorView      dir,
                 CudaContext&          ctx)
{
  if (lo.empty())
  {
    return;
  }
  lsDirKernel<<<blocks(lo.size()), kThreads, 0, stream(ctx)>>>(
      lo.size(),
      lo.data(),
      hi.data(),
      data.data(),
      wts.data(),
      lo_wt,
      hi_wt,
      scale,
      dir.data());
  device::checkLastError();
}

} // namespace detail
} // namespace inverse
} // namespace femx
