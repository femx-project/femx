#include <cuda_runtime_api.h>

#include <femx/inverse/DeviceTimeObjective.hpp>

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
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
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
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < size)
  {
    const Real res = lo_wt * lo[i] + hi_wt * hi[i] - data[i];
    dir[i]         = scale * wts[i] * res;
  }
}

__global__ void quadraticValueKernel(Index        size,
                                     const Index* rows,
                                     const Index* cols,
                                     const Real*  vals,
                                     const Real*  row_ref,
                                     const Real*  col_ref,
                                     const Real*  prm,
                                     Real*        scalar)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < size)
  {
    const Real row_val = prm[rows[i]] - row_ref[i];
    const Real col_val = prm[cols[i]] - col_ref[i];
    atomicAdd(scalar, 0.5 * vals[i] * row_val * col_val);
  }
}

__global__ void quadraticGradKernel(Index        size,
                                    const Index* rows,
                                    const Index* cols,
                                    const Real*  vals,
                                    const Real*  row_ref,
                                    const Real*  col_ref,
                                    const Real*  prm,
                                    Real*        out)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < size)
  {
    const Index row     = rows[i];
    const Index col     = cols[i];
    const Real  row_val = prm[row] - row_ref[i];
    const Real  col_val = prm[col] - col_ref[i];
    const Real  val     = 0.5 * vals[i];
    atomicAdd(out + row, val * col_val);
    atomicAdd(out + col, val * row_val);
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

using DeviceConstIndexView =
    VectorView<MemorySpace::Device, const Index>;

void launchQuadraticValue(DeviceConstIndexView  rows,
                          DeviceConstIndexView  cols,
                          DeviceConstVectorView vals,
                          DeviceConstVectorView row_ref,
                          DeviceConstVectorView col_ref,
                          DeviceConstVectorView prm,
                          DeviceVectorView      scalar,
                          CudaContext&          ctx)
{
  if (vals.empty())
  {
    return;
  }
  quadraticValueKernel<<<blocks(vals.size()), kThreads, 0, stream(ctx)>>>(
      vals.size(),
      rows.data(),
      cols.data(),
      vals.data(),
      row_ref.data(),
      col_ref.data(),
      prm.data(),
      scalar.data());
  device::checkLastError();
}

void launchQuadraticGrad(DeviceConstIndexView  rows,
                         DeviceConstIndexView  cols,
                         DeviceConstVectorView vals,
                         DeviceConstVectorView row_ref,
                         DeviceConstVectorView col_ref,
                         DeviceConstVectorView prm,
                         DeviceVectorView      out,
                         CudaContext&          ctx)
{
  if (vals.empty())
  {
    return;
  }
  quadraticGradKernel<<<blocks(vals.size()), kThreads, 0, stream(ctx)>>>(
      vals.size(),
      rows.data(),
      cols.data(),
      vals.data(),
      row_ref.data(),
      col_ref.data(),
      prm.data(),
      out.data());
  device::checkLastError();
}

} // namespace detail
} // namespace inverse
} // namespace femx
