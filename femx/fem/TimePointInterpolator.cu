#include <cuda_runtime_api.h>

#include <stdexcept>

#include <femx/fem/TimePointInterpolator.hpp>

namespace femx
{
namespace fem
{
namespace
{

constexpr int kThreads = 256;

__global__ void observeKernel(
    PointInterpolatorView<MemorySpace::Device> data,
    const Real*                                state,
    Real*                                      out)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < data.numObservations())
  {
    out[i] = data.eval(i, state);
  }
}

__global__ void addStateJacTKernel(
    PointInterpolatorView<MemorySpace::Device> data,
    const Real*                                dir,
    Real*                                      out)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= data.numObservations())
  {
    return;
  }

  const Real val = dir[i];
  for (Index k = data.begin(i); k < data.end(i); ++k)
  {
    atomicAdd(out + data.dof(k), data.wt(k) * val);
  }
}

cudaStream_t cudaStream(CudaContext& ctx)
{
  return static_cast<cudaStream_t>(ctx.stream());
}

unsigned int numBlocks(Index size)
{
  return static_cast<unsigned int>((size + kThreads - 1) / kThreads);
}

} // namespace

void observe(PointInterpolatorView<MemorySpace::Device> data,
             DeviceConstVectorView                      state,
             DeviceVectorView                           out,
             CudaContext&                               ctx)
{
  if (state.size() != data.numStates()
      || out.size() != data.numObservations())
  {
    throw std::runtime_error(
        "PointInterpolator Device observation size mismatch");
  }
  if (!out.empty() && state.data() == out.data())
  {
    throw std::runtime_error(
        "PointInterpolator observation output must not alias state");
  }
  if (out.empty())
  {
    return;
  }

  observeKernel<<<numBlocks(out.size()), kThreads, 0, cudaStream(ctx)>>>(
      data, state.data(), out.data());
  device::checkLastError();
}

void addStateJacT(PointInterpolatorView<MemorySpace::Device> data,
                  DeviceConstVectorView                      dir,
                  DeviceVectorView                           out,
                  CudaContext&                               ctx)
{
  if (dir.size() != data.numObservations()
      || out.size() != data.numStates())
  {
    throw std::runtime_error(
        "PointInterpolator Device transpose size mismatch");
  }
  if (!out.empty() && dir.data() == out.data())
  {
    throw std::runtime_error(
        "PointInterpolator transpose output must not alias direction");
  }
  if (dir.empty())
  {
    return;
  }

  addStateJacTKernel<<<numBlocks(dir.size()), kThreads, 0, cudaStream(ctx)>>>(
      data, dir.data(), out.data());
  device::checkLastError();
}

} // namespace fem
} // namespace femx
