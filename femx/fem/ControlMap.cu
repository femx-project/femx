#include <cuda_runtime_api.h>

#include <stdexcept>

#include <femx/fem/ControlMap.hpp>

namespace femx
{
namespace fem
{
namespace
{

constexpr int kThreads = 256;

FEMX_DEVICE Real ctrVal(ControlMapView<MemorySpace::Device> map,
                        Index                               step,
                        Index                               i,
                        const Real*                         prm)
{
  const Index lo    = map.lower[step];
  const Index hi    = map.upper[step];
  const Real  hi_wt = map.upper_wts[step];
  const Real  lo_wt = 1.0 - hi_wt;
  Real        val   = 0.0;
  for (Index k = map.ctrBegin(i); k < map.ctrEnd(i); ++k)
  {
    const Index col = map.ctr_cols[k];
    Real        q   = lo_wt * prm[map.ctrIndex(lo, col)];
    if (hi != lo && hi_wt != 0.0)
    {
      q += hi_wt * prm[map.ctrIndex(hi, col)];
    }
    val += map.ctr_wts[k] * q;
  }
  return val;
}

__global__ void controlValsKernel(
    ControlMapView<MemorySpace::Device> map,
    Index                               step,
    const Real*                         prm,
    Real*                               out)
{
  const Index ib =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ib < map.num_ctr)
  {
    out[ib] = ctrVal(map, step, ib, prm);
  }
  else if (ib < map.numBcs())
  {
    const Index i = ib - map.num_ctr;
    out[ib]       = map.fixedValue(step, i);
  }
}

__global__ void controlJacKernel(
    ControlMapView<MemorySpace::Device> map,
    Index                               step,
    const Real*                         dir,
    Real*                               out)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < map.num_ctr)
  {
    out[map.dof(i)] = -ctrVal(map, step, i, dir);
  }
}

__global__ void addControlJacTKernel(
    ControlMapView<MemorySpace::Device> map,
    Index                               step,
    const Real*                         adj,
    Real*                               grad)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= map.num_ctr)
  {
    return;
  }

  const Index lo    = map.lower[step];
  const Index hi    = map.upper[step];
  const Real  hi_wt = map.upper_wts[step];
  const Real  lo_wt = 1.0 - hi_wt;
  const Real  val   = -adj[map.dof(i)];
  for (Index k = map.ctrBegin(i); k < map.ctrEnd(i); ++k)
  {
    const Index col = map.ctr_cols[k];
    const Real  wt  = map.ctr_wts[k] * val;
    atomicAdd(grad + map.ctrIndex(lo, col), lo_wt * wt);
    if (hi != lo && hi_wt != 0.0)
    {
      atomicAdd(grad + map.ctrIndex(hi, col), hi_wt * wt);
    }
  }
}

__global__ void initialStateKernel(
    InitialStateMapView<MemorySpace::Device> map,
    const Real*                              prm,
    Real*                                    out)
{
  const Index row =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= map.num_states)
  {
    return;
  }

  Real val = map.mean[row];
  for (Index col = 0; col < map.num_modes; ++col)
  {
    val += map.mode(row, col) * prm[map.init_off + col];
  }
  out[row] = val;
}

__global__ void setInitialControlKernel(
    InitialStateMapView<MemorySpace::Device> map,
    const Real*                              prm,
    Real*                                    out)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= map.num_ctr)
  {
    return;
  }

  Real val = 0.0;
  for (Index k = map.ctrBegin(i); k < map.ctrEnd(i); ++k)
  {
    val += map.ctr_wts[k] * prm[map.ctr_off + map.ctr_cols[k]];
  }
  out[map.ctr_dofs[i]] = val;
}

__global__ void addInitialModesTKernel(
    InitialStateMapView<MemorySpace::Device> map,
    const Real*                              adj,
    Real*                                    grad)
{
  const Index col =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= map.num_modes)
  {
    return;
  }

  Real val = 0.0;
  for (Index row = 0; row < map.num_states; ++row)
  {
    val += map.mode(row, col) * adj[row];
  }
  atomicAdd(grad + map.init_off + col, val);
}

__global__ void addInitialControlTKernel(
    InitialStateMapView<MemorySpace::Device> map,
    const Real*                              adj,
    Real*                                    grad)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= map.num_ctr)
  {
    return;
  }

  const Real val = adj[map.ctr_dofs[i]];
  for (Index k = map.ctrBegin(i); k < map.ctrEnd(i); ++k)
  {
    atomicAdd(grad + map.ctr_off + map.ctr_cols[k],
              map.ctr_wts[k] * val);
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

void checkControl(ControlMapView<MemorySpace::Device> map,
                  Index                               step,
                  DeviceConstVectorView               in,
                  DeviceVectorView                    out,
                  Index                               out_size)
{
  if (step < 0 || step >= map.num_steps || in.size() != map.num_prm
      || out.size() != out_size)
  {
    throw std::runtime_error("ControlMap Device input mismatch");
  }
}

} // namespace

void controlVals(ControlMapView<MemorySpace::Device> map,
                 Index                               step,
                 DeviceConstVectorView               prm,
                 DeviceVectorView                    out,
                 CudaContext&                        ctx)
{
  checkControl(map, step, prm, out, map.numBcs());
  if (map.numBcs() == 0)
  {
    return;
  }
  controlValsKernel<<<numBlocks(map.numBcs()),
                      kThreads,
                      0,
                      cudaStream(ctx)>>>(map, step, prm.data(), out.data());
  device::checkLastError();
}

void controlJac(ControlMapView<MemorySpace::Device> map,
                Index                               step,
                DeviceConstVectorView               dir,
                DeviceVectorView                    out,
                CudaContext&                        ctx)
{
  checkControl(map, step, dir, out, map.num_states);
  if (!out.empty())
  {
    device::zero(out.data(),
                 static_cast<std::size_t>(out.size()) * sizeof(Real),
                 ctx.stream());
  }
  if (map.num_ctr == 0)
  {
    return;
  }
  controlJacKernel<<<numBlocks(map.num_ctr),
                     kThreads,
                     0,
                     cudaStream(ctx)>>>(map, step, dir.data(), out.data());
  device::checkLastError();
}

void addControlJacT(ControlMapView<MemorySpace::Device> map,
                    Index                               step,
                    DeviceConstVectorView               adj,
                    DeviceVectorView                    grad,
                    CudaContext&                        ctx)
{
  if (step < 0 || step >= map.num_steps
      || adj.size() != map.num_states || grad.size() != map.num_prm)
  {
    throw std::runtime_error("ControlMap Device transpose input mismatch");
  }
  if (map.num_ctr == 0)
  {
    return;
  }
  addControlJacTKernel<<<numBlocks(map.num_ctr),
                         kThreads,
                         0,
                         cudaStream(ctx)>>>(
      map, step, adj.data(), grad.data());
  device::checkLastError();
}

void initialState(InitialStateMapView<MemorySpace::Device> map,
                  DeviceConstVectorView                    prm,
                  DeviceVectorView                         out,
                  CudaContext&                             ctx)
{
  if (prm.size() != map.num_prm || out.size() != map.num_states)
  {
    throw std::runtime_error("InitialStateMap Device input mismatch");
  }
  if (map.num_states > 0)
  {
    initialStateKernel<<<numBlocks(map.num_states),
                         kThreads,
                         0,
                         cudaStream(ctx)>>>(map, prm.data(), out.data());
    device::checkLastError();
  }
  if (map.num_ctr > 0)
  {
    setInitialControlKernel<<<numBlocks(map.num_ctr),
                              kThreads,
                              0,
                              cudaStream(ctx)>>>(
        map, prm.data(), out.data());
    device::checkLastError();
  }
}

void addInitialJacT(InitialStateMapView<MemorySpace::Device> map,
                    DeviceConstVectorView                    adj,
                    DeviceVectorView                         grad,
                    CudaContext&                             ctx)
{
  if (adj.size() != map.num_states || grad.size() != map.num_prm)
  {
    throw std::runtime_error(
        "InitialStateMap Device transpose input mismatch");
  }
  if (map.num_modes > 0)
  {
    addInitialModesTKernel<<<numBlocks(map.num_modes),
                             kThreads,
                             0,
                             cudaStream(ctx)>>>(
        map, adj.data(), grad.data());
    device::checkLastError();
  }
  if (map.num_ctr > 0)
  {
    addInitialControlTKernel<<<numBlocks(map.num_ctr),
                               kThreads,
                               0,
                               cudaStream(ctx)>>>(
        map, adj.data(), grad.data());
    device::checkLastError();
  }
}

} // namespace fem
} // namespace femx
