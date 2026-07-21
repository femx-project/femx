#include <stdexcept>

#include "Components.hpp"
#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/CudaAssembly.hpp>
#include <femx/common/Checks.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

namespace femx::model::ns::detail
{
namespace
{

void checkRange(Index                              ie_begin,
                Index                              ie_end,
                const assembly::DeviceAssemblyMap& map)
{
  require(ie_begin == 0 && ie_end == map.numElems(),
          "CUDA Navier assembly requires the full element range");
}

#if defined(FEMX_HAS_ENZYME)

template <Index NumQpts, Index NumNodes, Index Dim>
__global__ void histVjpKernel(
    NavierOperator<MemorySpace::Device>            op,
    Index                                          step,
    Index                                          lag,
    assembly::AssemblyMapView<MemorySpace::Device> map,
    const Real*                                    hist,
    const Real*                                    nxt,
    const Real*                                    adj,
    Real*                                          out)
{
  constexpr Index ndof       = (Dim + 1) * NumNodes;
  constexpr Index elem_tasks = ndof * ndof;
  const Index     task       = static_cast<Index>(blockIdx.x * blockDim.x
                                        + threadIdx.x);
  const Index     ie         = task / elem_tasks;
  const Index     local      = task - ie * elem_tasks;
  const Index     row        = local / ndof;
  const Index     col        = local - row * ndof;
  if (ie >= map.num_elems)
  {
    return;
  }

  Real hist_e[kNumHist * ndof];
  Real hist_dir[kNumHist * ndof];
  Real nxt_e[ndof];

  for (Index i = 0; i < kNumHist * ndof; ++i)
  {
    const Index hist_lag = i / ndof;
    const Index hist_col = i - hist_lag * ndof;
    hist_e[i]            = hist[hist_lag * map.num_states
                     + map.stateDof(ie, hist_col)];
    hist_dir[i]          = 0.0;
  }
  hist_dir[lag * ndof + col] = 1.0;
  for (Index col = 0; col < ndof; ++col)
  {
    nxt_e[col] = nxt[map.stateDof(ie, col)];
  }
  const auto data = op.data();
  const Real val  = __enzyme_fwddiff<Real>(
      reinterpret_cast<void*>(
          evalNavierRowAdj<MemorySpace::Device, NumQpts, NumNodes, Dim>),
      enzyme_const,
      data.numElems(),
      enzyme_const,
      data.NData(),
      enzyme_const,
      data.dNdxData(),
      enzyme_const,
      data.JxWData(),
      enzyme_const,
      op.fluid().rho,
      enzyme_const,
      op.fluid().mu,
      enzyme_const,
      op.dt(),
      enzyme_const,
      ie,
      enzyme_const,
      row,
      enzyme_const,
      step,
      enzyme_dup,
      hist_e,
      hist_dir,
      enzyme_const,
      nxt_e,
      enzyme_const,
      adj[map.resDof(ie, row)]);

  atomicAdd(out + map.stateDof(ie, col), val);
}

template <Index NumQpts, Index NumNodes, Index Dim>
void launchHistVjp(
    const NavierOperator<MemorySpace::Device>& op,
    Index                                      step,
    Index                                      lag,
    const assembly::DeviceAssemblyMap&         map,
    DeviceConstVectorView                      hist,
    DeviceConstVectorView                      nxt,
    DeviceConstVectorView                      adj,
    DeviceVector&                              out,
    CudaContext&                               ctx)
{
  constexpr Index        ndof    = (Dim + 1) * NumNodes;
  constexpr unsigned int threads = 128;
  const Index            tasks   = map.numElems() * ndof * ndof;
  const unsigned int     blocks  = cuda::numBlocks(tasks, threads);
  const auto             stream  = static_cast<cudaStream_t>(ctx.stream());
  histVjpKernel<NumQpts, NumNodes, Dim>
      <<<blocks, threads, 0, stream>>>(op,
                                       step,
                                       lag,
                                       map.view(),
                                       hist.data(),
                                       nxt.data(),
                                       adj.data(),
                                       out.data());
}

#endif

} // namespace

void assembleNavierNext(
    const NavierOperator<MemorySpace::Device>& op,
    Index                                      step,
    Index                                      num_hist,
    Index                                      ie_begin,
    Index                                      ie_end,
    const assembly::DeviceAssemblyMap&         map,
    DeviceConstVectorView                      hist,
    DeviceConstVectorView                      nxt,
    DeviceVector&                              res,
    DeviceCsrMatrix&                           jac,
    CudaContext&                               ctx)
{
  checkRange(ie_begin, ie_end, map);
  assembly::assemble(op,
                     step,
                     num_hist,
                     state::VariableBlock::NextState,
                     map,
                     hist,
                     nxt,
                     res,
                     jac,
                     ctx);
}

void evalNavierRes(
    const NavierOperator<MemorySpace::Device>& op,
    Index                                      step,
    Index                                      num_hist,
    Index                                      ie_begin,
    Index                                      ie_end,
    const assembly::DeviceAssemblyMap&         map,
    DeviceConstVectorView                      hist,
    DeviceConstVectorView                      nxt,
    DeviceVector&                              out,
    CudaContext&                               ctx)
{
  checkRange(ie_begin, ie_end, map);
  assembly::assembleResidual(
      op, step, num_hist, map, hist, nxt, out, ctx);
}

void applyNavierHistJacT(
    const NavierOperator<MemorySpace::Device>& op,
    Index                                      step,
    Index                                      num_hist,
    Index                                      lag,
    Index                                      ie_begin,
    Index                                      ie_end,
    const assembly::DeviceAssemblyMap&         map,
    DeviceConstVectorView                      hist,
    DeviceConstVectorView                      nxt,
    DeviceConstVectorView                      adj,
    DeviceVector&                              out,
    CudaContext&                               ctx)
{
  checkRange(ie_begin, ie_end, map);
  require(num_hist == 2 && lag >= 0 && lag < num_hist,
          "CUDA Navier history VJP requires two valid history states");
  require(map.maxRes() <= kMaxNd && map.maxState() <= kMaxNd,
          "CUDA Navier history VJP element dimensions are unsupported");
  require(hist.size() == num_hist * map.numStates()
              && nxt.size() == map.numStates()
              && adj.size() == map.numRes(),
          "CUDA Navier history VJP dimensions do not match");
  if (out.size() != map.numStates())
  {
    out.resize(map.numStates());
  }
  linalg::CudaVectorHandler vec_handler(ctx);
  vec_handler.zero(out.view());

#if defined(FEMX_HAS_ENZYME)
  if (map.numElems() == 0)
  {
    return;
  }
  const auto data = op.data();
  if (data.numQpts() == 4 && data.numNodes() == 4 && data.dim() == 2)
  {
    launchHistVjp<4, 4, 2>(op, step, lag, map, hist, nxt, adj, out, ctx);
  }
  else if (data.numQpts() == 3 && data.numNodes() == 3
           && data.dim() == 2)
  {
    launchHistVjp<3, 3, 2>(op, step, lag, map, hist, nxt, adj, out, ctx);
  }
  else if (data.numQpts() == 4 && data.numNodes() == 4
           && data.dim() == 3)
  {
    launchHistVjp<4, 4, 3>(op, step, lag, map, hist, nxt, adj, out, ctx);
  }
  else
  {
    throw std::runtime_error(
        "CUDA Navier history VJP received unsupported element dimensions");
  }
  cuda::checkLastError();
#else
  (void) op;
  (void) step;
  throw std::runtime_error(
      "CUDA Navier history VJP requires Enzyme. Configure with "
      "-DFEMX_ENABLE_ENZYME=ON and use Clang as the CUDA compiler.");
#endif
}

} // namespace femx::model::ns::detail
