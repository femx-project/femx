#include <stdexcept>

#include "NavierElementKernel.hpp"
#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/CudaAssembly.hpp>
#include <femx/common/Checks.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>

namespace femx::model::navier::detail
{
namespace
{

void checkRange(Index                              ie_begin,
                Index                              ie_end,
                const assembly::DeviceAssemblyMap& map)
{
  require(ie_begin == 0 && ie_end == map.numElems(),
          "CUDA Navier-Stokes assembly requires the full element range");
}

#if defined(FEMX_HAS_ENZYME)

template <Index NumQpts, Index NumNodes, Index Dim>
__global__ void histVjpKernel(
    DeviceNavierElementKernel       kernel,
    Index                           step,
    Index                           lag,
    assembly::DeviceAssemblyMapView map,
    const Real*                     hist,
    const Real*                     nxt,
    const Real*                     adj,
    Real*                           out)
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
  Real nxt_dir[ndof];

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
    nxt_e[col]   = nxt[map.stateDof(ie, col)];
    nxt_dir[col] = 0.0;
  }
  const auto data = kernel.data();
  const Real val  = __enzyme_fwddiff<Real>(
      reinterpret_cast<void*>(
          evalResRowAdj<MemorySpace::Device, NumQpts, NumNodes, Dim>),
      enzyme_const,
      data.numElems(),
      enzyme_const,
      data.NData(),
      enzyme_const,
      data.dNdxData(),
      enzyme_const,
      data.JxWData(),
      enzyme_const,
      kernel.fluid().rho,
      enzyme_const,
      kernel.fluid().mu,
      enzyme_const,
      kernel.dt(),
      enzyme_const,
      ie,
      enzyme_const,
      row,
      enzyme_const,
      step,
      enzyme_dup,
      hist_e,
      hist_dir,
      enzyme_dup,
      nxt_e,
      nxt_dir,
      enzyme_const,
      adj[map.resDof(ie, row)]);

  atomicAdd(out + map.stateDof(ie, col), val);
}

template <Index NumQpts, Index NumNodes, Index Dim>
void launchHistVjp(
    const DeviceNavierElementKernel&   kernel,
    Index                              step,
    Index                              lag,
    const assembly::DeviceAssemblyMap& map,
    DeviceVectorView<const Real>       hist,
    DeviceVectorView<const Real>       nxt,
    DeviceVectorView<const Real>       adj,
    DeviceVector<Real>&                out,
    linalg::CudaContext&               ctx)
{
  constexpr Index        ndof    = (Dim + 1) * NumNodes;
  constexpr unsigned int threads = 128;
  const Index            tasks   = map.numElems() * ndof * ndof;
  const unsigned int     blocks  = cuda::numBlocks(tasks, threads);
  const auto             stream  = static_cast<cudaStream_t>(ctx.stream());
  histVjpKernel<NumQpts, NumNodes, Dim>
      <<<blocks, threads, 0, stream>>>(kernel,
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

void assembleNext(
    const DeviceNavierElementKernel&   kernel,
    Index                              step,
    Index                              num_hist,
    Index                              ie_begin,
    Index                              ie_end,
    const assembly::DeviceAssemblyMap& map,
    DeviceVectorView<const Real>       hist,
    DeviceVectorView<const Real>       nxt,
    DeviceVector<Real>&                res,
    linalg::CudaSystemMatrix&          jac,
    linalg::CudaContext&               ctx)
{
  checkRange(ie_begin, ie_end, map);
  assembly::assembleResidualAndJacobian(kernel,
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

void applyHistJacT(
    const DeviceNavierElementKernel&   kernel,
    Index                              step,
    Index                              num_hist,
    Index                              lag,
    Index                              ie_begin,
    Index                              ie_end,
    const assembly::DeviceAssemblyMap& map,
    DeviceVectorView<const Real>       hist,
    DeviceVectorView<const Real>       nxt,
    DeviceVectorView<const Real>       adj,
    DeviceVector<Real>&                out,
    linalg::CudaContext&               ctx)
{
  checkRange(ie_begin, ie_end, map);
  require(num_hist == 2 && lag >= 0 && lag < num_hist,
          "CUDA Navier-Stokes history VJP requires two valid history states");
  require(map.maxRes() <= kMaxNd && map.maxState() <= kMaxNd,
          "CUDA Navier-Stokes history VJP element dimensions are unsupported");
  require(hist.size() == num_hist * map.numStates()
              && nxt.size() == map.numStates()
              && adj.size() == map.numRes(),
          "CUDA Navier-Stokes history VJP dimensions do not match");
  if (out.size() != map.numStates())
  {
    out.resize(map.numStates());
  }
  auto& vec_handler = ctx.vectorHandler();
  vec_handler.zero(out.view());

#if defined(FEMX_HAS_ENZYME)
  if (map.numElems() == 0)
  {
    return;
  }
  const auto data = kernel.data();
  if (data.numQuadraturePoints() == 4 && data.numShapes() == 4 && data.dim() == 2)
  {
    launchHistVjp<4, 4, 2>(
        kernel, step, lag, map, hist, nxt, adj, out, ctx);
  }
  else if (data.numQuadraturePoints() == 3 && data.numShapes() == 3 && data.dim() == 2)
  {
    launchHistVjp<3, 3, 2>(
        kernel, step, lag, map, hist, nxt, adj, out, ctx);
  }
  else if (data.numQuadraturePoints() == 4 && data.numShapes() == 4 && data.dim() == 3)
  {
    launchHistVjp<4, 4, 3>(
        kernel, step, lag, map, hist, nxt, adj, out, ctx);
  }
  else
  {
    throw std::runtime_error(
        "CUDA Navier-Stokes history VJP received unsupported element dimensions");
  }
  cuda::checkLastError();
#else
  (void) kernel;
  (void) step;
  throw std::runtime_error(
      "CUDA Navier-Stokes history VJP requires Enzyme. Configure with "
      "-DFEMX_ENABLE_ENZYME=ON and use Clang as the CUDA compiler.");
#endif
}

} // namespace femx::model::navier::detail
