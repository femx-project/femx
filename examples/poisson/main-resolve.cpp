#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonForward.hpp"
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>

#if defined(FEMX_RESOLVE_USE_CUDA)
#include <cuda_runtime_api.h>

#include "PoissonOperator.hpp"
#include <femx/assembly/CudaAssembly.hpp>
#include <femx/common/Context.hpp>
#endif

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson;
using namespace femx::linalg;

#ifndef FEMX_POISSON_APP_NAME
#define FEMX_POISSON_APP_NAME "poisson-resolve"
#endif

namespace
{

#if defined(FEMX_RESOLVE_USE_CUDA)
constexpr int cuda_threads = 256;

__global__ void negateKernel(Index size, const Real* src, Real* dst)
{
  const Index i = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < size)
  {
    dst[i] = -src[i];
  }
}

__global__ void resNormKernel(Index        rows,
                              const Index* row_ptr,
                              const Index* col_ind,
                              const Real*  vals,
                              const Real*  rhs,
                              const Real*  sol,
                              Real*        norm2)
{
  const Index row =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows)
  {
    return;
  }

  Real val = -rhs[row];
  for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
  {
    val += vals[k] * sol[col_ind[k]];
  }
  atomicAdd(norm2, val * val);
}

unsigned int cudaBlocks(Index size)
{
  const std::int64_t blocks =
      (static_cast<std::int64_t>(size) + cuda_threads - 1) / cuda_threads;
  return static_cast<unsigned int>(blocks);
}

void solveDevice(const PoissonForwardProblem& problem,
                 HostVector&                  x,
                 Real&                        res_norm)
{
  CudaContext ctx;

  fem::DeviceGeometry         geom;
  assembly::DeviceAssemblyMap map;
  assembly::DeviceBoundaryMap bc_map;
  copy(problem.geom(), geom, ctx);
  assembly::copy(problem.map(), map, ctx);
  assembly::copy(problem.bcMap(), bc_map, ctx);

  DeviceVector state(problem.numDofs());
  DeviceVector res;
  DeviceVector rhs(problem.numDofs());
  DeviceVector bc_vals;
  copy(problem.bcVals(), bc_vals, ctx);

  DeviceCsrMatrix mat(map.graph());
  assembly::assemble(PoissonQuadQ1Operator{},
                     geom,
                     map,
                     state,
                     res,
                     mat,
                     ctx);
  negateKernel<<<cudaBlocks(res.size()),
                 cuda_threads,
                 0,
                 static_cast<cudaStream_t>(ctx.stream())>>>(
      res.size(), res.data(), rhs.data());
  device::checkLastError();
  assembly::prepareForwardSolve(bc_map, mat, rhs, bc_vals, ctx);

  ReSolveLinearSolver solver;
  DeviceVector sol;
  solver.solve(mat, rhs, sol, ctx);

  DeviceVector norm2(1);
  resNormKernel<<<cudaBlocks(mat.rows()),
                  cuda_threads,
                  0,
                  static_cast<cudaStream_t>(ctx.stream())>>>(
      mat.rows(),
      mat.rowPtrData(),
      mat.colIndData(),
      mat.valsData(),
      rhs.data(),
      sol.data(),
      norm2.data());
  device::checkLastError();

  HostVector host_norm2;
  copy(sol, x, ctx);
  copy(norm2, host_norm2, ctx);
  ctx.synchronize();
  res_norm = std::sqrt(host_norm2[0]);
}
#endif

int run(const Options& opts)
{
  ExampleHelper         helper("resolve", opts.backend, outputDir());
  PoissonForwardProblem problem(opts);

  HostVector x;
  Real       res_norm = 0.0;
  if (opts.backend == MemorySpace::Host)
  {
    HostCsrMatrix A(problem.map().graph());
    HostVector    rhs;
    problem.assemble(A, rhs);

    ReSolveLinearSolver solver;
    CpuContext          ctx;
    solver.solve(A, rhs, x, ctx);
    res_norm = helper.resNorm(A, rhs, x);
  }
  else
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    solveDevice(problem, x, res_norm);
#else
    throw std::runtime_error(
        "CUDA Poisson backend requires a CUDA-enabled ReSolve build");
#endif
  }

  printReport(std::cout,
              helper.name(),
              problem,
              problem.errorReport(x),
              res_norm);

  if (opts.write_output)
  {
    const std::string base = helper.outputBase(outputStem(opts));
    problem.writeSolution(x, base);
    helper.printVisualizationPath(base);
  }

  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    if (examples::hasHelp(argc, argv))
    {
      printUsage(FEMX_POISSON_APP_NAME, false);
      return 0;
    }
    return run(parseOptions(argc, argv, false));
  }
  catch (const std::exception& e)
  {
    return examples::reportError(FEMX_POISSON_APP_NAME, e);
  }
}
