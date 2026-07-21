#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonForward.hpp"
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>

#if defined(FEMX_RESOLVE_USE_CUDA)
#include "PoissonComponents.hpp"
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

void solveHost(const ExampleHelper&         helper,
               const PoissonForwardProblem& problem,
               HostVector&                  x,
               Real&                        res_norm)
{
  HostCsrMatrix mat(problem.map().pattern());
  HostVector    rhs;
  problem.assemble(mat, rhs);

  ReSolveLinearSolver solver;
  CpuContext          ctx;
  solver.solve(mat, rhs, x, ctx);
  res_norm = helper.resNorm(mat, rhs, x, ctx);
}

#if defined(FEMX_RESOLVE_USE_CUDA)
void solveDevice(const ExampleHelper&         helper,
                 const PoissonForwardProblem& problem,
                 HostVector&                  x,
                 Real&                        res_norm)
{
  CudaContext       ctx;
  CudaVectorHandler vec_handler(ctx);

  fem::DeviceGeometry              geom;
  fem::DeviceElementQuadratureData element_data;
  assembly::DeviceAssemblyMap      map;
  assembly::DeviceBoundaryMap      bc_map;
  copy(problem.geom(), geom, ctx);
  copy(problem.elementData(), element_data, ctx);
  assembly::copy(problem.map(), map, ctx);
  assembly::copy(problem.bcMap(), bc_map, ctx);

  DeviceVector state(problem.numDofs());
  DeviceVector res;
  DeviceVector rhs(problem.numDofs());
  DeviceVector bc_vals;
  vec_handler.copy(problem.bcVals(), bc_vals);

  DeviceCsrMatrix mat(map.pattern());
  assembly::assemble(PoissonComponents<MemorySpace::Device>(element_data.view()),
                     geom,
                     map,
                     state,
                     res,
                     mat,
                     ctx);
  vec_handler.axpby(-1.0, res.view(), 0.0, rhs.view());
  assembly::prepareForwardSolve(bc_map, mat, rhs, bc_vals, ctx);

  ReSolveLinearSolver solver;
  DeviceVector        sol;
  solver.solve(mat, rhs, sol, ctx);

  res_norm = helper.resNorm(mat, rhs, sol, ctx);
  vec_handler.copy(sol, x);
  ctx.sync();
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
    solveHost(helper, problem, x, res_norm);
  }
  else
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    solveDevice(helper, problem, x, res_norm);
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
