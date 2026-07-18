#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "../ExampleHelper.hpp"
#if defined(FEMX_POISSON_HAS_DEVICE_SOLVER)
#include "PoissonCuda.hpp"
#endif
#include "PoissonForward.hpp"
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson;
using namespace femx::linalg;

#ifndef FEMX_POISSON_APP_NAME
#define FEMX_POISSON_APP_NAME "poisson-resolve"
#endif

namespace
{

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
    solver.setOperator(A);
    solver.solve(rhs, x);
    res_norm = helper.resNorm(A, rhs, x);
  }
  else
  {
#if defined(FEMX_POISSON_HAS_DEVICE_SOLVER)
    CudaSolveResult result = solveCuda(problem);
    x                      = std::move(result.sol);
    res_norm               = result.res_norm;
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
