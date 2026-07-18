#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonForward.hpp"
#include <femx/linalg/native/DenseLinearSolver.hpp>
#include <femx/linalg/native/MapCsrMatrix.hpp>

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson;
using namespace femx::linalg;

#ifndef FEMX_POISSON_APP_NAME
#define FEMX_POISSON_APP_NAME "poisson"
#endif

namespace
{

int run(const Options& opts)
{
  if (opts.backend != MemorySpace::Host)
  {
    throw std::runtime_error("Dense Poisson backend supports only 'cpu'");
  }

  ExampleHelper         helper("dense", opts.backend, outputDir());
  PoissonForwardProblem problem(opts);

  MapCsrMatrix A(problem.map());
  HostVector   rhs;
  problem.assemble(A.mat(), rhs);

  DenseLinearSolver solver;

  HostVector x;
  solver.solve(A, rhs, x);

  printReport(std::cout,
              helper.name(),
              problem,
              problem.errorReport(x),
              helper.resNorm(A.mat(), rhs, x));

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
      printUsage(FEMX_POISSON_APP_NAME, false, "dense solver supports cpu only");
      return 0;
    }
    return run(parseOptions(argc, argv, false));
  }
  catch (const std::exception& e)
  {
    return examples::reportError(FEMX_POISSON_APP_NAME, e);
  }
}
