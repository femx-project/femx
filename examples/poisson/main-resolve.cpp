#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonForward.hpp"
#include <femx/common/Workspace.hpp>
#include <femx/linalg/native/CsrAssemblyMatrix.hpp>
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

int run(const PoissonOptions& opts)
{
  ExampleHelper         helper("resolve", opts.backend, outputDir());
  PoissonForwardProblem problem(opts);

  CsrAssemblyMatrix A(problem.pattern());
  Vector<Real>      rhs;
  problem.assemble(A, rhs);

  ReSolveLinearSolver solver(opts.backend);

  Vector<Real> x;
  solver.solve(A, rhs, x);

  printReport(std::cout,
              helper.backendName(),
              problem,
              problem.errorReport(x),
              helper.residualNorm(A, rhs, x));

  if (opts.write_output)
  {
    const std::string base = helper.outputBase(outputStem(opts));
    problem.writeSolution(x, base);
    helper.printVisualizationPath(base);
  }

  return 0;
}

bool hasHelp(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
    {
      return true;
    }
  }
  return false;
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    if (hasHelp(argc, argv))
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
