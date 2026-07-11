#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonForward.hpp"
#include <femx/common/Workspace.hpp>
#include <femx/linalg/native/DenseAssemblyMatrix.hpp>
#include <femx/linalg/native/DenseLinearSolver.hpp>

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson;
using namespace femx::linalg;

#ifndef FEMX_POISSON_APP_NAME
#define FEMX_POISSON_APP_NAME "poisson"
#endif

namespace
{

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

int run(const Options& opts)
{
  if (opts.backend != WorkspaceType::Cpu)
  {
    throw std::runtime_error("Dense Poisson backend supports only 'cpu'");
  }

  ExampleHelper         helper("dense", opts.backend, outputDir());
  PoissonForwardProblem problem(opts);

  DenseAssemblyMatrix A;
  Vector<Real>        rhs;
  problem.assemble(A, rhs);

  DenseLinearSolver solver;

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

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    if (hasHelp(argc, argv))
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
