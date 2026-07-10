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

constexpr Index dense_default_cells = 8;

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

bool hasOption(int argc, char** argv, const std::string& name)
{
  const std::string assignment = name + "=";
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == name || arg.rfind(assignment, 0) == 0)
    {
      return true;
    }
  }
  return false;
}

Options parseDenseOptions(int argc, char** argv)
{
  Options opts = parseOptions(argc, argv, false);
  if (!hasOption(argc, argv, "--nx"))
  {
    opts.num_x_cells = dense_default_cells;
  }
  if (!hasOption(argc, argv, "--ny"))
  {
    opts.num_y_cells = dense_default_cells;
  }
  return opts;
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
      printUsage(FEMX_POISSON_APP_NAME,
                 false,
                 "dense solver supports cpu only");
      std::cout << "  dense solver defaults to --nx " << dense_default_cells
                << " --ny " << dense_default_cells << '\n';
      return 0;
    }
    return run(parseDenseOptions(argc, argv));
  }
  catch (const std::exception& e)
  {
    return examples::reportError(FEMX_POISSON_APP_NAME, e);
  }
}
