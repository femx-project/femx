#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonForward.hpp"
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/native/DenseLinearSolver.hpp>
#include <femx/runtime/LinearSystemFactory.hpp>

using namespace femx;
using namespace femx::assembly;
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
  constexpr auto solver_type = runtime::SolverType::Dense;
  if (opts.execution_device != runtime::ExecutionDevice::Host)
  {
    throw std::runtime_error(
        "Dense Poisson supports only Host execution");
  }

  ExampleHelper         helper(solver_type, opts.execution_device, outputDir());
  PoissonForwardProblem problem(opts);

  HostCsrMatrix    A(problem.map().pattern());
  HostVector<Real> rhs;
  problem.assemble(A, rhs);

  DenseLinearSolver   native_solver;
  linalg::HostContext ctx;

  HostVector<Real> x;
  native_solver.solve(A, rhs, x, ctx);

  printReport(std::cout,
              helper.name(),
              problem,
              problem.errorReport(x),
              helper.resNorm(A, rhs, x, ctx));

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
      printUsage(
          FEMX_POISSON_APP_NAME,
          false,
          "dense solver supports Host execution only");
      return 0;
    }
    return run(parseOptions(argc, argv, false));
  }
  catch (const std::exception& e)
  {
    return examples::reportError(FEMX_POISSON_APP_NAME, e);
  }
}
