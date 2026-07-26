#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonForward.hpp"
#include <femx/runtime/LinearSystemFactory.hpp>
#include <femx/state/StateSolver.hpp>

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

  auto system = runtime::makeHostLinearSystem(solver_type);

  state::HostLinearStateSolver solver(problem, *system);
  const HostVector<Real>       prm;

  HostVector<Real> x;
  solver.solve(prm, x);

  printReport(std::cout,
              helper.name(),
              problem,
              problem.errorReport(x),
              helper.resNorm(
                  problem, x, prm, system->context()));

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
