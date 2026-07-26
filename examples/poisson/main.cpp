#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonProblem.hpp"
#include "PoissonResidual.hpp"
#include <femx/linalg/native/HostLinearSystem.hpp>
#include <femx/state/StateSolver.hpp>

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson;

#ifndef FEMX_POISSON_APP_NAME
#define FEMX_POISSON_APP_NAME "poisson"
#endif

namespace
{

int run(const Options& opts)
{
  constexpr auto solver_type = runtime::SolverType::Dense;
  if (opts.memspace != MemorySpace::Host)
  {
    throw std::runtime_error(
        "Dense Poisson supports only the CPU backend");
  }

  ExampleHelper  helper(solver_type, opts.memspace, outputDir());
  PoissonProblem problem(opts);

  linalg::HostLinearSystem system;

  HostPoissonResidual          res(problem);
  state::HostLinearStateSolver state_solver(res, system);

  HostVector<Real> result;
  state_solver.solve(result);

  printReport(std::cout,
              helper.name(),
              problem,
              problem.errorReport(result),
              helper.resNorm(res, result, system.context()));

  if (opts.write_output)
  {
    const std::string base = helper.outputBase(outputStem(opts));
    problem.writeSolution(result, base);
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
          "dense solver supports only the CPU backend");
      return 0;
    }
    return run(parseOptions(argc, argv, false));
  }
  catch (const std::exception& e)
  {
    return examples::reportError(FEMX_POISSON_APP_NAME, e);
  }
}
