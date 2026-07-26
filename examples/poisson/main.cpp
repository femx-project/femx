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
  if (opts.execution_device != runtime::ExecutionDevice::Host)
  {
    throw std::runtime_error(
        "Dense Poisson supports only Host execution");
  }

  ExampleHelper  helper(solver_type, opts.execution_device, outputDir());
  PoissonProblem prob(opts);

  linalg::HostLinearSystem system;

  HostPoissonResidual          res(prob);
  state::HostLinearStateSolver state_solver(res, system);
  const HostVector<Real>       prm;

  HostVector<Real> sol;
  state_solver.solve(prm, sol);

  printReport(std::cout,
              helper.name(),
              prob,
              prob.errorReport(sol),
              helper.resNorm(res, sol, prm, system.context()));

  if (opts.write_output)
  {
    const std::string base = helper.outputBase(outputStem(opts));
    prob.writeSolution(sol, base);
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
