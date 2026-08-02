#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonProblem.hpp"
#include "PoissonResidual.hpp"
#include <femx/linalg/host/HostLinearSystem.hpp>
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
  // Use the built-in dense linear solver and require Host execution.
  constexpr auto solver_type = runtime::SolverType::Dense;
  if (opts.memspace != MemorySpace::Host)
  {
    throw std::runtime_error(
        "Dense Poisson supports only the CPU backend");
  }

  ExampleHelper helper(solver_type, opts.memspace, outputDir());

  // The problem owns the FEM data needed to assemble the system.
  PoissonProblem problem(opts);

  // Create a native dense linear system in host memory. It owns
  // A, b, and the solver that provides the backend-specific interface.
  linalg::HostLinearSystem system;

  // Define the discrete Poisson residual R(x) = A x - b. The state solver
  // uses this residual and the dense linear system to find R(x) = 0.
  HostPoissonResidual          res(problem);
  state::HostLinearStateSolver state_solver(res, system);

  // Assemble R(0) = -b and J = dR/dx = A, then solve J x = -R(0),
  // which corresponds to the original linear system A x = b.
  HostVector<Real> result;
  state_solver.solve(result);

  // Report the computed solution.
  printReport(std::cout,
              helper.name(),
              problem,
              problem.errorReport(result),
              helper.resNorm(res, result, system.context()));

  // Optional: write visualization output.
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
