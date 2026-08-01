#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "../ExampleHelper.hpp"
#include "PoissonProblem.hpp"
#include "PoissonResidual.hpp"
#include <femx/linalg/host/HostLinearSystem.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/state/StateSolver.hpp>

#if defined(FEMX_RESOLVE_USE_CUDA)
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaLinearSystem.hpp>
#endif

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson;
using namespace femx::linalg;

#ifndef FEMX_POISSON_APP_NAME
#define FEMX_POISSON_APP_NAME "poisson-resolve"
#endif

namespace
{

Real solveHost(const ExampleHelper&  helper,
               const PoissonProblem& problem,
               HostVector<Real>&     h_x)
{
  // Build a ReSolve-backed Host linear system.
  auto             solver = std::make_unique<ReSolveLinearSolver>();
  HostLinearSystem system(std::move(solver));

  // Bind the Poisson residual and solve the state equation.
  HostPoissonResidual          res(problem);
  state::HostLinearStateSolver state_solver(res, system);

  state_solver.solve(h_x);

  // Evaluate the residual norm of the computed solution.
  const Real rnorm = helper.resNorm(res, h_x, system.context());

  return rnorm;
}

#if defined(FEMX_RESOLVE_USE_CUDA)
Real solveDevice(const ExampleHelper&  helper,
                 const PoissonProblem& problem,
                 HostVector<Real>&     h_x)
{
  // Build a ReSolve-backed CUDA linear system.
  auto             solver = std::make_unique<ReSolveLinearSolver>();
  CudaLinearSystem system(std::move(solver));

  auto& ctx = static_cast<linalg::CudaContext&>(system.context());

  // Copy the problem data to Device and solve the state equation there.
  CudaPoissonResidual            res(problem, ctx);
  state::DeviceLinearStateSolver state_solver(res, system);

  DeviceVector<Real> d_x;
  state_solver.solve(d_x);

  // Evaluate on Device, then return the solution to Host for reporting.
  const Real rnorm = helper.resNorm(res, d_x, ctx);
  ctx.vectorHandler().copy(d_x, h_x);
  ctx.sync();

  return rnorm;
}
#endif

int run(const Options& opts)
{
  // Construct the backend-independent problem and reporting helper.
  constexpr auto solver_type = runtime::SolverType::ReSolve;
  ExampleHelper  helper(solver_type, opts.memspace, outputDir());
  PoissonProblem problem(opts);

  // Solve with the selected backend while keeping the final result on Host.
  HostVector<Real> x;
  Real             rnorm;
  if (opts.memspace == MemorySpace::Host)
  {
    rnorm = solveHost(helper, problem, x);
  }
  else
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    rnorm = solveDevice(helper, problem, x);
#else
    throw std::runtime_error(
        "The CUDA backend requires a CUDA-enabled ReSolve build");
#endif
  }

  // Report the computed solution.
  printReport(std::cout,
              helper.name(),
              problem,
              problem.errorReport(x),
              rnorm);

  // Optional: write visualization output.
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
