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
  // Create a ReSolve linear system in Host memory.
  auto             solver = std::make_unique<ReSolveLinearSolver>();
  HostLinearSystem system(std::move(solver));

  // Define the discrete Poisson residual R(x) = A x - b. The state solver
  // uses this residual and the ReSolve linear system to find R(x) = 0.
  HostPoissonResidual          res(problem);
  state::HostLinearStateSolver state_solver(res, system);

  // Assemble R(0) = -b and J = dR/dx = A, then solve J x = -R(0),
  // which corresponds to the original linear system A x = b.
  state_solver.solve(h_x);

  // Evaluate the residual norm of the computed solution.
  const Real rnorm = helper.resNorm(res, h_x, system.context());

  return rnorm;
}

#if defined(FEMX_RESOLVE_USE_CUDA)
Real solveCuda(const ExampleHelper&  helper,
               const PoissonProblem& problem,
               HostVector<Real>&     h_x)
{
  // Create a ReSolve linear system in Device memory.
  auto             solver = std::make_unique<ReSolveLinearSolver>();
  CudaLinearSystem system(std::move(solver));

  auto& ctx = static_cast<linalg::CudaContext&>(system.context());

  // Copy the finite-element data to Device and define the same discrete
  // Poisson residual R(x) = A x - b there. The Device state solver uses
  // this residual and the ReSolve linear system to find R(x) = 0.
  CudaPoissonResidual            res(problem, ctx);
  state::DeviceLinearStateSolver state_solver(res, system);

  // Assemble R(0) = -b and J = dR/dx = A on Device, then solve
  // J x = -R(0), which corresponds to the original linear system A x = b.
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
  // Use ReSolve as the linear solver backend in the selected memory space.
  constexpr auto solver_type = runtime::SolverType::ReSolve;
  ExampleHelper  helper(solver_type, opts.memspace, outputDir());

  // The problem owns the FEM data needed to assemble the system.
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
    rnorm = solveCuda(helper, problem, x);
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
