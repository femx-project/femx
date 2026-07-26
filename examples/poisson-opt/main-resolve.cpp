#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "../ExampleHelper.hpp"
#include "PoissonOptProblem.hpp"
#include "PoissonOptResidual.hpp"
#include "PoissonOptSolve.hpp"
#include <femx/linalg/native/HostLinearSystem.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/runtime/PETScRuntime.hpp>
#include <femx/state/StateSolver.hpp>

#if defined(FEMX_RESOLVE_USE_CUDA)
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaLinearSystem.hpp>
#endif

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson_opt;

#ifndef FEMX_POISSON_OPT_APP_NAME
#define FEMX_POISSON_OPT_APP_NAME "poisson-opt-resolve"
#endif

namespace
{

Result optimizeHost(PoissonOptProblem& prob)
{
  auto fwd_solver = std::make_unique<linalg::ReSolveLinearSolver>();
  auto adj_solver = std::make_unique<linalg::ReSolveLinearSolver>();

  linalg::HostLinearSystem fwd_system(std::move(fwd_solver));
  linalg::HostLinearSystem adj_system(std::move(adj_solver));

  HostPoissonOptResidual       res(prob);
  state::HostLinearStateSolver state_solver(res, fwd_system);

  return optimize(prob, state_solver, adj_system, PETSC_COMM_SELF);
}

#if defined(FEMX_RESOLVE_USE_CUDA)

Result optimizeDevice(PoissonOptProblem& prob)
{
  auto fwd_solver = std::make_unique<linalg::ReSolveLinearSolver>();
  auto adj_solver = std::make_unique<linalg::ReSolveLinearSolver>();

  linalg::CudaLinearSystem fwd_system(std::move(fwd_solver));
  linalg::CudaLinearSystem adj_system(std::move(adj_solver));

  auto& ctx = static_cast<linalg::CudaContext&>(fwd_system.context());

  DevicePoissonOptResidual                      res(prob, ctx);
  state::LinearStateSolver<MemorySpace::Device> state_solver(res, fwd_system);

  return optimize(prob, state_solver, adj_system, PETSC_COMM_SELF);
}

#endif

int run(const Options& opts)
{
  ExampleHelper     helper(runtime::SolverType::ReSolve,
                           opts.memspace,
                           outputDir());
  PoissonOptProblem prob(opts);

  Result result;
  if (opts.memspace == MemorySpace::Host)
  {
    result = optimizeHost(prob);
  }
  else
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    result = optimizeDevice(prob);
#else
    throw std::runtime_error(
        "Device Poisson optimization requires a CUDA-enabled "
        "ReSolve build");
#endif
  }

  printReport(std::cout,
              helper.name(),
              prob,
              result.report,
              result.iterations,
              result.reason);

  if (opts.write_output)
  {
    const std::string base = helper.outputBase(outputStem(opts));
    prob.writeSolution(result.control, result.state, base);
    helper.printVisualizationPath(base);
    helper.printVisualizationPath(base + ".observations");
  }

  return result.converged ? 0 : 1;
}

} // namespace

int main(int argc, char* argv[])
{
  int status = 0;
  try
  {
    // TAO supplies the optimizer; ReSolve supplies both linear solves.
    runtime::PetscSession petsc(argc, argv);
    runtime::setSerialOpenMp();

    try
    {
      if (examples::hasHelp(argc, argv))
      {
        printUsage(
            std::cout, FEMX_POISSON_OPT_APP_NAME, true);
      }
      else
      {
        status = run(parseOptions(argc, argv, true));
      }
    }
    catch (const std::exception& error)
    {
      examples::reportError(FEMX_POISSON_OPT_APP_NAME, error);
      status = 1;
    }

    const PetscErrorCode error = petsc.finalize();
    if (error != PETSC_SUCCESS && status == 0)
    {
      status = 1;
    }
  }
  catch (const std::exception& error)
  {
    return examples::reportError(
        FEMX_POISSON_OPT_APP_NAME, error);
  }
  return status;
}
