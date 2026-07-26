#include <petscksp.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonOptProblem.hpp"
#include "PoissonOptResidual.hpp"
#include "PoissonOptSolve.hpp"
#include <femx/linalg/petsc/PETScLinearSystem.hpp>
#include <femx/runtime/PETScRuntime.hpp>
#include <femx/state/StateSolver.hpp>

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson_opt;

#ifndef FEMX_POISSON_OPT_APP_NAME
#define FEMX_POISSON_OPT_APP_NAME "poisson-opt-petsc"
#endif

namespace
{

int run(const Options& opts)
{
  if (opts.memspace != MemorySpace::Host)
  {
    throw std::runtime_error(
        "PETSc Poisson optimization supports only the CPU backend");
  }

  ExampleHelper helper(runtime::SolverType::PETSc,
                       opts.memspace,
                       outputDir());

  PoissonOptProblem problem(opts);

  linalg::PETScLinearSystem fwd_system(PETSC_COMM_WORLD);
  linalg::PETScLinearSystem adj_system(PETSC_COMM_WORLD);

  HostPoissonOptResidual       res(problem);
  state::HostLinearStateSolver state_solver(res, fwd_system);

  const Result result = optimize(problem, state_solver, adj_system, PETSC_COMM_WORLD);

  if (runtime::isRoot())
  {
    printReport(std::cout,
                helper.name(),
                problem,
                result.report,
                result.iterations,
                result.reason);

    if (opts.write_output)
    {
      const std::string base = helper.outputBase(outputStem(opts));
      problem.writeSolution(result.control, result.state, base);
      helper.printVisualizationPath(base);
      helper.printVisualizationPath(base + ".observations");
    }
  }

  return result.converged ? 0 : 1;
}

} // namespace

int main(int argc, char* argv[])
{
  int status = 0;
  try
  {
    runtime::PetscSession petsc(argc, argv);
    runtime::setSerialOpenMp();

    try
    {
      if (examples::hasHelp(argc, argv))
      {
        if (runtime::isRoot())
        {
          printUsage(
              std::cout, FEMX_POISSON_OPT_APP_NAME, true);
        }
      }
      else
      {
        status = run(parseOptions(argc, argv, true));
      }
    }
    catch (const std::exception& error)
    {
      if (runtime::isRoot())
      {
        examples::reportError(
            FEMX_POISSON_OPT_APP_NAME, error);
      }
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
