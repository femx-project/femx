#include <petscksp.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonProblem.hpp"
#include "PoissonResidual.hpp"
#include <femx/linalg/petsc/PETScLinearSystem.hpp>
#include <femx/runtime/PETScRuntime.hpp>
#include <femx/state/StateSolver.hpp>

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson;
using namespace femx::runtime;

#ifndef FEMX_POISSON_APP_NAME
#define FEMX_POISSON_APP_NAME "poisson-petsc"
#endif

namespace
{

int run(const Options& opts)
{
  // Validate the backend and construct the shared Poisson problem data.
  constexpr auto solver_type = runtime::SolverType::PETSc;
  if (opts.memspace != MemorySpace::Host)
  {
    throw std::runtime_error(
        "PETSc Poisson supports only the CPU backend");
  }

  ExampleHelper  helper(solver_type, opts.memspace, outputDir());
  PoissonProblem problem(opts);

  // Build the PETSc linear system on the global communicator.
  linalg::PETScLinearSystem system(PETSC_COMM_WORLD);

  // Bind the residual to the linear system and solve the state equation.
  HostPoissonResidual          res(problem);
  state::HostLinearStateSolver state_solver(res, system);

  HostVector<Real> x;

  state_solver.solve(x);

  // Evaluate the residual norm of the computed solution.
  const Real rnorm = helper.resNorm(res, x, system.context());

  // Produce output from the root MPI rank.
  if (isRoot())
  {
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
  }

  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  int status = 0;
  try
  {
    // Initialize PETSc/MPI and use one OpenMP thread per MPI rank.
    PetscSession petsc(argc, argv);
    setSerialOpenMp();

    // Parse options and run collectively; only the root rank prints messages.
    try
    {
      if (examples::hasHelp(argc, argv))
      {
        if (isRoot())
        {
          printUsage(FEMX_POISSON_APP_NAME, true);
        }
      }
      else
      {
        status = run(parseOptions(argc, argv, true));
      }
    }
    catch (const std::exception& e)
    {
      if (isRoot())
      {
        examples::reportError(FEMX_POISSON_APP_NAME, e);
      }
      status = 1;
    }

    // Finalize PETSc collectively without hiding an earlier application error.
    const PetscErrorCode ierr = petsc.finalize();
    if (ierr != PETSC_SUCCESS && status == 0)
    {
      return 1;
    }
  }
  catch (const std::exception& e)
  {
    return examples::reportError(FEMX_POISSON_APP_NAME, e);
  }
  return status;
}
