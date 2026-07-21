#include <petscksp.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonOpt.hpp"
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScBackend.hpp>
#include <femx/linalg/petsc/PETScOperator.hpp>
#include <femx/runtime/PETScRuntime.hpp>

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson_opt;
using namespace femx::linalg;
using namespace femx::runtime;

#ifndef FEMX_POISSON_OPT_APP_NAME
#define FEMX_POISSON_OPT_APP_NAME "poisson-opt-petsc"
#endif

namespace
{

int run(const Options& opts)
{
  ExampleHelper     helper("petsc", MemorySpace::Host, outputDir());
  PoissonOptProblem problem(opts);

  PETScOperator fwd_jac(PETSC_COMM_SELF);
  PETScOperator adj_jac(PETSC_COMM_SELF);
  fwd_jac.resize(problem.stateMap().pattern());
  adj_jac.resize(problem.stateMap().pattern());

  KspLinearSolver fwd_lin_solver(PETSC_COMM_SELF);
  KspLinearSolver adj_lin_solver(PETSC_COMM_SELF);
  PetscContext    ctx{PETSC_COMM_SELF};

  const Result result = solve<PetscBackend>(problem,
                                            fwd_jac,
                                            fwd_lin_solver,
                                            adj_jac,
                                            adj_lin_solver,
                                            ctx);

  printReport(std::cout,
              helper.name(),
              problem,
              result.report,
              result.tao_itr,
              result.tao_reason);

  if (opts.write_output)
  {
    const std::string base = helper.outputBase(outputStem(opts));
    problem.writeSolution(result.prm, result.state, base);
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
    PetscSession petsc(argc, argv);
    setSerialOpenMp();

    try
    {
      if (examples::hasHelp(argc, argv))
      {
        printUsage(std::cout, FEMX_POISSON_OPT_APP_NAME, true);
      }
      else
      {
        status = run(parseOptions(argc, argv, true));
      }
    }
    catch (const std::exception& e)
    {
      examples::reportError(FEMX_POISSON_OPT_APP_NAME, e);
      status = 1;
    }

    const PetscErrorCode ierr = petsc.finalize();
    if (ierr != PETSC_SUCCESS && status == 0)
    {
      return 1;
    }
  }
  catch (const std::exception& e)
  {
    return examples::reportError(FEMX_POISSON_OPT_APP_NAME, e);
  }
  return status;
}
