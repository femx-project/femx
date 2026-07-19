#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonOpt.hpp"
#include <femx/linalg/Backend.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/runtime/PETScRuntime.hpp>

using namespace femx;
using namespace femx::assembly;
using namespace femx::examples;
using namespace femx::examples::poisson_opt;
using namespace femx::linalg;
using namespace femx::runtime;

#ifndef FEMX_POISSON_OPT_APP_NAME
#define FEMX_POISSON_OPT_APP_NAME "poisson-opt-resolve"
#endif

namespace
{

int run(const Options& opts)
{
  ExampleHelper     helper("resolve", MemorySpace::Host, outputDir());
  PoissonOptProblem problem(opts);

  HostCsrMatrix       fwd_jac(problem.stateMap().graph());
  HostCsrMatrix       adj_jac(problem.stateMap().graph());
  ReSolveLinearSolver fwd_lin_solver;
  ReSolveLinearSolver adj_lin_solver;
  CpuContext          ctx;

  const Result result = solve<HostCsrBackend>(problem,
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
    // TAO is provided by PETSc, even though the linear solves use ReSolve.
    PetscSession petsc(argc, argv);
    setSerialOpenMp();

    try
    {
      if (examples::hasHelp(argc, argv))
      {
        printUsage(std::cout, FEMX_POISSON_OPT_APP_NAME, false);
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
