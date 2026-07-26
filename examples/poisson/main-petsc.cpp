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
  constexpr auto solver_type = runtime::SolverType::PETSc;
  if (opts.execution_device != runtime::ExecutionDevice::Host)
  {
    throw std::runtime_error(
        "PETSc Poisson supports only Host execution");
  }

  ExampleHelper  helper(solver_type, opts.execution_device, outputDir());
  PoissonProblem prob(opts);

  linalg::PETScLinearSystem system(PETSC_COMM_WORLD);

  HostPoissonResidual          res(prob);
  state::HostLinearStateSolver state_solver(res, system);

  const HostVector<Real> prm;
  HostVector<Real>       x;

  state_solver.solve(prm, x);

  const Real res_norm = helper.resNorm(res, x, prm, system.context());

  if (isRoot())
  {
    printReport(std::cout,
                helper.name(),
                prob,
                prob.errorReport(x),
                res_norm);

    if (opts.write_output)
    {
      const std::string base = helper.outputBase(outputStem(opts));
      prob.writeSolution(x, base);
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
    PetscSession petsc(argc, argv);
    setSerialOpenMp();

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
