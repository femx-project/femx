#include <petscksp.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonOpt.hpp"
#include <femx/common/Workspace.hpp>
#include <femx/linalg/native/DenseAssemblyMatrix.hpp>
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScAssemblyMatrix.hpp>
#include <femx/runtime/PETScRuntime.hpp>
#include <femx/state/Linearization.hpp>

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson_opt;
using namespace femx::linalg;
using namespace femx::state;
using namespace femx::runtime;

#ifndef FEMX_POISSON_OPT_APP_NAME
#define FEMX_POISSON_OPT_APP_NAME "poisson-opt-petsc"
#endif

namespace
{

void setLinearSolverOptions(KspLinearSolver& solver)
{
  auto& opts        = solver.opts();
  opts.type         = KSPPREONLY;
  opts.pc_type      = PCLU;
  opts.rtol         = 1.0e-12;
  opts.max_its      = 5000;
  opts.use_opts_db  = true;
  opts.check_finite = true;
}

int run(const Options& opts)
{
  if (opts.backend != WorkspaceType::Cpu)
  {
    throw std::runtime_error("PETSc Poisson optimization backend supports only 'cpu'");
  }

  ExampleHelper     helper("petsc", opts.backend, outputDir());
  PoissonOptProblem problem(opts);

  // Residual Jacobian with respect to the state u and the control m.
  PETScAssemblyMatrix dRdu(PETSC_COMM_SELF);
  DenseAssemblyMatrix dRdm;
  
  MatrixLinearization lin(dRdu, dRdm);

  // Linear solvers for forward/adjoint systems.
  KspLinearSolver fwd_lin_solver(PETSC_COMM_SELF);
  KspLinearSolver adj_lin_solver(PETSC_COMM_SELF);
  setLinearSolverOptions(fwd_lin_solver);
  setLinearSolverOptions(adj_lin_solver);

  const Result result = solve(
      problem, lin, fwd_lin_solver, adj_lin_solver);

  printReport(std::cout,
              helper.backendName(),
              problem,
              result.report,
              result.tao_itr,
              result.tao_reason);

  if (opts.write_output)
  {
    const std::string output_base = helper.outputBase(outputStem(opts));
    problem.writeSolution(result.prm, result.state, output_base);
    helper.printVisualizationPath(output_base);
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
      if (hasPoissonOptHelp(argc, argv))
      {
        printPoissonOptUsage(std::cout, FEMX_POISSON_OPT_APP_NAME, true);
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
