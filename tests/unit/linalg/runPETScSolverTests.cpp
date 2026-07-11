#include <petscksp.h>

#include "SolverTestFixtures.hpp"
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScAssemblyMatrix.hpp>

namespace femx::tests
{
namespace
{

void configureKsp(linalg::KspLinearSolver& solver, bool check_finite)
{
  auto& opts        = solver.opts();
  opts.type         = KSPGMRES;
  opts.pc_type      = PCNONE;
  opts.rtol         = 1.0e-12;
  opts.atol         = 1.0e-14;
  opts.max_its      = 30;
  opts.restart      = 10;
  opts.use_opts_db  = false;
  opts.check_finite = check_finite;
}

TestOutcome petscCsrShellSolvesForwardAndTranspose()
{
  CsrPattern                pattern = solver::makeDense3Pattern();
  linalg::CsrAssemblyMatrix op(pattern);
  solver::fillTestMatrix(op);

  linalg::KspLinearSolver lin_solver(PETSC_COMM_SELF);
  configureKsp(lin_solver, false);
  return solver::solvesForwardAndTranspose(__func__, lin_solver, op);
}

TestOutcome petscAssemblyMatrixSolvesForwardAndTranspose()
{
  linalg::PETScAssemblyMatrix op(PETSC_COMM_SELF);
  op.resize(3, 3);
  solver::fillTestMatrix(op);

  linalg::KspLinearSolver lin_solver(PETSC_COMM_SELF);
  configureKsp(lin_solver, true);
  return solver::solvesForwardAndTranspose(__func__, lin_solver, op);
}

} // namespace
} // namespace femx::tests

int main(int argc, char** argv)
{
  if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS)
  {
    return 1;
  }

  femx::tests::TestingResults results;
  results            += femx::tests::petscCsrShellSolvesForwardAndTranspose();
  results            += femx::tests::petscAssemblyMatrixSolvesForwardAndTranspose();
  const int failures  = results.summary();

  PetscFinalize();
  return failures;
}
