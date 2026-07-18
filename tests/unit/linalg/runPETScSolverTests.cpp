#include <petscksp.h>

#include "SolverTestFixtures.hpp"
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScOperator.hpp>

namespace femx::tests
{
namespace
{

void configureKsp(linalg::KspLinearSolver& solver)
{
  auto& opts       = solver.opts();
  opts.type        = KSPGMRES;
  opts.pc_type     = PCNONE;
  opts.rtol        = 1.0e-12;
  opts.atol        = 1.0e-14;
  opts.max_its     = 30;
  opts.restart     = 10;
  opts.use_opts_db = false;
}

TestOutcome petscCsrShellSolvesForwardAndTranspose()
{
  auto                 map = solver::makeDense3Map();
  linalg::MapCsrMatrix op(map);
  solver::fillTestMat(op);

  linalg::KspLinearSolver lin_solver(PETSC_COMM_SELF);
  configureKsp(lin_solver);
  return solver::solvesForwardAndTranspose(__func__, lin_solver, op);
}

TestOutcome petscMatrixOperatorSolvesForwardAndTranspose()
{
  linalg::PETScOperator op(PETSC_COMM_SELF);
  op.resize(3, 3);
  solver::fillTestMat(op);

  linalg::KspLinearSolver lin_solver(PETSC_COMM_SELF);
  configureKsp(lin_solver);
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
  results            += femx::tests::petscMatrixOperatorSolvesForwardAndTranspose();
  const int failures  = results.summary();

  PetscFinalize();
  return failures;
}
