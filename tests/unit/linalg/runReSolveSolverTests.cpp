#include "SolverTestFixtures.hpp"
#include <femx/common/Workspace.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <resolve/resolve_defs.hpp>

namespace femx::tests
{
namespace
{

#if defined(RESOLVE_USE_KLU)
linalg::ReSolveOptions kluOptions()
{
  linalg::ReSolveOptions opts;
  opts.factor   = "klu";
  opts.refactor = "none";
  opts.solve    = "klu";
  opts.precond  = "none";
  opts.ir       = "none";
  return opts;
}
#endif

TestOutcome resolveCpuDefaultSolvesForwardAndTranspose()
{
  constexpr Index nx = 16;
  constexpr Index ny = 16;

  CsrPattern                pattern = solver::makeGrid5PointPattern(nx, ny);
  linalg::CsrAssemblyMatrix op(pattern);
  solver::fillGrid5PointMatrix(op, nx, ny);

  linalg::ReSolveLinearSolver lin_solver(WorkspaceType::Cpu);
  return solver::solvesForwardAndTranspose(
      __func__, lin_solver, op, solver::expectedGridSolution(nx, ny), 1.0e-7);
}

#if defined(RESOLVE_USE_KLU)
TestOutcome resolveCpuKluSolvesForwardAndTranspose()
{
  CsrPattern                pattern = solver::makeDense3Pattern();
  linalg::CsrAssemblyMatrix op(pattern);
  solver::fillTestMatrix(op);

  linalg::ReSolveLinearSolver lin_solver(WorkspaceType::Cpu, kluOptions());
  return solver::solvesForwardAndTranspose(__func__, lin_solver, op);
}
#endif

TestOutcome resolveCpuStoredOperatorSolves()
{
  TestStatus status(__func__);

  try
  {
    constexpr Index nx = 16;
    constexpr Index ny = 16;

    CsrPattern                pattern = solver::makeGrid5PointPattern(nx, ny);
    linalg::CsrAssemblyMatrix op(pattern);
    solver::fillGrid5PointMatrix(op, nx, ny);

    linalg::ReSolveLinearSolver lin_solver(WorkspaceType::Cpu);
    lin_solver.setOperator(op.mat());

    const Vector<Real> expected = solver::expectedGridSolution(nx, ny);
    Vector<Real>       rhs;
    op.apply(expected, rhs);

    Vector<Real> x;
    lin_solver.solve(rhs, x);
    status *= solver::vectorNear(x, expected, 1.0e-7);

    op.setZero();
    solver::fillGrid5PointMatrix(op, nx, ny);
    lin_solver.setOperator(op.mat());

    op.apply(expected, rhs);
    lin_solver.solve(rhs, x);
    status *= solver::vectorNear(x, expected, 1.0e-7);
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }

  return status.report();
}

#if defined(FEMX_RESOLVE_USE_CUDA)
TestOutcome resolveCudaDefaultSolvesForwardAndTranspose()
{
  constexpr Index nx = 16;
  constexpr Index ny = 16;

  CsrPattern                pattern = solver::makeGrid5PointPattern(nx, ny);
  linalg::CsrAssemblyMatrix op(pattern);
  solver::fillGrid5PointMatrix(op, nx, ny);

  linalg::ReSolveLinearSolver lin_solver(WorkspaceType::Cuda);
  return solver::solvesForwardAndTranspose(
      __func__, lin_solver, op, solver::expectedGridSolution(nx, ny), 1.0e-7);
}
#endif

} // namespace
} // namespace femx::tests

int main(int, char**)
{
  femx::tests::TestingResults results;

  results += femx::tests::resolveCpuDefaultSolvesForwardAndTranspose();
#if defined(RESOLVE_USE_KLU)
  results += femx::tests::resolveCpuKluSolvesForwardAndTranspose();
#endif
  results += femx::tests::resolveCpuStoredOperatorSolves();
#if defined(FEMX_RESOLVE_USE_CUDA)
  results += femx::tests::resolveCudaDefaultSolvesForwardAndTranspose();
#endif

  return results.summary();
}
