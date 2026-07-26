#include "SolverTestFixtures.hpp"
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <resolve/resolve_defs.hpp>

namespace femx::tests
{
namespace
{

TestOutcome resolveOptionsHaveOneSharedDefault()
{
  TestStatus                   status(__func__);
  const linalg::ReSolveOptions opts;

  status *= opts.solve == "fgmres";
  status *= opts.precond == "ilu0";
  status *= opts.max_its == 1000;
  status *= opts.restart == 200;
  status *= opts.rtol == 1.0e-8;
  return status.report();
}

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

  const auto    map = solver::makeGrid5PointMap(nx, ny);
  HostCsrMatrix mat(map.pattern());
  solver::fillGrid5PointMat(mat, nx, ny);

  linalg::ReSolveLinearSolver lin_solver;
  return solver::solvesForwardAndTranspose(
      __func__, lin_solver, mat, solver::expectedGridSolution(nx, ny), 1.0e-7);
}

#if defined(RESOLVE_USE_KLU)
TestOutcome resolveCpuKluSolvesForwardAndTranspose()
{
  const auto    map = solver::makeDense3Map();
  HostCsrMatrix mat(map.pattern());
  solver::fillTestMat(mat);

  linalg::ReSolveLinearSolver lin_solver(kluOptions());
  return solver::solvesForwardAndTranspose(__func__, lin_solver, mat);
}
#endif

TestOutcome resolveCpuConcreteMatrixReusesStorage()
{
  TestStatus status(__func__);

  try
  {
    constexpr Index nx = 16;
    constexpr Index ny = 16;

    const auto    map = solver::makeGrid5PointMap(nx, ny);
    HostCsrMatrix mat(map.pattern());
    solver::fillGrid5PointMat(mat, nx, ny);

    linalg::ReSolveLinearSolver lin_solver;
    linalg::HostContext         ctx;
    linalg::HostJacobian        jacobian(ctx);

    const HostVector<Real> expected = solver::expectedGridSolution(nx, ny);
    HostVector<Real>       rhs(expected.size());
    jacobian.apply(mat, expected.view(), rhs.view());

    HostVector<Real> x;
    lin_solver.solve(mat, rhs, x, ctx);
    status *= solver::vecNear(x, expected, 1.0e-7);

    ctx.vectors().zero(mat.vals().view());
    solver::fillGrid5PointMat(mat, nx, ny);
    jacobian.apply(mat, expected.view(), rhs.view());
    lin_solver.solve(mat, rhs, x, ctx);
    status *= solver::vecNear(x, expected, 1.0e-7);
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }

  return status.report();
}

TestOutcome resolveZeroRhsReturnsZero()
{
  TestStatus status(__func__);

  try
  {
    const auto    map = solver::makeDense3Map();
    HostCsrMatrix mat(map.pattern());
    solver::fillTestMat(mat);
    const HostVector<Real> rhs(3, 0.0);
    HostVector<Real>       sol{1.0, 2.0, 3.0};
    linalg::HostContext    ctx;

    linalg::ReSolveLinearSolver h_solver;
    h_solver.solve(mat, rhs, sol, ctx);
    status *= solver::vecNear(sol, rhs, 0.0);
    sol     = {1.0, 2.0, 3.0};
    h_solver.solveT(mat, rhs, sol, ctx);
    status *= solver::vecNear(sol, rhs, 0.0);
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }

  return status.report();
}

} // namespace
} // namespace femx::tests

int main(int, char**)
{
  femx::tests::TestingResults results;

  results += femx::tests::resolveOptionsHaveOneSharedDefault();
  results += femx::tests::resolveCpuDefaultSolvesForwardAndTranspose();
#if defined(RESOLVE_USE_KLU)
  results += femx::tests::resolveCpuKluSolvesForwardAndTranspose();
#endif
  results += femx::tests::resolveCpuConcreteMatrixReusesStorage();
  results += femx::tests::resolveZeroRhsReturnsZero();
  return results.summary();
}
