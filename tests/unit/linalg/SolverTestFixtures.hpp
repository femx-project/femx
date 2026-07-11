#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "TestHelper.hpp"

#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/CsrAssemblyMatrix.hpp>

namespace femx::tests::solver
{

inline CsrPattern makeDense3Pattern()
{
  return CsrPattern(3,
                    3,
                    1,
                    [](Index, Vector<Index>& dofs)
                    {
                      dofs = {0, 1, 2};
                    });
}

inline Index gridNode(Index ix, Index iy, Index nx)
{
  return iy * nx + ix;
}

inline CsrPattern makeGrid5PointPattern(Index nx, Index ny)
{
  const Index num_horizontal_edges = (nx - 1) * ny;
  const Index num_vertical_edges   = nx * (ny - 1);

  return CsrPattern(
      nx * ny,
      nx * ny,
      num_horizontal_edges + num_vertical_edges,
      [nx, num_horizontal_edges](Index ie, Vector<Index>& dofs)
      {
        if (ie < num_horizontal_edges)
        {
          const Index iy = ie / (nx - 1);
          const Index ix = ie % (nx - 1);
          dofs           = {gridNode(ix, iy, nx),
                            gridNode(ix + 1, iy, nx)};
          return;
        }

        const Index edge = ie - num_horizontal_edges;
        const Index iy   = edge / nx;
        const Index ix   = edge % nx;
        dofs             = {gridNode(ix, iy, nx),
                            gridNode(ix, iy + 1, nx)};
      });
}

template <typename AssemblyMatrix>
void fillTestMatrix(AssemblyMatrix& matrix)
{
  matrix.set(0, 0, 4.0);
  matrix.set(0, 1, 1.0);
  matrix.set(0, 2, -1.0);

  matrix.set(1, 0, 2.0);
  matrix.set(1, 1, 5.0);
  matrix.set(1, 2, 1.0);

  matrix.set(2, 0, 1.0);
  matrix.set(2, 1, -2.0);
  matrix.set(2, 2, 3.0);

  matrix.finalize();
}

template <typename AssemblyMatrix>
void fillGrid5PointMatrix(AssemblyMatrix& matrix, Index nx, Index ny)
{
  for (Index iy = 0; iy < ny; ++iy)
  {
    for (Index ix = 0; ix < nx; ++ix)
    {
      const Index row = gridNode(ix, iy, nx);
      matrix.set(row, row, 4.0);

      if (ix + 1 < nx)
      {
        const Index col = gridNode(ix + 1, iy, nx);
        matrix.set(row, col, -1.0);
        matrix.set(col, row, -1.0);
      }
      if (iy + 1 < ny)
      {
        const Index col = gridNode(ix, iy + 1, nx);
        matrix.set(row, col, -1.0);
        matrix.set(col, row, -1.0);
      }
    }
  }

  matrix.finalize();
}

inline Vector<Real> expectedSolution()
{
  return {1.0, -2.0, 0.5};
}

inline Vector<Real> expectedGridSolution(Index nx, Index ny)
{
  Vector<Real> x(nx * ny);
  for (Index iy = 0; iy < ny; ++iy)
  {
    for (Index ix = 0; ix < nx; ++ix)
    {
      x[gridNode(ix, iy, nx)] =
          0.5
          + 0.01 * static_cast<Real>(ix)
          - 0.02 * static_cast<Real>(iy)
          + 0.001 * static_cast<Real>(ix * iy);
    }
  }
  return x;
}

inline Vector<Real> forwardRhs()
{
  return {1.5, -7.5, 6.5};
}

inline Vector<Real> transposeRhs()
{
  return {0.5, -10.0, -1.5};
}

inline bool near(Real actual, Real expected, Real tol)
{
  return std::abs(actual - expected) <= tol * (1.0 + std::abs(expected));
}

inline bool vectorNear(const Vector<Real>& actual,
                       const Vector<Real>& expected,
                       Real                tol)
{
  if (actual.size() != expected.size())
  {
    return false;
  }

  for (Index i = 0; i < actual.size(); ++i)
  {
    if (!near(actual[i], expected[i], tol))
    {
      std::cout << "    mismatch at " << i << ": got " << actual[i]
                << ", expected " << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

inline TestOutcome solvesForwardAndTranspose(
    const char*                   name,
    linalg::LinearSolver&         solver,
    const linalg::LinearOperator& op,
    const Vector<Real>&           expected,
    Real                          tol = 1.0e-8)
{
  TestStatus status(name);

  try
  {
    Vector<Real> rhs;
    op.apply(expected, rhs);

    Vector<Real> x;
    solver.solve(op, rhs, x);
    status *= vectorNear(x, expected, tol);

    Vector<Real> rhs_t;
    op.applyT(expected, rhs_t);

    Vector<Real> xt;
    solver.solveT(op, rhs_t, xt);
    status *= vectorNear(xt, expected, tol);
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }

  return status.report();
}

inline TestOutcome solvesForwardAndTranspose(const char*           name,
                                             linalg::LinearSolver& solver,
                                             const linalg::LinearOperator& op,
                                             Real                  tol = 1.0e-8)
{
  return solvesForwardAndTranspose(
      name, solver, op, expectedSolution(), tol);
}

} // namespace femx::tests::solver
