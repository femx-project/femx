#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/MapCsrMatrix.hpp>

namespace femx::tests::solver
{

inline assembly::HostAssemblyMap makeDense3Map()
{
  const Array<Array<Index>> dofs{{0, 1, 2}};
  return assembly::makeAssemblyMap(3, 3, dofs, dofs);
}

inline Index gridNode(Index ix, Index iy, Index nx)
{
  return iy * nx + ix;
}

inline assembly::HostAssemblyMap makeGrid5PointMap(Index nx, Index ny)
{
  const Index         num_horizontal_edges = (nx - 1) * ny;
  const Index         num_vertical_edges   = nx * (ny - 1);
  Array<Array<Index>> dofs;
  dofs.reserve(num_horizontal_edges + num_vertical_edges);

  for (Index iy = 0; iy < ny; ++iy)
  {
    for (Index ix = 0; ix + 1 < nx; ++ix)
    {
      dofs.push_back({gridNode(ix, iy, nx), gridNode(ix + 1, iy, nx)});
    }
  }
  for (Index iy = 0; iy + 1 < ny; ++iy)
  {
    for (Index ix = 0; ix < nx; ++ix)
    {
      dofs.push_back({gridNode(ix, iy, nx), gridNode(ix, iy + 1, nx)});
    }
  }

  return assembly::makeAssemblyMap(nx * ny, nx * ny, dofs, dofs);
}

template <typename MatrixOperator>
void fillTestMat(MatrixOperator& mat)
{
  mat.set(0, 0, 4.0);
  mat.set(0, 1, 1.0);
  mat.set(0, 2, -1.0);

  mat.set(1, 0, 2.0);
  mat.set(1, 1, 5.0);
  mat.set(1, 2, 1.0);

  mat.set(2, 0, 1.0);
  mat.set(2, 1, -2.0);
  mat.set(2, 2, 3.0);

  mat.finalize();
}

template <typename MatrixOperator>
void fillGrid5PointMat(MatrixOperator& mat, Index nx, Index ny)
{
  for (Index iy = 0; iy < ny; ++iy)
  {
    for (Index ix = 0; ix < nx; ++ix)
    {
      const Index row = gridNode(ix, iy, nx);
      mat.set(row, row, 4.0);

      if (ix + 1 < nx)
      {
        const Index col = gridNode(ix + 1, iy, nx);
        mat.set(row, col, -1.0);
        mat.set(col, row, -1.0);
      }
      if (iy + 1 < ny)
      {
        const Index col = gridNode(ix, iy + 1, nx);
        mat.set(row, col, -1.0);
        mat.set(col, row, -1.0);
      }
    }
  }

  mat.finalize();
}

inline HostVector expectedSolution()
{
  return {1.0, -2.0, 0.5};
}

inline HostVector expectedGridSolution(Index nx, Index ny)
{
  HostVector x(nx * ny);
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

inline HostVector forwardRhs()
{
  return {1.5, -7.5, 6.5};
}

inline HostVector trRhs()
{
  return {0.5, -10.0, -1.5};
}

inline bool near(Real actual, Real expected, Real tol)
{
  return std::abs(actual - expected) <= tol * (1.0 + std::abs(expected));
}

inline bool vecNear(const HostVector& actual,
                    const HostVector& expected,
                    Real              tol)
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
    const HostVector&             expected,
    Real                          tol = 1.0e-8)
{
  TestStatus status(name);

  try
  {
    HostVector rhs;
    op.apply(expected, rhs);

    HostVector x;
    solver.solve(op, rhs, x);
    status *= vecNear(x, expected, tol);

    HostVector rhs_t;
    op.applyT(expected, rhs_t);

    HostVector xt;
    solver.solveT(op, rhs_t, xt);
    status *= vecNear(xt, expected, tol);
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }

  return status.report();
}

inline TestOutcome solvesForwardAndTranspose(const char*                   name,
                                             linalg::LinearSolver&         solver,
                                             const linalg::LinearOperator& op,
                                             Real                          tol = 1.0e-8)
{
  return solvesForwardAndTranspose(
      name, solver, op, expectedSolution(), tol);
}

} // namespace femx::tests::solver
