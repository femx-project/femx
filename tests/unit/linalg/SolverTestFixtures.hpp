#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>

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

inline void setEntry(HostCsrMatrix& mat,
                     Index          row,
                     Index          col,
                     Real           val)
{
  for (Index k = mat.rowPtrData()[row]; k < mat.rowPtrData()[row + 1]; ++k)
  {
    if (mat.colIndData()[k] == col)
    {
      mat.valsData()[k] = val;
      return;
    }
  }
  throw std::runtime_error("Test entry is outside the CSR pattern");
}

template <class Matrix>
void setEntry(Matrix& mat, Index row, Index col, Real val)
{
  mat.set(row, col, val);
}

inline void finalize(HostCsrMatrix&)
{
}

template <class Matrix>
void finalize(Matrix& mat)
{
  mat.finalize();
}

template <class Matrix>
void fillTestMat(Matrix& mat)
{
  setEntry(mat, 0, 0, 4.0);
  setEntry(mat, 0, 1, 1.0);
  setEntry(mat, 0, 2, -1.0);

  setEntry(mat, 1, 0, 2.0);
  setEntry(mat, 1, 1, 5.0);
  setEntry(mat, 1, 2, 1.0);

  setEntry(mat, 2, 0, 1.0);
  setEntry(mat, 2, 1, -2.0);
  setEntry(mat, 2, 2, 3.0);

  finalize(mat);
}

inline void fillGrid5PointMat(HostCsrMatrix& mat, Index nx, Index ny)
{
  for (Index iy = 0; iy < ny; ++iy)
  {
    for (Index ix = 0; ix < nx; ++ix)
    {
      const Index row = gridNode(ix, iy, nx);
      setEntry(mat, row, row, 4.0);

      if (ix + 1 < nx)
      {
        const Index col = gridNode(ix + 1, iy, nx);
        setEntry(mat, row, col, -1.0);
        setEntry(mat, col, row, -1.0);
      }
      if (iy + 1 < ny)
      {
        const Index col = gridNode(ix, iy + 1, nx);
        setEntry(mat, row, col, -1.0);
        setEntry(mat, col, row, -1.0);
      }
    }
  }
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
    const char*                  name,
    linalg::HostCsrLinearSolver& solver,
    const HostCsrMatrix&         mat,
    const HostVector&            expected,
    Real                         tol = 1.0e-8)
{
  TestStatus status(name);

  try
  {
    CpuContext                ctx;
    linalg::HostMatrixHandler mat_handler(ctx);
    HostVector                rhs(mat.rows());
    mat_handler.matvec(mat, expected.view(), rhs.view());

    HostVector x;
    solver.solve(mat, rhs, x, ctx);
    status *= vecNear(x, expected, tol);

    HostVector rhs_t(mat.cols());
    mat_handler.matvecT(mat, expected.view(), rhs_t.view());

    HostVector xt;
    solver.solveT(mat, rhs_t, xt, ctx);
    status *= vecNear(xt, expected, tol);
  }
  catch (const std::exception& e)
  {
    std::cout << "    exception: " << e.what() << '\n';
    status *= false;
  }

  return status.report();
}

inline TestOutcome solvesForwardAndTranspose(
    const char*                  name,
    linalg::HostCsrLinearSolver& solver,
    const HostCsrMatrix&         mat,
    Real                         tol = 1.0e-8)
{
  return solvesForwardAndTranspose(
      name, solver, mat, expectedSolution(), tol);
}

} // namespace femx::tests::solver
