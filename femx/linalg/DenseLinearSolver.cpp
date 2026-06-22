#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/linalg/DenseLinearSolver.hpp>

using namespace std;

namespace femx
{
namespace linalg
{

DenseLinearSolver::DenseLinearSolver(Real pivot_tolerance)
  : pivot_tolerance_(pivot_tolerance)
{
}

void DenseLinearSolver::solve(const LinearOperator& op,
                              const Vector<Real>&   rhs,
                              Vector<Real>&         out)
{
  if (op.numRows() != op.numCols() || rhs.size() != op.numRows())
  {
    throw runtime_error(
        "DenseLinearSolver received inconsistent dimensions");
  }

  Vector<Real> mat;
  sample(op, false, mat);
  solveDense(mat, rhs, out, op.numCols());
}

void DenseLinearSolver::solveT(const LinearOperator& op,
                               const Vector<Real>&   rhs,
                               Vector<Real>&         out)
{
  if (op.numRows() != op.numCols() || rhs.size() != op.numCols())
  {
    throw runtime_error(
        "DenseLinearSolver received inconsistent transpose dimensions");
  }

  Vector<Real> mat;
  sample(op, true, mat);
  solveDense(mat, rhs, out, op.numRows());
}

void DenseLinearSolver::sample(const LinearOperator& op,
                               bool                  transpose,
                               Vector<Real>&         mat) const
{
  const Index n = transpose ? op.numRows() : op.numCols();
  mat.assign(n * n, 0.0);

  Vector<Real> basis(n);
  Vector<Real> column;
  for (Index j = 0; j < n; ++j)
  {
    basis.setZero();
    basis[j] = 1.0;

    if (transpose)
    {
      op.applyT(basis, column);
    }
    else
    {
      op.apply(basis, column);
    }

    if (column.size() != n)
    {
      throw runtime_error(
          "DenseLinearSolver sampled operator with inconsistent size");
    }

    for (Index i = 0; i < n; ++i)
    {
      mat[entry(i, j, n)] = column[i];
    }
  }
}

void DenseLinearSolver::solveDense(Vector<Real>        mat,
                                   const Vector<Real>& rhs,
                                   Vector<Real>&       out,
                                   Index               size) const
{
  Vector<Real> b(size, 0.0);

  for (Index i = 0; i < size; ++i)
  {
    b[i] = rhs[i];
  }

  for (Index k = 0; k < size; ++k)
  {
    Index pivot = k;
    Real  best  = abs(mat[entry(k, k, size)]);

    for (Index i = k + 1; i < size; ++i)
    {
      const Real candidate = abs(mat[entry(i, k, size)]);

      if (candidate > best)
      {
        best  = candidate;
        pivot = i;
      }
    }

    if (best <= pivot_tolerance_)
    {
      throw runtime_error("DenseLinearSolver detected singular matrix");
    }

    if (pivot != k)
    {
      for (Index j = k; j < size; ++j)
      {
        swap(mat[entry(k, j, size)], mat[entry(pivot, j, size)]);
      }
      swap(b[k], b[pivot]);
    }

    for (Index i = k + 1; i < size; ++i)
    {
      const Real factor      = mat[entry(i, k, size)] / mat[entry(k, k, size)];
      mat[entry(i, k, size)] = 0.0;

      for (Index j = k + 1; j < size; ++j)
      {
        mat[entry(i, j, size)] -= factor * mat[entry(k, j, size)];
      }
      b[i] -= factor * b[k];
    }
  }

  resizeOrZero(out, size);
  for (Index i = size; i-- > 0;)
  {
    Real sum = b[i];

    for (Index j = i + 1; j < size; ++j)
    {
      sum -= mat[entry(i, j, size)] * out[j];
    }
    out[i] = sum / mat[entry(i, i, size)];
  }
}

Index DenseLinearSolver::entry(Index row, Index col, Index size)
{
  return row * size + col;
}

} // namespace linalg
} // namespace femx
