#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/DenseLinearSolver.hpp>

namespace femx
{
namespace linalg
{

DenseLinearSolver::DenseLinearSolver(Real pivot_tolerance)
  : pivot_tolerance_(pivot_tolerance)
{
}

void DenseLinearSolver::solve(const LinearOperator& op,
                              const HostVector&     rhs,
                              HostVector&           out)
{
  if (op.numRows() != op.numCols() || rhs.size() != op.numRows())
  {
    throw std::runtime_error(
        "DenseLinearSolver received inconsistent dimensions");
  }

  HostVector mat;
  sample(op, false, mat);
  solveDense(mat, rhs, out, op.numCols());
}

void DenseLinearSolver::solveT(const LinearOperator& op,
                               const HostVector&     rhs,
                               HostVector&           out)
{
  if (op.numRows() != op.numCols() || rhs.size() != op.numCols())
  {
    throw std::runtime_error(
        "DenseLinearSolver received inconsistent transpose dimensions");
  }

  HostVector mat;
  sample(op, true, mat);
  solveDense(mat, rhs, out, op.numRows());
}

void DenseLinearSolver::sample(const LinearOperator& op,
                               bool                  tr,
                               HostVector&           mat) const
{
  const Index size = tr ? op.numRows() : op.numCols();
  mat.assign(size * size, 0.0);

  HostVector basis(size);
  HostVector col;
  for (Index j = 0; j < size; ++j)
  {
    basis.setZero();
    basis[j] = 1.0;

    if (tr)
    {
      op.applyT(basis, col);
    }
    else
    {
      op.apply(basis, col);
    }

    if (col.size() != size)
    {
      throw std::runtime_error(
          "DenseLinearSolver sampled operator with inconsistent size");
    }

    for (Index i = 0; i < size; ++i)
    {
      mat[entry(i, j, size)] = col[i];
    }
  }
}

void DenseLinearSolver::solveDense(HostVector        mat,
                                   const HostVector& rhs,
                                   HostVector&       out,
                                   Index             size) const
{
  HostVector b(size, 0.0);

  for (Index i = 0; i < size; ++i)
  {
    b[i] = rhs[i];
  }

  for (Index k = 0; k < size; ++k)
  {
    Index pivot = k;
    Real  best  = std::abs(mat[entry(k, k, size)]);

    for (Index i = k + 1; i < size; ++i)
    {
      const Real candidate = std::abs(mat[entry(i, k, size)]);

      if (candidate > best)
      {
        best  = candidate;
        pivot = i;
      }
    }

    if (best <= pivot_tolerance_)
    {
      throw std::runtime_error("DenseLinearSolver detected singular matrix");
    }

    if (pivot != k)
    {
      for (Index j = k; j < size; ++j)
      {
        std::swap(mat[entry(k, j, size)], mat[entry(pivot, j, size)]);
      }
      std::swap(b[k], b[pivot]);
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
