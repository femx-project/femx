#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/algebra/DenseLinearSolver.hpp>

namespace femx
{
namespace algebra
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
    throw std::runtime_error(
        "DenseLinearSolver received inconsistent dimensions");
  }

  std::vector<Real> mat;
  sample(op, false, mat);
  solveDense(mat, rhs, out, op.numCols());
}

void DenseLinearSolver::solveT(const LinearOperator& op,
                               const Vector<Real>&   rhs,
                               Vector<Real>&         out)
{
  if (op.numRows() != op.numCols() || rhs.size() != op.numCols())
  {
    throw std::runtime_error(
        "DenseLinearSolver received inconsistent transpose dimensions");
  }

  std::vector<Real> mat;
  sample(op, true, mat);
  solveDense(mat, rhs, out, op.numRows());
}

void DenseLinearSolver::sample(const LinearOperator& op,
                               bool                  transpose,
                               std::vector<Real>&    mat) const
{
  const Index n = transpose ? op.numRows() : op.numCols();
  mat.assign(static_cast<std::size_t>(n * n), 0.0);

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
      throw std::runtime_error(
          "DenseLinearSolver sampled operator with inconsistent size");
    }

    for (Index i = 0; i < n; ++i)
    {
      mat[entry(i, j, n)] = column[i];
    }
  }
}

void DenseLinearSolver::solveDense(std::vector<Real>   mat,
                                   const Vector<Real>& rhs,
                                   Vector<Real>&       out,
                                   Index               size) const
{
  std::vector<Real> b(static_cast<std::size_t>(size), 0.0);

  for (Index i = 0; i < size; ++i)
  {
    b[static_cast<std::size_t>(i)] = rhs[i];
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
      std::swap(b[static_cast<std::size_t>(k)],
                b[static_cast<std::size_t>(pivot)]);
    }

    for (Index i = k + 1; i < size; ++i)
    {
      const Real factor      = mat[entry(i, k, size)] / mat[entry(k, k, size)];
      mat[entry(i, k, size)] = 0.0;

      for (Index j = k + 1; j < size; ++j)
      {
        mat[entry(i, j, size)] -= factor * mat[entry(k, j, size)];
      }
      b[static_cast<std::size_t>(i)] -= factor * b[static_cast<std::size_t>(k)];
    }
  }

  resize(out, size);
  for (Index i = size; i-- > 0;)
  {
    Real sum = b[static_cast<std::size_t>(i)];

    for (Index j = i + 1; j < size; ++j)
    {
      sum -= mat[entry(i, j, size)] * out[j];
    }
    out[i] = sum / mat[entry(i, i, size)];
  }
}

std::size_t DenseLinearSolver::entry(Index row,
                                     Index col,
                                     Index size)
{
  return static_cast<std::size_t>(row * size + col);
}

void DenseLinearSolver::resize(Vector<Real>& out,
                               Index         size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

} // namespace algebra
} // namespace femx
