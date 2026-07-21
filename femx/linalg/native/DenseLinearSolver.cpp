#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>
#include <femx/linalg/native/DenseLinearSolver.hpp>

namespace femx::linalg
{

DenseLinearSolver::DenseLinearSolver(Real pivot_tolerance)
  : pivot_tolerance_(pivot_tolerance)
{
  require(pivot_tolerance >= 0.0,
          "DenseLinearSolver pivot tolerance must be non-negative");
}

void DenseLinearSolver::solve(const HostCsrMatrix& mat,
                              const HostVector&    rhs,
                              HostVector&          out,
                              CpuContext&          ctx)
{
  require(mat.rows() == mat.cols() && rhs.size() == mat.rows(),
          "DenseLinearSolver received inconsistent CSR dimensions");
  DenseMatrix dense;
  sample(mat, false, dense);
  solveDense(std::move(dense), rhs, out, ctx);
}

void DenseLinearSolver::solveT(const HostCsrMatrix& mat,
                               const HostVector&    rhs,
                               HostVector&          out,
                               CpuContext&          ctx)
{
  require(mat.rows() == mat.cols() && rhs.size() == mat.cols(),
          "DenseLinearSolver received inconsistent transposed CSR dimensions");
  DenseMatrix dense;
  sample(mat, true, dense);
  solveDense(std::move(dense), rhs, out, ctx);
}

void DenseLinearSolver::sample(const HostCsrMatrix& mat,
                               bool                 transpose,
                               DenseMatrix&         dense) const
{
  dense.resize(mat.rows(), mat.cols());
  for (Index row = 0; row < mat.rows(); ++row)
  {
    for (Index k = mat.rowPtrData()[row]; k < mat.rowPtrData()[row + 1]; ++k)
    {
      const Index col = mat.colIndData()[k];
      if (transpose)
      {
        dense(col, row) = mat.valsData()[k];
      }
      else
      {
        dense(row, col) = mat.valsData()[k];
      }
    }
  }
}

void DenseLinearSolver::solveDense(DenseMatrix       mat,
                                   const HostVector& rhs,
                                   HostVector&       out,
                                   CpuContext&       ctx) const
{
  const Index size = mat.rows();
  HostVector  b(rhs);

  for (Index k = 0; k < size; ++k)
  {
    Index pivot = k;
    Real  best  = std::abs(mat(k, k));
    for (Index row = k + 1; row < size; ++row)
    {
      const Real candidate = std::abs(mat(row, k));
      if (candidate > best)
      {
        best  = candidate;
        pivot = row;
      }
    }
    if (best <= pivot_tolerance_)
    {
      throw std::runtime_error(
          "DenseLinearSolver detected singular matrix");
    }
    if (pivot != k)
    {
      for (Index col = k; col < size; ++col)
      {
        std::swap(mat(k, col), mat(pivot, col));
      }
      std::swap(b[k], b[pivot]);
    }
    for (Index row = k + 1; row < size; ++row)
    {
      const Real factor = mat(row, k) / mat(k, k);
      mat(row, k)       = 0.0;
      for (Index col = k + 1; col < size; ++col)
      {
        mat(row, col) -= factor * mat(k, col);
      }
      b[row] -= factor * b[k];
    }
  }

  HostVectorHandler vec_handler(ctx);
  vec_handler.resizeOrZero(out, size);
  for (Index row = size; row-- > 0;)
  {
    Real sum = b[row];
    for (Index col = row + 1; col < size; ++col)
    {
      sum -= mat(row, col) * out[col];
    }
    out[row] = sum / mat(row, row);
  }
}

} // namespace femx::linalg
