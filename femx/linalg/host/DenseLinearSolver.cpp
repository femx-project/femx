#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/host/DenseLinearSolver.hpp>

namespace femx::linalg
{

DenseLinearSolver::DenseLinearSolver(Real pivot_tolerance)
  : pivot_tol_(pivot_tolerance)
{
  require(pivot_tolerance >= 0.0,
          "DenseLinearSolver pivot tolerance must be non-negative");
}

void DenseLinearSolver::solve(const HostCsrMatrix&        mat,
                              const HostVector<Real>&     rhs,
                              HostVector<Real>&           x,
                              Context<MemorySpace::Host>& ctx)
{
  require(mat.rows() == mat.cols() && rhs.size() == mat.rows(),
          "DenseLinearSolver received inconsistent CSR dimensions");
  DenseMatrix dense;
  copyToDense(mat, false, dense);
  solveDense(std::move(dense), rhs, x, ctx);
}

void DenseLinearSolver::solveT(const HostCsrMatrix&        mat,
                               const HostVector<Real>&     rhs,
                               HostVector<Real>&           x,
                               Context<MemorySpace::Host>& ctx)
{
  require(mat.rows() == mat.cols() && rhs.size() == mat.cols(),
          "DenseLinearSolver received inconsistent transposed CSR dimensions");
  DenseMatrix dense;
  copyToDense(mat, true, dense);
  solveDense(std::move(dense), rhs, x, ctx);
}

void DenseLinearSolver::copyToDense(const HostCsrMatrix& mat,
                                    bool                 transpose,
                                    DenseMatrix&         dense) const
{
  dense.resize(mat.rows(), mat.cols());
  for (Index i = 0; i < mat.rows(); ++i)
  {
    for (Index k = mat.rowPtrData()[i]; k < mat.rowPtrData()[i + 1]; ++k)
    {
      const Index col = mat.colIndData()[k];
      if (transpose)
      {
        dense(col, i) = mat.valsData()[k];
      }
      else
      {
        dense(i, col) = mat.valsData()[k];
      }
    }
  }
}

void DenseLinearSolver::solveDense(DenseMatrix                 mat,
                                   const HostVector<Real>&     rhs,
                                   HostVector<Real>&           x,
                                   Context<MemorySpace::Host>& ctx) const
{
  const Index      size = mat.rows();
  HostVector<Real> b(rhs);

  for (Index k = 0; k < size; ++k)
  {
    Index pivot = k;
    Real  best  = std::abs(mat(k, k));
    for (Index i = k + 1; i < size; ++i)
    {
      const Real candidate = std::abs(mat(i, k));
      if (candidate > best)
      {
        best  = candidate;
        pivot = i;
      }
    }
    if (best <= pivot_tol_)
    {
      throw std::runtime_error(
          "DenseLinearSolver detected singular matrix");
    }
    if (pivot != k)
    {
      for (Index j = k; j < size; ++j)
      {
        std::swap(mat(k, j), mat(pivot, j));
      }
      std::swap(b[k], b[pivot]);
    }
    for (Index i = k + 1; i < size; ++i)
    {
      const Real factor = mat(i, k) / mat(k, k);
      mat(i, k)         = 0.0;
      for (Index j = k + 1; j < size; ++j)
      {
        mat(i, j) -= factor * mat(k, j);
      }
      b[i] -= factor * b[k];
    }
  }

  auto& vec_handler = ctx.vectorHandler();
  vec_handler.assign(x, size, 0);
  for (Index i = size; i-- > 0;)
  {
    Real sum = b[i];
    for (Index j = i + 1; j < size; ++j)
    {
      sum -= mat(i, j) * x[j];
    }
    x[i] = sum / mat(i, i);
  }
}

} // namespace femx::linalg
