#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <femx/common/Types.hpp>
#include <femx/system/LinearOperator.hpp>
#include <femx/system/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace system
{

/** @brief Dense fallback solver that samples a LinearOperator into a matrix. */
class DenseLinearSolver final : public LinearSolver
{
public:
  explicit DenseLinearSolver(real_type pivot_tolerance = 1.0e-14)
    : pivot_tolerance_(pivot_tolerance)
  {
  }

  void solve(const LinearOperator& op,
             const Vector&         rhs,
             Vector&               out) override
  {
    if (op.numRows() != op.numCols() || rhs.size() != op.numRows())
    {
      throw std::runtime_error(
          "DenseLinearSolver received inconsistent dimensions");
    }

    std::vector<real_type> matrix;
    sample(op, false, matrix);
    solveDense(matrix, rhs, out, op.numCols());
  }

  void solveT(const LinearOperator& op,
              const Vector&         rhs,
              Vector&               out) override
  {
    if (op.numRows() != op.numCols() || rhs.size() != op.numCols())
    {
      throw std::runtime_error(
          "DenseLinearSolver received inconsistent transpose dimensions");
    }

    std::vector<real_type> matrix;
    sample(op, true, matrix);
    solveDense(matrix, rhs, out, op.numRows());
  }

private:
  void sample(const LinearOperator&   op,
              bool                    transpose,
              std::vector<real_type>& matrix) const
  {
    const index_type n = transpose ? op.numRows() : op.numCols();
    matrix.assign(static_cast<std::size_t>(n * n), 0.0);

    Vector basis(n);
    Vector column;
    for (index_type j = 0; j < n; ++j)
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

      for (index_type i = 0; i < n; ++i)
      {
        matrix[entry(i, j, n)] = column[i];
      }
    }
  }

  void solveDense(std::vector<real_type> matrix,
                  const Vector&          rhs,
                  Vector&                out,
                  index_type             size) const
  {
    std::vector<real_type> b(static_cast<std::size_t>(size), 0.0);
    for (index_type i = 0; i < size; ++i)
    {
      b[static_cast<std::size_t>(i)] = rhs[i];
    }

    for (index_type k = 0; k < size; ++k)
    {
      index_type pivot = k;
      real_type  best  = std::abs(matrix[entry(k, k, size)]);
      for (index_type i = k + 1; i < size; ++i)
      {
        const real_type candidate = std::abs(matrix[entry(i, k, size)]);
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
        for (index_type j = k; j < size; ++j)
        {
          std::swap(matrix[entry(k, j, size)], matrix[entry(pivot, j, size)]);
        }
        std::swap(b[static_cast<std::size_t>(k)],
                  b[static_cast<std::size_t>(pivot)]);
      }

      for (index_type i = k + 1; i < size; ++i)
      {
        const real_type factor =
            matrix[entry(i, k, size)] / matrix[entry(k, k, size)];
        matrix[entry(i, k, size)] = 0.0;
        for (index_type j = k + 1; j < size; ++j)
        {
          matrix[entry(i, j, size)] -= factor * matrix[entry(k, j, size)];
        }
        b[static_cast<std::size_t>(i)] -= factor * b[static_cast<std::size_t>(k)];
      }
    }

    resize(out, size);
    for (index_type i = size; i-- > 0;)
    {
      real_type sum = b[static_cast<std::size_t>(i)];
      for (index_type j = i + 1; j < size; ++j)
      {
        sum -= matrix[entry(i, j, size)] * out[j];
      }
      out[i] = sum / matrix[entry(i, i, size)];
    }
  }

  static std::size_t entry(index_type row,
                           index_type col,
                           index_type size)
  {
    return static_cast<std::size_t>(row * size + col);
  }

  static void resize(Vector& out, index_type size)
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

private:
  real_type pivot_tolerance_{1.0e-14};
};

} // namespace system
} // namespace femx
