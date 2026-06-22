#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace linalg
{

/** @brief Dense in-memory matrix operator and assembly target. */
class DenseMatrixOperator final : public MatrixOperator
{
public:
  Index numRows() const override
  {
    return mat_.rows();
  }

  Index numCols() const override
  {
    return mat_.cols();
  }

  void resize(Index rows, Index cols) override
  {
    mat_.resize(rows, cols);
  }

  void setZero() override
  {
    mat_.setZero();
  }

  void set(Index row, Index col, Real value) override
  {
    mat_(row, col) = value;
  }

  void add(Index row, Index col, Real value) override
  {
    mat_(row, col) += value;
  }

  void addAtomic(Index row, Index col, Real value) override
  {
    Real& entry = mat_(row, col);
#pragma omp atomic update
    entry += value;
  }

  void finalize() override
  {
  }

  void apply(const Vector<Real>& dir, Vector<Real>& out) const override
  {
    if (dir.size() != numCols())
    {
      throw std::runtime_error(
          "DenseMatrixOperator apply received incompatible vector");
    }

    resizeOrZero(out, numRows());
    for (Index i = 0; i < numRows(); ++i)
    {
      for (Index j = 0; j < numCols(); ++j)
      {
        out[i] += mat_(i, j) * dir[j];
      }
    }
  }

  void applyT(const Vector<Real>& dir, Vector<Real>& out) const override
  {
    if (dir.size() != numRows())
    {
      throw std::runtime_error(
          "DenseMatrixOperator transpose apply received incompatible vector");
    }

    resizeOrZero(out, numCols());
    for (Index i = 0; i < numRows(); ++i)
    {
      for (Index j = 0; j < numCols(); ++j)
      {
        out[j] += mat_(i, j) * dir[i];
      }
    }
  }

  DenseMatrix& mat()
  {
    return mat_;
  }

  const DenseMatrix& mat() const
  {
    return mat_;
  }

private:
  DenseMatrix mat_;
};

} // namespace linalg
} // namespace femx
