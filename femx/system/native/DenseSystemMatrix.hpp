#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/system/SystemMatrix.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace system
{

/** @brief Dense in-memory implementation of SystemMatrix. */
class DenseSystemMatrix final : public SystemMatrix
{
public:
  index_type numRows() const override
  {
    return matrix_.rows();
  }

  index_type numCols() const override
  {
    return matrix_.cols();
  }

  void resize(index_type rows, index_type cols) override
  {
    matrix_.resize(rows, cols);
  }

  void setZero() override
  {
    matrix_.setZero();
  }

  void set(index_type row, index_type col, real_type value) override
  {
    matrix_(row, col) = value;
  }

  void add(index_type row, index_type col, real_type value) override
  {
    matrix_(row, col) += value;
  }

  void addAtomic(index_type row, index_type col, real_type value) override
  {
    real_type& entry = matrix_(row, col);
#pragma omp atomic update
    entry += value;
  }

  void finalize() override
  {
  }

  void apply(const Vector& dir, Vector& out) const override
  {
    if (dir.size() != numCols())
    {
      throw std::runtime_error(
          "DenseSystemMatrix apply received incompatible vector");
    }

    resizeVector(out, numRows());
    for (index_type i = 0; i < numRows(); ++i)
    {
      for (index_type j = 0; j < numCols(); ++j)
      {
        out[i] += matrix_(i, j) * dir[j];
      }
    }
  }

  void applyT(const Vector& dir, Vector& out) const override
  {
    if (dir.size() != numRows())
    {
      throw std::runtime_error(
          "DenseSystemMatrix transpose apply received incompatible vector");
    }

    resizeVector(out, numCols());
    for (index_type i = 0; i < numRows(); ++i)
    {
      for (index_type j = 0; j < numCols(); ++j)
      {
        out[j] += matrix_(i, j) * dir[i];
      }
    }
  }

  DenseMatrix& matrix()
  {
    return matrix_;
  }

  const DenseMatrix& matrix() const
  {
    return matrix_;
  }

private:
  static void resizeVector(Vector& out, index_type size)
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
  DenseMatrix matrix_;
};

} // namespace system
} // namespace femx
