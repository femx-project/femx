#pragma once

#include <cstddef>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/system/SystemMatrix.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/MatrixBackend.hpp>
#include <femx/linalg/SparseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace system
{

/** @brief Sparse SystemMatrix backed by femx::SparseMatrix. */
class SparseSystemMatrix final : public SystemMatrix
{
public:
  explicit SparseSystemMatrix(const CsrPattern& pattern,
                              MatrixBackend               backend = MatrixBackend::HostCsr)
    : matrix_(pattern, backend)
  {
  }

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
    if (rows != numRows() || cols != numCols())
    {
      throw std::runtime_error(
          "SparseSystemMatrix cannot resize away from its sparsity pattern");
    }
  }

  void setZero() override
  {
    matrix_.setZero();
  }

  void set(index_type row, index_type col, real_type value) override
  {
    matrix_.valuesData()[findEntry(row, col)] = value;
  }

  void add(index_type row, index_type col, real_type value) override
  {
    matrix_.valuesData()[findEntry(row, col)] += value;
  }

  void addAtomic(index_type row, index_type col, real_type value) override
  {
    real_type*       values = matrix_.valuesData();
    const index_type entry  = findEntry(row, col);
#pragma omp atomic update
    values[static_cast<std::size_t>(entry)] += value;
  }

  void finalize() override
  {
  }

  void apply(const Vector& dir, Vector& out) const override
  {
    if (dir.size() != numCols())
    {
      throw std::runtime_error(
          "SparseSystemMatrix apply received incompatible vector");
    }

    resizeVector(out, numRows());
    const index_type* row_ptr = matrix_.rowPtrData();
    const index_type* col_ind = matrix_.colIndData();
    const real_type*  values  = matrix_.valuesData();

    for (index_type row = 0; row < numRows(); ++row)
    {
      real_type sum = 0.0;
      for (index_type k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
      {
        sum += values[k] * dir[col_ind[k]];
      }
      out[row] = sum;
    }
  }

  void applyT(const Vector& dir, Vector& out) const override
  {
    if (dir.size() != numRows())
    {
      throw std::runtime_error(
          "SparseSystemMatrix transpose apply received incompatible vector");
    }

    resizeVector(out, numCols());
    const index_type* row_ptr = matrix_.rowPtrData();
    const index_type* col_ind = matrix_.colIndData();
    const real_type*  values  = matrix_.valuesData();

    for (index_type row = 0; row < numRows(); ++row)
    {
      for (index_type k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
      {
        out[col_ind[k]] += values[k] * dir[row];
      }
    }
  }

  SparseMatrix& matrix()
  {
    return matrix_;
  }

  const SparseMatrix& matrix() const
  {
    return matrix_;
  }

private:
  index_type findEntry(index_type row, index_type col) const
  {
    if (row < 0 || row >= numRows() || col < 0 || col >= numCols())
    {
      throw std::runtime_error("SparseSystemMatrix index is out of range");
    }

    const index_type* row_ptr = matrix_.rowPtrData();
    const index_type* col_ind = matrix_.colIndData();
    for (index_type k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
    {
      if (col_ind[k] == col)
      {
        return k;
      }
    }

    throw std::runtime_error(
        "SparseSystemMatrix entry is outside the sparsity pattern");
  }

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
  SparseMatrix matrix_;
};

} // namespace system
} // namespace femx
