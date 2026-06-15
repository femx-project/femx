#pragma once

#include <cstddef>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/SparseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemMatrix.hpp>

namespace femx
{
namespace system
{

/** @brief Sparse SystemMatrix backed by femx::SparseMatrix. */
class SparseSystemMatrix final : public SystemMatrix
{
public:
  explicit SparseSystemMatrix(const CsrPattern& pattern)
    : mat_(pattern)
  {
  }

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
    if (rows != numRows() || cols != numCols())
    {
      throw std::runtime_error(
          "SparseSystemMatrix cannot resize away from its sparsity pattern");
    }
  }

  void setZero() override
  {
    mat_.setZero();
  }

  void set(Index row, Index col, Real value) override
  {
    mat_.valuesData()[findEntry(row, col)] = value;
  }

  void add(Index row, Index col, Real value) override
  {
    mat_.valuesData()[findEntry(row, col)] += value;
  }

  void addAtomic(Index row, Index col, Real value) override
  {
    Real*       values = mat_.valuesData();
    const Index entry  = findEntry(row, col);
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
    const Index* row_ptr = mat_.rowPtrData();
    const Index* col_ind = mat_.colIndData();
    const Real*  values  = mat_.valuesData();

    for (Index row = 0; row < numRows(); ++row)
    {
      Real sum = 0.0;
      for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
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
    const Index* row_ptr = mat_.rowPtrData();
    const Index* col_ind = mat_.colIndData();
    const Real*  values  = mat_.valuesData();

    for (Index row = 0; row < numRows(); ++row)
    {
      for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
      {
        out[col_ind[k]] += values[k] * dir[row];
      }
    }
  }

  SparseMatrix& matrix()
  {
    return mat_;
  }

  const SparseMatrix& matrix() const
  {
    return mat_;
  }

private:
  Index findEntry(Index row, Index col) const
  {
    if (row < 0 || row >= numRows() || col < 0 || col >= numCols())
    {
      throw std::runtime_error("SparseSystemMatrix index is out of range");
    }

    const Index* row_ptr = mat_.rowPtrData();
    const Index* col_ind = mat_.colIndData();
    for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
    {
      if (col_ind[k] == col)
      {
        return k;
      }
    }

    throw std::runtime_error(
        "SparseSystemMatrix entry is outside the sparsity pattern");
  }

  static void resizeVector(Vector& out, Index size)
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
  SparseMatrix mat_;
};

} // namespace system
} // namespace femx
