#pragma once

#include <cstddef>
#include <stdexcept>

#include <femx/core/Types.hpp>
#include <femx/algebra/CsrPattern.hpp>
#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/SparseMatrix.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/SystemMatrix.hpp>

namespace femx
{
namespace algebra
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

  bool addLocalMatrix(Index ic, const DenseMatrix& local, bool atomic) override
  {
    if (atomic)
    {
      addLocalMatrixAtomic(ic, local);
    }
    else
    {
      addLocalMatrixSerial(ic, local);
    }
    return true;
  }

  void finalize() override
  {
  }

  void apply(const Vector<Real>& dir, Vector<Real>& out) const override
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

  void applyT(const Vector<Real>& dir, Vector<Real>& out) const override
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
  void checkLocalMatrix(Index ic, const DenseMatrix& local) const
  {
    const CsrPattern& pattern = mat_.pattern();
    if (ic < 0 || ic >= pattern.numElems())
    {
      throw std::runtime_error("SparseSystemMatrix cell index is out of range");
    }

    const Index num_dofs = pattern.elemNumDofs(ic);
    if (local.rows() != num_dofs || local.cols() != num_dofs)
    {
      throw std::runtime_error(
          "SparseSystemMatrix local matrix size does not match cell dofs");
    }
  }

  void addLocalMatrixSerial(Index ic, const DenseMatrix& local)
  {
    checkLocalMatrix(ic, local);

    const CsrPattern& pattern    = mat_.pattern();
    const Index       num_dofs   = pattern.elemNumDofs(ic);
    const Index       coo_offset = pattern.elemCooOffset(ic);
    const Index*      coo_to_csr = pattern.cooToCsrData();
    const Real*       local_vals = local.data();
    Real*             values     = mat_.valuesData();

    for (Index i = 0; i < num_dofs; ++i)
    {
      for (Index j = 0; j < num_dofs; ++j)
      {
        const Index local_entry = i * num_dofs + j;
        const Index coo_entry   = coo_offset + local_entry;
        const Index csr_entry   = coo_to_csr[coo_entry];
        values[static_cast<std::size_t>(csr_entry)] +=
            local_vals[static_cast<std::size_t>(local_entry)];
      }
    }
  }

  void addLocalMatrixAtomic(Index ic, const DenseMatrix& local)
  {
    checkLocalMatrix(ic, local);

    const CsrPattern& pattern    = mat_.pattern();
    const Index       num_dofs   = pattern.elemNumDofs(ic);
    const Index       coo_offset = pattern.elemCooOffset(ic);
    const Index*      coo_to_csr = pattern.cooToCsrData();
    const Real*       local_vals = local.data();
    Real*             values     = mat_.valuesData();

    for (Index i = 0; i < num_dofs; ++i)
    {
      for (Index j = 0; j < num_dofs; ++j)
      {
        const Index local_entry = i * num_dofs + j;
        const Index coo_entry   = coo_offset + local_entry;
        const Index csr_entry   = coo_to_csr[coo_entry];
#pragma omp atomic update
        values[static_cast<std::size_t>(csr_entry)] +=
            local_vals[static_cast<std::size_t>(local_entry)];
      }
    }
  }

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

  static void resizeVector(Vector<Real>& out, Index size)
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

} // namespace algebra
} // namespace femx
