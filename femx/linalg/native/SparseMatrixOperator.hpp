#pragma once

#include <cstddef>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/SparseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace linalg
{

/** @brief Sparse matrix operator and assembly target. */
class SparseMatrixOperator final : public MatrixOperator
{
public:
  explicit SparseMatrixOperator(const CsrPattern& pat)
    : mat_(pat)
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
          "SparseMatrixOperator cannot resize away from its sparsity pattern");
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
    Real*       vals  = mat_.valuesData();
    const Index entry = findEntry(row, col);
#pragma omp atomic update
    vals[entry] += value;
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
          "SparseMatrixOperator apply received incompatible vector");
    }

    resizeOrZero(out, numRows());
    const Index* rp   = mat_.rowPtrData();
    const Index* ci   = mat_.colIndData();
    const Real*  vals = mat_.valuesData();

    for (Index row = 0; row < numRows(); ++row)
    {
      Real sum = 0.0;
      for (Index k = rp[row]; k < rp[row + 1]; ++k)
      {
        sum += vals[k] * dir[ci[k]];
      }
      out[row] = sum;
    }
  }

  void applyT(const Vector<Real>& dir, Vector<Real>& out) const override
  {
    if (dir.size() != numRows())
    {
      throw std::runtime_error(
          "SparseMatrixOperator transpose apply received incompatible vector");
    }

    resizeOrZero(out, numCols());
    const Index* rp   = mat_.rowPtrData();
    const Index* ci   = mat_.colIndData();
    const Real*  vals = mat_.valuesData();

    for (Index row = 0; row < numRows(); ++row)
    {
      for (Index k = rp[row]; k < rp[row + 1]; ++k)
      {
        out[ci[k]] += vals[k] * dir[row];
      }
    }
  }

  SparseMatrix& mat()
  {
    return mat_;
  }

  const SparseMatrix& mat() const
  {
    return mat_;
  }

private:
  void checkLocalMatrix(Index ic, const DenseMatrix& local) const
  {
    const CsrPattern& pat = mat_.pat();
    if (ic < 0 || ic >= pat.numElems())
    {
      throw std::runtime_error("SparseMatrixOperator cell index is out of range");
    }

    const Index nd = pat.elemNumDofs(ic);
    if (local.rows() != nd || local.cols() != nd)
    {
      throw std::runtime_error(
          "SparseMatrixOperator local matrix size does not match cell dofs");
    }
  }

  void addLocalMatrixSerial(Index ic, const DenseMatrix& local)
  {
    checkLocalMatrix(ic, local);

    const CsrPattern& pat        = mat_.pat();
    const Index       nd         = pat.elemNumDofs(ic);
    const Index       coo_offset = pat.elemCooOffset(ic);
    const Index*      coo_to_csr = pat.cooToCsrData();
    const Real*       local_vals = local.data();
    Real*             vals       = mat_.valuesData();

    for (Index i = 0; i < nd; ++i)
    {
      for (Index j = 0; j < nd; ++j)
      {
        const Index local_entry  = i * nd + j;
        const Index coo_entry    = coo_offset + local_entry;
        const Index csr_entry    = coo_to_csr[coo_entry];
        vals[csr_entry]         += local_vals[local_entry];
      }
    }
  }

  void addLocalMatrixAtomic(Index ic, const DenseMatrix& local)
  {
    checkLocalMatrix(ic, local);

    const CsrPattern& pat        = mat_.pat();
    const Index       nd         = pat.elemNumDofs(ic);
    const Index       coo_offset = pat.elemCooOffset(ic);
    const Index*      coo_to_csr = pat.cooToCsrData();
    const Real*       local_vals = local.data();
    Real*             vals       = mat_.valuesData();

    for (Index i = 0; i < nd; ++i)
    {
      for (Index j = 0; j < nd; ++j)
      {
        const Index local_entry = i * nd + j;
        const Index coo_entry   = coo_offset + local_entry;
        const Index csr_entry   = coo_to_csr[coo_entry];
#pragma omp atomic update
        vals[csr_entry] += local_vals[local_entry];
      }
    }
  }

  Index findEntry(Index row, Index col) const
  {
    if (row < 0 || row >= numRows() || col < 0 || col >= numCols())
    {
      throw std::runtime_error("SparseMatrixOperator index is out of range");
    }

    const Index* rp = mat_.rowPtrData();
    const Index* ci = mat_.colIndData();
    for (Index k = rp[row]; k < rp[row + 1]; ++k)
    {
      if (ci[k] == col)
      {
        return k;
      }
    }

    throw std::runtime_error(
        "SparseMatrixOperator entry is outside the sparsity pattern");
  }

  SparseMatrix mat_;
};

} // namespace linalg
} // namespace femx
