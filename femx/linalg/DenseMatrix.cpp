#include <algorithm>
#include <stdexcept>

#include <femx/linalg/DenseMatrix.hpp>

namespace femx
{

DenseMatrix::DenseMatrix()
  : rows_(0),
    cols_(0)
{
}

DenseMatrix::DenseMatrix(Index rows, Index cols)
  : rows_(rows),
    cols_(cols),
    vals_(rows * cols, Real{})
{
}

void DenseMatrix::resize(Index rows, Index cols)
{
  rows_ = rows;
  cols_ = cols;

  vals_.assign(rows_ * cols_, Real{});
}

void DenseMatrix::setZero()
{
  std::fill(vals_.begin(), vals_.end(), Real{});
}

Index DenseMatrix::numRows() const
{
  return rows_;
}

Index DenseMatrix::numCols() const
{
  return cols_;
}

void DenseMatrix::set(Index row, Index col, Real val)
{
  (*this)(row, col) = val;
}

void DenseMatrix::add(Index row, Index col, Real val)
{
  (*this)(row, col) += val;
}

void DenseMatrix::addAtomic(Index row, Index col, Real val)
{
  Real& entry = (*this)(row, col);
#pragma omp atomic update
  entry += val;
}

void DenseMatrix::finalize()
{
}

Index DenseMatrix::size() const
{
  return vals_.size();
}

Real& DenseMatrix::operator()(Index i, Index j)
{
  return vals_[i * cols_ + j];
}

Real DenseMatrix::operator()(Index i, Index j) const
{
  return vals_[i * cols_ + j];
}

void DenseMatrix::apply(const HostVector& x, HostVector& out) const
{
  if (cols_ != x.size())
  {
    throw std::runtime_error("DenseMatrix apply received incompatible vector");
  }

  resizeOrZero(out, rows_);
  for (Index i = 0; i < rows_; ++i)
  {
    Real sum = 0.0;
    for (Index j = 0; j < cols_; ++j)
    {
      sum += (*this)(i, j) * x[j];
    }
    out[i] = sum;
  }
}

void DenseMatrix::applyT(const HostVector& x, HostVector& out) const
{
  if (rows_ != x.size())
  {
    throw std::runtime_error(
        "DenseMatrix transpose apply received incompatible vector");
  }

  resizeOrZero(out, cols_);
  for (Index j = 0; j < cols_; ++j)
  {
    Real sum = 0.0;
    for (Index i = 0; i < rows_; ++i)
    {
      sum += (*this)(i, j) * x[i];
    }
    out[j] = sum;
  }
}

Real* DenseMatrix::data()
{
  return vals_.data();
}

const Real* DenseMatrix::data() const
{
  return vals_.data();
}

void linalg::MatrixOperator::addElem(Index,
                                     const Array<Index>& row_dofs,
                                     const Array<Index>& col_dofs,
                                     const DenseMatrix&  mat_e,
                                     bool                atomic)
{
  if (mat_e.numRows() != row_dofs.size()
      || mat_e.numCols() != col_dofs.size())
  {
    throw std::runtime_error(
        "MatrixOperator element matrix size does not match dofs");
  }

  for (Index i = 0; i < mat_e.numRows(); ++i)
  {
    const Index row = row_dofs[i];
    for (Index j = 0; j < mat_e.numCols(); ++j)
    {
      const Index col = col_dofs[j];
      if (atomic)
      {
        addAtomic(row, col, mat_e(i, j));
      }
      else
      {
        add(row, col, mat_e(i, j));
      }
    }
  }
}

} // namespace femx
