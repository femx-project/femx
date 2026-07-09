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

Index DenseMatrix::rows() const
{
  return rows_;
}

Index DenseMatrix::cols() const
{
  return cols_;
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

void DenseMatrix::matvec(const Vector<Real>& x, Vector<Real>& out) const
{
  if (cols_ != x.size())
  {
    throw std::runtime_error("DenseMatrix matvec received incompatible vector");
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

void DenseMatrix::matvecT(const Vector<Real>& x, Vector<Real>& out) const
{
  if (rows_ != x.size())
  {
    throw std::runtime_error(
        "DenseMatrix transpose matvec received incompatible vector");
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

} // namespace femx
