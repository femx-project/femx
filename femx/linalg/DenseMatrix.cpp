#include <algorithm>

#include <femx/linalg/DenseMatrix.hpp>

using namespace std;

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
  fill(vals_.begin(), vals_.end(), Real{});
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

Real* DenseMatrix::data()
{
  return vals_.data();
}

const Real* DenseMatrix::data() const
{
  return vals_.data();
}

} // namespace femx
