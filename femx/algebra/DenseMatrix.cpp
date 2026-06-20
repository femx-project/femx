#include <algorithm>

#include <femx/algebra/DenseMatrix.hpp>

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
    values_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
            Real{})
{
}

void DenseMatrix::resize(Index rows, Index cols)
{
  rows_ = rows;
  cols_ = cols;

  values_.assign(
      static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_),
      Real{});
}

void DenseMatrix::setZero()
{
  std::fill(values_.begin(), values_.end(), Real{});
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
  return static_cast<Index>(values_.size());
}

Real& DenseMatrix::operator()(Index i, Index j)
{
  return values_[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_) + static_cast<std::size_t>(j)];
}

Real DenseMatrix::operator()(Index i, Index j) const
{
  return values_[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_) + static_cast<std::size_t>(j)];
}

Real* DenseMatrix::data()
{
  return values_.data();
}

const Real* DenseMatrix::data() const
{
  return values_.data();
}

} // namespace femx
