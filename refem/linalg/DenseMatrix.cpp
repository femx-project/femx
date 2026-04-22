#include <algorithm>

#include <refem/linalg/DenseMatrix.hpp>

namespace refem
{

DenseMatrix::DenseMatrix()
  : rows_(0),
    cols_(0)
{
}

DenseMatrix::DenseMatrix(index_type rows, index_type cols)
  : rows_(rows),
    cols_(cols),
    values_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
            real_type{})
{
}

void DenseMatrix::resize(index_type rows, index_type cols)
{
  rows_ = rows;
  cols_ = cols;

  values_.assign(
      static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_),
      real_type{});
}

void DenseMatrix::setZero()
{
  std::fill(values_.begin(), values_.end(), real_type{});
}

index_type DenseMatrix::rows() const
{
  return rows_;
}

index_type DenseMatrix::cols() const
{
  return cols_;
}

index_type DenseMatrix::size() const
{
  return static_cast<index_type>(values_.size());
}

real_type& DenseMatrix::operator()(index_type i, index_type j)
{
  return values_[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_) + static_cast<std::size_t>(j)];
}

real_type DenseMatrix::operator()(index_type i, index_type j) const
{
  return values_[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_) + static_cast<std::size_t>(j)];
}

real_type* DenseMatrix::data()
{
  return values_.data();
}

const real_type* DenseMatrix::data() const
{
  return values_.data();
}

} // namespace refem
