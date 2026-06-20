#include <algorithm>

#include <femx/algebra/CsrPattern.hpp>
#include <femx/algebra/SparseMatrix.hpp>

namespace femx
{

SparseMatrix::SparseMatrix(const CsrPattern& pattern)
  : pattern_(&pattern),
    values_(static_cast<std::size_t>(pattern.nnz()), Real{})
{
}

void SparseMatrix::setZero()
{
  std::fill(values_.begin(), values_.end(), Real{});
}

Index SparseMatrix::rows() const
{
  return pattern_->rows();
}

Index SparseMatrix::cols() const
{
  return pattern_->cols();
}

Index SparseMatrix::nnz() const
{
  return pattern_->nnz();
}

const CsrPattern& SparseMatrix::pattern() const
{
  return *pattern_;
}

const Index* SparseMatrix::rowPtrData() const
{
  return pattern_->rowPtrData();
}

const Index* SparseMatrix::colIndData() const
{
  return pattern_->colIndData();
}

Real* SparseMatrix::valuesData()
{
  return values_.data();
}

const Real* SparseMatrix::valuesData() const
{
  return values_.data();
}

} // namespace femx
