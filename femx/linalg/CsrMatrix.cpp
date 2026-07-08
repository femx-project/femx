#include <algorithm>

#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/CsrPattern.hpp>

namespace femx
{

CsrMatrix::CsrMatrix(const CsrPattern& pattern)
  : pattern_(&pattern),
    vals_(pattern.nnz(), Real{})
{
}

void CsrMatrix::setZero()
{
  std::fill(vals_.begin(), vals_.end(), Real{});
}

Index CsrMatrix::rows() const
{
  return pattern_->rows();
}

Index CsrMatrix::cols() const
{
  return pattern_->cols();
}

Index CsrMatrix::nnz() const
{
  return pattern_->nnz();
}

const CsrPattern& CsrMatrix::pattern() const
{
  return *pattern_;
}

const Index* CsrMatrix::rowPtrData() const
{
  return pattern_->rowPtrData();
}

const Index* CsrMatrix::colIndData() const
{
  return pattern_->colIndData();
}

Real* CsrMatrix::valuesData()
{
  return vals_.data();
}

const Real* CsrMatrix::valuesData() const
{
  return vals_.data();
}

} // namespace femx
