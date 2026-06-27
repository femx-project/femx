#include <algorithm>

#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/CsrMatrix.hpp>

using namespace std;

namespace femx
{

CsrMatrix::CsrMatrix(const CsrPattern& pettern)
  : pattern_(&pettern),
    vals_(pettern.nnz(), Real{})
{
}

void CsrMatrix::setZero()
{
  fill(vals_.begin(), vals_.end(), Real{});
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

const CsrPattern& CsrMatrix::pettern() const
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
