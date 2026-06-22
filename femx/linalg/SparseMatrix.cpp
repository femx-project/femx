#include <algorithm>

#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/SparseMatrix.hpp>

using namespace std;

namespace femx
{

SparseMatrix::SparseMatrix(const CsrPattern& pat)
  : pattern_(&pat),
    vals_(pat.nnz(), Real{})
{
}

void SparseMatrix::setZero()
{
  fill(vals_.begin(), vals_.end(), Real{});
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

const CsrPattern& SparseMatrix::pat() const
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
  return vals_.data();
}

const Real* SparseMatrix::valuesData() const
{
  return vals_.data();
}

} // namespace femx
