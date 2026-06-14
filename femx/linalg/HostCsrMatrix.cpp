#include <algorithm>
#include <stdexcept>

#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/HostCsrMatrix.hpp>

namespace femx
{

HostCsrMatrixImpl::HostCsrMatrixImpl(const CsrPattern& pattern)
  : pattern_(&pattern),
    rows_(pattern.rows()),
    cols_(pattern.cols()),
    nnz_(pattern.nnz()),
    values_(static_cast<std::size_t>(nnz_), Real{})
{
}

void HostCsrMatrixImpl::setZero()
{
  std::fill(values_.begin(), values_.end(), Real{});
}

Index HostCsrMatrixImpl::rows() const
{
  return rows_;
}

Index HostCsrMatrixImpl::cols() const
{
  return cols_;
}

Index HostCsrMatrixImpl::nnz() const
{
  return nnz_;
}

MatrixBackend HostCsrMatrixImpl::backend() const
{
  return MatrixBackend::HostCsr;
}

const Index* HostCsrMatrixImpl::rowPtrData() const
{
  return pattern_->rowPtrData();
}

const Index* HostCsrMatrixImpl::colIndData() const
{
  return pattern_->colIndData();
}

Real* HostCsrMatrixImpl::valuesData()
{
  return values_.data();
}

const Real* HostCsrMatrixImpl::valuesData() const
{
  return values_.data();
}

} // namespace femx
