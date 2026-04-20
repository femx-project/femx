#include <algorithm>
#include <stdexcept>

#include <refem/linalg/DenseMatrix.hpp>
#include <refem/linalg/FixedSparsityPattern.hpp>
#include <refem/linalg/HostCsrMatrix.hpp>

namespace refem
{

HostCsrMatrixImpl::HostCsrMatrixImpl(const FixedSparsityPattern& pattern)
  : pattern_(&pattern),
    rows_(pattern.rows()),
    cols_(pattern.cols()),
    nnz_(pattern.nnz()),
    values_(static_cast<std::size_t>(nnz_), real_type{})
{
}

void HostCsrMatrixImpl::setZero()
{
  std::fill(values_.begin(), values_.end(), real_type{});
}

index_type HostCsrMatrixImpl::rows() const
{
  return rows_;
}

index_type HostCsrMatrixImpl::cols() const
{
  return cols_;
}

index_type HostCsrMatrixImpl::nnz() const
{
  return nnz_;
}

MatrixBackend HostCsrMatrixImpl::backend() const
{
  return MatrixBackend::HostCsr;
}

const index_type* HostCsrMatrixImpl::rowPtrData() const
{
  return pattern_->rowPtrData();
}

const index_type* HostCsrMatrixImpl::colIndData() const
{
  return pattern_->colIndData();
}

real_type* HostCsrMatrixImpl::valuesData()
{
  return values_.data();
}

const real_type* HostCsrMatrixImpl::valuesData() const
{
  return values_.data();
}

} // namespace refem
