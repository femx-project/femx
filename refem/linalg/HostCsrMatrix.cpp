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

void HostCsrMatrixImpl::addLocalMatrix(index_type cell,
                                       const DenseMatrix& Ke)
{
  if (cell >= pattern_->numCells())
  {
    throw std::runtime_error("Cell index is out of range");
  }

  const index_type ndofs = pattern_->elemNumDofs(cell);

  if (Ke.rows() != ndofs || Ke.cols() != ndofs)
  {
    throw std::runtime_error("Local matrix size does not match cell dofs");
  }

  const index_type offset = pattern_->elemCooOffset(cell);

  index_type local_index = 0;

  for (index_type i = 0; i < ndofs; ++i)
  {
    for (index_type j = 0; j < ndofs; ++j)
    {
      const index_type coo_index = offset + local_index;
      const index_type csr_index = pattern_->mapToCsr(coo_index);

      values_[static_cast<std::size_t>(csr_index)] += Ke(i, j);

      ++local_index;
    }
  }
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
