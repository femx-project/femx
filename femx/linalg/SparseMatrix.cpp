#include <memory>
#include <stdexcept>

#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/HostCsrMatrix.hpp>
#include <femx/linalg/SparseMatrix.hpp>
#include <femx/linalg/SparseMatrixImpl.hpp>

namespace femx
{

SparseMatrix::SparseMatrix(const CsrPattern& pattern,
                           MatrixBackend     backend)
  : pattern_(&pattern)
{
  switch (backend)
  {
  case MatrixBackend::HostCsr:
    impl_ = std::make_unique<HostCsrMatrixImpl>(pattern);
    break;

  case MatrixBackend::CudaCsr:
    throw std::runtime_error("CudaCsr matrix backend is not supported yet");

  default:
    throw std::runtime_error("Unknown matrix backend");
  }
}

SparseMatrix::~SparseMatrix() = default;

void SparseMatrix::setZero()
{
  impl_->setZero();
}

Index SparseMatrix::rows() const
{
  return impl_->rows();
}

Index SparseMatrix::cols() const
{
  return impl_->cols();
}

Index SparseMatrix::nnz() const
{
  return impl_->nnz();
}

const CsrPattern& SparseMatrix::pattern() const
{
  return *pattern_;
}

MatrixBackend SparseMatrix::backend() const
{
  return impl_->backend();
}

const Index* SparseMatrix::rowPtrData() const
{
  return impl_->rowPtrData();
}

const Index* SparseMatrix::colIndData() const
{
  return impl_->colIndData();
}

Real* SparseMatrix::valuesData()
{
  return impl_->valuesData();
}

const Real* SparseMatrix::valuesData() const
{
  return impl_->valuesData();
}

} // namespace femx
