#pragma once

#include <memory>

#include <femx/common/Types.hpp>
#include <femx/linalg/MatrixBackend.hpp>

namespace femx
{

class DenseMatrix;
class CsrPattern;
class SparseMatrixImpl;

class SparseMatrix
{
public:
  explicit SparseMatrix(
      const CsrPattern& pattern,
      MatrixBackend               backend = MatrixBackend::HostCsr);

  ~SparseMatrix();

  void setZero();

  index_type rows() const;
  index_type cols() const;
  index_type nnz() const;

  const CsrPattern& pattern() const;

  MatrixBackend backend() const;

  const index_type* rowPtrData() const;
  const index_type* colIndData() const;
  real_type*        valuesData();
  const real_type*  valuesData() const;

private:
  const CsrPattern*       pattern_{nullptr};
  std::unique_ptr<SparseMatrixImpl> impl_;
};

} // namespace femx
