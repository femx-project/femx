#pragma once

#include <memory>

#include <refem/common/Types.hpp>
#include <refem/linalg/MatrixBackend.hpp>

namespace refem
{

class DenseMatrix;
class FixedSparsityPattern;
class SparseMatrixImpl;

class SparseMatrix
{
public:
  explicit SparseMatrix(
      const FixedSparsityPattern& pattern,
      MatrixBackend               backend = MatrixBackend::HostCsr);

  ~SparseMatrix();

  void setZero();

  index_type rows() const;
  index_type cols() const;
  index_type nnz() const;

  const FixedSparsityPattern& pattern() const;

  MatrixBackend backend() const;

  const index_type* rowPtrData() const;
  const index_type* colIndData() const;
  real_type*        valuesData();
  const real_type*  valuesData() const;

private:
  const FixedSparsityPattern*       pattern_{nullptr};
  std::unique_ptr<SparseMatrixImpl> impl_;
};

} // namespace refem
