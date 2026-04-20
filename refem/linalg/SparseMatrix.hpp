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

  void addLocalMatrix(index_type ic, const DenseMatrix& Ke);

  index_type rows() const;
  index_type cols() const;
  index_type nnz() const;

  MatrixBackend backend() const;

  const index_type* rowPtrData() const;
  const index_type* colIndData() const;
  real_type*        valuesData();
  const real_type*  valuesData() const;

private:
  std::unique_ptr<SparseMatrixImpl> impl_;
};

} // namespace refem
