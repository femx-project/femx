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
      MatrixBackend     backend = MatrixBackend::HostCsr);

  ~SparseMatrix();

  void setZero();

  Index rows() const;
  Index cols() const;
  Index nnz() const;

  const CsrPattern& pattern() const;

  MatrixBackend backend() const;

  const Index* rowPtrData() const;
  const Index* colIndData() const;
  Real*        valuesData();
  const Real*  valuesData() const;

private:
  const CsrPattern*                 pattern_{nullptr};
  std::unique_ptr<SparseMatrixImpl> impl_;
};

} // namespace femx
