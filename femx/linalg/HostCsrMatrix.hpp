#pragma once

#include <vector>

#include <femx/linalg/SparseMatrixImpl.hpp>

namespace femx
{

class CsrPattern;

class HostCsrMatrixImpl final : public SparseMatrixImpl
{
public:
  explicit HostCsrMatrixImpl(const CsrPattern& pattern);

  void setZero() override;

  Index rows() const override;
  Index cols() const override;
  Index nnz() const override;

  MatrixBackend backend() const override;

  const Index* rowPtrData() const override;
  const Index* colIndData() const override;
  Real*        valuesData() override;
  const Real*  valuesData() const override;

private:
  const CsrPattern* pattern_;

  Index rows_;
  Index cols_;
  Index nnz_;

  std::vector<Real> values_;
};

} // namespace femx
