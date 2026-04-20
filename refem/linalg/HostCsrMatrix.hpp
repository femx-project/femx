#pragma once

#include <vector>

#include <refem/linalg/SparseMatrixImpl.hpp>

namespace refem
{

class FixedSparsityPattern;

class HostCsrMatrixImpl final : public SparseMatrixImpl
{
public:
  explicit HostCsrMatrixImpl(const FixedSparsityPattern& pattern);

  void setZero() override;

  index_type rows() const override;
  index_type cols() const override;
  index_type nnz() const override;

  MatrixBackend backend() const override;

  const index_type* rowPtrData() const override;
  const index_type* colIndData() const override;
  real_type*        valuesData() override;
  const real_type*  valuesData() const override;

private:
  const FixedSparsityPattern* pattern_;

  index_type rows_;
  index_type cols_;
  index_type nnz_;

  std::vector<real_type> values_;
};

} // namespace refem
