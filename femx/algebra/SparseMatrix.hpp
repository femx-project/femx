#pragma once

#include <vector>

#include <femx/core/Types.hpp>

namespace femx
{

class CsrPattern;

class SparseMatrix
{
public:
  explicit SparseMatrix(const CsrPattern& pattern);

  void setZero();

  Index rows() const;
  Index cols() const;
  Index nnz() const;

  const CsrPattern& pattern() const;

  const Index* rowPtrData() const;
  const Index* colIndData() const;
  Real*        valuesData();
  const Real*  valuesData() const;

private:
  const CsrPattern* pattern_{nullptr};
  std::vector<Real> values_;
};

} // namespace femx
