#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class CsrPattern;

class CsrMatrix
{
public:
  explicit CsrMatrix(const CsrPattern& pettern);

  void setZero();

  Index rows() const;
  Index cols() const;
  Index nnz() const;

  const CsrPattern& pettern() const;

  const Index* rowPtrData() const;
  const Index* colIndData() const;
  Real*        valuesData();
  const Real*  valuesData() const;

private:
  const CsrPattern* pattern_{nullptr};
  Vector<Real>      vals_;
};

} // namespace femx
