#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class CsrPattern;

/**
 * @brief Numeric values for a fixed compressed sparse row pattern.
 *
 * CsrMatrix owns only the values array; row pointers and column indices are
 * supplied by an immutable CsrPattern.  This split lets multiple matrices share
 * sparsity discovered from finite-element assembly.
 */
class CsrMatrix
{
public:
  explicit CsrMatrix(const CsrPattern& pattern);

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
  Vector<Real>      vals_;
};

} // namespace femx
