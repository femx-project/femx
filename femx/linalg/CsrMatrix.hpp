#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

/**
 * @brief Numeric values for an immutable CSR pattern in one memory space.
 *
 * A matrix retains an immutable shared pattern handle, allowing multiple
 * matrices and maps to reuse one sparse structure without lifetime coupling.
 * Numeric values are owned exclusively by this matrix in Space.
 */
template <MemorySpace Space>
class CsrMatrix
{
public:
  using Pattern = CsrPattern<Space>;
  using Vals    = Vector<Space>;

  /** @brief Construct an empty matrix. */
  CsrMatrix()
    : pattern_(), vals_(0)
  {
  }

  /** @brief Allocate zero-initialized values for `pattern`. */
  explicit CsrMatrix(const Pattern& pattern)
    : pattern_(pattern), vals_(pattern.nnz())
  {
  }

  Index rows() const noexcept
  {
    return pattern_.rows();
  }

  Index cols() const noexcept
  {
    return pattern_.cols();
  }

  Index nnz() const noexcept
  {
    return pattern_.nnz();
  }

  const Pattern& pattern() const noexcept
  {
    return pattern_;
  }

  const Index* rowPtrData() const noexcept
  {
    return pattern_.rowPtrData();
  }

  const Index* colIndData() const noexcept
  {
    return pattern_.colIndData();
  }

  Real* valsData() noexcept
  {
    return vals_.data();
  }

  const Real* valsData() const noexcept
  {
    return vals_.data();
  }

  Vals& vals() noexcept
  {
    return vals_;
  }

  const Vals& vals() const noexcept
  {
    return vals_;
  }

private:
  Pattern pattern_;
  Vals    vals_;
};

} // namespace femx
