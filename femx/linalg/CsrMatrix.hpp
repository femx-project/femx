#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/linalg/CsrPattern.hpp>

namespace femx
{

/**
 * @brief Own numeric values for an immutable CSR pattern.
 */
template <MemorySpace Space>
class CsrMatrix
{
public:
  using Pattern = CsrPattern<Space>;
  using Vector  = Vector<Space>;

  /**
   * @brief Construct an empty zero-by-zero matrix.
   */
  CsrMatrix()
    : pattern_(), vals_(0)
  {
  }

  /**
   * @brief Construct a zero-valued matrix for a CSR pattern.
   *
   * @param[in] pattern - Immutable sparsity pattern.
   * @throws std::runtime_error If validation fails.
   */
  explicit CsrMatrix(const Pattern& pattern)
    : pattern_(pattern), vals_(pattern.nnz())
  {
  }

  /**
   * @brief Return the number of rows.
   */
  Index rows() const noexcept
  {
    return pattern_.rows();
  }

  /**
   * @brief Return the number of columns.
   */
  Index cols() const noexcept
  {
    return pattern_.cols();
  }

  /**
   * @brief Return the number of stored entries.
   */
  Index nnz() const noexcept
  {
    return pattern_.nnz();
  }

  /**
   * @brief Return the immutable sparsity pattern.
   */
  const Pattern& pattern() const noexcept
  {
    return pattern_;
  }

  /**
   * @brief Return the CSR row-offset data.
   */
  const Index* rowPtrData() const noexcept
  {
    return pattern_.rowPtrData();
  }

  /**
   * @brief Return the CSR column-index data.
   */
  const Index* colIndData() const noexcept
  {
    return pattern_.colIndData();
  }

  /**
   * @brief Return the numeric CSR value data.
   */
  Real* valsData() noexcept
  {
    return vals_.data();
  }

  /**
   * @brief Return the numeric CSR value data.
   */
  const Real* valsData() const noexcept
  {
    return vals_.data();
  }

  /**
   * @brief Return the owned numeric values.
   */
  Vector& vals() noexcept
  {
    return vals_;
  }

  /**
   * @brief Return the owned numeric values.
   */
  const Vector& vals() const noexcept
  {
    return vals_;
  }

private:
  Pattern pattern_; ///< Shared immutable CSR pattern.
  Vector  vals_;    ///< Numeric values in CSR order.
};

} // namespace femx
