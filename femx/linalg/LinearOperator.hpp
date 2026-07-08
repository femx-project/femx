#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace linalg
{

/**
 * @brief Linear operator y = A x with transpose application.
 *
 * Algorithms depend on this interface instead of concrete matrix storage.
 * Implementations are responsible for resizing or assigning the output vector.
 */
class LinearOperator
{
public:
  virtual ~LinearOperator() = default;

  virtual Index numRows() const = 0;
  virtual Index numCols() const = 0;

  /** @brief Compute out = A dir. */
  virtual void apply(const Vector<Real>& dir,
                     Vector<Real>&       out) const = 0;

  /** @brief Compute out = A^T dir. */
  virtual void applyT(const Vector<Real>& dir,
                      Vector<Real>&       out) const = 0;
};

} // namespace linalg
} // namespace femx
