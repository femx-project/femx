#pragma once

#include <femx/linalg/Vector.hpp>
#include <femx/common/Types.hpp>

namespace femx
{
namespace linalg
{

/** @brief Linear operator y = A x with optional transpose application. */
class LinearOperator
{
public:
  virtual ~LinearOperator() = default;

  virtual Index numRows() const = 0;
  virtual Index numCols() const = 0;

  virtual void apply(const Vector<Real>& dir,
                     Vector<Real>&       out) const = 0;

  virtual void applyT(const Vector<Real>& dir,
                      Vector<Real>&       out) const = 0;
};

} // namespace linalg
} // namespace femx
