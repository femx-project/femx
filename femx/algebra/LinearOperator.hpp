#pragma once

#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>

namespace femx
{
namespace algebra
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

} // namespace algebra
} // namespace femx
