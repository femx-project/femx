#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace system
{

/** @brief Matrix-free linear operator used by equation solvers. */
class LinearOperator
{
public:
  virtual ~LinearOperator() = default;

  virtual index_type numRows() const = 0;
  virtual index_type numCols() const = 0;

  virtual void apply(const Vector& dir, Vector& out) const  = 0;
  virtual void applyT(const Vector& dir, Vector& out) const = 0;
};

} // namespace system
} // namespace femx
