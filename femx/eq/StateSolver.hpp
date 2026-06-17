#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace eq
{

/** @brief Solver for the state u(m) satisfying a residual equation. */
class StateSolver
{
public:
  virtual ~StateSolver() = default;

  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  virtual void solve(const Vector<Real>& params, Vector<Real>& state) = 0;
};

} // namespace eq
} // namespace femx
