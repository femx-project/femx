#pragma once

#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace solve
{

/** @brief Solver for the state u(m) satisfying a residual equation. */
class StateSolver
{
public:
  virtual ~StateSolver() = default;

  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  virtual void solve(const Vector<Real>& prm, Vector<Real>& state) = 0;
};

} // namespace solve
} // namespace femx
