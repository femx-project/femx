#pragma once

#include <femx/common/Types.hpp>

namespace femx
{
namespace state
{

/**
 * @brief Solver for a parameter-dependent stationary state.
 *
 * StateSolver maps a parameter/control vector m to a state vector u(m) by
 * enforcing a residual equation.  Reduced optimization algorithms depend only
 * on this abstract interface.
 */
class StateSolver
{
public:
  virtual ~StateSolver() = default;

  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;
  virtual Index numRes() const    = 0;

  /** @brief Solve the state equation for a parameter vector. */
  virtual void solve(const HostVector& prm, HostVector& state) = 0;
};

} // namespace state
} // namespace femx
