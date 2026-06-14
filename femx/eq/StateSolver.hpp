#pragma once

#include <femx/core/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace equation
{

/** @brief Solver for the state u(m) satisfying a residual equation. */
class StateSolver
{
public:
  virtual ~StateSolver() = default;

  virtual index_type numStates() const = 0;
  virtual index_type numParams() const = 0;

  virtual void solve(const Vector& params, Vector& state) = 0;
};

} // namespace equation
} // namespace femx
