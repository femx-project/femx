#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace state
{

/** @brief Solver for a parameter-dependent stationary state. */
class StateSolver
{
public:
  virtual ~StateSolver() = default;

  virtual Index numStates() const    = 0;
  virtual Index numParams() const    = 0;
  virtual Index numResiduals() const = 0;

  virtual void solve(const Vector<Real>& prm, Vector<Real>& state) = 0;
};

} // namespace state
} // namespace femx
