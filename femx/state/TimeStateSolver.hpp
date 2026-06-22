#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace state
{

/** @brief Solver for a parameter-dependent time history. */
class TimeStateSolver
{
public:
  virtual ~TimeStateSolver() = default;

  virtual Index numSteps() const  = 0;
  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  virtual void solve(const Vector<Real>& prm,
                     TimeTrajectory&     tr) = 0;
};

} // namespace state
} // namespace femx
