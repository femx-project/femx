#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace eq
{

/** @brief Solver for a parameter-dependent time history. */
class TimeStateSolver
{
public:
  virtual ~TimeStateSolver() = default;

  virtual Index numSteps() const  = 0;
  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  virtual void solve(const Vector<Real>&  prm,
                     TimeStateTrajectory& tr) = 0;
};

} // namespace eq
} // namespace femx
