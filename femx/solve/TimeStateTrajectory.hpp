#pragma once

#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace solve
{

/** @brief State values on all time levels of a time-marching solve. */
class TimeStateTrajectory
{
public:
  TimeStateTrajectory();

  TimeStateTrajectory(Index num_steps,
                      Index num_states);

  void resize(Index num_steps,
              Index num_states);

  bool empty() const;

  Index numSteps() const;

  Index numTimeLevels() const;

  Index numLevels() const;

  Index numStates() const;

  Vector<Real> operator[](Index level);

  Vector<Real> operator[](Index level) const;

  void setZero();

private:
  void checkLevel(Index level) const;

private:
  Vector<Real> data_;
  Index        num_steps_{0};
  Index        num_states_{0};
};

} // namespace solve
} // namespace femx
