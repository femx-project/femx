#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace eq
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

} // namespace eq
} // namespace femx
