#pragma once

#include <stdexcept>

#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>

namespace femx
{
namespace solve
{

/** @brief State values on all time levels of a time-marching solve. */
class TimeTrajectory
{
public:
  TimeTrajectory() = default;

  TimeTrajectory(Index num_steps,
                 Index num_states)
  {
    resize(num_steps, num_states);
  }

  void resize(Index num_steps,
              Index num_states)
  {
    if (num_steps < 0 || num_states < 0)
    {
      throw std::runtime_error("TimeTrajectory received invalid dimensions");
    }

    num_steps_  = num_steps;
    num_states_ = num_states;
    data_.resize((num_steps_ + 1) * num_states_);
  }

  bool empty() const
  {
    return data_.empty();
  }

  Index numSteps() const
  {
    return empty() ? 0 : num_steps_;
  }

  Index numTimeLevels() const
  {
    return empty() ? 0 : num_steps_ + 1;
  }

  Index numLevels() const
  {
    return numTimeLevels();
  }

  Index numStates() const
  {
    return empty() ? 0 : num_states_;
  }

  Vector<Real> operator[](Index level)
  {
    checkLevel(level);
    return Vector<Real>::view(data_.data() + level * num_states_, num_states_);
  }

  Vector<Real> operator[](Index level) const
  {
    checkLevel(level);
    return Vector<Real>::view(
        const_cast<Real*>(data_.data()) + level * num_states_, num_states_);
  }

  void setZero()
  {
    data_.setZero();
  }

private:
  void checkLevel(Index level) const
  {
    if (level < 0 || level >= numTimeLevels())
    {
      throw std::runtime_error("TimeTrajectory level is out of range");
    }
  }

private:
  Vector<Real> data_;
  Index        num_steps_{0};
  Index        num_states_{0};
};

} // namespace solve
} // namespace femx
