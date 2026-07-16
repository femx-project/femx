#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{
namespace state
{

/**
 * @brief State values on all time levels of a time-marching solve.
 *
 * TimeTrajectory stores contiguous state blocks so each time level can be
 * accessed as a VectorView.
 */
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

  Index size() const
  {
    return data_.size();
  }

  Real* data()
  {
    return data_.data();
  }

  const Real* data() const
  {
    return data_.data();
  }

  VectorView<Real> level(Index level)
  {
    checkLevel(level);
    return VectorView<Real>(data_.data() + level * num_states_, num_states_);
  }

  VectorView<const Real> level(Index level) const
  {
    checkLevel(level);
    return VectorView<const Real>(data_.data() + level * num_states_,
                                  num_states_);
  }

  VectorView<Real> operator[](Index level)
  {
    return this->level(level);
  }

  VectorView<const Real> operator[](Index level) const
  {
    return this->level(level);
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

} // namespace state
} // namespace femx
