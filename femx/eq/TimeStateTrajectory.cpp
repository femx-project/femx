#include <stdexcept>

#include <femx/eq/TimeStateTrajectory.hpp>

namespace femx
{
namespace eq
{

TimeStateTrajectory::TimeStateTrajectory() = default;

TimeStateTrajectory::TimeStateTrajectory(Index num_steps,
                                         Index num_states)
{
  resize(num_steps, num_states);
}

void TimeStateTrajectory::resize(Index num_steps,
                                 Index num_states)
{
  if (num_steps < 0 || num_states < 0)
  {
    throw std::runtime_error(
        "TimeStateTrajectory received invalid dimensions");
  }

  num_steps_  = num_steps;
  num_states_ = num_states;
  data_.resize((num_steps_ + 1) * num_states_);
}

bool TimeStateTrajectory::empty() const
{
  return data_.empty();
}

Index TimeStateTrajectory::numSteps() const
{
  return empty() ? 0 : num_steps_;
}

Index TimeStateTrajectory::numTimeLevels() const
{
  return empty() ? 0 : num_steps_ + 1;
}

Index TimeStateTrajectory::numLevels() const
{
  return numTimeLevels();
}

Index TimeStateTrajectory::numStates() const
{
  return empty() ? 0 : num_states_;
}

Vector<Real> TimeStateTrajectory::operator[](Index level)
{
  checkLevel(level);
  return Vector<Real>::view(data_.data() + level * num_states_, num_states_);
}

Vector<Real> TimeStateTrajectory::operator[](Index level) const
{
  checkLevel(level);
  return Vector<Real>::view(
      const_cast<Real*>(data_.data()) + level * num_states_, num_states_);
}

void TimeStateTrajectory::setZero()
{
  data_.setZero();
}

void TimeStateTrajectory::checkLevel(Index level) const
{
  if (level < 0 || level >= numTimeLevels())
  {
    throw std::runtime_error("TimeStateTrajectory level is out of range");
  }
}

} // namespace eq
} // namespace femx
