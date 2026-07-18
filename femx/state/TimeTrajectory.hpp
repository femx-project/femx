#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace state
{

/** @brief Contiguous host storage for all time levels of a state. */
class TimeTrajectory
{
public:
  TimeTrajectory() = default;

  TimeTrajectory(Index num_steps, Index num_states)
  {
    resize(num_steps, num_states);
  }

  void resize(Index num_steps, Index num_states)
  {
    if (num_steps < 0 || num_states < 0)
    {
      throw std::runtime_error("TimeTrajectory received invalid dimensions");
    }

    const std::int64_t size =
        (static_cast<std::int64_t>(num_steps) + 1) * num_states;
    if (size > std::numeric_limits<Index>::max())
    {
      throw std::runtime_error("TimeTrajectory exceeds the Index range");
    }

    data_.resize(static_cast<Index>(size));
    num_steps_  = num_steps;
    num_states_ = num_states;
  }

  Index numSteps() const noexcept
  {
    return data_.empty() ? 0 : num_steps_;
  }

  Index numTimeLevels() const noexcept
  {
    return data_.empty() ? 0 : num_steps_ + 1;
  }

  Index numStates() const noexcept
  {
    return data_.empty() ? 0 : num_states_;
  }

  Index size() const noexcept
  {
    return data_.size();
  }

  Real* data() noexcept
  {
    return data_.data();
  }

  const Real* data() const noexcept
  {
    return data_.data();
  }

  HostVectorView level(Index level)
  {
    checkLevel(level);
    return {data_.data() + level * num_states_, num_states_};
  }

  HostConstVectorView level(Index level) const
  {
    checkLevel(level);
    return {data_.data() + level * num_states_, num_states_};
  }

  HostVectorView operator[](Index level)
  {
    return this->level(level);
  }

  HostConstVectorView operator[](Index level) const
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

  HostVector data_;
  Index      num_steps_{0};
  Index      num_states_{0};
};

} // namespace state
} // namespace femx
