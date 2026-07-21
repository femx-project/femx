#pragma once

#include <cstdint>
#include <limits>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

namespace femx::state
{

/** @brief Host-owned state values for all time levels of one solve. */
class TimeTrajectory
{
public:
  TimeTrajectory() = default;

  TimeTrajectory(Index num_steps, Index num_states)
  {
    resize(num_steps, num_states);
  }

  /** @brief Set dimensions, retaining an allocation of the required size. */
  void resize(Index num_steps, Index num_states)
  {
    const Index size = checkedSize(num_steps, num_states);
    if (data_.size() != size)
    {
      data_.resize(size);
    }
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
    return {data() + level * num_states_, num_states_};
  }

  HostConstVectorView level(Index level) const
  {
    checkLevel(level);
    return {data() + level * num_states_, num_states_};
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
    CpuContext                ctx;
    linalg::HostVectorHandler vec_handler(ctx);
    vec_handler.zero(data_.view());
  }

private:
  static Index checkedSize(Index num_steps, Index num_states)
  {
    require(num_steps >= 0 && num_states >= 0,
            "TimeTrajectory received invalid dimensions");

    const std::int64_t size =
        (static_cast<std::int64_t>(num_steps) + 1) * num_states;
    require(size <= std::numeric_limits<Index>::max(),
            "TimeTrajectory exceeds the Index range");
    return static_cast<Index>(size);
  }

  void checkLevel(Index level) const
  {
    require(level >= 0 && level < numTimeLevels(),
            "TimeTrajectory level is out of range");
  }

  HostVector data_;
  Index      num_steps_{0};
  Index      num_states_{0};
};

} // namespace femx::state
