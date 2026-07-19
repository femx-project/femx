#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/View.hpp>

namespace femx
{
namespace state
{

/** @brief Non-owning access to contiguous trajectory storage. */
template <MemorySpace Space, class T>
class TrajectoryView
{
public:
  FEMX_HOST_DEVICE TrajectoryView() = default;

  FEMX_HOST_DEVICE TrajectoryView(T*    data,
                                  Index num_levels,
                                  Index num_states)
    : data_(data),
      num_levels_(num_levels),
      num_states_(num_states)
  {
  }

  /** @brief Convert a compatible mutable view to a const view. */
  template <class U,
            typename std::enable_if<std::is_convertible<U*, T*>::value,
                                    int>::type = 0>
  FEMX_HOST_DEVICE TrajectoryView(const TrajectoryView<Space, U>& other)
    : data_(other.data()),
      num_levels_(other.numTimeLevels()),
      num_states_(other.numStates())
  {
  }

  /** @brief Return the number of residual steps represented by this view. */
  FEMX_HOST_DEVICE Index numSteps() const
  {
    return num_levels_ == 0 ? 0 : num_levels_ - 1;
  }

  /** @brief Return the number of stored time levels. */
  FEMX_HOST_DEVICE Index numTimeLevels() const
  {
    return num_levels_;
  }

  /** @brief Return the state size at each time level. */
  FEMX_HOST_DEVICE Index numStates() const
  {
    return num_states_;
  }

  /** @brief Return the number of scalar values in the view. */
  FEMX_HOST_DEVICE Index size() const
  {
    return num_levels_ * num_states_;
  }

  /** @brief Return the first address in this view's memory space. */
  FEMX_HOST_DEVICE T* data() const
  {
    return data_;
  }

  /** @brief Return one time level without bounds checking. */
  FEMX_HOST_DEVICE VectorView<Space, T> level(Index level) const
  {
    return {data_ + level * num_states_, num_states_};
  }

  /** @brief Return one time level without bounds checking. */
  FEMX_HOST_DEVICE VectorView<Space, T> operator[](Index level) const
  {
    return this->level(level);
  }

private:
  T*    data_{nullptr};
  Index num_levels_{0};
  Index num_states_{0};
};

/** @brief Memory-space-native contiguous storage for all state time levels. */
template <MemorySpace Space>
class Trajectory
{
public:
  using VectorType = Vector<Space>;

  Trajectory() = default;

  Trajectory(Index num_steps, Index num_states)
  {
    resize(num_steps, num_states);
  }

  Trajectory(const Trajectory&)                = default;
  Trajectory(Trajectory&&) noexcept            = default;
  Trajectory& operator=(const Trajectory&)     = default;
  Trajectory& operator=(Trajectory&&) noexcept = default;

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

  TrajectoryView<Space, Real> view() noexcept
  {
    return {data(), numTimeLevels(), numStates()};
  }

  TrajectoryView<Space, const Real> view() const noexcept
  {
    return {data(), numTimeLevels(), numStates()};
  }

  VectorView<Space, Real> level(Index level)
  {
    checkLevel(level);
    return view().level(level);
  }

  VectorView<Space, const Real> level(Index level) const
  {
    checkLevel(level);
    return view().level(level);
  }

  VectorView<Space, Real> operator[](Index level)
  {
    return this->level(level);
  }

  VectorView<Space, const Real> operator[](Index level) const
  {
    return this->level(level);
  }

  template <MemorySpace S                                              = Space,
            typename std::enable_if<S == MemorySpace::Host, int>::type = 0>
  void setZero()
  {
    data_.setZero();
  }

  template <MemorySpace S                                                = Space,
            typename std::enable_if<S == MemorySpace::Device, int>::type = 0>
  void setZero(CudaContext& ctx)
  {
    data_.setZero(ctx);
  }

private:
  static Index checkedSize(Index num_steps, Index num_states)
  {
    require(num_steps >= 0 && num_states >= 0,
            "Trajectory received invalid dimensions");

    const std::int64_t size =
        (static_cast<std::int64_t>(num_steps) + 1) * num_states;
    require(size <= std::numeric_limits<Index>::max(),
            "Trajectory exceeds the Index range");
    return static_cast<Index>(size);
  }

  void checkLevel(Index level) const
  {
    require(level >= 0 && level < numTimeLevels(),
            "Trajectory level is out of range");
  }

private:
  VectorType data_;
  Index      num_steps_{0};
  Index      num_states_{0};
};

/** @brief Conventional host trajectory used by current state interfaces. */
using TimeTrajectory = Trajectory<MemorySpace::Host>;

/** @brief CUDA device trajectory with the same layout and view contract. */
using DeviceTimeTrajectory = Trajectory<MemorySpace::Device>;

/**
 * @brief Copy a complete trajectory across a boundary involving Device memory.
 *
 * The destination retains an existing same-sized allocation. Source and
 * destination storage must remain alive until queued work on `ctx` completes.
 */
template <MemorySpace Src,
          MemorySpace Dst,
          typename std::enable_if<Src != MemorySpace::Host
                                      || Dst != MemorySpace::Host,
                                  int>::type = 0>
void copy(const Trajectory<Src>& src,
          Trajectory<Dst>&       dst,
          CudaContext&           ctx)
{
  if constexpr (Src == Dst)
  {
    if (&src == &dst)
    {
      return;
    }
  }

  dst.resize(src.numSteps(), src.numStates());
  if (src.size() > 0)
  {
    device::copy(dst.data(),
                 Dst,
                 src.data(),
                 Src,
                 static_cast<std::size_t>(src.size()) * sizeof(Real),
                 ctx.stream());
  }
}

template <MemorySpace Src,
          MemorySpace Dst,
          typename std::enable_if<Src != MemorySpace::Host
                                      || Dst != MemorySpace::Host,
                                  int>::type = 0>
void copy(Trajectory<Src>&&, Trajectory<Dst>&, CudaContext&) = delete;

} // namespace state
} // namespace femx
