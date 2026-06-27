#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{
namespace state
{

/** @brief State values on all time levels of a time-marching solve. */
class TimeTrajectory
{
public:
  TimeTrajectory() = default;

  TimeTrajectory(Index nt,
                 Index nst)
  {
    resize(nt, nst);
  }

  void resize(Index nt,
              Index nst)
  {
    if (nt < 0 || nst < 0)
    {
      throw std::runtime_error("TimeTrajectory received invalid dimensions");
    }

    nt_  = nt;
    nst_ = nst;
    data_.resize((nt_ + 1) * nst_);
  }

  bool empty() const
  {
    return data_.empty();
  }

  Index numSteps() const
  {
    return empty() ? 0 : nt_;
  }

  Index numTimeLevels() const
  {
    return empty() ? 0 : nt_ + 1;
  }

  Index numLevels() const
  {
    return numTimeLevels();
  }

  Index numStates() const
  {
    return empty() ? 0 : nst_;
  }

  VectorView<Real> operator[](Index level)
  {
    checkLevel(level);
    return VectorView<Real>(data_.data() + level * nst_, nst_);
  }

  VectorView<const Real> operator[](Index level) const
  {
    checkLevel(level);
    return VectorView<const Real>(data_.data() + level * nst_, nst_);
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
  Index        nt_{0};
  Index        nst_{0};
};

} // namespace state
} // namespace femx
