#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <femx/common/Types.hpp>

namespace femx
{

class Vector
{
public:
  Vector() = default;

  explicit Vector(Index size)
    : values_(static_cast<std::size_t>(size), Real{})
  {
  }

  void resize(Index size)
  {
    values_.assign(static_cast<std::size_t>(size), Real{});
  }

  Index size() const
  {
    return static_cast<Index>(values_.size());
  }

  Real& operator[](Index i)
  {
    return values_[static_cast<std::size_t>(i)];
  }

  Real operator[](Index i) const
  {
    return values_[static_cast<std::size_t>(i)];
  }

  Real* data()
  {
    return values_.data();
  }

  const Real* data() const
  {
    return values_.data();
  }

  void setZero()
  {
    std::fill(values_.begin(), values_.end(), Real{});
  }

private:
  std::vector<Real> values_;
};

} // namespace femx
