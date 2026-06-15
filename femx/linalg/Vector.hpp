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

  bool empty() const
  {
    return values_.empty();
  }

  Real& front()
  {
    return values_.front();
  }

  Real front() const
  {
    return values_.front();
  }

  Real& back()
  {
    return values_.back();
  }

  Real back() const
  {
    return values_.back();
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

  auto begin()
  {
    return values_.begin();
  }

  auto begin() const
  {
    return values_.begin();
  }

  auto end()
  {
    return values_.end();
  }

  auto end() const
  {
    return values_.end();
  }

  void setZero()
  {
    std::fill(values_.begin(), values_.end(), Real{});
  }

private:
  std::vector<Real> values_;
};

} // namespace femx
