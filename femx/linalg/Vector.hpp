#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <femx/core/Types.hpp>

namespace femx
{

class Vector
{
public:
  Vector() = default;

  explicit Vector(index_type size)
    : values_(static_cast<std::size_t>(size), real_type{})
  {
  }

  void resize(index_type size)
  {
    values_.assign(static_cast<std::size_t>(size), real_type{});
  }

  index_type size() const
  {
    return static_cast<index_type>(values_.size());
  }

  real_type& operator[](index_type i)
  {
    return values_[static_cast<std::size_t>(i)];
  }

  real_type operator[](index_type i) const
  {
    return values_[static_cast<std::size_t>(i)];
  }

  real_type* data()
  {
    return values_.data();
  }

  const real_type* data() const
  {
    return values_.data();
  }

  void setZero()
  {
    std::fill(values_.begin(), values_.end(), real_type{});
  }

private:
  std::vector<real_type> values_;
};

} // namespace femx
