#pragma once

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include <femx/common/Types.hpp>

namespace femx
{

template <typename T>
class Vector
{
public:
  Vector() = default;

  Vector(std::initializer_list<T> values)
    : values_(values)
  {
  }

  explicit Vector(Index size)
    : values_(static_cast<std::size_t>(size), T{})
  {
  }

  void resize(Index size)
  {
    values_.assign(static_cast<std::size_t>(size), T{});
  }

  Index size() const
  {
    return static_cast<Index>(values_.size());
  }

  bool empty() const
  {
    return values_.empty();
  }

  void clear()
  {
    values_.clear();
  }

  void reserve(Index size)
  {
    values_.reserve(static_cast<std::size_t>(size));
  }

  void push_back(const T& value)
  {
    values_.push_back(value);
  }

  T& front()
  {
    return values_.front();
  }

  T front() const
  {
    return values_.front();
  }

  T& back()
  {
    return values_.back();
  }

  T back() const
  {
    return values_.back();
  }

  T& operator[](Index i)
  {
    return values_[static_cast<std::size_t>(i)];
  }

  T operator[](Index i) const
  {
    return values_[static_cast<std::size_t>(i)];
  }

  T* data()
  {
    return values_.data();
  }

  const T* data() const
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
    std::fill(values_.begin(), values_.end(), T{});
  }

private:
  std::vector<T> values_;
};

} // namespace femx
