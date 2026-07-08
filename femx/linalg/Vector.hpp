#pragma once

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

#include <femx/common/Types.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{

template <typename T>
class Vector
{
public:
  Vector() = default;

  Vector(std::initializer_list<T> vals)
    : vals_(vals)
  {
  }

  explicit Vector(Index size)
    : vals_(checkedSize(size), T{})
  {
  }

  Vector(Index size, const T& value)
    : vals_(checkedSize(size), value)
  {
  }

  template <class U>
  Vector(VectorView<U> view)
  {
    assign(view);
  }

  Vector(const Vector&) = default;
  Vector(Vector&&) noexcept = default;

  Vector& operator=(const Vector&) = default;
  Vector& operator=(Vector&&) noexcept = default;

  Vector& operator=(std::initializer_list<T> vals)
  {
    vals_.assign(vals.begin(), vals.end());
    return *this;
  }

  template <class U>
  Vector& operator=(VectorView<U> view)
  {
    assign(view);
    return *this;
  }

  void resize(Index size)
  {
    vals_.assign(checkedSize(size), T{});
  }

  void assign(Index size, const T& value)
  {
    vals_.assign(checkedSize(size), value);
  }

  Index size() const
  {
    return static_cast<Index>(vals_.size());
  }

  bool empty() const
  {
    return size() == 0;
  }

  void clear()
  {
    vals_.clear();
  }

  void reserve(Index size)
  {
    vals_.reserve(checkedSize(size));
  }

  void push_back(const T& value)
  {
    vals_.push_back(value);
  }

  void push_back(T&& value)
  {
    vals_.push_back(std::move(value));
  }

  template <class... Args>
  T& emplace_back(Args&&... args)
  {
    return vals_.emplace_back(std::forward<Args>(args)...);
  }

  T& front()
  {
    return data()[0];
  }

  const T& front() const
  {
    return data()[0];
  }

  T& back()
  {
    return data()[size() - 1];
  }

  const T& back() const
  {
    return data()[size() - 1];
  }

  T& operator[](Index i)
  {
    return data()[static_cast<std::size_t>(i)];
  }

  const T& operator[](Index i) const
  {
    return data()[static_cast<std::size_t>(i)];
  }

  T* data()
  {
    return vals_.data();
  }

  const T* data() const
  {
    return vals_.data();
  }

  T* begin()
  {
    return data();
  }

  const T* begin() const
  {
    return data();
  }

  T* end()
  {
    return data() + size();
  }

  const T* end() const
  {
    return data() + size();
  }

  void setZero()
  {
    std::fill(begin(), end(), T{});
  }

private:
  static std::size_t checkedSize(Index size)
  {
    if (size < 0)
    {
      throw std::runtime_error("Vector size must be non-negative");
    }
    return static_cast<std::size_t>(size);
  }

  template <class U>
  void assign(VectorView<U> view)
  {
    const Index size = view.size();
    vals_.resize(checkedSize(size));
    for (Index i = 0; i < size; ++i)
    {
      vals_[static_cast<std::size_t>(i)] = view[i];
    }
  }

  std::vector<T> vals_;
};

template <typename T>
void resizeOrZero(Vector<T>& out, Index size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

} // namespace femx
