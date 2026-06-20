#pragma once

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <femx/core/Types.hpp>

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

  Vector(const Vector& other)
    : values_(other.begin(), other.end())
  {
  }

  Vector(Vector&& other) noexcept
    : values_(std::move(other.values_)),
      view_data_(other.view_data_),
      view_size_(other.view_size_)
  {
    other.view_data_ = nullptr;
    other.view_size_ = 0;
  }

  Vector& operator=(const Vector& other)
  {
    assign(other.begin(), other.end(), other.size());
    return *this;
  }

  Vector& operator=(Vector&& other)
  {
    if (isView())
    {
      assign(other.begin(), other.end(), other.size());
    }
    else if (other.isView())
    {
      values_.assign(other.begin(), other.end());
    }
    else
    {
      values_ = std::move(other.values_);
    }
    return *this;
  }

  Vector& operator=(std::initializer_list<T> values)
  {
    assign(values.begin(),
           values.end(),
           static_cast<Index>(values.size()));
    return *this;
  }

  static Vector view(T*    data,
                     Index size)
  {
    if (size < 0)
    {
      throw std::runtime_error("Vector view size must be non-negative");
    }
    Vector out;
    out.view_data_ = data;
    out.view_size_ = size;
    return out;
  }

  void resize(Index size)
  {
    if (size < 0)
    {
      throw std::runtime_error("Vector size must be non-negative");
    }
    if (isView())
    {
      if (size != view_size_)
      {
        throw std::runtime_error("Cannot resize a Vector view");
      }
      setZero();
      return;
    }
    values_.assign(static_cast<std::size_t>(size), T{});
  }

  Index size() const
  {
    return isView() ? view_size_ : static_cast<Index>(values_.size());
  }

  bool empty() const
  {
    return size() == 0;
  }

  void clear()
  {
    ensureOwning("clear");
    values_.clear();
  }

  void reserve(Index size)
  {
    ensureOwning("reserve");
    values_.reserve(static_cast<std::size_t>(size));
  }

  void push_back(const T& value)
  {
    ensureOwning("push_back");
    values_.push_back(value);
  }

  T& front()
  {
    return data()[0];
  }

  T front() const
  {
    return data()[0];
  }

  T& back()
  {
    return data()[size() - 1];
  }

  T back() const
  {
    return data()[size() - 1];
  }

  T& operator[](Index i)
  {
    return data()[static_cast<std::size_t>(i)];
  }

  T operator[](Index i) const
  {
    return data()[static_cast<std::size_t>(i)];
  }

  T* data()
  {
    return isView() ? view_data_ : values_.data();
  }

  const T* data() const
  {
    return isView() ? view_data_ : values_.data();
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
  bool isView() const
  {
    return view_data_ != nullptr;
  }

  void ensureOwning(const char* operation) const
  {
    if (isView())
    {
      throw std::runtime_error(std::string("Cannot ") + operation
                               + " a Vector view");
    }
  }

  template <typename It>
  void assign(It    begin_it,
              It    end_it,
              Index input_size)
  {
    if (isView())
    {
      if (input_size != view_size_)
      {
        throw std::runtime_error("Vector view assignment size mismatch");
      }
      std::copy(begin_it, end_it, data());
      return;
    }
    values_.assign(begin_it, end_it);
  }

  std::vector<T> values_;
  T*             view_data_{nullptr};
  Index          view_size_{0};
};

} // namespace femx
