#pragma once

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <femx/common/Types.hpp>

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
    : vals_(static_cast<std::size_t>(size), T{})
  {
  }

  Vector(Index size, const T& value)
    : vals_(static_cast<std::size_t>(size), value)
  {
  }

  Vector(const Vector& other)
    : vals_(other.begin(), other.end())
  {
  }

  Vector(Vector&& other) noexcept
    : vals_(std::move(other.vals_)),
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
      vals_.assign(other.begin(), other.end());
    }
    else
    {
      vals_ = std::move(other.vals_);
    }
    return *this;
  }

  Vector& operator=(std::initializer_list<T> vals)
  {
    assign(vals.begin(),
           vals.end(),
           static_cast<Index>(vals.size()));
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
    vals_.assign(static_cast<std::size_t>(size), T{});
  }

  void assign(Index size, const T& value)
  {
    if (size < 0)
    {
      throw std::runtime_error("Vector size must be non-negative");
    }
    if (isView())
    {
      if (size != view_size_)
      {
        throw std::runtime_error("Vector view assignment size mismatch");
      }
      std::fill(begin(), end(), value);
      return;
    }
    vals_.assign(static_cast<std::size_t>(size), value);
  }

  Index size() const
  {
    return isView() ? view_size_ : static_cast<Index>(vals_.size());
  }

  bool empty() const
  {
    return size() == 0;
  }

  void clear()
  {
    ensureOwning("clear");
    vals_.clear();
  }

  void reserve(Index size)
  {
    ensureOwning("reserve");
    vals_.reserve(static_cast<std::size_t>(size));
  }

  void push_back(const T& value)
  {
    ensureOwning("push_back");
    vals_.push_back(value);
  }

  void push_back(T&& value)
  {
    ensureOwning("push_back");
    vals_.push_back(std::move(value));
  }

  template <class... Args>
  T& emplace_back(Args&&... args)
  {
    ensureOwning("emplace_back");
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
    return isView() ? view_data_ : vals_.data();
  }

  const T* data() const
  {
    return isView() ? view_data_ : vals_.data();
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
    vals_.assign(begin_it, end_it);
  }

  std::vector<T> vals_;
  T*             view_data_{nullptr};
  Index          view_size_{0};
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
