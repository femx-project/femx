#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include <femx/common/Types.hpp>

namespace femx
{

template <class T>
class VectorView
{
public:
  VectorView() = default;

  VectorView(T* data, Index size)
    : data_(data), size_(size)
  {
  }

  template <class Values>
  VectorView& operator=(const Values& vals)
  {
    if (vals.size() != size_)
    {
      throw std::runtime_error("VectorView assignment size mismatch");
    }
    for (Index i = 0; i < size_; ++i)
    {
      data_[i] = vals[i];
    }
    return *this;
  }

  T& operator[](Index i) const
  {
    return data_[i];
  }

  T* data() const
  {
    return data_;
  }

  Index size() const
  {
    return size_;
  }

  bool empty() const
  {
    return size_ == 0;
  }

  T* begin() const
  {
    return data_;
  }

  T* end() const
  {
    return data_ + size_;
  }

  void setZero() const
  {
    std::fill(begin(), end(), T{});
  }

private:
  T*    data_ = nullptr;
  Index size_ = 0;
};

} // namespace femx
