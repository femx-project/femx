#pragma once

#include <cstddef>
#include <stdexcept>

#include <femx/core/Types.hpp>

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

private:
  T*    data_ = nullptr;
  Index size_ = 0;
};

} // namespace femx
