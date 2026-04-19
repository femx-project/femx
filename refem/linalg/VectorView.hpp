#pragma once

#include <cstddef>
#include <stdexcept>

#include <refem/common/Types.hpp>

namespace refem
{

template <class T>
class VectorView
{
public:
  VectorView() = default;

  VectorView(T* data, index_type size)
    : data_(data), size_(size)
  {
  }

  T& operator[](index_type i) const
  {
    return data_[i];
  }

  T* data() const
  {
    return data_;
  }

  index_type size() const
  {
    return size_;
  }

private:
  T*         data_ = nullptr;
  index_type size_ = 0;
};

} // namespace refem
