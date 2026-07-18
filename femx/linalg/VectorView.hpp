#pragma once

#include <algorithm>
#include <stdexcept>

#include <femx/common/Types.hpp>

namespace femx
{

/** @brief Lightweight non-owning view of a memory-space-specific array. */
template <MemorySpace Space, class T>
class VectorView
{
public:
  FEMX_HOST_DEVICE VectorView() = default;

  FEMX_HOST_DEVICE VectorView(T* data, Index size)
    : data_(data), size_(size)
  {
  }

  template <class U>
  FEMX_HOST_DEVICE VectorView(const VectorView<Space, U>& other)
    : data_(other.data()), size_(other.size())
  {
  }

  FEMX_HOST_DEVICE T& operator[](Index i) const
  {
    return data_[i];
  }

  FEMX_HOST_DEVICE T* data() const
  {
    return data_;
  }

  FEMX_HOST_DEVICE Index size() const
  {
    return size_;
  }

  FEMX_HOST_DEVICE bool empty() const
  {
    return size_ == 0;
  }

  FEMX_HOST_DEVICE VectorView subview(Index offset, Index count) const
  {
    return VectorView(data_ + offset, count);
  }

  T* begin() const
  {
    static_assert(Space == MemorySpace::Host,
                  "Device views are not host-iterable");
    return data_;
  }

  T* end() const
  {
    return begin() + size_;
  }

  template <class Values>
  VectorView& operator=(const Values& vals)
  {
    static_assert(Space == MemorySpace::Host,
                  "Assign device views with an explicit copy operation");
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

  void setZero() const
  {
    static_assert(Space == MemorySpace::Host,
                  "Zero device views through a CudaContext");
    std::fill(begin(), end(), T{});
  }

private:
  T*    data_{nullptr};
  Index size_{0};
};

} // namespace femx
