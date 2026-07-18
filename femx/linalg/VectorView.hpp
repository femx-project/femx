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

  /** @brief View `size` contiguous values beginning at `data`. */
  FEMX_HOST_DEVICE VectorView(T* data, Index size)
    : data_(data), size_(size)
  {
  }

  /** @brief Convert a compatible view without changing ownership. */
  template <class U>
  FEMX_HOST_DEVICE VectorView(const VectorView<Space, U>& other)
    : data_(other.data()), size_(other.size())
  {
  }

  /** @brief Access value `i` without bounds checking. */
  FEMX_HOST_DEVICE T& operator[](Index i) const
  {
    return data_[i];
  }

  /** @brief Return the first viewed address in `Space`. */
  FEMX_HOST_DEVICE T* data() const
  {
    return data_;
  }

  /** @brief Return the number of viewed values. */
  FEMX_HOST_DEVICE Index size() const
  {
    return size_;
  }

  /** @brief Return whether the view is empty. */
  FEMX_HOST_DEVICE bool empty() const
  {
    return size_ == 0;
  }

  /** @brief Return a subview without bounds checking. */
  FEMX_HOST_DEVICE VectorView subview(Index offset, Index count) const
  {
    return VectorView(data_ + offset, count);
  }

  /** @brief Return the first host iterator. */
  T* begin() const
  {
    static_assert(Space == MemorySpace::Host,
                  "Device views are not host-iterable");
    return data_;
  }

  /** @brief Return the one-past-last host iterator. */
  T* end() const
  {
    return begin() + size_;
  }

  /** @brief Copy same-sized host values into this view. */
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

  /** @brief Set all viewed host values to zero. */
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
