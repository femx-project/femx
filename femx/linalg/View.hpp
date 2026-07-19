#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <femx/common/Checks.hpp>
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
    require(vals.size() == size_,
            "VectorView assignment size mismatch");
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

/** @brief Non-owning row-major view of equal-sized vector blocks. */
template <MemorySpace Space, class T>
class BlockVectorView
{
public:
  FEMX_HOST_DEVICE BlockVectorView() = default;

  /** @brief View `blocks` contiguous blocks of `block_size` values. */
  FEMX_HOST_DEVICE BlockVectorView(T*    data,
                                   Index blocks,
                                   Index block_size)
    : data_(data), blocks_(blocks), block_size_(block_size)
  {
  }

  /** @brief Access one value without bounds checking. */
  FEMX_HOST_DEVICE T& operator()(Index block, Index i) const
  {
    return data_[block * block_size_ + i];
  }

  /** @brief Return a non-owning view of block `i`. */
  FEMX_HOST_DEVICE VectorView<Space, T> block(Index i) const
  {
    return {data_ + i * block_size_, block_size_};
  }

  /** @brief Return a flat view of all blocks. */
  FEMX_HOST_DEVICE VectorView<Space, T> flat() const
  {
    return {data_, size()};
  }

  /** @brief Return the first viewed address. */
  FEMX_HOST_DEVICE T* data() const
  {
    return data_;
  }

  /** @brief Return the number of blocks. */
  FEMX_HOST_DEVICE Index blocks() const
  {
    return blocks_;
  }

  /** @brief Return the number of values per block. */
  FEMX_HOST_DEVICE Index blockSize() const
  {
    return block_size_;
  }

  /** @brief Return the total number of viewed values. */
  FEMX_HOST_DEVICE Index size() const
  {
    return blocks_ * block_size_;
  }

private:
  T*    data_{nullptr};
  Index blocks_{0};
  Index block_size_{0};
};

/** @brief Non-owning row-major dense matrix view in one memory space. */
template <MemorySpace Space, class T>
class MatrixView
{
public:
  MatrixView() = default;

  /** @brief View a contiguous `rows` by `cols` row-major matrix. */
  MatrixView(T* data, Index rows, Index cols)
    : data_(data), rows_(rows), cols_(cols)
  {
  }

  /** @brief Convert a compatible matrix view without changing ownership. */
  template <class U>
  MatrixView(const MatrixView<Space, U>& other)
    : data_(other.data()), rows_(other.rows()), cols_(other.cols())
  {
  }

  /** @brief Access entry `(i, j)` without bounds checking. */
  T& operator()(Index i, Index j) const
  {
    return data_[i * cols_ + j];
  }

  /** @brief Return the first viewed address. */
  T* data() const
  {
    return data_;
  }

  /** @brief Return the row count. */
  Index rows() const
  {
    return rows_;
  }

  /** @brief Return the column count. */
  Index cols() const
  {
    return cols_;
  }

private:
  T*    data_{nullptr};
  Index rows_{0};
  Index cols_{0};
};

template <class T>
using HostMatrixView = MatrixView<MemorySpace::Host, T>;

template <class T>
using DeviceMatrixView = MatrixView<MemorySpace::Device, T>;

/// @cond INTERNAL
namespace detail
{

/** @brief Return whether two contiguous pointer ranges share any byte. */
template <class T, class U>
inline bool overlaps(const T* lhs,
                     Index    lhs_size,
                     const U* rhs,
                     Index    rhs_size) noexcept
{
  if (lhs == nullptr || rhs == nullptr || lhs_size <= 0 || rhs_size <= 0)
  {
    return false;
  }
  const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
  const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
  const auto lhs_end   = lhs_begin + static_cast<std::size_t>(lhs_size) * sizeof(T);
  const auto rhs_end   = rhs_begin + static_cast<std::size_t>(rhs_size) * sizeof(U);
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

/** @brief Return whether two contiguous views share any byte. */
template <MemorySpace Space, class T, class U>
inline bool overlaps(VectorView<Space, T> lhs,
                     VectorView<Space, U> rhs) noexcept
{
  return overlaps(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}

} // namespace detail

/// @endcond

} // namespace femx
