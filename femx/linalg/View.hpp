#pragma once

#include <cstddef>
#include <cstdint>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>

namespace femx
{

/** @brief Provide a non-owning view of a contiguous array. */
template <MemorySpace Space, class T>
class VectorView
{
public:
  FEMX_HOST_DEVICE VectorView() = default;

  /**
   * @brief Construct a view from contiguous storage.
   *
   * @param[in] data - Address of the first value.
   * @param[in] size - Number of viewed values.
   */
  FEMX_HOST_DEVICE VectorView(T* data, Index size)
    : data_(data), size_(size)
  {
  }

  /**
   * @brief Convert a compatible vector view.
   *
   * @param[in] other - Source view.
   */
  template <class U>
  FEMX_HOST_DEVICE VectorView(const VectorView<Space, U>& other)
    : data_(other.data()), size_(other.size())
  {
  }

  /**
   * @brief Access a value without bounds checking.
   *
   * @param[in] i - Value index.
   * @return Reference to the indexed value.
   */
  FEMX_HOST_DEVICE T& operator[](Index i) const
  {
    return data_[i];
  }

  /**
   * @brief Return the address of the first viewed value.
   *
   * @return Pointer to the first value, or `nullptr` for an empty default view.
   */
  FEMX_HOST_DEVICE T* data() const
  {
    return data_;
  }

  /** @brief Return the number of viewed values. */
  FEMX_HOST_DEVICE Index size() const
  {
    return size_;
  }

  /** @brief Report whether the view is empty. */
  FEMX_HOST_DEVICE bool empty() const
  {
    return size_ == 0;
  }

  /**
   * @brief Report whether the pointer and size form a valid view.
   *
   * @return `true` when the size is nonnegative and any nonempty view has data.
   */
  FEMX_HOST_DEVICE bool isValid() const
  {
    return size_ >= 0 && (size_ == 0 || data_ != nullptr);
  }

  /**
   * @brief Return a subview without bounds checking.
   *
   * @param[in] offset - Offset of the first value in the subview.
   * @param[in] count - Number of values in the subview.
   * @return View of the requested range.
   */
  FEMX_HOST_DEVICE VectorView subview(Index offset, Index count) const
  {
    return VectorView(data_ + offset, count);
  }

  /** @brief Return a Host iterator to the first viewed value. */
  T* begin() const
  {
    static_assert(Space == MemorySpace::Host,
                  "Device views are not host-iterable");
    return data_;
  }

  /** @brief Return a Host iterator past the last viewed value. */
  T* end() const
  {
    return begin() + size_;
  }

  /**
   * @brief Copy same-sized Host values into this view.
   *
   * @param[in] vals - Values to copy.
   * @return This view.
   * @throws std::runtime_error - If the source and destination sizes differ.
   */
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

private:
  T*    data_{nullptr}; ///< Address of the first viewed value.
  Index size_{0};       ///< Number of viewed values.
};

/** @brief Provide a non-owning row-major view of equal-sized vector blocks. */
template <MemorySpace Space, class T>
class BlockVectorView
{
public:
  FEMX_HOST_DEVICE BlockVectorView() = default;

  /**
   * @brief Construct a view of equal-sized contiguous blocks.
   *
   * @param[in] data - Address of the first value.
   * @param[in] blocks - Number of blocks.
   * @param[in] block_size - Number of values per block.
   */
  FEMX_HOST_DEVICE BlockVectorView(T*    data,
                                   Index blocks,
                                   Index block_size)
    : data_(data), blocks_(blocks), block_size_(block_size)
  {
  }

  /**
   * @brief Access a value without bounds checking.
   *
   * @param[in] block - Block index.
   * @param[in] i - Value index within the block.
   * @return Reference to the indexed value.
   */
  FEMX_HOST_DEVICE T& operator()(Index block, Index i) const
  {
    return data_[block * block_size_ + i];
  }

  /**
   * @brief Return one block without bounds checking.
   *
   * @param[in] i - Block index.
   * @return View of the selected block.
   */
  FEMX_HOST_DEVICE VectorView<Space, T> block(Index i) const
  {
    return {data_ + i * block_size_, block_size_};
  }

  /** @brief Return a flat view of all values. */
  FEMX_HOST_DEVICE VectorView<Space, T> flat() const
  {
    return {data_, size()};
  }

  /** @brief Return the address of the first viewed value. */
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

  /** @brief Return the total number of values. */
  FEMX_HOST_DEVICE Index size() const
  {
    return blocks_ * block_size_;
  }

private:
  T*    data_{nullptr}; ///< Address of the first viewed value.
  Index blocks_{0};     ///< Number of blocks.
  Index block_size_{0}; ///< Number of values per block.
};

/** @brief Provide a non-owning row-major dense matrix view. */
template <MemorySpace Space, class T>
class MatrixView
{
public:
  MatrixView() = default;

  /**
   * @brief Construct a row-major matrix view.
   *
   * @param[in] data - Address of the first entry.
   * @param[in] rows - Number of rows.
   * @param[in] cols - Number of columns.
   */
  MatrixView(T* data, Index rows, Index cols)
    : data_(data), rows_(rows), cols_(cols)
  {
  }

  /**
   * @brief Convert a compatible matrix view.
   *
   * @param[in] other - Source view.
   */
  template <class U>
  MatrixView(const MatrixView<Space, U>& other)
    : data_(other.data()), rows_(other.rows()), cols_(other.cols())
  {
  }

  /**
   * @brief Access an entry without bounds checking.
   *
   * @param[in] i - Row index.
   * @param[in] j - Column index.
   * @return Reference to the indexed entry.
   */
  T& operator()(Index i, Index j) const
  {
    return data_[i * cols_ + j];
  }

  /** @brief Return the address of the first viewed entry. */
  T* data() const
  {
    return data_;
  }

  /** @brief Return the number of rows. */
  Index rows() const
  {
    return rows_;
  }

  /** @brief Return the number of columns. */
  Index cols() const
  {
    return cols_;
  }

private:
  T*    data_{nullptr}; ///< Address of the first viewed entry.
  Index rows_{0};       ///< Number of rows.
  Index cols_{0};       ///< Number of columns.
};

/** @brief Provide a Host dense matrix view. */
template <class T>
using HostMatrixView = MatrixView<MemorySpace::Host, T>;
/** @brief Provide a Device dense matrix view. */
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
