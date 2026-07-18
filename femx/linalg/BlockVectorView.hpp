#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{

/** @brief Non-owning row-major view of equal-sized host vector blocks. */
template <class T>
class BlockVectorView
{
public:
  BlockVectorView() = default;

  /** @brief View `blocks` contiguous blocks of `block_size` values. */
  BlockVectorView(T* data, Index blocks, Index block_size)
    : data_(data),
      blocks_(blocks),
      block_size_(block_size)
  {
    if (blocks_ < 0 || block_size_ < 0)
    {
      throw std::runtime_error("BlockVectorView received invalid dimensions");
    }
    if (blocks_ > 0 && block_size_ > 0 && data_ == nullptr)
    {
      throw std::runtime_error("BlockVectorView received null data");
    }
  }

  /** @brief Access one value without bounds checking. */
  T& operator()(Index block, Index i) const
  {
    return data_[block * block_size_ + i];
  }

  /** @brief Return a non-owning view of block `i`. */
  HostArrayView<T> block(Index i) const
  {
    return HostArrayView<T>(data_ + i * block_size_, block_size_);
  }

  /** @brief Return a flat view of all blocks. */
  HostArrayView<T> flat() const
  {
    return HostArrayView<T>(data_, size());
  }

  /** @brief Return the first viewed address. */
  T* data() const
  {
    return data_;
  }

  /** @brief Return the number of blocks. */
  Index blocks() const
  {
    return blocks_;
  }

  /** @brief Return the number of values per block. */
  Index blockSize() const
  {
    return block_size_;
  }

  /** @brief Return the total number of viewed values. */
  Index size() const
  {
    return blocks_ * block_size_;
  }

private:
  T*    data_       = nullptr;
  Index blocks_     = 0;
  Index block_size_ = 0;
};

} // namespace femx
