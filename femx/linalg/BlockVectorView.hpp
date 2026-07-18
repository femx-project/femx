#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{

template <class T>
class BlockVectorView
{
public:
  BlockVectorView() = default;

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

  T& operator()(Index block, Index i) const
  {
    return data_[block * block_size_ + i];
  }

  HostArrayView<T> block(Index i) const
  {
    return HostArrayView<T>(data_ + i * block_size_, block_size_);
  }

  HostArrayView<T> flat() const
  {
    return HostArrayView<T>(data_, size());
  }

  T* data() const
  {
    return data_;
  }

  Index blocks() const
  {
    return blocks_;
  }

  Index blockSize() const
  {
    return block_size_;
  }

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
