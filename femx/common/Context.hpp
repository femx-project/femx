#pragma once

#include <memory>
#include <utility>

#include <femx/common/Cuda.hpp>

namespace femx
{

namespace detail
{
struct CudaContextAccess;
} // namespace detail

/** @brief Tag selecting serial CPU execution. */
struct CpuContext
{
  /** @brief Complete pending CPU work; serial execution is already complete. */
  void sync() const noexcept
  {
  }
};

/** @brief Owner of the CUDA stream used by backend operations. */
class CudaContext
{
public:
  /** @brief Create a context owning one non-blocking CUDA stream. */
  CudaContext()
    : stream_(cuda::createStream())
  {
  }

  /** @brief Destroy the owned stream after its queued work completes. */
  ~CudaContext()
  {
    // Backend workspaces can own resources associated with this stream and
    // therefore must be released before the stream itself.
    sparse_state_.reset();
    cuda::destroyStream(stream_);
  }

  CudaContext(const CudaContext&)            = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  /** @brief Transfer stream ownership from another context. */
  CudaContext(CudaContext&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)),
      sparse_state_(std::move(other.sparse_state_))
  {
  }

  /** @brief Replace this stream with one moved from another context. */
  CudaContext& operator=(CudaContext&& other) noexcept
  {
    if (this != &other)
    {
      sparse_state_.reset();
      cuda::destroyStream(stream_);
      stream_       = std::exchange(other.stream_, nullptr);
      sparse_state_ = std::move(other.sparse_state_);
    }
    return *this;
  }

  /** @brief Return the opaque native CUDA stream handle. */
  void* stream() const noexcept
  {
    return stream_;
  }

  /** @brief Wait for all work queued on this context. */
  void sync() const
  {
    cuda::sync(stream_);
  }

  /** @brief Return whether a usable CUDA device is available. */
  static bool available() noexcept
  {
    return cuda::available();
  }

private:
  friend struct detail::CudaContextAccess;

  void*                         stream_{nullptr};
  mutable std::shared_ptr<void> sparse_state_;
};

} // namespace femx
