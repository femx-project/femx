#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include <femx/common/Types.hpp>

namespace femx
{

namespace detail
{
struct CudaContextAccess;
} // namespace detail

/// @cond INTERNAL
namespace device
{
bool available() noexcept;

void* allocate(std::size_t bytes);
void  release(void* ptr) noexcept;

void copy(void*       dst,
          MemorySpace dst_memspace,
          const void* src,
          MemorySpace src_memspace,
          std::size_t bytes,
          void*       stream = nullptr);

void zero(void* ptr, std::size_t bytes, void* stream = nullptr);

void* createStream();
void  destroyStream(void* stream) noexcept;
void  synchronize(void* stream);
void  checkLastError();
} // namespace device

/// @endcond

/** @brief Tag selecting serial CPU execution. */
struct CpuContext
{
  /** @brief Complete pending CPU work; serial execution is already complete. */
  void synchronize() const noexcept
  {
  }
};

/** @brief Owner of the CUDA stream used by backend operations. */
class CudaContext
{
public:
  /** @brief Create a context owning one non-blocking CUDA stream. */
  CudaContext()
    : stream_(device::createStream())
  {
  }

  /** @brief Destroy the owned stream after its queued work completes. */
  ~CudaContext()
  {
    // Backend workspaces can own resources associated with this stream and
    // therefore must be released before the stream itself.
    sparse_state_.reset();
    device::destroyStream(stream_);
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
      device::destroyStream(stream_);
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
  void synchronize() const
  {
    device::synchronize(stream_);
  }

  /** @brief Return whether a usable CUDA device is available. */
  static bool available() noexcept
  {
    return device::available();
  }

private:
  friend struct detail::CudaContextAccess;

  void*                         stream_{nullptr};
  mutable std::shared_ptr<void> sparse_state_;
};

} // namespace femx
