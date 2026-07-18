#pragma once

#include <cstddef>
#include <utility>

#include <femx/common/Types.hpp>

namespace femx
{

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

/** @brief Tag selecting serial CPU execution. */
struct CpuContext
{
};

/** @brief Owner of the CUDA stream used by backend operations. */
class CudaContext
{
public:
  CudaContext()
    : stream_(device::createStream())
  {
  }

  ~CudaContext()
  {
    device::destroyStream(stream_);
  }

  CudaContext(const CudaContext&)            = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  CudaContext(CudaContext&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr))
  {
  }

  CudaContext& operator=(CudaContext&& other) noexcept
  {
    if (this != &other)
    {
      device::destroyStream(stream_);
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  void* stream() const noexcept
  {
    return stream_;
  }

  void synchronize() const
  {
    device::synchronize(stream_);
  }

  static bool available() noexcept
  {
    return device::available();
  }

private:
  void* stream_{nullptr};
};

} // namespace femx
