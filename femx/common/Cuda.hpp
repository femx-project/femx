#pragma once

#include <cstddef>

#include <femx/common/Types.hpp>

#if defined(__CUDACC__)
#include <cuda_runtime_api.h>
#endif

namespace femx::cuda
{

bool available() noexcept;

#if defined(__CUDACC__)
void check(cudaError_t status, const char* operation);
#endif

inline unsigned int numBlocks(Index        work_items,
                              unsigned int threads_per_block) noexcept
{
  if (work_items <= 0)
  {
    return 0;
  }
  const auto count   = static_cast<std::int64_t>(work_items);
  const auto threads = static_cast<std::int64_t>(threads_per_block);
  return static_cast<unsigned int>((count + threads - 1) / threads);
}

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
void  sync(void* stream);
void  checkLastError();

} // namespace femx::cuda
