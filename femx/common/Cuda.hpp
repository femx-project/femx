#pragma once

#include <cstddef>

#include <femx/common/Types.hpp>

namespace femx::cuda
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
void  sync(void* stream);
void  checkLastError();

} // namespace femx::cuda
