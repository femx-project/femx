#include <stdexcept>

#include <femx/common/Cuda.hpp>

namespace femx::cuda
{

namespace
{
[[noreturn]] void unavailable()
{
  throw std::runtime_error(
      "femx was built without the CUDA execution backend");
}
} // namespace

bool available() noexcept
{
  return false;
}

void* allocate(std::size_t)
{
  unavailable();
}

void release(void*) noexcept
{
}

void copy(void*, MemorySpace, const void*, MemorySpace, std::size_t bytes, void*)
{
  if (bytes != 0)
  {
    unavailable();
  }
}

void zero(void*, std::size_t bytes, void*)
{
  if (bytes != 0)
  {
    unavailable();
  }
}

void* createStream()
{
  unavailable();
}

void destroyStream(void*) noexcept
{
}

void sync(void*)
{
  unavailable();
}

void checkLastError()
{
  unavailable();
}

} // namespace femx::cuda
