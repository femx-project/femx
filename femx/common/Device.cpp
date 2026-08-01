#include <stdexcept>

#include <femx/common/Device.hpp>

namespace femx::device
{

namespace
{
[[noreturn]] void unavailable()
{
  throw std::runtime_error(
      "femx was built without Device execution support");
}
} // namespace

void* allocate(std::size_t bytes)
{
  if (bytes != 0)
  {
    unavailable();
  }
  return nullptr;
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

void fill(Real*, Index size, Real, void*)
{
  if (size != 0)
  {
    unavailable();
  }
}

void fill(Index*, Index size, Index, void*)
{
  if (size != 0)
  {
    unavailable();
  }
}

void sync(void*)
{
  unavailable();
}

} // namespace femx::device
