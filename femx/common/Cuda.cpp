#include <stdexcept>

#include <femx/common/Cuda.hpp>

namespace femx::cuda
{

namespace
{
[[noreturn]] void unavailable()
{
  throw std::runtime_error(
      "femx was built without CUDA execution support");
}
} // namespace

bool available() noexcept
{
  return false;
}

void* createStream()
{
  unavailable();
}

void destroyStream(void*) noexcept
{
}

void checkLastError()
{
  unavailable();
}

} // namespace femx::cuda
