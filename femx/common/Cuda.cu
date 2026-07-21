#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include <femx/common/Cuda.hpp>

namespace femx::cuda
{

namespace
{
void check(cudaError_t status, const char* op)
{
  if (status != cudaSuccess)
  {
    throw std::runtime_error(std::string(op) + ": "
                             + cudaGetErrorString(status));
  }
}

cudaStream_t asStream(void* stream)
{
  return static_cast<cudaStream_t>(stream);
}

cudaMemcpyKind copyKind(MemorySpace dst, MemorySpace src)
{
  if (src == MemorySpace::Host && dst == MemorySpace::Device)
  {
    return cudaMemcpyHostToDevice;
  }
  if (src == MemorySpace::Device && dst == MemorySpace::Host)
  {
    return cudaMemcpyDeviceToHost;
  }
  if (src == MemorySpace::Device && dst == MemorySpace::Device)
  {
    return cudaMemcpyDeviceToDevice;
  }
  return cudaMemcpyHostToHost;
}
} // namespace

bool available() noexcept
{
  int        count  = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess)
  {
    cudaGetLastError();
    return false;
  }
  return count > 0;
}

void* allocate(std::size_t bytes)
{
  if (bytes == 0)
  {
    return nullptr;
  }
  void* ptr = nullptr;
  check(cudaMalloc(&ptr, bytes), "cudaMalloc failed");
  return ptr;
}

void release(void* ptr) noexcept
{
  if (ptr != nullptr)
  {
    cudaFree(ptr);
  }
}

void copy(void*       dst,
          MemorySpace dst_memspace,
          const void* src,
          MemorySpace src_memspace,
          std::size_t bytes,
          void*       stream)
{
  if (bytes == 0)
  {
    return;
  }
  const cudaMemcpyKind kind = copyKind(dst_memspace, src_memspace);
  if (stream != nullptr)
  {
    check(cudaMemcpyAsync(dst, src, bytes, kind, asStream(stream)),
          "cudaMemcpyAsync failed");
  }
  else
  {
    check(cudaMemcpy(dst, src, bytes, kind), "cudaMemcpy failed");
  }
}

void zero(void* ptr, std::size_t bytes, void* stream)
{
  if (bytes == 0)
  {
    return;
  }
  if (stream != nullptr)
  {
    check(cudaMemsetAsync(ptr, 0, bytes, asStream(stream)),
          "cudaMemsetAsync failed");
  }
  else
  {
    check(cudaMemset(ptr, 0, bytes), "cudaMemset failed");
  }
}

void* createStream()
{
  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags failed");
  return stream;
}

void destroyStream(void* stream) noexcept
{
  if (stream != nullptr)
  {
    cudaStreamDestroy(asStream(stream));
  }
}

void sync(void* stream)
{
  check(cudaStreamSynchronize(asStream(stream)),
        "cudaStreamSynchronize failed");
}

void checkLastError()
{
  check(cudaGetLastError(), "CUDA kernel launch failed");
}

} // namespace femx::cuda
