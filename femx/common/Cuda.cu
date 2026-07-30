#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include <femx/common/Cuda.hpp>

namespace femx::cuda
{

void check(cudaError_t status, const char* operation)
{
  if (status != cudaSuccess)
  {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
  }
}

namespace
{
constexpr unsigned int kThreads = 256;

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

template <class T>
__global__ void fillKernel(Index size, T val, T* out)
{
  const Index i =
      static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < size)
  {
    out[i] = val;
  }
}

template <class T>
bool hasZeroBytes(const T& val)
{
  const auto* begin = reinterpret_cast<const unsigned char*>(&val);
  return std::all_of(begin,
                     begin + sizeof(T),
                     [](unsigned char byte)
                     { return byte == 0; });
}

template <class T>
void fillValues(T* ptr, Index size, T val, void* stream)
{
  if (size <= 0)
  {
    return;
  }
  if (hasZeroBytes(val))
  {
    zero(ptr, static_cast<std::size_t>(size) * sizeof(T), stream);
    return;
  }

  fillKernel<<<numBlocks(size, kThreads),
               kThreads,
               0,
               asStream(stream)>>>(size, val, ptr);
  checkLastError();
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

void fill(Real* ptr, Index size, Real val, void* stream)
{
  fillValues(ptr, size, val, stream);
}

void fill(Index* ptr, Index size, Index val, void* stream)
{
  fillValues(ptr, size, val, stream);
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
