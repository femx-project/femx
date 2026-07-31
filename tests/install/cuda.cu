#include <cuda_runtime.h>

namespace
{

__global__ void setValue(int* value)
{
  *value = 1;
}

} // namespace

int main()
{
  int* value = nullptr;
  if (cudaMallocManaged(&value, sizeof(int)) != cudaSuccess)
  {
    return 1;
  }

  setValue<<<1, 1>>>(value);
  if (cudaDeviceSynchronize() != cudaSuccess)
  {
    cudaFree(value);
    return 1;
  }

  const int result = *value;
  cudaFree(value);
  return result == 1 ? 0 : 1;
}
