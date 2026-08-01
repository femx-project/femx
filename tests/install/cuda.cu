#include <cuda_runtime.h>

#include <femx/common/Vector.hpp>

namespace
{

__global__ void setValue(int* value)
{
  *value = 1;
}

} // namespace

int main()
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
  {
    cudaGetLastError();
    return 0;
  }

  femx::DeviceVector<int> value(1);

  setValue<<<1, 1>>>(value.data());
  if (cudaDeviceSynchronize() != cudaSuccess)
  {
    return 1;
  }

  int result = 0;
  if (cudaMemcpy(&result,
                 value.data(),
                 sizeof(result),
                 cudaMemcpyDeviceToHost)
      != cudaSuccess)
  {
    return 1;
  }
  return result == 1 ? 0 : 1;
}
