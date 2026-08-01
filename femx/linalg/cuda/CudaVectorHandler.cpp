#include <stdexcept>

#include <femx/common/Device.hpp>
#include <femx/linalg/cuda/CudaVectorHandler.hpp>

namespace femx::linalg
{

#if !defined(FEMX_HAS_CUDA)
namespace
{
[[noreturn]] void cudaUnavailable()
{
  throw std::runtime_error(
      "femx was built without CUDA execution support");
}
} // namespace

void CudaVectorHandler::copy(DeviceVectorView<const Real>,
                             DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void CudaVectorHandler::copy(DeviceVectorView<const Real>,
                             DeviceVector<Real>&) const
{
  cudaUnavailable();
}

void CudaVectorHandler::copy(HostVectorView<const Real>,
                             DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void CudaVectorHandler::copy(HostVectorView<const Real>,
                             DeviceVector<Real>&) const
{
  cudaUnavailable();
}

void CudaVectorHandler::copy(DeviceVectorView<const Real>,
                             HostVectorView<Real>) const
{
  cudaUnavailable();
}

void CudaVectorHandler::copy(DeviceVectorView<const Real>,
                             HostVector<Real>&) const
{
  cudaUnavailable();
}

void CudaVectorHandler::assign(DeviceVector<Real>&, Index, Real) const
{
  cudaUnavailable();
}

void CudaVectorHandler::assign(DeviceVector<Index>&, Index, Index) const
{
  cudaUnavailable();
}

void CudaVectorHandler::zero(DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void CudaVectorHandler::axpby(Real,
                              DeviceVectorView<const Real>,
                              Real,
                              DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void CudaVectorHandler::gather(DeviceVectorView<const Real>,
                               DeviceVectorView<const Index>,
                               DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void CudaVectorHandler::scatter(DeviceVectorView<const Real>,
                                DeviceVectorView<const Index>,
                                DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void CudaVectorHandler::dot(DeviceVectorView<const Real>,
                            DeviceVectorView<const Real>,
                            DeviceVectorView<Real>) const
{
  cudaUnavailable();
}
#endif

void CudaVectorHandler::copyBytes(const void* src,
                                  MemorySpace src_space,
                                  void*       dst,
                                  MemorySpace dst_space,
                                  std::size_t bytes) const
{
  device::copy(dst, dst_space, src, src_space, bytes, stream());
}

} // namespace femx::linalg
