#include <stdexcept>

#include <femx/linalg/handler/VectorHandler.hpp>

namespace femx::linalg
{

void VectorHandler<HostCsrBackend>::zero(HostVectorView vals) const
{
  std::fill(vals.begin(), vals.end(), Real{});
}

void VectorHandler<HostCsrBackend>::axpby(Real                a,
                                          HostConstVectorView x,
                                          Real                b,
                                          HostVectorView      y) const
{
  require(x.size() == y.size(),
          "Host axpby requires equal vector sizes");
  require(x.data() == y.data() || !femx::detail::overlaps(x, y),
          "Host axpby does not support partial overlap");
  for (Index i = 0; i < x.size(); ++i)
  {
    y[i] = a * x[i] + b * y[i];
  }
}

Real VectorHandler<HostCsrBackend>::dot(HostConstVectorView x,
                                        HostConstVectorView y) const
{
  require(x.size() == y.size(), "Host dot requires equal vector sizes");
  Real val = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    val += x[i] * y[i];
  }
  return val;
}

Real VectorHandler<HostCsrBackend>::squaredNorm(
    HostConstVectorView x) const
{
  return dot(x, x);
}

void VectorHandler<HostCsrBackend>::gather(HostConstVectorView src,
                                           HostConstIndexView  indices,
                                           HostVectorView      dst) const
{
  require(indices.size() == dst.size(),
          "Host gather output size mismatch");
  require(!femx::detail::overlaps(src, dst),
          "Host gather does not support aliased vectors");
  for (Index i = 0; i < indices.size(); ++i)
  {
    require(indices[i] >= 0 && indices[i] < src.size(),
            "Host gather index is out of range");
    dst[i] = src[indices[i]];
  }
}

void VectorHandler<HostCsrBackend>::scatter(HostConstVectorView src,
                                            HostConstIndexView  indices,
                                            HostVectorView      dst) const
{
  require(src.size() == indices.size(),
          "Host scatter input size mismatch");
  require(!femx::detail::overlaps(src, dst),
          "Host scatter does not support aliased vectors");
  for (Index i = 0; i < indices.size(); ++i)
  {
    require(indices[i] >= 0 && indices[i] < dst.size(),
            "Host scatter index is out of range");
    dst[indices[i]] = src[i];
  }
}

#if !defined(FEMX_HAS_CUDA)
namespace
{
[[noreturn]] void cudaUnavailable()
{
  throw std::runtime_error(
      "femx was built without the CUDA execution backend");
}
} // namespace

void VectorHandler<CudaCsrBackend>::copy(DeviceConstVectorView,
                                         DeviceVectorView) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::copy(DeviceConstVectorView,
                                         DeviceVector&) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::copy(HostConstVectorView,
                                         DeviceVectorView) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::copy(HostConstVectorView,
                                         DeviceVector&) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::copy(DeviceConstVectorView,
                                         HostVectorView) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::copy(DeviceConstVectorView,
                                         HostVector&) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::zero(DeviceVectorView) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::axpby(Real,
                                          DeviceConstVectorView,
                                          Real,
                                          DeviceVectorView) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::gather(DeviceConstVectorView,
                                           DeviceConstIndexView,
                                           DeviceVectorView) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::scatter(DeviceConstVectorView,
                                            DeviceConstIndexView,
                                            DeviceVectorView) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::dot(DeviceConstVectorView,
                                        DeviceConstVectorView,
                                        DeviceVectorView) const
{
  cudaUnavailable();
}
#endif

} // namespace femx::linalg
