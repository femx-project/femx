#include <stdexcept>

#include <femx/linalg/handler/VectorHandler.hpp>

namespace femx::linalg
{

void VectorHandler<HostCsrBackend>::zero(HostVectorView<Real> vals) const
{
  std::fill(vals.begin(), vals.end(), Real{});
}

void VectorHandler<HostCsrBackend>::axpby(Real                       a,
                                          HostVectorView<const Real> x,
                                          Real                       b,
                                          HostVectorView<Real>       y) const
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

Real VectorHandler<HostCsrBackend>::dot(HostVectorView<const Real> x,
                                        HostVectorView<const Real> y) const
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
    HostVectorView<const Real> x) const
{
  return dot(x, x);
}

void VectorHandler<HostCsrBackend>::gather(HostVectorView<const Real>  src,
                                           HostVectorView<const Index> indices,
                                           HostVectorView<Real>        dst) const
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

void VectorHandler<HostCsrBackend>::scatter(HostVectorView<const Real>  src,
                                            HostVectorView<const Index> indices,
                                            HostVectorView<Real>        dst) const
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

void VectorHandler<CudaCsrBackend>::copy(DeviceVectorView<const Real>,
                                         DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::copy(DeviceVectorView<const Real>,
                                         DeviceVector<Real>&) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::copy(HostVectorView<const Real>,
                                         DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::copy(HostVectorView<const Real>,
                                         DeviceVector<Real>&) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::copy(DeviceVectorView<const Real>,
                                         HostVectorView<Real>) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::copy(DeviceVectorView<const Real>,
                                         HostVector<Real>&) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::zero(DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::axpby(Real,
                                          DeviceVectorView<const Real>,
                                          Real,
                                          DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::gather(DeviceVectorView<const Real>,
                                           DeviceVectorView<const Index>,
                                           DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::scatter(DeviceVectorView<const Real>,
                                            DeviceVectorView<const Index>,
                                            DeviceVectorView<Real>) const
{
  cudaUnavailable();
}

void VectorHandler<CudaCsrBackend>::dot(DeviceVectorView<const Real>,
                                        DeviceVectorView<const Real>,
                                        DeviceVectorView<Real>) const
{
  cudaUnavailable();
}
#endif

} // namespace femx::linalg
