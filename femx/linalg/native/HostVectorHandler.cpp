#include <femx/linalg/native/HostVectorHandler.hpp>

namespace femx::linalg
{

void HostVectorHandler::zero(HostVectorView<Real> vals) const
{
  std::fill(vals.begin(), vals.end(), Real{});
}

void HostVectorHandler::axpby(Real                       a,
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

Real HostVectorHandler::dot(HostVectorView<const Real> x,
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

Real HostVectorHandler::squaredNorm(HostVectorView<const Real> x) const
{
  return dot(x, x);
}

void HostVectorHandler::gather(HostVectorView<const Real>  src,
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

void HostVectorHandler::scatter(HostVectorView<const Real>  src,
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

} // namespace femx::linalg
