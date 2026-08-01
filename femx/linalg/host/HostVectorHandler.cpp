#include <femx/linalg/host/HostVectorHandler.hpp>

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

void HostVectorHandler::gather(HostVectorView<const Real>  src,
                               HostVectorView<const Index> idx,
                               HostVectorView<Real>        dst) const
{
  require(idx.size() == dst.size(),
          "Host gather output size mismatch");
  require(!femx::detail::overlaps(src, dst),
          "Host gather does not support aliased vectors");
  for (Index i = 0; i < idx.size(); ++i)
  {
    require(idx[i] >= 0 && idx[i] < src.size(),
            "Host gather index is out of range");
    dst[i] = src[idx[i]];
  }
}

void HostVectorHandler::scatter(HostVectorView<const Real>  src,
                                HostVectorView<const Index> idx,
                                HostVectorView<Real>        dst) const
{
  require(src.size() == idx.size(),
          "Host scatter input size mismatch");
  require(!femx::detail::overlaps(src, dst),
          "Host scatter does not support aliased vectors");
  for (Index i = 0; i < idx.size(); ++i)
  {
    require(idx[i] >= 0 && idx[i] < dst.size(),
            "Host scatter index is out of range");
    dst[idx[i]] = src[i];
  }
}

} // namespace femx::linalg
