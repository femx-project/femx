#include <cmath>
#include <stdexcept>

#include <femx/common/Math.hpp>

namespace femx
{

Real dot(const HostVector<Real>& x, const HostVector<Real>& y)
{
  if (x.size() != y.size())
  {
    throw std::runtime_error("dot received incompatible vectors");
  }

  Real result = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    result += x[i] * y[i];
  }

  return result;
}

Real dot(const Point3& x, const Point3& y)
{
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

Real squaredNorm(const HostVector<Real>& x)
{
  return dot(x, x);
}

Real squaredNorm(const Point3& x)
{
  return dot(x, x);
}

Real norm(const HostVector<Real>& x)
{
  return std::sqrt(squaredNorm(x));
}

Real norm(const Point3& x)
{
  return std::sqrt(squaredNorm(x));
}

Real rootMeanSquareError(const HostVector<Real>& x,
                         const HostVector<Real>& y)
{
  if (x.size() != y.size())
  {
    throw std::runtime_error(
        "rootMeanSquareError received incompatible vectors");
  }

  Real sum = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    const Real diff  = x[i] - y[i];
    sum             += diff * diff;
  }

  return std::sqrt(sum / x.size());
}

HostVector<Real> difference(const HostVector<Real>& x,
                            const HostVector<Real>& y)
{
  if (x.size() != y.size())
  {
    throw std::runtime_error("difference received incompatible vectors");
  }

  HostVector<Real> diff(x.size());
  for (Index i = 0; i < x.size(); ++i)
  {
    diff[i] = x[i] - y[i];
  }

  return diff;
}

Point3 difference(const Point3& x, const Point3& y)
{
  return {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
}

Point3 cross(const Point3& x, const Point3& y)
{
  return {x[1] * y[2] - x[2] * y[1],
          x[2] * y[0] - x[0] * y[2],
          x[0] * y[1] - x[1] * y[0]};
}

Point3 normalized(const Point3& x)
{
  const Real length = norm(x);
  if (length <= 0.0)
  {
    throw std::runtime_error("cannot normalize zero vector");
  }
  return {x[0] / length, x[1] / length, x[2] / length};
}

Real squaredDistance(const Point3& x, const Point3& y)
{
  return squaredNorm(difference(x, y));
}

Real distance(const Point3& x, const Point3& y)
{
  return std::sqrt(squaredDistance(x, y));
}

Real triangleArea(const Point3& a, const Point3& b, const Point3& c)
{
  return 0.5 * norm(cross(difference(b, a), difference(c, a)));
}

Real squaredDistanceToLine(const Point3& point,
                           const Point3& line_point,
                           const Point3& line_direction)
{
  const Point3 delta            = difference(point, line_point);
  const Point3 unit_direction   = normalized(line_direction);
  const Real   projection       = dot(delta, unit_direction);
  const Real   squared_distance = squaredNorm(delta) - projection * projection;

  return squared_distance > 0.0 ? squared_distance : 0.0;
}

} // namespace femx
