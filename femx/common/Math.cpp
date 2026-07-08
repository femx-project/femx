#include <cmath>
#include <stdexcept>

#include <femx/common/Math.hpp>

using namespace std;

namespace femx
{

Real dot(const Vector<Real>& x, const Vector<Real>& y)
{
  if (x.size() != y.size())
  {
    throw runtime_error("dot received incompatible vectors");
  }

  Real value = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    value += x[i] * y[i];
  }
  return value;
}

Real squaredNorm(const Vector<Real>& x)
{
  return dot(x, x);
}

Real norm(const Vector<Real>& x)
{
  return sqrt(squaredNorm(x));
}

Real rmse(const Vector<Real>& x, const Vector<Real>& y)
{
  if (x.size() != y.size())
  {
    throw runtime_error("rmse received incompatible vectors");
  }

  Real sum = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    const Real diff  = x[i] - y[i];
    sum             += diff * diff;
  }
  return sqrt(sum / x.size());
}

Vector<Real> difference(const Vector<Real>& x, const Vector<Real>& y)
{
  if (x.size() != y.size())
  {
    throw runtime_error("difference received incompatible vectors");
  }

  Vector<Real> diff(x.size());
  for (Index i = 0; i < x.size(); ++i)
  {
    diff[i] = x[i] - y[i];
  }
  return diff;
}

Real dot(const Point3& x, const Point3& y)
{
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
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

Real squaredNorm(const Point3& x)
{
  return dot(x, x);
}

Real norm(const Point3& x)
{
  return sqrt(squaredNorm(x));
}

Point3 unit(const Point3& x)
{
  const Real len = norm(x);
  if (len <= 0.0)
  {
    throw runtime_error("unit received zero vector");
  }
  return {x[0] / len, x[1] / len, x[2] / len};
}

Real sqDist(const Point3& x, const Point3& y)
{
  return squaredNorm(difference(x, y));
}

Real distance(const Point3& x, const Point3& y)
{
  return sqrt(sqDist(x, y));
}

Real triArea(const Point3& a, const Point3& b, const Point3& c)
{
  return 0.5 * norm(cross(difference(b, a), difference(c, a)));
}

Real radialSq(const Point3& point, const Point3& origin, const Point3& axis)
{
  const Point3 delta     = difference(point, origin);
  const Point3 axis_unit = unit(axis);
  const Real   axial     = dot(delta, axis_unit);
  const Real   radial    = squaredNorm(delta) - axial * axial;
  return radial > 0.0 ? radial : 0.0;
}

} // namespace femx
