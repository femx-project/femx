#pragma once

#include <cmath>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

struct LinearInterpolation
{
  Index lower        = 0;
  Index upper        = 0;
  Real  upper_weight = 0.0;

  Real lowerWeight() const
  {
    return 1.0 - upper_weight;
  }

  Real upperWeight() const
  {
    return upper_weight;
  }

  bool hasUpper() const
  {
    return upper != lower && upperWeight() != 0.0;
  }

  bool isValid() const
  {
    return lower >= 0 && upper >= lower && upper_weight >= 0.0
           && upper_weight <= 1.0 && std::isfinite(upper_weight)
           && (upper != lower || upper_weight == 0.0);
  }

  template <class F>
  void forEachWeight(F&& f) const
  {
    f(lower, lowerWeight());
    if (hasUpper())
    {
      f(upper, upperWeight());
    }
  }
};

inline LinearInterpolation linearInterpolation(const Vector<Real>& pts,
                                               Real                x)
{
  if (pts.empty() || !std::isfinite(x))
  {
    throw std::runtime_error("linearInterpolation received invalid inputs");
  }
  if (pts.size() == 1 || x <= pts.front())
  {
    return {0, 0, 0.0};
  }

  const Index last = pts.size() - 1;
  if (x >= pts.back())
  {
    return {last, last, 0.0};
  }

  for (Index level = 0; level < last; ++level)
  {
    const Real lo = pts[level];
    const Real hi = pts[level + 1];
    if (hi <= lo || !std::isfinite(lo) || !std::isfinite(hi))
    {
      throw std::runtime_error(
          "linearInterpolation points must be finite and increasing");
    }
    if (x >= lo && x <= hi)
    {
      return {level, level + 1, (x - lo) / (hi - lo)};
    }
  }

  throw std::runtime_error("failed to bracket interpolation point");
}

} // namespace femx
