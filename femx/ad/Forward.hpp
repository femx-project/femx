#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace ad
{

class Forward
{
public:
  Forward() = default;

  explicit Forward(Real value)
    : value_(value)
  {
  }

  Forward(Real value, Index num_derivatives)
    : value_(value),
      derivative_(num_derivatives)
  {
  }

  static Forward variable(Real value, Index num_derivatives, Index index)
  {
    Forward out(value, num_derivatives);
    if (index < 0 || index >= num_derivatives)
    {
      throw std::runtime_error("Forward AD derivative index is out of range");
    }
    out.derivative_[index] = 1.0;
    return out;
  }

  Real value() const
  {
    return value_;
  }

  Index numDerivatives() const
  {
    return derivative_.size();
  }

  Real derivative(Index i) const
  {
    return derivative_[i];
  }

  void setDerivative(Index i, Real value)
  {
    derivative_[i] = value;
  }

  Forward& operator+=(const Forward& other)
  {
    syncSize(other);
    value_ += other.value_;
    for (Index i = 0; i < derivative_.size(); ++i)
    {
      if (i < other.derivative_.size())
      {
        derivative_[i] += other.derivative_[i];
      }
    }
    return *this;
  }

  Forward& operator-=(const Forward& other)
  {
    syncSize(other);
    value_ -= other.value_;
    for (Index i = 0; i < derivative_.size(); ++i)
    {
      if (i < other.derivative_.size())
      {
        derivative_[i] -= other.derivative_[i];
      }
    }
    return *this;
  }

  Forward& operator*=(Real scalar)
  {
    value_ *= scalar;
    for (Index i = 0; i < derivative_.size(); ++i)
    {
      derivative_[i] *= scalar;
    }
    return *this;
  }

  Forward& operator/=(Real scalar)
  {
    value_ /= scalar;
    for (Index i = 0; i < derivative_.size(); ++i)
    {
      derivative_[i] /= scalar;
    }
    return *this;
  }

private:
  void syncSize(const Forward& other)
  {
    if (derivative_.empty())
    {
      derivative_.resize(other.derivative_.size());
    }
    else if (!other.derivative_.empty()
             && derivative_.size() != other.derivative_.size())
    {
      throw std::runtime_error("Forward AD derivative size mismatch");
    }
  }

private:
  Real         value_{0.0};
  Vector<Real> derivative_;
};

inline Forward operator+(Forward lhs, const Forward& rhs)
{
  lhs += rhs;
  return lhs;
}

inline Forward operator+(Forward lhs, Real rhs)
{
  lhs += Forward(rhs, lhs.numDerivatives());
  return lhs;
}

inline Forward operator+(Real lhs, Forward rhs)
{
  rhs += Forward(lhs, rhs.numDerivatives());
  return rhs;
}

inline Forward operator-(Forward lhs, const Forward& rhs)
{
  lhs -= rhs;
  return lhs;
}

inline Forward operator-(Forward lhs, Real rhs)
{
  lhs -= Forward(rhs, lhs.numDerivatives());
  return lhs;
}

inline Forward operator-(Real lhs, const Forward& rhs)
{
  Forward out(lhs, rhs.numDerivatives());
  out -= rhs;
  return out;
}

inline Forward operator-(Forward value)
{
  value *= -1.0;
  return value;
}

inline Forward operator*(const Forward& lhs, const Forward& rhs)
{
  Forward out(lhs.value() * rhs.value(),
              std::max(lhs.numDerivatives(), rhs.numDerivatives()));
  for (Index i = 0; i < out.numDerivatives(); ++i)
  {
    const Real dl = i < lhs.numDerivatives() ? lhs.derivative(i) : 0.0;
    const Real dr = i < rhs.numDerivatives() ? rhs.derivative(i) : 0.0;
    out.setDerivative(i, dl * rhs.value() + lhs.value() * dr);
  }
  return out;
}

inline Forward operator*(Forward lhs, Real rhs)
{
  lhs *= rhs;
  return lhs;
}

inline Forward operator*(Real lhs, Forward rhs)
{
  rhs *= lhs;
  return rhs;
}

inline Forward operator/(const Forward& lhs, const Forward& rhs)
{
  Forward out(lhs.value() / rhs.value(),
              std::max(lhs.numDerivatives(), rhs.numDerivatives()));
  const Real denom = rhs.value() * rhs.value();
  for (Index i = 0; i < out.numDerivatives(); ++i)
  {
    const Real dl = i < lhs.numDerivatives() ? lhs.derivative(i) : 0.0;
    const Real dr = i < rhs.numDerivatives() ? rhs.derivative(i) : 0.0;
    out.setDerivative(i, (dl * rhs.value() - lhs.value() * dr) / denom);
  }
  return out;
}

inline Forward operator/(Forward lhs, Real rhs)
{
  lhs /= rhs;
  return lhs;
}

inline Forward operator/(Real lhs, const Forward& rhs)
{
  return Forward(lhs, rhs.numDerivatives()) / rhs;
}

inline Forward sqrt(const Forward& x)
{
  const Real root = std::sqrt(x.value());
  Forward    out(root, x.numDerivatives());
  if (root == 0.0)
  {
    return out;
  }
  for (Index i = 0; i < x.numDerivatives(); ++i)
  {
    out.setDerivative(i, 0.5 * x.derivative(i) / root);
  }
  return out;
}

inline Real sqrt(Real x)
{
  return std::sqrt(x);
}

inline Forward abs(const Forward& x)
{
  const Real sign = x.value() > 0.0 ? 1.0 : (x.value() < 0.0 ? -1.0 : 0.0);
  Forward    out(std::abs(x.value()), x.numDerivatives());
  for (Index i = 0; i < x.numDerivatives(); ++i)
  {
    out.setDerivative(i, sign * x.derivative(i));
  }
  return out;
}

inline Real abs(Real x)
{
  return std::abs(x);
}

inline Real value(Real x)
{
  return x;
}

inline Real value(const Forward& x)
{
  return x.value();
}

} // namespace ad
} // namespace femx
