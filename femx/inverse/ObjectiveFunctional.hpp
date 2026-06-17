#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Scalar objective functional J(u, m). */
class ObjectiveFunctional
{
public:
  virtual ~ObjectiveFunctional() = default;

  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  virtual Real value(const Vector<Real>& state,
                     const Vector<Real>& params) const = 0;

  virtual void stateGrad(const Vector<Real>& state,
                         const Vector<Real>& params,
                         Vector<Real>&       out) const = 0;

  virtual void paramGrad(const Vector<Real>& state,
                         const Vector<Real>& params,
                         Vector<Real>&       out) const = 0;
};

} // namespace inverse
} // namespace femx
