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

  virtual Real value(const Vector& state,
                     const Vector& params) const = 0;

  virtual void stateGrad(const Vector& state,
                         const Vector& params,
                         Vector&       out) const = 0;

  virtual void paramGrad(const Vector& state,
                         const Vector& params,
                         Vector&       out) const = 0;
};

} // namespace inverse
} // namespace femx
