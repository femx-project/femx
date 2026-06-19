#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Reduced objective F(m) seen by optimizers. */
class ReducedFunctional
{
public:
  virtual ~ReducedFunctional() = default;

  virtual Index numParams() const = 0;

  virtual Real value(const Vector<Real>& prm) = 0;

  virtual void grad(const Vector<Real>& prm, Vector<Real>& out) = 0;

  virtual Real valueGrad(const Vector<Real>& prm, Vector<Real>& grad_out);
};

} // namespace inverse
} // namespace femx
