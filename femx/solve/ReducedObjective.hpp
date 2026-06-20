#pragma once

// Compatibility reduced-functional interface. Prefer
// <femx/solve/ReducedObjective.hpp> for new adjoint reduced objectives.
#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace solve
{

/** @brief Reduced objective F(m) seen by optimizers. */
class ReducedObjective
{
public:
  virtual ~ReducedObjective() = default;

  virtual Index numParams() const = 0;

  virtual Real value(const Vector<Real>& prm) = 0;

  virtual void grad(const Vector<Real>& prm, Vector<Real>& out) = 0;

  virtual Real valueGrad(const Vector<Real>& prm, Vector<Real>& grad_out);
};

} // namespace solve
} // namespace femx
