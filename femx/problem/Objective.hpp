#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace problem
{

/** @brief Scalar objective functional J(u, m). */
class Objective
{
public:
  virtual ~Objective() = default;

  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  virtual Real value(const Vector<Real>& state,
                     const Vector<Real>& prm) const = 0;

  virtual void stateGrad(const Vector<Real>& state,
                         const Vector<Real>& prm,
                         Vector<Real>&       out) const = 0;

  virtual void paramGrad(const Vector<Real>& state,
                         const Vector<Real>& prm,
                         Vector<Real>&       out) const = 0;
};

} // namespace problem
} // namespace femx
