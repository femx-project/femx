#pragma once

#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace problem
{

/** @brief Nonlinear state equation represented by a residual R(u, m) = 0. */
class ResidualEquation
{
public:
  virtual ~ResidualEquation() = default;

  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;
  virtual Index numRes() const    = 0;

  virtual void res(const Vector<Real>& state,
                   const Vector<Real>& prm,
                   Vector<Real>&       out) const = 0;

  virtual void applyStateJac(const Vector<Real>& state,
                             const Vector<Real>& prm,
                             const Vector<Real>& dir,
                             Vector<Real>&       out) const = 0;

  virtual void applyStateJacT(const Vector<Real>& state,
                              const Vector<Real>& prm,
                              const Vector<Real>& lambda,
                              Vector<Real>&       out) const = 0;

  virtual void applyParamJac(const Vector<Real>& state,
                             const Vector<Real>& prm,
                             const Vector<Real>& dir,
                             Vector<Real>&       out) const = 0;

  virtual void applyParamJacT(const Vector<Real>& state,
                              const Vector<Real>& prm,
                              const Vector<Real>& lambda,
                              Vector<Real>&       out) const = 0;
};

} // namespace problem
} // namespace femx
