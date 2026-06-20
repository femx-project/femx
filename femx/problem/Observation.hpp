#pragma once

#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>

namespace femx
{
namespace problem
{

/** @brief Observation map y = H(u, m) used by objective functions. */
class Observation
{
public:
  virtual ~Observation() = default;

  virtual Index numStates() const       = 0;
  virtual Index numParams() const       = 0;
  virtual Index numObservations() const = 0;

  virtual void observe(const Vector<Real>& state,
                       const Vector<Real>& prm,
                       Vector<Real>&       out) const = 0;

  virtual void applyStateJac(const Vector<Real>& state,
                             const Vector<Real>& prm,
                             const Vector<Real>& dir,
                             Vector<Real>&       out) const = 0;

  virtual void applyStateJacT(const Vector<Real>& state,
                              const Vector<Real>& prm,
                              const Vector<Real>& dir,
                              Vector<Real>&       out) const = 0;

  virtual void applyParamJac(const Vector<Real>& state,
                             const Vector<Real>& prm,
                             const Vector<Real>& dir,
                             Vector<Real>&       out) const = 0;

  virtual void applyParamJacT(const Vector<Real>& state,
                              const Vector<Real>& prm,
                              const Vector<Real>& dir,
                              Vector<Real>&       out) const = 0;
};

} // namespace problem
} // namespace femx
