#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace problem
{

/** @brief Observation map y_l = H_l(u_l, m) for time objectives. */
class TimeObservationOperator
{
public:
  virtual ~TimeObservationOperator() = default;

  virtual Index numSteps() const        = 0;
  virtual Index numStates() const       = 0;
  virtual Index numParams() const       = 0;
  virtual Index numObservations() const = 0;

  virtual void observe(Index               level,
                       const Vector<Real>& state,
                       const Vector<Real>& prm,
                       Vector<Real>&       out) const = 0;

  virtual void applyStateJac(Index               level,
                             const Vector<Real>& state,
                             const Vector<Real>& prm,
                             const Vector<Real>& dir,
                             Vector<Real>&       out) const = 0;

  virtual void applyStateJacT(Index               level,
                              const Vector<Real>& state,
                              const Vector<Real>& prm,
                              const Vector<Real>& dir,
                              Vector<Real>&       out) const = 0;

  virtual void applyParamJac(Index               level,
                             const Vector<Real>& state,
                             const Vector<Real>& prm,
                             const Vector<Real>& dir,
                             Vector<Real>&       out) const = 0;

  virtual void applyParamJacT(Index               level,
                              const Vector<Real>& state,
                              const Vector<Real>& prm,
                              const Vector<Real>& dir,
                              Vector<Real>&       out) const = 0;
};

} // namespace problem
} // namespace femx
