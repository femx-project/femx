#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Observation map y = H(u, m) used by inverse objectives. */
class ObservationOperator
{
public:
  virtual ~ObservationOperator() = default;

  virtual Index numStates() const       = 0;
  virtual Index numParams() const       = 0;
  virtual Index numObservations() const = 0;

  virtual void observe(const Vector<Real>& state,
                       const Vector<Real>& params,
                       Vector<Real>&       out) const = 0;

  virtual void applyStateJac(const Vector<Real>& state,
                             const Vector<Real>& params,
                             const Vector<Real>& dir,
                             Vector<Real>&       out) const = 0;

  virtual void applyStateJacT(const Vector<Real>& state,
                              const Vector<Real>& params,
                              const Vector<Real>& dir,
                              Vector<Real>&       out) const = 0;

  virtual void applyParamJac(const Vector<Real>& state,
                             const Vector<Real>& params,
                             const Vector<Real>& dir,
                             Vector<Real>&       out) const = 0;

  virtual void applyParamJacT(const Vector<Real>& state,
                              const Vector<Real>& params,
                              const Vector<Real>& dir,
                              Vector<Real>&       out) const = 0;
};

} // namespace inverse
} // namespace femx
