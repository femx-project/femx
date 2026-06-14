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

  virtual index_type numStates() const       = 0;
  virtual index_type numParams() const       = 0;
  virtual index_type numObservations() const = 0;

  virtual void observe(const Vector& state,
                       const Vector& params,
                       Vector&       out) const = 0;

  virtual void applyStateJac(const Vector& state,
                             const Vector& params,
                             const Vector& dir,
                             Vector&       out) const = 0;

  virtual void applyStateJacT(const Vector& state,
                              const Vector& params,
                              const Vector& dir,
                              Vector&       out) const = 0;

  virtual void applyParamJac(const Vector& state,
                             const Vector& params,
                             const Vector& dir,
                             Vector&       out) const = 0;

  virtual void applyParamJacT(const Vector& state,
                              const Vector& params,
                              const Vector& dir,
                              Vector&       out) const = 0;
};

} // namespace inverse
} // namespace femx
