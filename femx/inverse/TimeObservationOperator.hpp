#pragma once

#include <femx/common/Types.hpp>

namespace femx
{
namespace inverse
{

/**
 * @brief Observation map y_l = H_l(u_l, m) for time objectives.
 *
 * Implementations evaluate observations and provide Jacobian and
 * transpose-Jacobian products with respect to state and parameter variables.
 */
class TimeObservationOperator
{
public:
  virtual ~TimeObservationOperator() = default;

  virtual Index numSteps() const        = 0;
  virtual Index numStates() const       = 0;
  virtual Index numParams() const       = 0;
  virtual Index numObservations() const = 0;

  virtual void observe(Index             level,
                       const HostVector& state,
                       const HostVector& prm,
                       HostVector&       out) const = 0;

  virtual void applyStateJac(Index             level,
                             const HostVector& state,
                             const HostVector& prm,
                             const HostVector& dir,
                             HostVector&       out) const = 0;

  virtual void applyStateJacT(Index             level,
                              const HostVector& state,
                              const HostVector& prm,
                              const HostVector& dir,
                              HostVector&       out) const = 0;

  virtual void applyParamJac(Index             level,
                             const HostVector& state,
                             const HostVector& prm,
                             const HostVector& dir,
                             HostVector&       out) const = 0;

  virtual void applyParamJacT(Index             level,
                              const HostVector& state,
                              const HostVector& prm,
                              const HostVector& dir,
                              HostVector&       out) const = 0;
};

} // namespace inverse
} // namespace femx
