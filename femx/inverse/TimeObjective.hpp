#pragma once

#include <femx/common/Types.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace inverse
{

/**
 * @brief Objective functional J(u_0, ..., u_N, m) for time trajectories.
 *
 * Implementations provide the scalar value plus gradients with respect to
 * each time level and the shared parameter vector.
 */
class TimeObjective
{
public:
  virtual ~TimeObjective() = default;

  virtual Index numSteps() const  = 0;
  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  virtual Real value(const state::TimeTrajectory& tr,
                     const HostVector&            prm) const = 0;

  virtual void stateGrad(Index                        level,
                         const state::TimeTrajectory& tr,
                         const HostVector&            prm,
                         HostVector&                  out) const = 0;

  virtual void paramGrad(const state::TimeTrajectory& tr,
                         const HostVector&            prm,
                         HostVector&                  out) const = 0;
};

} // namespace inverse
} // namespace femx
