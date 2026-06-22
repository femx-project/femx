#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace problem
{

/** @brief Objective functional J(u_0, ..., u_N, m) for time trajectories. */
class TimeObjective
{
public:
  virtual ~TimeObjective() = default;

  virtual Index numSteps() const  = 0;
  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  virtual Real value(const state::TimeTrajectory& tr,
                     const Vector<Real>&          prm) const = 0;

  virtual void stateGrad(Index                        level,
                         const state::TimeTrajectory& tr,
                         const Vector<Real>&          prm,
                         Vector<Real>&                out) const = 0;

  virtual void paramGrad(const state::TimeTrajectory& tr,
                         const Vector<Real>&          prm,
                         Vector<Real>&                out) const = 0;
};

} // namespace problem
} // namespace femx
