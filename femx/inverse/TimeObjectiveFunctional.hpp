#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Objective functional J(u_0, ..., u_N, m) for time trajectories. */
class TimeObjectiveFunctional
{
public:
  virtual ~TimeObjectiveFunctional() = default;

  virtual Index numSteps() const  = 0;
  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  virtual Real value(const eq::TimeStateTrajectory& tr,
                     const Vector<Real>&            prm) const = 0;

  virtual void stateGrad(Index                          level,
                         const eq::TimeStateTrajectory& tr,
                         const Vector<Real>&            prm,
                         Vector<Real>&                  out) const = 0;

  virtual void paramGrad(const eq::TimeStateTrajectory& tr,
                         const Vector<Real>&            prm,
                         Vector<Real>&                  out) const = 0;
};

} // namespace inverse
} // namespace femx
