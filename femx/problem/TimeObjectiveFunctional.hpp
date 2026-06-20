#pragma once

#include <femx/core/Types.hpp>
#include <femx/solve/TimeStateTrajectory.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace problem
{

/** @brief Objective functional J(u_0, ..., u_N, m) for time trajectories. */
class TimeObjectiveFunctional
{
public:
  virtual ~TimeObjectiveFunctional() = default;

  virtual Index numSteps() const  = 0;
  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;

  virtual Real value(const solve::TimeStateTrajectory& tr,
                     const Vector<Real>&            prm) const = 0;

  virtual void stateGrad(Index                          level,
                         const solve::TimeStateTrajectory& tr,
                         const Vector<Real>&            prm,
                         Vector<Real>&                  out) const = 0;

  virtual void paramGrad(const solve::TimeStateTrajectory& tr,
                         const Vector<Real>&            prm,
                         Vector<Real>&                  out) const = 0;
};

} // namespace problem
} // namespace femx
