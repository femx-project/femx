#pragma once

#include <vector>

#include <femx/core/Types.hpp>
#include <femx/solve/TimeStateTrajectory.hpp>
#include <femx/problem/TimeObjectiveFunctional.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace problem
{

/** @brief Non-owning sum of time-dependent objective terms. */
class SumTimeObjectiveFunctional final : public TimeObjectiveFunctional
{
public:
  SumTimeObjectiveFunctional(Index num_steps,
                             Index num_states,
                             Index num_prm);

  SumTimeObjectiveFunctional& add(const TimeObjectiveFunctional& term);

  Index numSteps() const override;

  Index numStates() const override;

  Index numParams() const override;

  Real value(const solve::TimeStateTrajectory& tr,
             const Vector<Real>&            prm) const override;

  void stateGrad(Index                          level,
                 const solve::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override;

  void paramGrad(const solve::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override;

private:
  static void resize(Vector<Real>& out,
                     Index         size);

  static void addInto(const Vector<Real>& input,
                      Vector<Real>&       out,
                      Index               size);

private:
  Index num_steps_{0};
  Index num_states_{0};
  Index num_prm_{0};

  std::vector<const TimeObjectiveFunctional*> terms_;
};

} // namespace problem
} // namespace femx
