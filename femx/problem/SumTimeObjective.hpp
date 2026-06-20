#pragma once

#include <vector>

#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/solve/TimeTrajectory.hpp>

namespace femx
{
namespace problem
{

/** @brief Non-owning sum of time-dependent objective terms. */
class SumTimeObjective final : public TimeObjective
{
public:
  SumTimeObjective(Index num_steps, Index num_states, Index num_prm);

  SumTimeObjective& add(const TimeObjective& term);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(const solve::TimeTrajectory& tr,
             const Vector<Real>& prm) const override;

  void stateGrad(Index level,
                 const solve::TimeTrajectory& tr,
                 const Vector<Real>& prm,
                 Vector<Real>& out) const override;

  void paramGrad(const solve::TimeTrajectory& tr,
                 const Vector<Real>& prm,
                 Vector<Real>& out) const override;

private:
  static void resize(Vector<Real>& out, Index size);
  static void addInto(const Vector<Real>& input, Vector<Real>& out, Index size);

private:
  Index num_steps_{0};
  Index num_states_{0};
  Index num_prm_{0};
  std::vector<const TimeObjective*> terms_;
};

} // namespace problem
} // namespace femx
