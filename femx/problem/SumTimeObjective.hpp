#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace problem
{

/** @brief Non-owning sum of time-dependent objective terms. */
class SumTimeObjective final : public TimeObjective
{
public:
  SumTimeObjective(Index nt, Index nst, Index nprm);

  SumTimeObjective& add(const TimeObjective& term);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(const state::TimeTrajectory& tr,
             const Vector<Real>&          prm) const override;

  void stateGrad(Index                        level,
                 const state::TimeTrajectory& tr,
                 const Vector<Real>&          prm,
                 Vector<Real>&                out) const override;

  void paramGrad(const state::TimeTrajectory& tr,
                 const Vector<Real>&          prm,
                 Vector<Real>&                out) const override;

private:
  static void checkSize(const Vector<Real>& value, Index exp);
  static void addInto(const Vector<Real>& input, Vector<Real>& out, Index size);

private:
  Index                        nt_{0};
  Index                        nst_{0};
  Index                        nprm_{0};
  Vector<const TimeObjective*> terms_;
};

} // namespace problem
} // namespace femx
