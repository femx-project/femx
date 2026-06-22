#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace problem
{

class TimeRegularization final : public TimeObjective
{
public:
  TimeRegularization(Index               nt,
                     Index               nst,
                     Index               nl,
                     Index               block_size,
                     Real                beta_difference,
                     Real                beta_value = 0.0,
                     const Vector<Real>& reference  = {});

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
  Index index(Index level, Index comp) const;
  Real  centered(const Vector<Real>& prm, Index level, Index comp) const;
  void  checkParamSize(const Vector<Real>& prm) const;

private:
  Index        nt_{0};
  Index        nst_{0};
  Index        nl_{0};
  Index        block_size_{0};
  Real         beta_difference_{0.0};
  Real         beta_value_{0.0};
  Vector<Real> reference_;
};

} // namespace problem
} // namespace femx
