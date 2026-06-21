#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/solve/TimeTrajectory.hpp>

namespace femx
{
namespace problem
{

class TimeRegularization final : public TimeObjective
{
public:
  TimeRegularization(Index               num_steps,
                     Index               num_states,
                     Index               num_levels,
                     Index               block_size,
                     Real                beta_difference,
                     Real                beta_value = 0.0,
                     const Vector<Real>& reference  = {});

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(const solve::TimeTrajectory& tr,
             const Vector<Real>&          prm) const override;

  void stateGrad(Index                        level,
                 const solve::TimeTrajectory& tr,
                 const Vector<Real>&          prm,
                 Vector<Real>&                out) const override;

  void paramGrad(const solve::TimeTrajectory& tr,
                 const Vector<Real>&          prm,
                 Vector<Real>&                out) const override;

private:
  Index       index(Index level, Index component) const;
  Real        centered(const Vector<Real>& prm, Index level, Index component) const;
  void        checkParamSize(const Vector<Real>& prm) const;
  static void resize(Vector<Real>& out, Index size);

private:
  Index        num_steps_{0};
  Index        num_states_{0};
  Index        num_levels_{0};
  Index        block_size_{0};
  Real         beta_difference_{0.0};
  Real         beta_value_{0.0};
  Vector<Real> reference_;
};

} // namespace problem
} // namespace femx
