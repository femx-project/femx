#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/inverse/TimeObjective.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace inverse
{

class TimeRegularization final : public TimeObjective
{
public:
  TimeRegularization(Index                   num_steps,
                     Index                   num_states,
                     Index                   num_levels,
                     Index                   block_size,
                     Real                    beta_diff,
                     Real                    beta_value = 0.0,
                     const HostVector<Real>& reference  = {});

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(const state::TimeTrajectory& traj,
             const HostVector<Real>&      prm) const override;

  void stateGrad(Index                        level,
                 const state::TimeTrajectory& traj,
                 const HostVector<Real>&      prm,
                 HostVector<Real>&            out) const override;

  void paramGrad(const state::TimeTrajectory& traj,
                 const HostVector<Real>&      prm,
                 HostVector<Real>&            out) const override;

private:
  Index index(Index level, Index comp) const;
  Real  centered(const HostVector<Real>& prm, Index level, Index comp) const;
  void  checkParamSize(const HostVector<Real>& prm) const;

private:
  Index            num_steps_{0};
  Index            num_states_{0};
  Index            num_levels_{0};
  Index            block_size_{0};
  Real             beta_diff_{0.0};
  Real             beta_value_{0.0};
  HostVector<Real> reference_;
};

} // namespace inverse
} // namespace femx
