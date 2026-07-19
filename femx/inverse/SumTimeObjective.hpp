#pragma once

#include <femx/common/Types.hpp>
#include <femx/inverse/TimeObjective.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace inverse
{

/**
 * @brief Non-owning sum of time-dependent objective terms.
 *
 * SumTimeObjective stores references to compatible terms and accumulates
 * their values and gradients without owning the terms.
 */
class SumTimeObjective final : public TimeObjective
{
public:
  SumTimeObjective(Index num_steps, Index num_states, Index num_param);

  /** @brief Append a compatible non-owning objective term. */
  SumTimeObjective& add(const TimeObjective& term);

  /** @brief Return the objective terms in insertion order. */
  const Array<const TimeObjective*>& terms() const noexcept;

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(const state::TimeTrajectory& tr,
             const HostVector&            prm) const override;

  void stateGrad(Index                        level,
                 const state::TimeTrajectory& tr,
                 const HostVector&            prm,
                 HostVector&                  out) const override;

  void paramGrad(const state::TimeTrajectory& tr,
                 const HostVector&            prm,
                 HostVector&                  out) const override;

private:
  static void checkSize(const HostVector& val, Index exp);
  static void addInto(const HostVector& src, HostVector& out, Index size);

private:
  Index                       num_steps_{0};
  Index                       num_states_{0};
  Index                       num_param_{0};
  Array<const TimeObjective*> terms_;
};

} // namespace inverse
} // namespace femx
