#pragma once

#include <femx/common/Types.hpp>
#include <femx/inverse/Objective.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/**
 * @brief Sum of stationary objective terms with the same dimensions.
 *
 * SumObjective stores references to compatible terms and accumulates their
 * values, state gradients, and parameter gradients.
 */
class SumObjective final : public Objective
{
public:
  SumObjective(Index num_states, Index num_param);

  SumObjective& add(const Objective& term);

  Index numStates() const override;
  Index numParams() const override;

  Real value(const HostVector& state,
             const HostVector& prm) const override;

  void stateGrad(const HostVector& state,
                 const HostVector& prm,
                 HostVector&       out) const override;

  void paramGrad(const HostVector& state,
                 const HostVector& prm,
                 HostVector&       out) const override;

private:
  static void addInto(const HostVector& src,
                      HostVector&       out,
                      Index             size);

private:
  Index                   num_states_{0};
  Index                   num_param_{0};
  Array<const Objective*> terms_;
};

} // namespace inverse
} // namespace femx
