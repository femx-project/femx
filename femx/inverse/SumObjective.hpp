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

  Real value(const HostVector<Real>& state,
             const HostVector<Real>& prm) const override;

  void stateGrad(const HostVector<Real>& state,
                 const HostVector<Real>& prm,
                 HostVector<Real>&       out) const override;

  void paramGrad(const HostVector<Real>& state,
                 const HostVector<Real>& prm,
                 HostVector<Real>&       out) const override;

private:
  static void addInto(const HostVector<Real>& src,
                      HostVector<Real>&       out,
                      Index                   size);

private:
  Index                        num_states_{0};
  Index                        num_param_{0};
  HostVector<const Objective*> terms_;
};

} // namespace inverse
} // namespace femx
